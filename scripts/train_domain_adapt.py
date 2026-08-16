# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "unsloth",
#   "datasets",
#   "huggingface_hub",
# ]
# ///
# transformers/peft/accelerate/bitsandbytes deliberately NOT pinned here:
# unsloth declares its own compatible version ranges for those, and pinning
# them ourselves (as earlier plain-transformers attempts on this same script
# did) previously resolved to a transformers version too OLD for this
# Unsloth release (ModuleNotFoundError: transformers.models.qwen3 -- a model
# family Unsloth's install expects that 4.46.3 predates). Let uv's resolver
# satisfy unsloth's own constraints instead of guessing a compatible pin.
"""QLoRA domain-adaptation fine-tune of Phi-3.5 Mini on the broad lipid/
spectroscopy abstract corpus (PROGRESS.md Step 4).

Objective, decided and documented here (not left implicit, per the task):
**next-token / causal-LM domain adaptation on raw title+abstract text**, not
an instruction format. There is no specific downstream instruction task this
model needs to learn -- the goal is domain fluency (vocabulary, phrasing,
lipid/Raman terminology) to feed into the existing grounded-retrieval +
citation-checking pipeline, which is a separate concern from what the model
learned during fine-tuning. `scripts/finetune_phi3.py` already covers the
instruction-tuned (peaks + evidence -> interpretation) case on the small
retrieval corpus; this is deliberately a different, complementary objective
on the different, much larger corpus (see docs/solutions.md, 2026-08-14
"Fine-tuning corpus kept fully separate from the retrieval corpus").

Runs as an `hf jobs uv run` job (GPU required, Turing/sm_75+ for Unsloth --
A10G/Ampere qualifies; the P100 Kaggle defaults to does not):

    hf jobs uv run scripts/train_domain_adapt.py \
        --flavor a10g-large --timeout 6h \
        --secrets HF_TOKEN \
        -- --push-to-hub srikarjy025/lipidos-phi3-domain-adapt

Uses Unsloth's `FastLanguageModel` instead of plain transformers +
BitsAndBytesConfig for the 4-bit load and LoRA wrap -- same QLoRA config
(rank/alpha/dropout/target_modules) as the plain-transformers version that
was mid-training successfully on HF Jobs before this swap, just with
Unsloth's patched attention/backward kernels for faster throughput on the
same A10G. See docs/solutions.md for why Kaggle+Unsloth was tried and
abandoned (Kaggle's API can't select a Unsloth-compatible GPU) in favor of
this.

Evaluates held-out perplexity for BASE and FINE-TUNED model on the same
val set, printed at the end so the acceptance criterion (measured, not
estimated, both numbers) has a citable log.
"""

import argparse
import json
import math
import os
import sys

# unsloth must be imported before transformers/peft to apply its patches --
# importing it after (as an earlier version of this script did) triggers
# "Unsloth should be imported before transformers, peft" and silently loses
# some of the speedup this whole platform switch was for.
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

CORPUS_REPO = "srikarjy025/lipidos-finetune-corpus"
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
# Checkpoint persistence lives on the Hub, NOT the /data bucket mount --
# confirmed live that /data does NOT survive across separate `hf jobs uv run`
# invocations (a checkpoint written by one job was invisible to a follow-up
# job trying to read it back), unlike a Hub repo, which is genuinely global.
# Two live runs died to exit code 143 (SIGTERM, no Python traceback --
# GPU-capacity preemption, not an app error) after 315 and ~1060 steps with
# zero recoverable progress under the old /data-only design.
CHECKPOINT_REPO = "srikarjy025/lipidos-domain-adapt-checkpoints"
# Measured on 500 random train examples: mean 239 words / ~310 tokens,
# p95 372 words / ~480 tokens. 512 covers the p95 case with room to spare
# without wasting compute padding short abstracts out to 1024.
MAX_LEN = 512


def load_split(repo: str, filename: str):
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    return load_dataset("json", data_files=path, split="train")


class HubCheckpointCallback(TrainerCallback):
    """Pushes each local checkpoint to a fixed "latest/" path in a Hub repo
    right after Trainer writes it to disk. Overwriting one fixed path (not a
    new path per step) keeps repo size bounded and keeps resume-lookup
    trivial -- Trainer checkpoint filenames (trainer_state.json, optimizer.pt,
    etc.) are the same regardless of step number, so each push cleanly
    replaces the previous one's contents rather than accumulating them.
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.api = HfApi()

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return
        print(f"Pushing checkpoint at step {state.global_step} to {self.repo_id}...")
        self.api.upload_folder(
            repo_id=self.repo_id, repo_type="model",
            folder_path=ckpt_dir, path_in_repo="latest",
            commit_message=f"checkpoint step {state.global_step}",
        )


def tokenize_fn(tokenizer):
    def fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=MAX_LEN,
                         padding=False)
        return out
    return fn


@torch.inference_mode()
def eval_perplexity(model, tokenizer, val_ds, batch_size=4) -> float:
    """Mean token-level cross-entropy -> perplexity over the held-out set."""
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(val_ds), batch_size):
        batch = [val_ds[j] for j in range(i, min(i + batch_size, len(val_ds)))]
        enc = collator(batch).to(model.device)
        out = model(**enc)
        # out.loss is already mean over non-masked tokens for this batch;
        # weight by token count so batches of different lengths combine correctly.
        n_tok = (enc["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    mean_loss = total_loss / total_tokens
    return math.exp(mean_loss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1,
                     help="-1 (default) trains the full epoch(s) over all 105,665 "
                          "examples; set a positive value to cap wall-clock time instead")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--output-dir", type=str,
                     default="/data/lipidos-phi3-domain-adapt",
                     help="local scratch dir for Trainer's own checkpoint "
                          "writes within a single job run. NOT relied on for "
                          "cross-job persistence -- confirmed live that /data "
                          "does not survive between separate `hf jobs uv run` "
                          "invocations despite being bucket-backed. Real "
                          "resume persistence is CHECKPOINT_REPO on the Hub.")
    ap.add_argument("--push-to-hub", type=str, default=None,
                     help="HF Hub repo id to push the adapter to")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available.", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading corpus splits from {CORPUS_REPO}...")
    train_ds = load_split(CORPUS_REPO, "train.jsonl")
    val_ds = load_split(CORPUS_REPO, "val.jsonl")
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    # trust_remote_code deliberately NOT used (same reasoning as the earlier
    # plain-transformers version): the model repo's custom modeling_phi3.py
    # calls a DynamicCache API removed from current transformers. Unsloth
    # loads its own patched model class regardless of trust_remote_code, so
    # this mainly documents why the base model id (not an Unsloth-specific
    # pre-quantized repo) is safe to pass here.
    print(f"Loading base model via Unsloth: {BASE_MODEL}")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        dtype=None,  # auto-detect: bf16 on this A10G (Ampere)
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_tok = train_ds.map(tokenize_fn(tokenizer), batched=True,
                              remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_fn(tokenizer), batched=True,
                          remove_columns=val_ds.column_names)

    FastLanguageModel.for_inference(base_model)
    print("Evaluating BASE model perplexity on held-out set...")
    base_ppl = eval_perplexity(base_model, tokenizer, val_tok)
    print(f"BASE held-out perplexity: {base_ppl:.4f}")

    # Same QLoRA config (rank/alpha/dropout/target_modules) as the
    # plain-transformers version this replaces -- only the loading/kernel
    # path changed, not the adapter shape.
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        # "unsloth" mode, not False: Unsloth's own checkpointing is highly
        # optimized (their headline claim is near-zero speed cost for full
        # memory savings, unlike vanilla HF checkpointing). Needed after a
        # live OOM at batch_size=16/no-checkpointing: "CUDA out of memory...
        # 22.28 GiB in use" of the A10G's 22.3 GiB at step 0.
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    # No FastLanguageModel.for_training(model) call here: that's only needed
    # to undo a prior for_inference() call on the SAME object. base_model
    # (not model) was the one passed to for_inference above, and a freshly
    # wrapped get_peft_model() result is already trainable -- calling
    # for_training(model) crashed with AttributeError since the internal
    # _flag_for_generation attribute for_training tries to delete was never
    # set on this new PeftModelForCausalLM wrapper.

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="steps",
        # 250, not 1000: two live runs were killed by exit code 143 (SIGTERM,
        # no Python traceback -- looks like GPU-capacity preemption, not an
        # app error) before ever reaching a 1000-step checkpoint, losing all
        # progress. Finer-grained saves + --resume (below) mean a preemption
        # costs at most ~250 steps of work, not the whole run.
        save_steps=250,
        save_total_limit=2,
        bf16=True,
        # Gradient checkpointing off (set at get_peft_model above, this is
        # just consistent with that): a 3.8B model in 4-bit QLoRA fits an
        # A10G's 24GB comfortably at this batch size without it, and
        # checkpointing was trading throughput for VRAM headroom this run
        # doesn't need -- measured live on the pre-Unsloth version: 0.03
        # epochs completed in 34 min with it on, i.e. ~19h/epoch, well past
        # any reasonable job timeout.
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=42,
        dataloader_num_workers=2,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_tok,
                       data_collator=collator,
                       callbacks=[HubCheckpointCallback(CHECKPOINT_REPO)])

    # Resume support, via the Hub (see CHECKPOINT_REPO comment above for why
    # /data alone doesn't work): try to pull a checkpoint a PREVIOUS,
    # preempted run of this same script pushed. First run ever: repo has no
    # "latest/" yet, snapshot_download raises, and that's the normal
    # fresh-start case, not an error.
    # Require trainer_state.json AND an actual adapter weights file, not just
    # the state file -- a partial/test upload with only trainer_state.json
    # (this happened once, from a persistence smoke-test) passes a
    # state-file-only check but fails Trainer's own stricter validation with
    # "Can't find a valid checkpoint", crashing after model load. Checking
    # for real weights here fails closed (falls back to fresh start) instead.
    resume_dir = None
    try:
        local_repo = snapshot_download(repo_id=CHECKPOINT_REPO, repo_type="model",
                                        allow_patterns=["latest/*"])
        candidate = os.path.join(local_repo, "latest")
        has_state = os.path.isfile(os.path.join(candidate, "trainer_state.json"))
        has_weights = any(
            os.path.isfile(os.path.join(candidate, f))
            for f in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if has_state and has_weights:
            resume_dir = candidate
            print(f"Found existing checkpoint on {CHECKPOINT_REPO}, "
                  f"resuming from {resume_dir}")
        else:
            print(f"Found {candidate} but it's incomplete "
                  f"(state={has_state}, weights={has_weights}); starting fresh.")
    except Exception as e:
        print(f"No usable checkpoint on {CHECKPOINT_REPO} ({e}); starting fresh.")

    print("Starting QLoRA fine-tune...")
    trainer.train(resume_from_checkpoint=resume_dir)

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Evaluating FINE-TUNED model perplexity on the same held-out set...")
    FastLanguageModel.for_inference(model)
    ft_ppl = eval_perplexity(model, tokenizer, val_tok)
    print(f"FINE-TUNED held-out perplexity: {ft_ppl:.4f}")

    result = {
        "base_model": BASE_MODEL,
        "held_out_examples": len(val_tok),
        "base_perplexity": base_ppl,
        "finetuned_perplexity": ft_ppl,
        "relative_improvement_pct": 100 * (base_ppl - ft_ppl) / base_ppl,
    }
    print("RESULT_JSON " + json.dumps(result))

    if args.push_to_hub:
        print(f"Pushing adapter to {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
