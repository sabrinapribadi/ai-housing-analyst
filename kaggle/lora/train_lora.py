"""P9 (ADR-009): LoRA on Qwen3-1.7B from filtered teacher answers; seeds 0, 1, 2 (two T4s, two seeds at a time).
Filter (configs/rag_eval.yaml lora.train_filter): 25-180 words, >= 1 citation, every cited id in the context.
Prompt text = the serving chat template (thinking off), so training and inference see identical prompts.
Checkpoint per seed = the epoch with the lower validation loss. Outputs adapters/seed{N}/ and train_meta.json."""
import glob, json, os, subprocess, sys, time, textwrap
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl==1.17.1", "peft==0.21.2", "torchao>0.16"], check=True)
IN = os.path.dirname(glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)[0])
TEACH = glob.glob("/kaggle/input/**/teacher_answers.parquet", recursive=True)[0]
sys.path.insert(0, IN)
import pandas as pd
from answers import ANSWER_SYSTEM, answer_user, code_checks

q = pd.read_parquet(f"{IN}/questions.parquet").set_index("qid")
text = dict(zip(*[pd.read_parquet(f"{IN}/corpus.parquet")[c] for c in ("doc_id", "text")]))
t = pd.read_parquet(TEACH)
chk = pd.DataFrame([code_checks(a, list(c)) for a, c in zip(t.answer, t.context_ids)])
t = pd.concat([t.reset_index(drop=True), chk], axis=1)
keep = t.citations_valid & t.words.between(25, 180) & (t.n_citations >= 1)
meta = {"teacher_rows": int(len(t)), "kept": {s: int((keep & (t.set == s)).sum()) for s in ("train", "val")},
        "dropped_reasons": {"abstained": int(t.abstained.sum()), "invalid_or_no_citation": int((~t.citations_valid & ~t.abstained).sum()),
                            "length": int((t.citations_valid & ~t.words.between(25, 180)).sum())}}
t = t[keep]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
def row(r):
    prompt = tok.apply_chat_template([{"role": "system", "content": ANSWER_SYSTEM},
                                      {"role": "user", "content": answer_user(q.loc[r.qid, "question"], [(i, text[i]) for i in r.context_ids])}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return {"prompt": prompt, "completion": r.answer + tok.eos_token}
for s in ("train", "val"):
    pd.DataFrame([row(r) for r in t[t.set == s].itertuples()]).to_json(f"/kaggle/working/{s}.jsonl", orient="records", lines=True)
json.dump(meta, open("/kaggle/working/train_meta.json", "w"), indent=1); print(meta)

train_one = textwrap.dedent('''
    import json, sys, time, torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    seed = int(sys.argv[1]); out = f"/kaggle/working/run_seed{seed}"
    ds = load_dataset("json", data_files={"train": "/kaggle/working/train.jsonl", "val": "/kaggle/working/val.jsonl"})
    cfg = SFTConfig(output_dir=out, seed=seed, num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8, per_device_eval_batch_size=4,
                    gradient_checkpointing=True, fp16=True, bf16=False, max_length=2048, completion_only_loss=True,
                    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
                    metric_for_best_model="eval_loss", greater_is_better=False, logging_steps=10, report_to=[])
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    tr = SFTTrainer(model="Qwen/Qwen3-1.7B", args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"], peft_config=lora)
    t0 = time.time(); tr.train(); secs = time.time() - t0
    tr.save_model(f"/kaggle/working/adapters/seed{seed}")
    evals = [l for l in tr.state.log_history if "eval_loss" in l]
    json.dump({"seed": seed, "train_s": round(secs), "evals": evals, "best": tr.state.best_model_checkpoint},
              open(f"/kaggle/working/adapters/seed{seed}/train_log.json", "w"), indent=1)
    print("DONE", seed, round(secs), evals)
''')
open("/kaggle/working/train_one.py", "w").write(train_one)
os.makedirs("/kaggle/working/adapters", exist_ok=True)
def launch(seed, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen([sys.executable, "/kaggle/working/train_one.py", str(seed)], env=env,
                            stdout=open(f"/kaggle/working/seed{seed}.log", "w"), stderr=subprocess.STDOUT)
t0 = time.time()
for batch in ((0, 1), (2,)):
    procs = [launch(s, g) for g, s in enumerate(batch)]
    for p in procs: p.wait()
meta["total_train_s"] = round(time.time() - t0)
for s in (0, 1, 2):
    p = f"/kaggle/working/adapters/seed{s}/train_log.json"
    meta[f"seed{s}"] = json.load(open(p)) if os.path.exists(p) else {"error": open(f"/kaggle/working/seed{s}.log").read()[-3000:]}
subprocess.run("rm -rf /kaggle/working/run_seed*", shell=True)       # keep only the chosen adapters
json.dump(meta, open("/kaggle/working/train_meta.json", "w"), indent=1); print(json.dumps(meta)[:3000])
