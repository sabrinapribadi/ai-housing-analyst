"""P9 (ADR-009 step 3): teacher answers from Qwen3-8B (open, Apache-2.0) for the 2,000 train and 100 validation questions.
Contexts = top 10 of the retrieval system chosen by gate G1 (system.json in the dataset). Prompt = src/rag_eval/answers.py.
Outputs teacher_answers.parquet: qid, set, context_ids, answer, tokens_in, tokens_out."""
import glob, json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.33.0", "torchao>0.16"], check=True)
IN = os.path.dirname(glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)[0])
sys.path.insert(0, IN)
import pandas as pd
from answers import ANSWER_SYSTEM, answer_user          # same prompt as every other system
from vllm import LLM, SamplingParams

system = json.load(open(f"{IN}/system.json"))["retrieval_system"]
q = pd.read_parquet(f"{IN}/questions.parquet"); q = q[q.set.isin(["train", "val"])]
text = dict(zip(*[pd.read_parquet(f"{IN}/corpus.parquet")[c] for c in ("doc_id", "text")]))
run = pd.read_parquet(f"{IN}/{system}.parquet")
ctx = {qid: g.sort_values("rank").doc_id.head(10).tolist() for qid, g in run[run.qid.isin(q.qid)].groupby("qid")}

llm = LLM(model="Qwen/Qwen3-8B", dtype="half", tensor_parallel_size=2, max_model_len=4096, gpu_memory_utilization=0.9, seed=0)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "system", "content": ANSWER_SYSTEM},
                                    {"role": "user", "content": answer_user(r.question, [(i, text[i]) for i in ctx[r.qid]])}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in q.itertuples()]
t0 = time.time()
gens = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=256, seed=0))
secs = time.time() - t0
out = pd.DataFrame({"qid": q.qid.to_numpy(), "set": q.set.to_numpy(), "context_ids": [ctx[x] for x in q.qid],
                    "answer": [g.outputs[0].text.strip() for g in gens],
                    "tokens_in": [len(g.prompt_token_ids) for g in gens], "tokens_out": [len(g.outputs[0].token_ids) for g in gens]})
out.to_parquet("/kaggle/working/teacher_answers.parquet", index=False)
json.dump({"retrieval_system": system, "n": len(out), "generate_s": round(secs, 1), "gpu": "2x T4, tensor parallel"},
          open("/kaggle/working/teacher_meta.json", "w"), indent=1)
print(out.answer.head(3).tolist())
