"""P10 (ADR-009 steps 9-10): vLLM OpenAI-compatible server on ONE T4. S0 = Qwen3-1.7B, S1 = + LoRA (seeds 0, 1, 2).
Answers for the 300 test questions per system (temperature 0), latency at concurrency 1 and 8, throughput for cost.
Secondary: the seed-0 adapter merged into the weights, same measurements. Outputs answers_*.parquet and bench.json."""


def main():
    import glob, json, os, subprocess, sys, time
    import concurrent.futures as cf
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.33.0", "torchao>0.16", "peft==0.21.2"], check=True)
    IN = os.path.dirname(glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)[0])
    ADP = os.path.dirname(glob.glob("/kaggle/input/**/adapters/seed0/adapter_config.json", recursive=True)[0])
    ADP = os.path.dirname(ADP)
    sys.path.insert(0, IN)
    import pandas as pd, requests
    from answers import ANSWER_SYSTEM, answer_user
    from transformers import AutoTokenizer

    q = pd.read_parquet(f"{IN}/questions.parquet"); q = q[q.set == "test"].reset_index(drop=True)
    text = dict(zip(*[pd.read_parquet(f"{IN}/corpus.parquet")[c] for c in ("doc_id", "text")]))
    run = pd.read_parquet(f"{IN}/R4.parquet")
    ctx = {k: g.sort_values("rank").doc_id.head(10).tolist() for k, g in run[run.qid.isin(q.qid)].groupby("qid")}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts = {r.qid: tok.apply_chat_template([{"role": "system", "content": ANSWER_SYSTEM},
                                               {"role": "user", "content": answer_user(r.question, [(i, text[i]) for i in ctx[r.qid]])}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in q.itertuples()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")      # one T4, as priced in ADR-007
    URL = "http://127.0.0.1:8000"

    def serve(args, log):
        p = subprocess.Popen(["vllm", "serve"] + args + ["--dtype", "half", "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
                                                          "--port", "8000", "--seed", "0"],
                             env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        for _ in range(240):
            try:
                if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                    return p
            except Exception:
                pass
            time.sleep(5)
        p.kill(); raise RuntimeError(open(log).read()[-3000:])

    def call(model, qid):
        t0 = time.time()
        r = requests.post(f"{URL}/v1/completions", json={"model": model, "prompt": prompts[qid], "max_tokens": 256,
                                                         "temperature": 0, "seed": 0}, timeout=120).json()
        lat = time.time() - t0
        return {"qid": qid, "answer": r["choices"][0]["text"].strip(), "tokens_in": r["usage"]["prompt_tokens"],
                "tokens_out": r["usage"]["completion_tokens"], "latency_s": lat}

    def measure(model, conc):
        for qid in list(prompts)[:5]:
            call(model, qid)                                  # warm-up, not timed
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=conc) as ex:
            rows = list(ex.map(lambda k: call(model, k), list(prompts)))
        wall = time.time() - t0
        lat = pd.Series([r["latency_s"] for r in rows])
        return rows, {"concurrency": conc, "n": len(rows), "wall_s": round(wall, 2), "answers_per_s": round(len(rows) / wall, 3),
                      "p50_s": round(lat.quantile(.5), 3), "p95_s": round(lat.quantile(.95), 3),
                      "tokens_out_mean": round(pd.Series([r["tokens_out"] for r in rows]).mean(), 1)}

    bench = {"gpu": subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.split()[0:2]}
    W = "/kaggle/working"
    p = serve(["Qwen/Qwen3-1.7B", "--enable-lora", "--max-lora-rank", "16", "--max-loras", "3",
               "--lora-modules", f"s1_seed0={ADP}/seed0", f"s1_seed1={ADP}/seed1", f"s1_seed2={ADP}/seed2"], f"{W}/serve_lora.log")
    for name, model in (("S0", "Qwen/Qwen3-1.7B"), ("S1_seed0", "s1_seed0")):
        rows, m1 = measure(model, 1)
        _, m8 = measure(model, 8)
        pd.DataFrame(rows).assign(system=name).to_parquet(f"{W}/answers_{name}.parquet", index=False)
        bench[name] = {"concurrency_1": m1, "concurrency_8": m8}
        json.dump(bench, open(f"{W}/bench.json", "w"), indent=1)
    for name, model in (("S1_seed1", "s1_seed1"), ("S1_seed2", "s1_seed2")):
        rows, m8 = measure(model, 8)
        pd.DataFrame(rows).assign(system=name).to_parquet(f"{W}/answers_{name}.parquet", index=False)
        bench[name] = {"concurrency_8": m8}
    p.terminate(); p.wait(); time.sleep(10)

    merge = ("import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel\n"
             f"m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype=torch.float16), '{ADP}/seed0')\n"
             f"m.merge_and_unload().save_pretrained('{W}/merged'); AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B').save_pretrained('{W}/merged')")
    subprocess.run([sys.executable, "-c", merge], check=True, env=env)
    p = serve([f"{W}/merged", "--served-model-name", "s1_merged"], f"{W}/serve_merged.log")
    rows, m1 = measure("s1_merged", 1)
    _, m8 = measure("s1_merged", 8)
    pd.DataFrame(rows).assign(system="S1_merged").to_parquet(f"{W}/answers_S1_merged.parquet", index=False)
    bench["S1_merged"] = {"concurrency_1": m1, "concurrency_8": m8}
    p.terminate(); p.wait()
    subprocess.run(f"rm -rf {W}/merged", shell=True)
    json.dump(bench, open(f"{W}/bench.json", "w"), indent=1); print(json.dumps(bench, indent=1))


if __name__ == "__main__":
    main()
