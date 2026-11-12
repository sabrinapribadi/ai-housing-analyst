"""P10 (ADR-009 step 7): cross-check judge Qwen3-8B (opposite bias to the API judge: it taught S1).
Same rubric and inputs as the primary judge (judge.py from the dataset). Blind: one answer per prompt, no system names.
Inputs: answers_<system>.parquet in the dataset. Outputs cross_<system>.parquet and cross_meta.json."""


def main():
    import glob, json, os, subprocess, sys, time
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.33.0", "torchao>0.16"], check=True)
    IN = os.path.dirname(glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)[0])
    sys.path.insert(0, IN)
    import pandas as pd
    from judge import ANSWER_JUDGE_SYSTEM, answer_judge_input, parse_answer_judgement
    from vllm import LLM, SamplingParams

    q = pd.read_parquet(f"{IN}/questions.parquet").set_index("qid")
    text = dict(zip(*[pd.read_parquet(f"{IN}/corpus.parquet")[c] for c in ("doc_id", "text")]))
    files = sorted(glob.glob(f"{IN}/answers_*.parquet"))
    llm = LLM(model="Qwen/Qwen3-8B", dtype="half", tensor_parallel_size=2, max_model_len=6144, gpu_memory_utilization=0.9, seed=0)
    tok = llm.get_tokenizer()
    meta = {}
    for f in files:
        system = os.path.basename(f)[len("answers_"):-len(".parquet")]
        ans = pd.read_parquet(f)
        items = [(r.qid, *answer_judge_input(q.loc[r.qid, "question"], [(i, text[i]) for i in r.context_ids], r.answer)) for r in ans.itertuples()]
        prompts = [tok.apply_chat_template([{"role": "system", "content": ANSWER_JUDGE_SYSTEM}, {"role": "user", "content": u}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, u, _ in items]
        t0 = time.time()
        gens = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=300, seed=0))
        rows, bad = [], 0
        for (qid, _, n), g in zip(items, gens):
            out = g.outputs[0].text.strip()
            if out.startswith("```"):
                out = out.strip("`").removeprefix("json").strip()
            try:
                j = parse_answer_judgement(out, n)
                rows.append({"qid": qid, "system": system, "n_sentences": n, "supported": j["supported"], "answers_question": j["answers_question"]})
            except Exception:
                bad += 1
        pd.DataFrame(rows).to_parquet(f"/kaggle/working/cross_{system}.parquet", index=False)
        meta[system] = {"judged": len(rows), "unparseable": bad, "seconds": round(time.time() - t0, 1)}
        json.dump(meta, open("/kaggle/working/cross_meta.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":   # vLLM tensor parallel spawns workers that re-import this file
    main()
