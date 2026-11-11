"""P9 feasibility (ADR-007): vLLM 0.33 + Qwen3-1.7B + LoRA on a Kaggle T4; Qwen3.5-2B load trial. Writes feasibility.json."""
import json, os, subprocess, sys, time, textwrap

OUT = "/kaggle/working/feasibility.json"
res = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()

res["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
t0 = time.time()
res["pip"] = sh(f"{sys.executable} -m pip install -q vllm==0.33.0 peft 2>&1 | tail -5")
res["pip_seconds"] = round(time.time() - t0)
res["versions"] = sh(f"{sys.executable} -c \"import vllm, torch, transformers, peft; print(vllm.__version__, torch.__version__, transformers.__version__, peft.__version__, torch.version.cuda)\"")
json.dump(res, open(OUT, "w"), indent=1)

# 1) dummy LoRA adapter (random B so the LoRA path really changes outputs)
make_adapter = textwrap.dedent('''
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float16)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", init_lora_weights=False))
    m.save_pretrained("/kaggle/working/dummy_lora")
    print("adapter ok")
''')
open("/kaggle/working/make_adapter.py", "w").write(make_adapter)
res["adapter"] = sh(f"{sys.executable} /kaggle/working/make_adapter.py 2>&1 | tail -3")

# 2) vLLM with and without LoRA, each in its own process so GPU memory is released
run_vllm = textwrap.dedent('''
    import json, sys, time
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    model, lora = sys.argv[1], sys.argv[2] == "1"
    out = {"model": model, "lora": lora}
    t0 = time.time()
    try:
        llm = LLM(model=model, dtype="half", max_model_len=4096, gpu_memory_utilization=0.85,
                  enable_lora=lora, max_lora_rank=16, enforce_eager=False)
        out["load_s"] = round(time.time() - t0, 1)
        tok = llm.get_tokenizer()
        msgs = [[{"role": "user", "content": f"In two sentences, what do guests usually value in a Tokyo apartment? ({i})"}] for i in range(8)]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        sp = SamplingParams(temperature=0, max_tokens=128)
        req = LoRARequest("dummy", 1, "/kaggle/working/dummy_lora") if lora else None
        t1 = time.time()
        gens = llm.generate(prompts, sp, lora_request=req)
        out["batch8_s"] = round(time.time() - t1, 2)
        out["tokens_out"] = sum(len(g.outputs[0].token_ids) for g in gens)
        out["sample"] = gens[0].outputs[0].text[:200]
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = repr(e)[:1500]
    print("RESULT " + json.dumps(out))
''')
open("/kaggle/working/run_vllm.py", "w").write(run_vllm)
res["runs"] = []
for model, lora in (("Qwen/Qwen3-1.7B", "0"), ("Qwen/Qwen3-1.7B", "1"), ("Qwen/Qwen3.5-2B", "0")):
    log = sh(f"{sys.executable} /kaggle/working/run_vllm.py {model} {lora}")
    line = [l for l in log.splitlines() if l.startswith("RESULT ")]
    res["runs"].append(json.loads(line[-1][7:]) if line else {"model": model, "lora": lora == "1", "ok": False, "log_tail": log[-2500:]})
    json.dump(res, open(OUT, "w"), indent=1)

res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])
