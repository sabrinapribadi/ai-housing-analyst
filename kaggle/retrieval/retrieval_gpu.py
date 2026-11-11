"""ADR-008 on a Kaggle T4: R2 = bge-m3 dense (ward filter), R3 = RRF(R0, R1, R2) k=60, R4 = bge-reranker-v2-m3 on R3 top 50.
Inputs: /kaggle/input/aha-rag-eval/{corpus,questions,R0,R1}.parquet. Outputs: R2/R3/R4.parquet, timings.json."""
import json, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import glob, os
OUT = "/kaggle/working"
_hits = glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)   # mount path differs between Kaggle versions
assert _hits, "corpus.parquet not found under /kaggle/input: " + str(glob.glob("/kaggle/input/**", recursive=True)[:20])
IN = os.path.dirname(_hits[0])
corpus = pd.read_parquet(f"{IN}/corpus.parquet"); qs = pd.read_parquet(f"{IN}/questions.parquet")
timing = {"gpu": torch.cuda.get_device_name(0)}

# R2: bge-m3 dense, normalised, cosine; ward-scoped questions rank only that ward's reviews
t0 = time.time()
emb = SentenceTransformer("BAAI/bge-m3", device="cuda"); emb.half(); emb.max_seq_length = 512
D = emb.encode(corpus.text.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
Q = emb.encode(qs.question.tolist(), batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
timing["r2_encode_s"] = round(time.time() - t0, 1)
np.save(f"{OUT}/corpus_bge_m3.npy", D.astype(np.float16))
wards, ids = corpus.ward.to_numpy(), corpus.doc_id.to_numpy()
S = Q @ D.T
rows = []
for i, q in enumerate(qs.itertuples()):
    s = S[i].astype(np.float32)
    if q.scope != "ALL":
        s = np.where(wards == q.scope, s, -np.inf)
    top = np.argsort(-s, kind="stable")[:100]
    top = top[np.isfinite(s[top])]
    rows += [{"qid": q.qid, "rank": r, "doc_id": ids[j], "score": float(s[j])} for r, j in enumerate(top, start=1)]
r2 = pd.DataFrame(rows); r2.to_parquet(f"{OUT}/R2.parquet", index=False)
del emb; torch.cuda.empty_cache()

# R3: reciprocal rank fusion (same code as src/rag_eval/retrieval.rrf)
def rrf(runs, k=60, depth=100):
    a = pd.concat(runs, ignore_index=True); a["rrf"] = 1.0 / (k + a["rank"])
    f = a.groupby(["qid", "doc_id"], as_index=False)["rrf"].sum().sort_values(["qid", "rrf", "doc_id"], ascending=[True, False, True])
    f["rank"] = f.groupby("qid").cumcount() + 1
    return f[f["rank"] <= depth].rename(columns={"rrf": "score"})[["qid", "rank", "doc_id", "score"]].reset_index(drop=True)
r3 = rrf([pd.read_parquet(f"{IN}/R0.parquet"), pd.read_parquet(f"{IN}/R1.parquet"), r2]); r3.to_parquet(f"{OUT}/R3.parquet", index=False)

# R4: cross-encoder reranks R3's top 50
t0 = time.time()
tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rm = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16).cuda().eval()
text = dict(zip(corpus.doc_id, corpus.text)); qtext = dict(zip(qs.qid, qs.question))
cand = r3[r3["rank"] <= 50].copy()
pairs = [(qtext[q], text[d]) for q, d in zip(cand.qid, cand.doc_id)]
scores = []
with torch.no_grad():
    for i in range(0, len(pairs), 64):
        b = tok([p[0] for p in pairs[i:i + 64]], [p[1] for p in pairs[i:i + 64]], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        scores += rm(**b).logits.view(-1).float().cpu().tolist()
cand["score"] = scores
cand = cand.sort_values(["qid", "score", "doc_id"], ascending=[True, False, True])
cand["rank"] = cand.groupby("qid").cumcount() + 1
cand[["qid", "rank", "doc_id", "score"]].to_parquet(f"{OUT}/R4.parquet", index=False)
timing["r4_rerank_s"] = round(time.time() - t0, 1); timing["r4_pairs"] = len(pairs)
json.dump(timing, open(f"{OUT}/timings.json", "w"), indent=1); print(timing)
