"""
RAG agent: semantic review search using ChromaDB + OpenAI text-embedding-3-small.

Build the index once before deploying:
    python scripts/build_review_index.py
"""

import os
import logging
from typing import Optional

import pandas as pd
import chromadb
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

CHROMA_PATH  = "data/chroma"
COLLECTION   = "tokyo_reviews"
N_SAMPLES    = 25_000   # balanced sample across neighbourhoods
EMBED_DIMS   = 256      # text-embedding-3-small supports matryoshka reduction
EMBED_MODEL  = "text-embedding-3-small"
EMBED_BATCH  = 100      # safe OpenAI batch size

_SYSTEM_PROMPT = """You answer questions about Tokyo Airbnb guest experiences
using ONLY the review excerpts provided below.

Rules:
1. Ground every claim in the retrieved text — quote or paraphrase directly.
2. If reviews are mixed (some positive, some negative), reflect that honestly.
3. Mention neighbourhoods when relevant patterns emerge.
4. Be concise: 3–5 sentences, use bullet points for lists.
5. Never invent information not present in the reviews."""


class ReviewRAGAgent:
    """Semantic review search: embed → retrieve → synthesise with GPT-4o-mini."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._chroma: Optional[chromadb.PersistentClient] = None
        self._col = None
        self._llm: Optional[ChatOpenAI] = None

    # ── Lazy properties ───────────────────────────────────────────────────────

    @property
    def _openai(self) -> OpenAI:
        return OpenAI(api_key=self._api_key)

    @property
    def chroma(self) -> chromadb.PersistentClient:
        if self._chroma is None:
            self._chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        return self._chroma

    @property
    def collection(self):
        if self._col is None:
            self._col = self.chroma.get_or_create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._col

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=self._api_key,
            )
        return self._llm

    # ── Index management ──────────────────────────────────────────────────────

    def is_index_built(self) -> bool:
        """Return True if the ChromaDB collection contains reviews."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def index_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def build_index(
        self,
        reviews_path: str = "data/processed/reviews_processed.parquet",
        listings_path: str = "data/processed/listings_processed.parquet",
    ) -> int:
        """Embed N_SAMPLES reviews and upsert into ChromaDB.
        Returns the total document count after indexing."""

        reviews  = pd.read_parquet(reviews_path)[["listing_id", "comments"]]
        listings = pd.read_parquet(listings_path)[["id", "neighbourhood_cleansed"]]

        df = reviews.merge(listings, left_on="listing_id", right_on="id", how="left")
        df = df[df["comments"].notna() & (df["comments"].str.strip().str.len() > 30)]
        df = df.dropna(subset=["neighbourhood_cleansed"]).reset_index(drop=True)

        if len(df) > N_SAMPLES:
            df = df.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

        logger.info(f"Embedding {len(df):,} reviews …")

        texts     = df["comments"].astype(str).tolist()
        ids       = [f"rev_{i}" for i in range(len(df))]
        metadatas = [
            {
                "listing_id":    str(row["listing_id"]),
                "neighbourhood": str(row["neighbourhood_cleansed"]),
            }
            for _, row in df.iterrows()
        ]

        # Embed in batches via OpenAI
        all_embeddings: list = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            resp  = self._openai.embeddings.create(
                model=EMBED_MODEL, input=batch, dimensions=EMBED_DIMS
            )
            all_embeddings.extend([e.embedding for e in resp.data])
            if start % 2000 == 0:
                logger.info(f"  {start:,}/{len(texts):,} embedded")

        # Upsert in batches
        for start in range(0, len(ids), EMBED_BATCH):
            sl = slice(start, start + EMBED_BATCH)
            self.collection.upsert(
                ids=ids[sl],
                documents=texts[sl],
                embeddings=all_embeddings[sl],
                metadatas=metadatas[sl],
            )

        count = self.collection.count()
        logger.info(f"Index complete: {count:,} documents.")
        return count

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        neighbourhood: str = "",
        n_results: int = 10,
    ) -> dict:
        """Embed question → retrieve similar reviews → synthesise answer.

        Returns:
            {"answer": str, "sources": list[{"text", "neighbourhood", "similarity"}]}
        """
        # Embed the question
        q_emb = self._openai.embeddings.create(
            model=EMBED_MODEL, input=[question], dimensions=EMBED_DIMS
        ).data[0].embedding

        where = {"neighbourhood": {"$eq": neighbourhood}} if neighbourhood else None

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        if not docs:
            return {
                "answer": "No relevant reviews found for that query and filter.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[{m.get('neighbourhood', 'Unknown')}] {d}"
            for d, m in zip(docs, metas)
        )

        response = self.llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nReviews:\n{context}"),
        ])

        sources = [
            {
                "text":          d,
                "neighbourhood": m.get("neighbourhood", ""),
                "similarity":    round(1.0 - dist, 3),
            }
            for d, m, dist in zip(docs, metas, dists)
        ]

        return {"answer": response.content, "sources": sources}
