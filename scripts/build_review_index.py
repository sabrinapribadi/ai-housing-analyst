"""
One-time script: build the ChromaDB review index for the RAG "Ask the Reviews" feature.

Run from the project root:
    python scripts/build_review_index.py

Add --rebuild to force a full rebuild even if the index already exists.

Cost estimate: 25,000 reviews × ~100 tokens × $0.020/1M tokens ≈ $0.05 one-time.
"""

import sys
import os
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from src.agent.rag_agent import ReviewRAGAgent, CHROMA_PATH, N_SAMPLES, EMBED_DIMS

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    agent = ReviewRAGAgent()

    if agent.is_index_built() and "--rebuild" not in sys.argv:
        logger.info(
            f"Index already contains {agent.index_count():,} documents. "
            "Pass --rebuild to force a full rebuild."
        )
        return

    if "--rebuild" in sys.argv and agent.is_index_built():
        logger.info("--rebuild flag set: deleting existing collection …")
        agent.chroma.delete_collection("tokyo_reviews")
        agent._col = None

    logger.info(
        f"Building index: {N_SAMPLES:,} reviews, "
        f"text-embedding-3-small @ {EMBED_DIMS} dims → {CHROMA_PATH}/"
    )
    t0    = time.time()
    count = agent.build_index()
    elapsed = time.time() - t0

    # Report ChromaDB directory size
    chroma_size = sum(
        f.stat().st_size for f in Path(CHROMA_PATH).rglob("*") if f.is_file()
    ) / (1024 ** 2)

    logger.info(f"Done in {elapsed:.0f}s — {count:,} documents indexed")
    logger.info(f"ChromaDB size on disk: {chroma_size:.1f} MB")
    logger.info("Next step: git add data/chroma/ && git push")


if __name__ == "__main__":
    main()
