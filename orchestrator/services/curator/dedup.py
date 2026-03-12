"""
MinHashDeduplicator
-------------------
Near-duplicate detection using MinHash + Locality Sensitive Hashing (LSH).

Two samples are considered duplicates if their Jaccard similarity
(measured over character 5-grams) exceeds the configured threshold.

The first occurrence of a near-duplicate cluster is kept; the rest dropped.
"""
import re
from datasketch import MinHash, MinHashLSH

from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

_NUM_PERM = 128   # MinHash permutations — higher = more accurate, slower


def _shingling(text: str, n: int = 5) -> set[bytes]:
    """Return the set of character n-grams for a text string."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    return {text[i:i+n].encode("utf-8") for i in range(len(text) - n + 1)}


def _make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_NUM_PERM)
    for shingle in _shingling(text):
        m.update(shingle)
    return m


class MinHashDeduplicator:

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def run(self, samples: list[dict]) -> list[dict]:
        """
        Remove near-duplicate samples. Returns deduplicated list.
        Preserves original order — first occurrence of a cluster is kept.
        """
        if not samples:
            return []

        lsh = MinHashLSH(threshold=self.threshold, num_perm=_NUM_PERM)
        kept = []
        dropped = 0

        for idx, sample in enumerate(samples):
            text = f"{sample.get('prompt', '')} {sample.get('response', '')}"
            mh = _make_minhash(text)
            key = str(idx)

            # Check if any near-duplicate already in index
            results = lsh.query(mh)
            if results:
                dropped += 1
                continue

            lsh.insert(key, mh)
            kept.append(sample)

        logger.info("dedup_complete",
                    input=len(samples), kept=len(kept), dropped=dropped,
                    threshold=self.threshold)
        return kept
