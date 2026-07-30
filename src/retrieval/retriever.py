from typing import List, Dict, Any, Tuple
from src.utils import logger
from src.embeddings.embedder import BugEmbedder
from src.retrieval.faiss_store import FaissBugStore

class BugRetriever:
    """Retrieves top-K historical bug context from FAISS and formats it for internal LLM reasoning."""

    def __init__(self, embedder: BugEmbedder, faiss_store: FaissBugStore, top_k: int = 5):
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.top_k = top_k

    def retrieve(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Encodes query text, searches vector index, and returns structured result list."""
        k = top_k or self.top_k
        query_vec = self.embedder.encode_query(query_text)
        distances, indices, metadatas = self.faiss_store.search(query_vec, top_k=k)

        results = []
        for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadatas)):
            similarity = float(dist) if self.faiss_store.metric in ["cosine", "inner_product", "ip"] else float(1.0 / (1.0 + dist))
            title = meta.get("title_clean") or meta.get("title") or "Untitled Bug"
            desc = meta.get("description_clean") or meta.get("description") or ""
            labels = meta.get("labels", [])
            resolution = meta.get("resolution_hint") or meta.get("closed_at") or "No explicit resolution"

            results.append({
                "rank": i + 1,
                "similarity_score": round(similarity, 4),
                "distance": float(dist),
                "faiss_index": int(idx),
                "title": title,
                "description": desc,
                "labels": labels,
                "resolution": resolution,
                "metadata": meta
            })

        logger.info(f"Retrieved {len(results)} relevant historical bugs for query snippet.")
        return results

    @staticmethod
    def format_for_llm(retrieval_results: List[Dict[str, Any]]) -> str:
        """Formats historical bug search results into prompt context text."""
        if not retrieval_results:
            return "No historical bug context available."

        lines = ["Top relevant historical bugs (for internal reasoning only):"]
        for res in retrieval_results:
            lines.append(f"\n--- Historical Bug #{res['rank']} (Similarity: {res['similarity_score']:.2f}) ---")
            lines.append(f"Title: {res['title']}")
            lines.append(f"Description: {res['description'][:300]}...")
            lines.append(f"Labels: {', '.join(res['labels']) if res['labels'] else 'None'}")
            lines.append(f"Resolution: {res['resolution']}")

        return "\n".join(lines)

    def retrieve_and_format(self, query_text: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Runs retrieval and returns formatted prompt string along with raw structured list."""
        raw_results = self.retrieve(query_text, top_k=top_k)
        formatted_context = self.format_for_llm(raw_results)
        return formatted_context, raw_results
