import re
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
from src.utils import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    KeyBERT = None

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    yake = None

class KeyphraseExtractor:
    """Hybrid keyphrase extractor combining KeyBERT, YAKE, spaCy Noun Chunks, and TF-IDF fallback."""

    def __init__(self, spacy_model: str = "en_core_web_sm", method: str = "hybrid", top_n: int = 10):
        self.method = method
        self.top_n = top_n

        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception:
                try:
                    self.nlp = spacy.blank("en")
                except Exception:
                    self.nlp = None

        self.kw_model = None
        if KEYBERT_AVAILABLE:
            try:
                self.kw_model = KeyBERT(model="all-MiniLM-L6-v2")
            except Exception:
                self.kw_model = None

        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
            "should", "would", "will", "this", "that", "these", "those", "it", "its", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
        }

    def extract_keybert(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keyphrases using KeyBERT with MMR diversity."""
        if not self.kw_model or not text.strip():
            return []
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                use_mmr=True,
                diversity=0.5,
                top_n=top_n
            )
            return [(kw[0].lower(), float(kw[1])) for kw in keywords]
        except Exception as e:
            return []

    def extract_yake(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keyphrases using YAKE algorithm."""
        if not YAKE_AVAILABLE or not text.strip():
            return []
        try:
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.9,
                top=top_n,
                features=None
            )
            keywords = kw_extractor.extract_keywords(text)
            results = []
            for kw, score in keywords:
                norm_score = max(0.0, 1.0 - float(score))
                results.append((kw.lower(), norm_score))
            return results
        except Exception:
            return []

    def extract_spacy(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keyphrases using spaCy Noun Chunks and POS frequency analysis."""
        if not text.strip():
            return []

        phrases = []
        if self.nlp:
            doc = self.nlp(text)
            if doc.has_annotation("DEP"):
                for chunk in doc.noun_chunks:
                    clean_chunk = chunk.text.lower().strip()
                    if len(clean_chunk) > 3 and clean_chunk not in self.stopwords:
                        phrases.append(clean_chunk)

            for token in doc:
                if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop and len(token.text) > 2:
                    phrases.append(token.text.lower())

        if not phrases:
            return self.extract_tfidf_fallback(text, top_n=top_n)

        counts = Counter(phrases)
        total = max(sum(counts.values()), 1)
        return [(phrase, count / total) for phrase, count in counts.most_common(top_n)]

    def extract_tfidf_fallback(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Fallback frequency-based keyphrase extractor."""
        if not text.strip():
            return []
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in self.stopwords]

        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
        combined = filtered + bigrams
        counts = Counter(combined)
        total = max(sum(counts.values()), 1)

        return [(item, count / total) for item, count in counts.most_common(top_n)]

    def extract_hybrid(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Combines KeyBERT, YAKE, spaCy, and TF-IDF methods with score aggregation."""
        scores_map: Dict[str, float] = {}

        kb_res = self.extract_keybert(text, top_n=top_n)
        yake_res = self.extract_yake(text, top_n=top_n)
        spacy_res = self.extract_spacy(text, top_n=top_n)
        fallback_res = self.extract_tfidf_fallback(text, top_n=top_n)

        for phrase, score in kb_res:
            scores_map[phrase] = scores_map.get(phrase, 0.0) + (score * 0.4)

        for phrase, score in yake_res:
            scores_map[phrase] = scores_map.get(phrase, 0.0) + (score * 0.3)

        for phrase, score in spacy_res:
            scores_map[phrase] = scores_map.get(phrase, 0.0) + (score * 0.2)

        for phrase, score in fallback_res:
            scores_map[phrase] = scores_map.get(phrase, 0.0) + (score * 0.1)

        sorted_phrases = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)

        final_list: List[Tuple[str, float]] = []
        for phrase, score in sorted_phrases:
            if not any(phrase in existing[0] and phrase != existing[0] for existing in final_list):
                final_list.append((phrase, round(score, 4)))
            if len(final_list) >= top_n:
                break

        return final_list

    def extract(self, text: str, method: Optional[str] = None, top_n: Optional[int] = None) -> Dict[str, Any]:
        """Unified keyphrase extraction entry point."""
        use_method = method or self.method
        use_top_n = top_n or self.top_n

        if use_method == "keybert" and KEYBERT_AVAILABLE and self.kw_model:
            scored = self.extract_keybert(text, top_n=use_top_n)
        elif use_method == "yake" and YAKE_AVAILABLE:
            scored = self.extract_yake(text, top_n=use_top_n)
        elif use_method == "spacy" and SPACY_AVAILABLE and self.nlp:
            scored = self.extract_spacy(text, top_n=use_top_n)
        else:
            scored = self.extract_hybrid(text, top_n=use_top_n)

        keyphrases = [item[0] for item in scored]
        search_query = " ".join(keyphrases[:5])

        return {
            "method": use_method,
            "keyphrases": keyphrases,
            "scored_keyphrases": scored,
            "search_query": search_query
        }
