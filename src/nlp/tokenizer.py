import re
from typing import List, Dict, Any, Optional
from src.utils import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class BugTokenizer:
    """Provides spaCy tokenization, POS tagging, lemmatization, and noun chunk extraction with pure Python fallback."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.model_name = spacy_model
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception:
                try:
                    self.nlp = spacy.blank("en")
                except Exception:
                    self.nlp = None

    def tokenize(
        self,
        text: str,
        keep_stop: bool = True,
        keep_punct: bool = False,
        lemmatize: bool = False
    ) -> List[str]:
        """Tokenizes text into a list of clean string tokens matching PDF spec e.g. ['app','crashes','uploading','images','20mb']."""
        if not text:
            return []

        if self.nlp:
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if not keep_punct and (token.is_punct or token.is_space):
                    if not any(c.isalnum() for c in token.text):
                        continue
                if not keep_stop and token.is_stop:
                    continue
                val = token.lemma_.lower() if lemmatize else token.text.lower()
                val = val.strip()
                if len(val) > 1 or val in ["x", "y", "z", "v"]:
                    tokens.append(val)
            return tokens
        else:
            raw_tokens = re.findall(r'\b[a-zA-Z0-9_\>]+', text.lower())
            stop_words = {"the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "with", "by", "and", "or", "it", "this", "that"}
            tokens = []
            for t in raw_tokens:
                if not keep_stop and t in stop_words:
                    continue
                if len(t) > 1:
                    tokens.append(t)
            return tokens

    def tokenize_detailed(self, text: str) -> List[Dict[str, Any]]:
        """Extracts comprehensive token attributes including POS tags, lemmas, and entity flags."""
        if not text:
            return []

        if self.nlp:
            doc = self.nlp(text)
            detailed = []
            for token in doc:
                detailed.append({
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "is_stop": token.is_stop,
                    "is_punct": token.is_punct,
                    "is_entity": token.ent_type_ != ""
                })
            return detailed
        else:
            tokens = self.tokenize(text, keep_stop=True)
            return [{"text": t, "lemma": t, "pos": "NOUN", "tag": "NN", "is_stop": False, "is_punct": False, "is_entity": False} for t in tokens]

    def get_pos_distribution(self, text: str) -> Dict[str, int]:
        """Calculates Part-of-Speech tag frequencies."""
        if not text:
            return {}

        if self.nlp:
            doc = self.nlp(text)
            pos_counts = {}
            for token in doc:
                pos = token.pos_ or "NOUN"
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            return pos_counts
        else:
            tokens = self.tokenize(text, keep_stop=False)
            return {"NOUN": len(tokens)}

    def extract_noun_chunks(self, text: str) -> List[str]:
        """Extracts noun chunks using spaCy parser or fallback bi-gram grouping."""
        if not text:
            return []

        if self.nlp:
            doc = self.nlp(text)
            if doc.has_annotation("DEP"):
                return [chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]

        words = self.tokenize(text, keep_stop=False)
        return [" ".join(words[i:i+2]) for i in range(len(words)-1)]
