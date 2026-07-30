import re
import html
from typing import Dict, List, Any
from src.utils import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class TextPreprocessor:
    """Performs raw log/text sanitization, normalization, and sentence segmentation using spaCy."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.model_name = spacy_model
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model '{spacy_model}' for text preprocessing.")
            except Exception as e:
                logger.warning(f"Could not load spaCy model '{spacy_model}': {e}. Falling back to spacy.blank('en').")
                try:
                    self.nlp = spacy.blank("en")
                    if "sentencizer" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("sentencizer")
                except Exception:
                    self.nlp = None

    def clean_raw(self, text: str) -> str:
        """Sanitizes raw text, log outputs, stack traces, and code fences."""
        if not text:
            return ""

        text = html.unescape(text)
        text = re.sub(r'File ".*?", line \d+.*', ' [STACK_TRACE_LINE] ', text)
        text = re.sub(r'at [\w\.\$]+\([\w\.\$]+:\d+\)', ' [STACK_TRACE_LINE] ', text)
        text = re.sub(r'\b(INFO|DEBUG|ERROR|WARNING|FATAL|CRITICAL):\s*', '', text)
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`[^`\n]+`', ' ', text)
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[#\*_~=\-\+]{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize(self, text: str, lowercase: bool = True, remove_stop: bool = False) -> str:
        """Normalizes text by optional lowercasing and stopword removal."""
        cleaned = self.clean_raw(text)
        if self.nlp:
            doc = self.nlp(cleaned)
            tokens = []
            for token in doc:
                if remove_stop and token.is_stop:
                    continue
                t_text = token.text.lower() if lowercase else token.text
                if t_text.strip():
                    tokens.append(t_text)
            return " ".join(tokens)
        else:
            words = cleaned.split()
            if lowercase:
                words = [w.lower() for w in words]
            return " ".join(words)

    def segment_sentences(self, text: str) -> List[str]:
        """Segments text into individual sentence strings using spaCy or regex boundary fallback."""
        cleaned = self.clean_raw(text)
        if not cleaned:
            return []
        if self.nlp:
            doc = self.nlp(cleaned)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            raw_sents = re.split(r'(?<=[.!?])\s+', cleaned)
            return [s.strip() for s in raw_sents if s.strip()]

    def combine_title_desc(self, title: str, description: str) -> Dict[str, Any]:
        """Combines title and description into structured sanitized records."""
        title_clean = self.clean_raw(title)
        desc_clean = self.clean_raw(description)
        combined = f"{title_clean} - {desc_clean}" if desc_clean else title_clean
        sentences = self.segment_sentences(combined)

        return {
            "title_clean": title_clean,
            "description_clean": desc_clean,
            "combined": combined,
            "sentences": sentences
        }

    def preprocess_for_llm(self, title: str, description: str) -> Dict[str, Any]:
        """Runs complete preprocessing pipeline tailored for LLM prompt context preparation."""
        combined_data = self.combine_title_desc(title, description)
        combined_text = combined_data["combined"]
        tokens_approx = len(combined_text.split())

        return {
            "title_clean": combined_data["title_clean"],
            "description_clean": combined_data["description_clean"],
            "clean_text": combined_text,
            "combined_lower": combined_text.lower(),
            "sentences": combined_data["sentences"],
            "token_count_approx": tokens_approx
        }
