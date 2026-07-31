import re
from typing import Dict, List, Any, Optional
from src.utils import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class EntityExtractor:
    """Extracts software-specific entities (Actions, Components, Errors, Conditions, Paths, OS, Versions) and spaCy NER."""

    def __init__(self, spacy_model: str = "en_core_web_sm", custom_patterns: Optional[Dict[str, str]] = None):
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

        patterns = custom_patterns or {}
        self.file_path_regex = patterns.get(
            "file_path_regex",
            r"\b[\w\/\.-]+\.(py|js|ts|java|cpp|go|rb|tsx|jsx)\b"
        )
        self.error_code_regex = patterns.get(
            "error_code_regex",
            r"\b([A-Z]+_\d+|ERR_[A-Z0-9_]+|E\d{4}|HTTP\s*\d{3})\b"
        )
        self.version_regex = patterns.get(
            "version_regex",
            r"\bv?\d+\.\d+(?:\.\d+)?\b"
        )

        self.action_keywords = [
            "crash", "crashes", "freeze", "freezes", "fail", "fails", "failing",
            "error", "upload", "uploading", "download", "downloading", "login",
            "timeout", "hang", "hangs", "submit", "submitting", "connect", "connecting",
            "charge", "charging", "search", "searching", "render", "rendering"
        ]

        self.component_keywords = [
            "api", "database", "frontend", "backend", "ui", "server", "client",
            "auth", "authentication", "image", "images", "file", "files",
            "payment", "search", "gateway", "pool", "view", "controller",
            "queue", "buffer", "session", "token", "query", "cache", "redis"
        ]

        self.error_keywords = [
            "crash", "freeze", "timeout", "exception", "oom", "out of memory",
            "deadlock", "nullpointer", "syntaxerror", "runtimeerror", "double charge",
            "empty results", "unhandled error", "500", "504", "403", "404"
        ]

        self.os_browser_patterns = [
            "windows", "win11", "win10", "linux", "ubuntu", "debian", "macos",
            "ios", "android", "chrome", "firefox", "safari", "edge"
        ]

    def extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extracts standard spaCy Named Entities (ORG, PERSON, PRODUCT, GPE, DATE, etc.)."""
        if not text or not self.nlp:
            return {}

        doc = self.nlp(text)
        entities: Dict[str, List[str]] = {}

        # ``ENT`` is not a valid annotation name in newer spaCy releases
        # (for example 3.8).  ``doc.ents`` is safe for both a trained pipeline
        # and the blank-English fallback, so iterate over it directly.
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            if ent.text.strip() not in entities[label]:
                entities[label].append(ent.text.strip())

        return entities

    def extract_software_entities(self, text: str) -> Dict[str, Any]:
        """Extracts domain-specific software engineering entities (Actions, Components, Errors, Conditions, Paths)."""
        if not text:
            return {
                "actions": [], "components": [], "error_types": [], "conditions": [],
                "file_paths": [], "error_codes": [], "versions": [], "os_browsers": [],
                "structured_summary": {"Action": [], "Component": [], "Error": [], "Condition": []}
            }

        text_lower = text.lower()

        # 1. File Paths (use re.finditer for full string match)
        file_paths = []
        for m in re.finditer(self.file_path_regex, text, re.IGNORECASE):
            val = m.group(0).strip()
            if val and val not in file_paths:
                file_paths.append(val)

        # 2. Error Codes
        error_codes = []
        for m in re.finditer(self.error_code_regex, text, re.IGNORECASE):
            val = m.group(0).strip()
            if val and val not in error_codes:
                error_codes.append(val)

        # 3. Version Numbers
        versions = []
        for m in re.finditer(self.version_regex, text, re.IGNORECASE):
            val = m.group(0).strip()
            if val and val not in versions:
                versions.append(val)

        # 4. OS / Browsers
        os_browsers = [ob for ob in self.os_browser_patterns if ob in text_lower]

        # 5. Actions
        actions = []
        for kw in self.action_keywords:
            if kw in text_lower:
                actions.append(kw.capitalize())

        context_actions = re.findall(r'\b(while|after|when)\s+([a-z_]+(?:\s+[a-z_]+)?)\b', text_lower)
        for _, act in context_actions:
            if act not in actions:
                actions.append(act.title())

        # 6. Components
        components = []
        for kw in self.component_keywords:
            if kw in text_lower:
                components.append(kw.capitalize())

        # 7. Error Types
        error_types = []
        for kw in self.error_keywords:
            if kw in text_lower:
                error_types.append(kw.upper() if len(kw) <= 4 else kw.title())

        # 8. Conditions
        condition_patterns = [
            r'>\s*\d+\s*(?:mb|gb|kb)',
            r'\d+\s*(?:mb|gb|kb)',
            r'after\s+\d+\s*(?:s|sec|seconds|ms)',
            r'on\s+[a-z0-9\s]+(?:network|conditions)',
            r'when\s+[a-z0-9\s]+(?:contains|includes)[a-z0-9\s%_]+'
        ]

        conditions = []
        for pat in condition_patterns:
            for m in re.finditer(pat, text_lower):
                val = m.group(0).strip()
                if val and val not in conditions:
                    conditions.append(val)

        # Deduplicate & cap to 10 each
        actions = list(dict.fromkeys(actions))[:10]
        components = list(dict.fromkeys(components))[:10]
        error_types = list(dict.fromkeys(error_types))[:10]
        conditions = list(dict.fromkeys(conditions))[:10]
        file_paths = file_paths[:10]
        error_codes = error_codes[:10]
        versions = versions[:10]
        os_browsers = list(dict.fromkeys(os_browsers))[:10]

        structured_summary = {
            "Action": actions[:3] if actions else ["General Execution"],
            "Component": components[:3] if components else ["System Component"],
            "Error": error_types[:3] if error_types else ["System Failure"],
            "Condition": conditions[:3] if conditions else ["Standard Environment"]
        }

        return {
            "actions": actions,
            "components": components,
            "error_types": error_types,
            "conditions": conditions,
            "file_paths": file_paths,
            "error_codes": error_codes,
            "versions": versions,
            "os_browsers": os_browsers,
            "structured_summary": structured_summary
        }

    def extract_all(self, text: str) -> Dict[str, Any]:
        """Extracts both spaCy NER entities and software domain entities."""
        spacy_ents = self.extract_with_spacy(text)
        software_ents = self.extract_software_entities(text)

        return {
            "spacy_entities": spacy_ents,
            "software_entities": software_ents,
            "combined_text": text
        }
