import re
import html
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils import logger

class BugDatasetCleaner:
    """Cleans, filters, deduplicates, and sanitizes GitHub bug datasets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        cleaning_cfg = config.get("cleaning", {})
        self.remove_bot_comments = cleaning_cfg.get("remove_bot_comments", True)
        self.bot_keywords = cleaning_cfg.get("bot_keywords", ["dependabot", "stale", "github-actions", "codecov", "snyk-bot", "probot"])
        self.min_desc_len = cleaning_cfg.get("min_description_length", 20)
        self.max_desc_len = cleaning_cfg.get("max_description_length", 5000)
        self.deduplicate_flag = cleaning_cfg.get("deduplicate", True)

    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitizes raw issue text by stripping code fences, URLs, HTML tags, and markdown images."""
        if not text:
            return ""

        text = html.unescape(text)
        text = re.sub(r"```[\s\S]*?```", " [CODE_BLOCK] ", text)
        text = re.sub(r"`[^`\n]+`", " [CODE] ", text)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"https?://\S+", " [URL] ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_bot_issue(self, issue: Dict[str, Any]) -> bool:
        """Checks if an issue was submitted by automated bots or dependency updaters."""
        title = (issue.get("title") or "").lower()
        desc = (issue.get("description") or "").lower()
        combined = f"{title} {desc}"

        for kw in self.bot_keywords:
            if kw in combined:
                return True

        if re.search(r"^(bump|update)\s+.*?\s+from\s+.*?\s+to\s+", title):
            return True

        return False

    def is_valid_issue(self, issue: Dict[str, Any]) -> bool:
        """Checks whether the issue meets minimal quality requirements."""
        title = (issue.get("title") or "").strip()
        desc = (issue.get("description") or "").strip()

        if not title:
            return False

        if len(desc) < self.min_desc_len:
            return False

        return True

    def compute_md5(self, title: str, desc: str) -> str:
        """Generates MD5 hash signature for deduplication based on title and description snippet."""
        key = f"{title.lower().strip()}_{desc[:200].lower().strip()}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def clean_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans title, description, and constructs sanitized combined fields."""
        cleaned = dict(issue)

        raw_title = issue.get("title", "")
        raw_desc = issue.get("description", "")

        title_clean = self.clean_text(raw_title)
        desc_clean = self.clean_text(raw_desc)

        if len(desc_clean) > self.max_desc_len:
            desc_clean = desc_clean[:self.max_desc_len] + "..."

        combined_clean = f"{title_clean} - {desc_clean}"

        cleaned["title_clean"] = title_clean
        cleaned["description_clean"] = desc_clean
        cleaned["combined_clean"] = combined_clean

        return cleaned

    def clean_all(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes full cleaning pipeline: bot filter -> quality validation -> deduplication -> sanitization."""
        logger.info(f"Starting dataset cleaning pipeline on {len(issues)} raw issues...")
        cleaned_list = []
        seen_hashes = set()

        bot_count = 0
        invalid_count = 0
        dedup_count = 0

        for issue in issues:
            if self.remove_bot_comments and self.is_bot_issue(issue):
                bot_count += 1
                continue

            if not self.is_valid_issue(issue):
                invalid_count += 1
                continue

            if self.deduplicate_flag:
                issue_hash = self.compute_md5(issue.get("title", ""), issue.get("description", ""))
                if issue_hash in seen_hashes:
                    dedup_count += 1
                    continue
                seen_hashes.add(issue_hash)

            cleaned_issue = self.clean_issue(issue)
            cleaned_list.append(cleaned_issue)

        logger.info(f"Cleaning finished: {len(cleaned_list)} valid issues retained. "
                    f"(Filtered: {bot_count} bots, {invalid_count} low-quality, {dedup_count} duplicates).")
        return cleaned_list

    def save(self, issues: List[Dict[str, Any]], output_path: str):
        """Saves cleaned dataset to JSON file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(issues)} cleaned issues to {out_file}")
