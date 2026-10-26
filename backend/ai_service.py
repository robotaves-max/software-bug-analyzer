import sys
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import logger
from src.pipeline import BugAnalysisPipeline
from backend.db import save_bug_analysis

class AIService:
    """Service bridge connecting AI pipeline to application."""

    def __init__(self):
        logger.info("Initializing AIService singleton...")
        self.pipeline = BugAnalysisPipeline()

        index_path = ROOT_DIR / "faiss_index" / "bug_index.faiss"
        pkl_path = ROOT_DIR / "faiss_index" / "bug_index.pkl"
        if not index_path.exists() and not pkl_path.exists():
            logger.info("Building initial FAISS index for AIService...")
            self.pipeline.build_faiss_index()

        self.pipeline.initialize()

    def analyze_bug(self, title: str, description: str, user: str ) -> Dict[str, Any]:
        """Executes full bug analysis and persists result into SQLite database."""
        result = self.pipeline.analyze(title, description)
        result["user"] = user

        bug_id = save_bug_analysis(result)
        result["bug_id"] = bug_id

        return result
