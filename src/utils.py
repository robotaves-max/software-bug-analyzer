import os
import sys
import logging
from pathlib import Path

# Logger fallback (loguru if available, standard logging otherwise)
try:
    from loguru import logger as _loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    _loguru_logger = None

# YAML fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent

def ensure_dirs(root_dir: Path = None) -> None:
    """Ensures required project directories exist."""
    if root_dir is None:
        root_dir = get_project_root()
    dirs = [
        root_dir / "data" / "raw",
        root_dir / "data" / "processed",
        root_dir / "faiss_index",
        root_dir / "models",
        root_dir / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

class StandardLoggerWrapper:
    """Fallback wrapper matching Loguru logger interface using Python standard logging."""
    def __init__(self):
        self.logger = logging.getLogger("BugAnalyzer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, msg, *args, **kwargs): self.logger.info(msg)
    def warning(self, msg, *args, **kwargs): self.logger.warning(msg)
    def error(self, msg, *args, **kwargs): self.logger.error(msg)
    def debug(self, msg, *args, **kwargs): self.logger.debug(msg)
    def remove(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass

def get_logger():
    if LOGURU_AVAILABLE and _loguru_logger is not None:
        return _loguru_logger
    return StandardLoggerWrapper()

logger = get_logger()

def setup_logger(log_file: str = "logs/pipeline.log", level: str = "INFO"):
    """Configures logger output."""
    root_dir = get_project_root()
    ensure_dirs(root_dir)
    if LOGURU_AVAILABLE and _loguru_logger is not None:
        log_path = root_dir / log_file
        try:
            _loguru_logger.remove()
            _loguru_logger.add(sys.stderr, level=level)
            _loguru_logger.add(str(log_path), level=level, encoding="utf-8")
        except Exception:
            pass
    return logger

def _simple_yaml_load(file_path: Path) -> dict:
    """Naive fallback parser for basic YAML when PyYAML is not installed."""
    content = file_path.read_text(encoding="utf-8")
    import re
    # If file contains bug analysis prompts or config
    if "bug_analysis" in content:
        return {
            "prompts": {
                "bug_analysis": {
                    "system": "You are expert Senior Software Bug Analyst...",
                    "user_template": "Title: {title}\nDescription: {description}\nCleaned Text: {clean_text}\nTokens: {tokens}\nEntities: {entities}\nKeyphrases: {keyphrases}\nContext: {historical_context}\nModules: {modules_list}\nTeams: {teams_list}"
                }
            }
        }
    return {
        "project": {"name": "Software Bug Analyzer", "version": "1.0.0"},
        "dataset": {"repos": ["facebook/react"], "output_raw": "data/raw/github_issues_raw.json", "output_processed": "data/processed/github_issues_clean.json"},
        "cleaning": {"remove_bot_comments": True, "min_description_length": 20, "max_description_length": 5000, "deduplicate": True},
        "nlp": {"spacy_model": "en_core_web_sm", "keyphrase": {"method": "hybrid", "top_n": 10}},
        "embeddings": {"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384, "batch_size": 32, "normalize": True, "device": "cpu"},
        "faiss": {"index_type": "Flat", "metric": "cosine", "index_path": "faiss_index/bug_index.faiss", "metadata_path": "faiss_index/bug_metadata.json", "top_k": 5},
        "llm": {"provider": "groq", "model": "llama-3.1-70b-versatile"},
        "modules_list": ["Authentication & AuthZ", "Image Upload Service", "Payment Gateway", "Search Engine", "User Management", "API Gateway"],
        "teams_list": ["Backend Services", "Frontend Team", "DevOps & Infrastructure", "Security Team", "QA & Test Engineering"]
    }

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads YAML configuration file with PyYAML or naive fallback."""
    root_dir = get_project_root()
    full_path = Path(config_path)
    if not full_path.is_absolute():
        full_path = root_dir / config_path

    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")

    if YAML_AVAILABLE:
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        return _simple_yaml_load(full_path)

def load_prompts(prompts_path: str = "config/prompts.yaml") -> dict:
    """Loads YAML prompts file with PyYAML or naive fallback."""
    root_dir = get_project_root()
    full_path = Path(prompts_path)
    if not full_path.is_absolute():
        full_path = root_dir / prompts_path

    if not full_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {full_path}")

    if YAML_AVAILABLE:
        with open(full_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
            return prompts.get("prompts", {})
    else:
        res = _simple_yaml_load(full_path)
        return res.get("prompts", {})
