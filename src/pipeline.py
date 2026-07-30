import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from src.utils import get_project_root, ensure_dirs, setup_logger, load_config, load_prompts, logger
from src.dataset.collector import GitHubIssueCollector
from src.dataset.cleaner import BugDatasetCleaner
from src.nlp.preprocessor import TextPreprocessor
from src.nlp.tokenizer import BugTokenizer
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.keyphrase_extractor import KeyphraseExtractor
from src.embeddings.embedder import BugEmbedder
from src.retrieval.faiss_store import FaissBugStore
from src.retrieval.retriever import BugRetriever
from src.llm.prompt_manager import PromptManager
from src.llm.llm_client import LLMClient
from src.llm.predictor import BugPredictor

class BugAnalysisPipeline:
    """Master orchestrator combining NLP analysis, FAISS vector retrieval, and LLM bug prediction."""

    def __init__(self, config_path: str = "config/config.yaml", prompts_path: str = "config/prompts.yaml"):
        self.root_dir = get_project_root()
        ensure_dirs(self.root_dir)
        setup_logger()

        self.config_path = config_path
        self.prompts_path = prompts_path
        self.config = load_config(config_path)

        self.initialized = False

        self.preprocessor = None
        self.tokenizer = None
        self.entity_extractor = None
        self.keyphrase_extractor = None
        self.embedder = None
        self.faiss_store = None
        self.retriever = None
        self.prompt_manager = None
        self.llm_client = None
        self.predictor = None

    def initialize(self, load_faiss: bool = True):
        """Loads all NLP models, embedding encoders, FAISS stores, and LLM predictors."""
        logger.info("Initializing Bug Analysis Pipeline components...")

        nlp_cfg = self.config.get("nlp", {})
        spacy_model = nlp_cfg.get("spacy_model", "en_core_web_sm")
        custom_patterns = nlp_cfg.get("entity_custom_patterns", {})
        kp_cfg = nlp_cfg.get("keyphrase", {})

        self.preprocessor = TextPreprocessor(spacy_model=spacy_model)
        self.tokenizer = BugTokenizer(spacy_model=spacy_model)
        self.entity_extractor = EntityExtractor(spacy_model=spacy_model, custom_patterns=custom_patterns)
        self.keyphrase_extractor = KeyphraseExtractor(
            spacy_model=spacy_model,
            method=kp_cfg.get("method", "hybrid"),
            top_n=kp_cfg.get("top_n", 10)
        )

        emb_cfg = self.config.get("embeddings", {})
        self.embedder = BugEmbedder(
            model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
            device=emb_cfg.get("device", "cpu"),
            normalize=emb_cfg.get("normalize", True),
            batch_size=emb_cfg.get("batch_size", 32)
        )

        faiss_cfg = self.config.get("faiss", {})
        self.faiss_store = FaissBugStore(
            embedding_dim=self.embedder.embedding_dim,
            metric=faiss_cfg.get("metric", "cosine"),
            index_type=faiss_cfg.get("index_type", "Flat")
        )

        index_path = str(self.root_dir / faiss_cfg.get("index_path", "faiss_index/bug_index.faiss"))
        metadata_path = str(self.root_dir / faiss_cfg.get("metadata_path", "faiss_index/bug_metadata.json"))

        if load_faiss:
            if Path(index_path).exists() or Path(index_path).with_suffix(".pkl").exists():
                try:
                    self.faiss_store.load(index_path, metadata_path)
                except Exception as e:
                    logger.warning(f"Could not load existing FAISS index ({e}). Index remains empty until build.")
            else:
                logger.warning(f"FAISS index file not found at {index_path}. Build index via pipeline.build_faiss_index().")

        self.retriever = BugRetriever(
            embedder=self.embedder,
            faiss_store=self.faiss_store,
            top_k=faiss_cfg.get("top_k", 5)
        )

        llm_cfg = self.config.get("llm", {})
        self.prompt_manager = PromptManager(self.prompts_path)
        self.llm_client = LLMClient(
            provider=llm_cfg.get("provider", "groq"),
            model=llm_cfg.get("model", "llama-3.1-70b-versatile"),
            config=llm_cfg
        )

        modules_list = self.config.get("modules_list", [])
        teams_list = self.config.get("teams_list", [])

        self.predictor = BugPredictor(
            llm_client=self.llm_client,
            prompt_manager=self.prompt_manager,
            modules_list=modules_list,
            teams_list=teams_list
        )

        self.initialized = True
        logger.info("Bug Analysis Pipeline initialized successfully.")

    def process_nlp(self, title: str, description: str) -> Dict[str, Any]:
        """Executes 6-stage NLP processing: preprocessing, tokenization, entity extraction, and keyphrase extraction."""
        if not self.initialized:
            self.initialize()

        prep_res = self.preprocessor.preprocess_for_llm(title, description)
        clean_text = prep_res["clean_text"]

        tokens = self.tokenizer.tokenize(clean_text, keep_stop=False, keep_punct=False)
        entities_res = self.entity_extractor.extract_all(clean_text)
        kp_res = self.keyphrase_extractor.extract(clean_text)

        return {
            "title": title,
            "description": description,
            "clean_text": clean_text,
            "combined_clean": clean_text,
            "sentences": prep_res["sentences"],
            "tokens": tokens,
            "entities": entities_res,
            "keyphrases": kp_res["keyphrases"],
            "keyphrase_details": kp_res,
            "search_query": kp_res["search_query"]
        }

    def retrieve_context(self, nlp_data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Queries FAISS store using keyphrases / combined text and formats prompt context."""
        query = nlp_data.get("search_query") or nlp_data.get("clean_text", "")
        if not query or self.faiss_store.index.ntotal == 0:
            return "No historical context available in vector store.", []

        try:
            context_str, raw_results = self.retriever.retrieve_and_format(query)
            return context_str, raw_results
        except Exception as e:
            logger.warning(f"Retrieval error: {e}")
            return "No historical context available due to retrieval error.", []

    def analyze(self, title: str, description: str) -> Dict[str, Any]:
        """End-to-End Bug Analysis entry point."""
        if not self.initialized:
            self.initialize()

        nlp_data = self.process_nlp(title, description)
        historical_context, raw_retrieval = self.retrieve_context(nlp_data)
        prediction = self.predictor.predict(
            title=title,
            description=description,
            nlp_data=nlp_data,
            historical_context=historical_context
        )

        output = {
            "input": {
                "title": title,
                "description": description
            },
            "nlp": {
                "clean_text": nlp_data["clean_text"],
                "tokens": nlp_data["tokens"][:20],
                "entities": nlp_data["entities"]["software_entities"],
                "keyphrases": nlp_data["keyphrases"]
            },
            "prediction": prediction,
            "_internal": {
                "retrieval_count": len(raw_retrieval),
                "historical_used": len(raw_retrieval) > 0
            }
        }

        return output

    def collect_dataset(self) -> str:
        """Executes issue collector to fetch GitHub issues or fallback sample data."""
        ds_cfg = self.config.get("dataset", {})
        repos = ds_cfg.get("repos", [])
        max_per_repo = ds_cfg.get("max_issues_per_repo", 50)
        output_raw = str(self.root_dir / ds_cfg.get("output_raw", "data/raw/github_issues_raw.json"))

        collector = GitHubIssueCollector()
        issues = collector.collect_all(repos, max_per_repo=max_per_repo)
        collector.save(issues, json_path=output_raw, csv_path=output_raw.replace(".json", ".csv"))
        return output_raw

    def clean_dataset(self, raw_path: Optional[str] = None) -> str:
        """Sanitizes and deduplicates collected issue datasets."""
        ds_cfg = self.config.get("dataset", {})
        if raw_path is None:
            raw_path = str(self.root_dir / ds_cfg.get("output_raw", "data/raw/github_issues_raw.json"))

        out_processed = str(self.root_dir / ds_cfg.get("output_processed", "data/processed/github_issues_clean.json"))

        if not Path(raw_path).exists():
            logger.info("Raw dataset file missing. Generating raw dataset first...")
            raw_path = self.collect_dataset()

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_issues = json.load(f)

        cleaner = BugDatasetCleaner(self.config)
        cleaned_issues = cleaner.clean_all(raw_issues)
        cleaner.save(cleaned_issues, out_processed)

        return out_processed

    def build_faiss_index(self, cleaned_path: Optional[str] = None) -> Dict[str, Any]:
        """Computes embeddings for cleaned issues and builds FAISS vector index."""
        if not self.initialized:
            self.initialize(load_faiss=False)

        ds_cfg = self.config.get("dataset", {})
        if cleaned_path is None:
            cleaned_path = str(self.root_dir / ds_cfg.get("output_processed", "data/processed/github_issues_clean.json"))

        if not Path(cleaned_path).exists():
            logger.info("Cleaned dataset missing. Running cleaner first...")
            cleaned_path = self.clean_dataset()

        with open(cleaned_path, "r", encoding="utf-8") as f:
            issues = json.load(f)

        if not issues:
            logger.warning("Empty dataset for indexing. Loading sample dataset...")
            issues = GitHubIssueCollector.load_sample_dataset()

        embeddings = self.embedder.embed_issues(issues)
        self.faiss_store.build_from_scratch(embeddings, metadatas=issues)

        faiss_cfg = self.config.get("faiss", {})
        index_path = str(self.root_dir / faiss_cfg.get("index_path", "faiss_index/bug_index.faiss"))
        metadata_path = str(self.root_dir / faiss_cfg.get("metadata_path", "faiss_index/bug_metadata.json"))

        self.faiss_store.save(index_path, metadata_path)
        stats = self.faiss_store.get_stats()
        logger.info(f"FAISS vector index built successfully: {stats}")
        return stats
