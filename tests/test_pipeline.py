import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset.collector import GitHubIssueCollector
from src.dataset.cleaner import BugDatasetCleaner
from src.nlp.preprocessor import TextPreprocessor
from src.nlp.tokenizer import BugTokenizer
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.keyphrase_extractor import KeyphraseExtractor
from src.embeddings.embedder import BugEmbedder
from src.retrieval.faiss_store import FaissBugStore
from src.retrieval.retriever import BugRetriever
from src.pipeline import BugAnalysisPipeline

class TestBugAnalyzerPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = BugAnalysisPipeline()

    def test_01_sample_dataset_generator(self):
        issues = GitHubIssueCollector.load_sample_dataset()
        self.assertEqual(len(issues), 50)
        self.assertIn("title", issues[0])
        self.assertIn("description", issues[0])

    def test_02_dataset_cleaner(self):
        cleaner = BugDatasetCleaner()
        raw = "Check log at File \"main.py\", line 12. Fix bug https://example.com ```code block```"
        cleaned = cleaner.clean_text(raw)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("code block", cleaned)

    def test_03_preprocessor_and_tokenizer(self):
        prep = TextPreprocessor()
        tok = BugTokenizer()

        text = "Application crashes while uploading images >20MB."
        res = prep.preprocess_for_llm("Upload Crash", text)
        self.assertIn("clean_text", res)

        tokens = tok.tokenize(text, keep_stop=False)
        self.assertIn("crashes", tokens)
        self.assertIn("uploading", tokens)

    def test_04_entity_extractor(self):
        extractor = EntityExtractor()
        text = "App crashes in auth_controller.py on line 42 with ERR_OOM_500 in v1.2.3 on Windows 11."
        res = extractor.extract_software_entities(text)

        self.assertIn("auth_controller.py", res["file_paths"])
        self.assertIn("ERR_OOM_500", res["error_codes"])
        self.assertIn("v1.2.3", res["versions"])
        self.assertIn("windows", res["os_browsers"])

    def test_05_keyphrase_extractor(self):
        kp_extractor = KeyphraseExtractor()
        text = "Application freezes after clicking login button due to session token validation deadlock."
        kp_res = kp_extractor.extract(text)
        self.assertGreater(len(kp_res["keyphrases"]), 0)

    def test_06_embedder_and_faiss_store(self):
        embedder = BugEmbedder()
        issues = GitHubIssueCollector.load_sample_dataset()[:5]
        embeddings = embedder.embed_issues(issues)
        self.assertEqual(embeddings.shape[0], 5)

        store = FaissBugStore(embedding_dim=embedder.embedding_dim)
        store.build_from_scratch(embeddings, metadatas=issues)
        self.assertEqual(store.index.ntotal, 5)

        retriever = BugRetriever(embedder, store, top_k=2)
        results = retriever.retrieve("Login freezes on click")
        self.assertEqual(len(results), 2)

    def test_07_full_pipeline_analyze_and_schema(self):
        title = "Application crashes while uploading large profile pictures"
        desc = "Uploading images larger than 20MB causes the application to terminate unexpectedly with OutOfMemory error."

        result = self.pipeline.analyze(title, desc)

        self.assertIn("input", result)
        self.assertIn("nlp", result)
        self.assertIn("prediction", result)
        self.assertIn("_internal", result)

        pred = result["prediction"]
        self.assertIn("category", pred)
        self.assertIn("severity", pred)
        self.assertIn("affected_module", pred)
        self.assertIn("possible_causes", pred)
        self.assertIn("suggested_team", pred)
        self.assertIn("recommended_tests", pred)
        self.assertIn("confidence_score", pred)
        self.assertIn("explanation", pred)

        self.assertIsInstance(pred["possible_causes"], list)
        self.assertIsInstance(pred["recommended_tests"], list)

if __name__ == "__main__":
    unittest.main()
