import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

load_dotenv()

from src.pipeline import BugAnalysisPipeline

console = Console()

TEST_BUGS = [
    {
        "title": "Application crashes while uploading large profile pictures",
        "description": "Uploading images larger than 20MB causes the application to terminate unexpectedly with an OutOfMemory error. Occurs in image_uploader.ts on line 88."
    },
    {
        "title": "App freezes after clicking login button in auth module (v1.4.0)",
        "description": "When user submits credentials on line 42 in auth_controller.py, the UI thread hangs indefinitely. Stack trace: RuntimeError at AuthClient.connect() after 3000ms. Windows 11 / Chrome 120."
    },
    {
        "title": "API Gateway timeout / Requests fail after 30s during database query execution",
        "description": "POST /api/v1/analytics returns HTTP 504 Gateway Timeout after 30 seconds. Database connector db_pool.py fails to release connection locks under heavy concurrency."
    },
    {
        "title": "Payment double charge on slow network connection / Idempotency key missing",
        "description": "User is charged twice when clicking pay button during slow 3G network conditions in checkout_view.jsx. Missing idempotency token in request headers."
    }
]

def main():
    console.print(
        Panel.fit(
            "[bold green]Software Bug Analyzer - AI Pipeline Demo[/bold green]", 
            subtitle="RAG + LLM"
        )
    )

    pipeline = BugAnalysisPipeline()

    index_path = ROOT_DIR / "faiss_index" / "bug_index.faiss"
    pkl_path = ROOT_DIR / "faiss_index" / "bug_index.pkl"
    if not index_path.exists() and not pkl_path.exists():
        console.print("[yellow]Building initial FAISS knowledge base index from sample dataset...[/yellow]")
        pipeline.build_faiss_index()

    pipeline.initialize()

    for idx, test_case in enumerate(TEST_BUGS, 1):
        console.rule(f"[bold cyan]Test Scenario #{idx}: {test_case['title']}[/bold cyan]")
        console.print(f"[bold gray]Description:[/bold gray] {test_case['description']}\n")

        with console.status(
            "[bold green]Analyzing Bug (Preprocessing -> Embeddings -> FAISS -> LLM)...[/bold green]"
        ):
            result = pipeline.analyze(test_case["title"], test_case["description"])

        nlp_info = result["nlp"]
        console.print("[bold yellow]1. Extracted Features:[/bold yellow]")
        console.print(f"  * Tokens: {nlp_info['tokens']}")
        console.print(f"  * Entities: {nlp_info['entities']}")
        console.print(f"  * Keyphrases: {nlp_info['keyphrases']}")
        console.print(f"  * Cleaned Text: {nlp_info['clean_text'][:120]}...\n")

        pred_info = result["prediction"]
        console.print("[bold yellow]2. AI Prediction Output:[/bold yellow]")
        console.print(JSON(json.dumps(pred_info)))
        console.print("\n" + "="*80 + "\n")

    console.print(
        "[bold green]SUCCESS: All 4 Test Scenarios Completed Successfully![/bold green]")

if __name__ == "__main__":
    main()
