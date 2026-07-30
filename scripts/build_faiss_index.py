import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

load_dotenv()

from src.pipeline import BugAnalysisPipeline

def main():
    print("=== STEP 3: Building FAISS Vector Index ===")
    pipeline = BugAnalysisPipeline()
    stats = pipeline.build_faiss_index()
    print(f"FAISS Index successfully built with stats: {stats}")

if __name__ == "__main__":
    main()
