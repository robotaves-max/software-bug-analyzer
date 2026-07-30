import sys
from pathlib import Path
from dotenv import load_dotenv

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

load_dotenv()

from src.pipeline import BugAnalysisPipeline

def main():
    print("=== STEP 1: Collecting GitHub Issues Dataset ===")
    pipeline = BugAnalysisPipeline()
    output_path = pipeline.collect_dataset()
    print(f"Dataset collection complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
