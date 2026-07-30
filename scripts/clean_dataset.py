import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

load_dotenv()

from src.pipeline import BugAnalysisPipeline

def main():
    print("=== STEP 2: Cleaning GitHub Issues Dataset ===")
    pipeline = BugAnalysisPipeline()
    output_path = pipeline.clean_dataset()
    print(f"Dataset cleaning complete. Sanitized file saved to: {output_path}")

if __name__ == "__main__":
    main()
