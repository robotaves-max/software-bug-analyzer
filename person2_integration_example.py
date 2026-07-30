"""
Person 2 Integration Example Script

Demonstrates how Person 2 (Application Developer) imports and consumes Person 1 (AI Engineer Pipeline).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

load_dotenv()

from src.pipeline import BugAnalysisPipeline

def main():
    print("=== Person 2 Integration Example ===")
    
    # 1. Initialize Person 1 Pipeline
    pipeline = BugAnalysisPipeline(config_path="config/config.yaml", prompts_path="config/prompts.yaml")
    pipeline.initialize()

    # 2. Sample User Input from Person 2 Form
    title = "Application crashes while uploading large profile pictures"
    description = "Uploading images larger than 20MB causes the application to terminate unexpectedly with OutOfMemory error."

    print(f"\nUser Submitted Bug:")
    print(f"Title: {title}")
    print(f"Description: {description}\n")

    # 3. Call Person 1 Analysis
    result = pipeline.analyze(title, description)

    # 4. Inspect Output (Person 2 displays 'prediction' and 'nlp', hides '_internal')
    print("=== Received Output Payload from Person 1 ===")
    print("Prediction Category:", result["prediction"]["category"])
    print("Prediction Severity:", result["prediction"]["severity"])
    print("Affected Module:", result["prediction"]["affected_module"])
    print("Suggested Team:", result["prediction"]["suggested_team"])
    print("Confidence Score:", result["prediction"]["confidence_score"])
    print("Possible Causes:", result["prediction"]["possible_causes"])
    print("Recommended Tests:", result["prediction"]["recommended_tests"])
    print("Explanation:", result["prediction"]["explanation"])
    
    print("\nExtracted Keyphrases:", result["nlp"]["keyphrases"])
    print("Extracted Software Entities:", result["nlp"]["entities"])

if __name__ == "__main__":
    main()
