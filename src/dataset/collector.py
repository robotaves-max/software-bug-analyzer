import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils import logger

class GitHubIssueCollector:
    """Collects historical bug issues from public GitHub repositories via REST API v3."""

    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SoftwareBugAnalyzer/1.0"
        }
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

    def fetch_repo_issues(self, repo: str, max_issues: int = 100) -> List[Dict[str, Any]]:
        """Fetch issues for a specific GitHub repository using pagination and rate limit handling."""
        issues = []
        page = 1
        per_page = min(100, max_issues)
        url = f"https://api.github.com/repos/{repo}/issues"

        logger.info(f"Fetching issues for repository: {repo}")

        while len(issues) < max_issues:
            params = {
                "state": "closed",
                "per_page": per_page,
                "page": page
            }
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)

                if response.status_code == 403:
                    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                    sleep_duration = max(reset_time - int(time.time()), 1) + 2
                    logger.warning(f"Rate limited by GitHub API. Sleeping for {sleep_duration} seconds...")
                    time.sleep(min(sleep_duration, 10))
                    break

                if response.status_code != 200:
                    logger.error(f"Failed to fetch from {repo}: HTTP {response.status_code}")
                    break

                batch = response.json()
                if not batch:
                    break

                for item in batch:
                    if "pull_request" in item:
                        continue

                    labels = [label["name"] if isinstance(label, dict) else str(label) for label in item.get("labels", [])]
                    body = item.get("body") or ""
                    title = item.get("title") or ""
                    closed_at = item.get("closed_at") or "Unknown"

                    issue_dict = {
                        "id": item.get("id"),
                        "number": item.get("number"),
                        "repo": repo,
                        "title": title,
                        "description": body,
                        "labels": labels,
                        "state": item.get("state"),
                        "created_at": item.get("created_at"),
                        "closed_at": closed_at,
                        "comments_count": item.get("comments"),
                        "url": item.get("html_url"),
                        "combined_text": f"{title} - {body}",
                        "resolution_hint": f"Closed at {closed_at}. Labels: {', '.join(labels)}"
                    }
                    issues.append(issue_dict)
                    if len(issues) >= max_issues:
                        break

                page += 1
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching issues for {repo}: {e}")
                break

        logger.info(f"Collected {len(issues)} issues from {repo}")
        return issues

    def collect_all(self, repos: List[str], max_per_repo: int = 50) -> List[Dict[str, Any]]:
        """Collect issues across multiple repositories."""
        all_issues = []
        if self.github_token:
            for repo in repos:
                repo_issues = self.fetch_repo_issues(repo, max_issues=max_per_repo)
                all_issues.extend(repo_issues)

        if not all_issues:
            logger.warning("No live GitHub issues collected (missing token or rate limit). Loading fallback sample dataset.")
            all_issues = self.load_sample_dataset()

        return all_issues

    def save(self, issues: List[Dict[str, Any]], json_path: str, csv_path: Optional[str] = None):
        """Saves collected issues to JSON and optional CSV file."""
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(issues)} issues to JSON: {json_file}")

        if csv_path:
            try:
                csv_file = Path(csv_path)
                csv_file.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(issues)
                df.to_csv(csv_file, index=False, encoding="utf-8")
                logger.info(f"Saved {len(issues)} issues to CSV: {csv_file}")
            except Exception as e:
                logger.warning(f"CSV export failed: {e}")

    @staticmethod
    def load_sample_dataset() -> List[Dict[str, Any]]:
        """Generates a dataset of 50 sample bug issues derived from 5 core software bug templates."""
        templates = [
            {
                "title_template": "Login freezes / App freezes after clicking login button in auth module (v{version})",
                "desc_template": "When user submits credentials on line 42 in auth_controller.py, the UI thread hangs indefinitely. Stack trace: RuntimeError at AuthClient.connect() after 3000ms. Windows 11 / Chrome 120.",
                "labels": ["bug", "auth", "ui-freeze"],
                "resolution": "Fixed session token validation deadlocks in auth worker queue."
            },
            {
                "title_template": "Upload crash / Crash on uploading avatar images >20MB",
                "desc_template": "App crashes with OutOfMemory error while uploading profile pictures >20MB in image_uploader.ts. File path: /src/upload/service.ts. Error code: ERR_OOM_500. Linux kernel 5.15.",
                "labels": ["crash", "backend", "image-processor"],
                "resolution": "Added streaming file size validation and buffer limits prior to processing."
            },
            {
                "title_template": "API Gateway timeout / Requests fail after 30s during database query execution",
                "desc_template": "POST /api/v1/analytics returns HTTP 504 Gateway Timeout after 30 seconds. Database connector db_pool.py fails to release connection locks under heavy concurrency.",
                "labels": ["performance", "api", "database"],
                "resolution": "Increased connection pool size and added 10s query execution timeout."
            },
            {
                "title_template": "Payment double charge on network drop / Idempotency key missing",
                "desc_template": "User is charged twice when clicking pay button during slow 3G network conditions in checkout_view.jsx. Missing idempotency token in headers.",
                "labels": ["critical", "payment", "race-condition"],
                "resolution": "Enforced client-side idempotency keys and disabled retry buttons during in-flight payments."
            },
            {
                "title_template": "Search engine returns empty results when query contains special characters like % and _",
                "desc_template": "SQL wildcard characters % and _ in search query cause unescaped ILIKE queries in search_repository.py. Returns 0 records instead of exact matches.",
                "labels": ["search", "logic-error", "sql"],
                "resolution": "Escaped SQL wildcard tokens in query builder prior to database execution."
            }
        ]

        issues = []
        issue_id = 1001
        for i in range(50):
            tmpl = templates[i % len(templates)]
            version = f"1.{i % 5}.{i % 3}"
            title = tmpl["title_template"].format(version=version)
            desc = tmpl["desc_template"]
            repo_name = ["facebook/react", "python/cpython", "microsoft/vscode", "django/django", "nodejs/node"][i % 5]

            issues.append({
                "id": issue_id + i,
                "number": 100 + i,
                "repo": repo_name,
                "title": title,
                "description": desc,
                "labels": tmpl["labels"],
                "state": "closed",
                "created_at": "2024-01-15T10:00:00Z",
                "closed_at": "2024-01-16T14:30:00Z",
                "comments_count": 5 + (i % 7),
                "url": f"https://github.com/{repo_name}/issues/{100 + i}",
                "combined_text": f"{title} - {desc}",
                "resolution_hint": tmpl["resolution"]
            })

        logger.info(f"Generated {len(issues)} synthetic fallback bug issues across 5 core bug archetypes.")
        return issues
