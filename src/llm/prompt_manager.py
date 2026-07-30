import json
from pathlib import Path
from typing import Dict, Any, Tuple
from src.utils import logger, load_prompts

class PromptManager:
    """Manages prompt template loading and variable rendering for LLM calls."""

    def __init__(self, prompts_path: str = "config/prompts.yaml"):
        self.prompts_path = prompts_path
        self.prompts = load_prompts(prompts_path)
        logger.info(f"Loaded prompt manager with templates: {list(self.prompts.keys())}")

    def get(self, key: str) -> Dict[str, str]:
        """Retrieves system and user template dictionary for a given key."""
        if key not in self.prompts:
            raise KeyError(f"Prompt key '{key}' not found in {self.prompts_path}")
        return self.prompts[key]

    def render(self, key: str, **kwargs) -> Tuple[str, str]:
        """Renders system prompt and user template with keyword arguments."""
        tmpl = self.get(key)
        system_prompt = tmpl.get("system", "").strip()
        user_template = tmpl.get("user_template", "").strip()

        try:
            user_prompt = user_template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing key in prompt format: {e}. Performing safe fallback replace.")
            user_prompt = user_template
            for k, v in kwargs.items():
                user_prompt = user_prompt.replace(f"{{{k}}}", str(v))

        return system_prompt, user_prompt

    def render_bug_analysis(
        self,
        title: str,
        description: str,
        clean_text: str,
        tokens: list,
        entities: dict,
        keyphrases: list,
        historical_context: str,
        modules_list: list,
        teams_list: list
    ) -> Tuple[str, str]:
        """Specific renderer for bug analysis prompt with token & phrase truncation."""
        tokens_str = json.dumps(tokens[:50])
        entities_str = json.dumps(entities.get("structured_summary", entities))
        keyphrases_str = json.dumps(keyphrases[:10])
        modules_str = json.dumps(modules_list)
        teams_str = json.dumps(teams_list)

        return self.render(
            "bug_analysis",
            title=title,
            description=description,
            clean_text=clean_text,
            tokens=tokens_str,
            entities=entities_str,
            keyphrases=keyphrases_str,
            historical_context=historical_context or "No historical context available.",
            modules_list=modules_str,
            teams_list=teams_str
        )
