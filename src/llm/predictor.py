from typing import Dict, List, Any, Optional
from src.utils import logger
from src.llm.llm_client import LLMClient
from src.llm.prompt_manager import PromptManager

class BugPredictor:
    """Predicts bug category, severity, module, causes, team, tests, confidence, and technical explanation."""

    REQUIRED_FIELDS = [
        "category", "severity", "affected_module", "possible_causes",
        "suggested_team", "recommended_tests", "confidence_score", "explanation"
    ]

    CATEGORIES = [
        "Crash", "Performance", "Security", "UI/UX", "Logic Error",
        "Compatibility", "Network", "Data Corruption", "Authentication", "Configuration"
    ]

    SEVERITIES = ["Critical", "High", "Medium", "Low"]

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        modules_list: List[str],
        teams_list: List[str]
    ):
        self.llm = llm_client
        self.prompts = prompt_manager
        self.modules_list = modules_list
        self.teams_list = teams_list

    def validate_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and enforces output JSON schema invariants and fallback values."""
        validated = dict(pred)

        cat = validated.get("category", "")
        matched_cat = None
        for allowed in self.CATEGORIES:
            if allowed.lower() in str(cat).lower():
                matched_cat = allowed
                break
        validated["category"] = matched_cat or "Logic Error"

        sev = validated.get("severity", "")
        matched_sev = None
        for allowed in self.SEVERITIES:
            if allowed.lower() in str(sev).lower():
                matched_sev = allowed
                break
        validated["severity"] = matched_sev or "Medium"

        mod = validated.get("affected_module", "")
        matched_mod = None
        for allowed in self.modules_list:
            if allowed.lower() in str(mod).lower():
                matched_mod = allowed
                break
        validated["affected_module"] = matched_mod or (self.modules_list[0] if self.modules_list else "General Component")

        team = validated.get("suggested_team", "")
        matched_team = None
        for allowed in self.teams_list:
            if allowed.lower() in str(team).lower():
                matched_team = allowed
                break
        validated["suggested_team"] = matched_team or (self.teams_list[0] if self.teams_list else "Backend Services")

        causes = validated.get("possible_causes", [])
        if not isinstance(causes, list) or not causes:
            causes = ["Unhandled exception during operation processing.", "Boundary condition check missing."]
        validated["possible_causes"] = [str(c) for c in causes[:4]]

        tests = validated.get("recommended_tests", [])
        if not isinstance(tests, list) or not tests:
            tests = [
                "Execute standard unit test suite for target module.",
                "Perform load test under high concurrency conditions.",
                "Verify boundary value validation on input fields.",
                "Check error logging and graceful failure fallback."
            ]
        validated["recommended_tests"] = [str(t) for t in tests[:6]]

        try:
            val = float(validated.get("confidence_score", 0.85))
            validated["confidence_score"] = round(min(max(val, 0.0), 1.0), 2)
        except Exception:
            validated["confidence_score"] = 0.85

        exp = validated.get("explanation", "").strip()
        if not exp:
            validated["explanation"] = "Target module encountered an unhandled exception under specified execution parameters. Further investigation of stack trace log points is recommended."
        validated["explanation"] = exp

        return validated

    def fallback_heuristic(self, title: str, description: str, nlp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based heuristic prediction engine for standalone execution without LLM keys."""
        logger.info("Executing Rule-Based NLP Heuristic Engine for bug analysis...")
        text_lower = f"{title} {description}".lower()

        if any(k in text_lower for k in ["crash", "hang", "freeze", "terminate", "oom", "out of memory", "fatal"]):
            category = "Crash"
        elif any(k in text_lower for k in ["slow", "lag", "timeout", "30s", "latency", "performance", "high cpu"]):
            category = "Performance"
        elif any(k in text_lower for k in ["auth", "login", "token", "session", "jwt", "permission"]):
            category = "Authentication"
        elif any(k in text_lower for k in ["security", "xss", "sqli", "injection", "vulnerability", "leak"]):
            category = "Security"
        elif any(k in text_lower for k in ["ui", "button", "css", "align", "display", "render", "screen"]):
            category = "UI/UX"
        elif any(k in text_lower for k in ["network", "http", "504", "502", "connection", "socket"]):
            category = "Network"
        elif any(k in text_lower for k in ["payment", "charge", "idempotency", "billing"]):
            category = "Logic Error"
        else:
            category = "Logic Error"

        if any(k in text_lower for k in ["crash", "freeze", "fatal", "security", "double charge", "data loss", "cannot login"]):
            severity = "Critical"
        elif any(k in text_lower for k in ["fail", "error", "timeout", "broken", "major"]):
            severity = "High"
        elif any(k in text_lower for k in ["minor", "typo", "cosmetic", "alignment"]):
            severity = "Low"
        else:
            severity = "Medium"

        default_mod = self.modules_list[0] if self.modules_list else "General Component"
        default_team = self.teams_list[0] if self.teams_list else "Backend Services"

        if any(k in text_lower for k in ["upload", "image", "file", "avatar", "20mb", "picture"]):
            module = "Image Upload Service"
            team = "Backend Services"
            causes = [
                "Missing file size payload validation prior to memory buffer allocation.",
                "Uncaught memory limit exception in image processing pipeline."
            ]
            tests = [
                "Upload PNG image smaller than 10MB.",
                "Upload JPEG image larger than 20MB to verify size validation.",
                "Upload corrupted or zero-byte image files.",
                "Verify server memory usage and graceful error response under load.",
                "Test upload retry after connection reset."
            ]
        elif any(k in text_lower for k in ["auth", "login", "session", "freeze"]):
            module = "Authentication & AuthZ"
            team = "Backend Services"
            causes = [
                "Session token validation deadlock in worker queue thread.",
                "Synchronous database call blocking main event loop during credential check."
            ]
            tests = [
                "Submit valid user login credentials.",
                "Submit invalid credentials to test lockouts.",
                "Simulate high concurrent login requests.",
                "Verify session token expiration behavior."
            ]
        elif any(k in text_lower for k in ["payment", "charge", "idempotency"]):
            module = "Payment Gateway"
            team = "Backend Services"
            causes = [
                "Missing client-side idempotency header token.",
                "Network timeout triggering automatic non-idempotent retry."
            ]
            tests = [
                "Submit payment request with slow network throttle (3G).",
                "Double-click payment submit button rapidly.",
                "Verify idempotency key enforcement on API gateway.",
                "Verify transaction rollback on gateway timeout."
            ]
        else:
            module = default_mod
            team = default_team
            causes = [
                "Unhandled edge condition in logic workflow.",
                "Missing input sanitization or validation boundary check."
            ]
            tests = [
                "Execute boundary value test suite.",
                "Verify error log outputs.",
                "Test invalid payload formats.",
                "Perform standard functional workflow regression test."
            ]

        explanation = (
            f"Rule-Based NLP Analysis: Identified critical patterns for '{category}' category "
            f"within '{module}'. The issue exhibits characteristics of {severity.lower()} impact requiring immediate review."
        )

        return {
            "category": category,
            "severity": severity,
            "affected_module": module,
            "possible_causes": causes,
            "suggested_team": team,
            "recommended_tests": tests,
            "confidence_score": 0.65,
            "explanation": explanation,
            "_fallback": True
        }

    def predict(
        self,
        title: str,
        description: str,
        nlp_data: Dict[str, Any],
        historical_context: str = ""
    ) -> Dict[str, Any]:
        """Runs LLM prediction with automated fallback to rule-based engine on missing keys or network errors."""
        clean_text = nlp_data.get("clean_text") or f"{title} - {description}"
        tokens = nlp_data.get("tokens", [])
        entities = nlp_data.get("entities", {})
        keyphrases = nlp_data.get("keyphrases", [])

        system_prompt, user_prompt = self.prompts.render_bug_analysis(
            title=title,
            description=description,
            clean_text=clean_text,
            tokens=tokens,
            entities=entities,
            keyphrases=keyphrases,
            historical_context=historical_context,
            modules_list=self.modules_list,
            teams_list=self.teams_list
        )

        prediction = None
        if self.llm.is_available():
            try:
                raw_pred = self.llm.generate_json(system_prompt, user_prompt)
                prediction = self.validate_prediction(raw_pred)
                prediction["_fallback"] = False
                logger.info(f"LLM Prediction successfully generated via {self.llm.provider}.")
            except Exception as e:
                logger.error(f"LLM Prediction failed ({e}). Reverting to rule-based heuristic.")
                prediction = None

        if prediction is None:
            prediction = self.fallback_heuristic(title, description, nlp_data)

        prediction["_meta"] = {
            "provider": self.llm.provider if self.llm.is_available() else "heuristic_fallback",
            "model": self.llm.model if self.llm.is_available() else "rule-based-nlp-v1",
            "nlp_tokens_count": len(tokens),
            "keyphrases_used": len(keyphrases)
        }

        return prediction
