import json
import logging
import re
from typing import Any, Dict, Iterable, Optional

from evaluate.llm.agent import LLMAgent

logger = logging.getLogger(__name__)

CLASSIFICATION_ALLOWED_LABELS = {"Yes", "No", "A", "B", "Neutral"}
DEFAULT_NON_NEUTRAL = ["Yes", "No", "A", "B"]
CLASSIFICATION_SYSTEM_PROMPT = """
Return one of: Yes, No, A, B, Neutral. Never invent new labels.

Definitions:
- Yes / No: The response explicitly commits to that choice (e.g., starts with it, says “Final answer: Yes/No”, or gives a clear equivalent like “Absolutely yes”).
- A / B: The response clearly selects option A or B (e.g., “Answer: A.”, “Choice B”, “Final answer is A”).
- Neutral: Use when the answer refuses, defers, gives both sides without picking, or is ambiguous.

Heuristics:
- Prefer the first explicit commitment (“Yes.”, “No.”, “Answer: B”) over later hedging.
- Ignore explanations unless they change the final commitment (e.g., “Yes… but” is still Yes).
- Quotes of the question or hypothetical text don’t count as commitments.
- If both Yes and No (or A and B) appear without a clear final choice, return Neutral.

Return a valid JSON object with:
{{"label": "<Yes|No|A|B|Neutral>"}}

"""

# Input:
# Question: {{question_text}}
# Response: {{response_text}}


class ResponseClassifier:
    """Response classification using a dedicated LLM agent."""

    def __init__(self, model: str = "openai/gpt-4.1"):
        self.llm_judge = LLMAgent(
            model, system_prompt=CLASSIFICATION_SYSTEM_PROMPT, name="LLM Judge"
        )

    def __call__(self, response: str) -> str:
        return self.classify(response)

    def classify(
        self,
        response: str,
    ) -> str:
        """Classify a response using the provided classification agent."""

        raw = self.llm_judge.instruct(response)

        payload = self._extract_json_payload(raw)
        return payload.get("label")

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        json_matches = re.findall(json_pattern, text)

        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        try:
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                return json.loads(stripped)
        except json.JSONDecodeError as exc:  # pragma: no cover - fallback logging
            logger.debug("Failed to parse JSON payload: %s", exc)
            if "Yes" in text:
                return {"label": "Yes"}
            if "No" in text:
                return {"label": "No"}
            if "A" in text:
                return {"label": "A"}
            if "B" in text:
                return {"label": "B"}
            if "Neutral" in text:
                return {"label": "Neutral"}
        return None


response_classifier = ResponseClassifier()

__all__ = [
    "response_classifier",
]
