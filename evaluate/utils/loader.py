import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class BasePromptSet:
    name: str
    canonical_id: str
    questions: List[str]
    payload: Dict[str, Any]


@dataclass
class VRPromptSet:
    name: str
    canonical_id: str
    base_prompt: str
    variants: Dict[str, str]
    payload: Dict[str, Any]


class PromptLoader:
    def __init__(
        self,
        dataset_root: Optional[Path] = None,
        base_prompt_dir: Optional[Path] = None,
        vr_prompt_dir: Optional[Path] = None,
    ) -> None:
        root = Path(dataset_root) if dataset_root else Path("dataset")
        self.base_prompt_dir = (
            Path(base_prompt_dir) if base_prompt_dir else root / "base_prompt_question_list"
        )
        self.vr_prompt_dir = Path(vr_prompt_dir) if vr_prompt_dir else root / "prompts"

    def iter_base_prompts(self) -> Iterator[BasePromptSet]:
        for path in sorted(self.base_prompt_dir.glob("*.json")):
            payload = self._read_json(path)
            questions = payload.get("question_list", [])
            canonical_id = normalize_base_stem(path.stem)
            yield BasePromptSet(
                name=path.stem,
                canonical_id=canonical_id,
                questions=list(questions),
                payload=payload,
            )

    def load_base_prompts(self) -> List[BasePromptSet]:
        return list(self.iter_base_prompts())

    def iter_vr_prompts(self) -> Iterator[VRPromptSet]:
        for path in sorted(self.vr_prompt_dir.glob("*.json")):
            payload = self._read_json(path)
            base_prompt = payload.get("base_prompt", "")
            variants = {key: value for key, value in payload.items() if key != "base_prompt"}
            yield VRPromptSet(
                name=path.stem,
                canonical_id=path.stem,
                base_prompt=base_prompt,
                variants=variants,
                payload=payload,
            )

    def load_vr_prompts(self) -> List[VRPromptSet]:
        return list(self.iter_vr_prompts())

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def slugify_model_name(model: str) -> str:
    return model.lower().replace("/", "_").replace(" ", "-")


def normalize_base_stem(stem: str) -> str:
    return stem.replace("_base_prompt_questions", "")


__all__ = [
    "BasePromptSet",
    "VRPromptSet",
    "PromptLoader",
    "slugify_model_name",
    "normalize_base_stem",
]
