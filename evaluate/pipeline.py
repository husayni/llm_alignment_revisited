import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich import print
from tqdm.auto import tqdm

from .llm.agent import LLMAgent
from .utils import (
    PromptLoader,
    PromptMetrics,
    compute_prompt_metrics,
    response_classifier,
    slugify_model_name,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(
        self,
        agent: LLMAgent,
        loader: PromptLoader,
        base_output: Path,
        vr_output: Path,
        classifier_agent,
    ) -> None:
        self.agent = agent
        self.loader = loader
        self.base_output = base_output
        self.vr_output = vr_output
        self.classifier = classifier_agent
        self.model_suffix = slugify_model_name(self.agent.model)
        self._base_stats: Dict[str, Dict[str, Any]] = {}
        self._context_stats: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, PromptMetrics] = {}

    def run_base(self) -> None:
        logger.info("Running evaluation on base prompts")

        promt_sets = self.loader.load_base_prompts()

        for prompt_set in promt_sets:
            logger.info(f"Processing {prompt_set.name}")
            payload = prompt_set.payload.copy()
            counts: Counter[str] = Counter()

            for idx, question in enumerate(
                tqdm(
                    prompt_set.questions,
                    desc=f"{prompt_set.name} questions: ",
                    leave=False,
                    unit="question",
                )
            ):
                response = self.agent.instruct(question)
                label = self.classifier(response)
                counts[label] += 1
                payload[idx] = {
                    "question": question,
                    "response": response,
                    "label": label,
                }

            out_path = self.base_output / f"{prompt_set.name}.json"
            with out_path.open("w", encoding="utf-8") as handle:
                logger.info(f"Writing output to {out_path}")
                json.dump(payload, handle, indent=2)
            self._base_stats[prompt_set.canonical_id] = {
                "counts": counts,
                "total": len(prompt_set.questions),
            }

    def run_vr(self) -> None:
        logger.info("Running evaluation on VR prompts")

        prompt_sets = self.loader.load_vr_prompts()

        for prompt_set in prompt_sets:
            payload = prompt_set.payload.copy()
            counts: Counter[str] = Counter()

            for key, question in tqdm(
                prompt_set.variants.items(),
                desc=f"{prompt_set.name} variants",
                leave=False,
                unit="variant",
            ):
                response = self.agent.instruct(question)
                label = self.classifier(response)
                counts[label] += 1
                payload[f"{key}-response"] = response
                payload[f"{key}-label"] = label

            out_path = self.vr_output / f"{prompt_set.name}.json"
            with out_path.open("w", encoding="utf-8") as handle:
                logger.info(f"Writing output to {out_path}")
                json.dump(payload, handle, indent=2)

            self._context_stats[prompt_set.canonical_id] = {
                "counts": counts,
                "total": len(prompt_set.variants),
            }

    def run(self, evaluate_base: bool, evaluate_vr: bool) -> Dict[str, PromptMetrics]:
        if evaluate_base:
            self.run_base()

        if evaluate_vr:
            self.run_vr()

        return self._build_metrics()
        

    def _build_metrics(self) -> None:
        
        neutral_prompts = 0
        total_prompts = 0
        for stats in self._base_stats.values(): 
            neutral_prompts += stats['counts'].get("Neutral", 0)
            total_prompts += stats['total']

        base_neutrality = neutral_prompts / total_prompts

        for stats in self._context_stats.values(): 
            neutral_prompts += stats['counts'].get("Neutral", 0)
            total_prompts += stats['total']

        vr_neutrality = neutral_prompts / total_prompts

        
        
        metrics = [{"base_neutrality": base_neutrality, "vr_neutrality": vr_neutrality}]
        for conical_id in self._context_stats.keys() & self._base_stats.keys():
            base_results = self._base_stats[conical_id]['counts']
            context_results = self._context_stats[conical_id]['counts']
            prompt_metrics = compute_prompt_metrics(base_results, context_results, canonical_id=conical_id)
            metrics.append(prompt_metrics)
        return metrics

                


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


@dataclass
class EvaluateConfig:
    model: str
    evaluate_base: bool = True
    evaluate_vr: bool = True
    metrics: bool = True
    dataset_root: Optional[Path] = Path("dataset")
    output: Path = Path("runs")

    def as_serializable_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of the config."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return {key: convert(value) for key, value in self.__dict__.items()}


def main(cfg: EvaluateConfig) -> None:
    if not (cfg.evaluate_base or cfg.evaluate_vr):
        raise ValueError("Enable at least one of --evaluate_base or --evaluate_vr")

    loader = PromptLoader(dataset_root=cfg.dataset_root)

    agent = LLMAgent(name="Evaluation Agent", model=cfg.model)

    model_slug = slugify_model_name(cfg.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output / f"{model_slug}_{timestamp}"
    base_output = run_dir / "base"
    vr_output = run_dir / "vr"
    if cfg.evaluate_base:
        base_output.mkdir(parents=True, exist_ok=True)
    if cfg.evaluate_vr:
        vr_output.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.as_serializable_dict(), f, indent=2)

    runner = EvaluationPipeline(
        agent=agent,
        loader=loader,
        base_output=base_output,
        vr_output=vr_output,
        classifier_agent=response_classifier,
    )
    metrics = runner.run(evaluate_base=cfg.evaluate_base, evaluate_vr=cfg.evaluate_vr)

    for metric in metrics:
        print(metric)
    # if cfg.metrics:
    #     metrics_path = run_dir / "metrics.json"
    #     metrics_path.parent.mkdir(parents=True, exist_ok=True)
    #     with metrics_path.open("w", encoding="utf-8") as handle:
    #         json.dump(
    #             {key: metric.to_dict() for key, metric in metrics.items()},
    #             handle,
    #             indent=2,
    #         )

    #     for canonical_id, metric in sorted(metrics.items()):
    #         base_neutral = _format_ratio(metric.base_neutrality)
    #         context_neutral = _format_ratio(metric.context_neutrality)
    #         abs_dev = (
    #             f"{metric.absolute_deviation:.3f}"
    #             if metric.absolute_deviation is not None
    #             else "n/a"
    #         )
    #         kl_value = f"{metric.kl_divergence:.3f}" if metric.kl_divergence is not None else "n/a"
    #         print(
    #             f"{canonical_id}: base neutral {base_neutral}, context neutral {context_neutral}, "
    #             f"abs dev {abs_dev}, KL {kl_value}"
    #         )
