"""
A/B Testing Analytics Module

Provides comprehensive A/B testing framework for comparing retrieval methods.
Tracks metrics, stores results, and enables comparison data export.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class TestVariant(Enum):
    """A/B test variants for retrieval methods."""
    CONTROL = "semantic"  # Semantic search only (baseline)
    VARIANT_A = "bm25"  # BM25 only
    VARIANT_B = "hybrid"  # Hybrid (BM25 + semantic)
    VARIANT_C = "hybrid_rerank"  # Hybrid with re-ranking


@dataclass
class RetrievalMetrics:
    """
    Metrics for a single retrieval operation.

    Attributes:
        latency_ms: Time taken for retrieval in milliseconds
        num_results: Number of results returned
        top_score: Highest relevance score
        avg_score: Average relevance score
        score_variance: Variance of relevance scores
        retrieval_method: Method used for retrieval
    """
    latency_ms: float
    num_results: int
    top_score: float
    avg_score: float
    score_variance: float
    retrieval_method: str


@dataclass
class ABTestResult:
    """
    Result of a single A/B test trial.

    Attributes:
        test_id: Unique test identifier
        query: Test query
        variant: Test variant used
        metrics: Retrieval metrics
        retrieved_content_preview: Preview of retrieved content
        timestamp: When the test was run
        user_feedback: Optional user feedback (1-5 rating)
    """
    test_id: str
    query: str
    variant: str
    metrics: RetrievalMetrics
    retrieved_content_preview: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_feedback: Optional[int] = None  # 1-5 rating


@dataclass
class ABTestExperiment:
    """
    An A/B testing experiment containing multiple trials.

    Attributes:
        experiment_id: Unique experiment identifier
        name: Human-readable experiment name
        description: Experiment description
        created_at: Experiment creation timestamp
        variants: List of variants being tested
        results: Test results for each variant
        metadata: Experiment metadata
    """
    experiment_id: str
    name: str
    description: str
    created_at: str
    variants: List[str] = field(default_factory=list)
    results: Dict[str, List[ABTestResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingManager:
    """
    Manages A/B testing experiments for retrieval method comparison.

    Provides functionality to:
    - Run comparative tests across retrieval variants
    - Track latency, relevance scores, and result quality
    - Store results for analysis
    - Export comparison data

    Attributes:
        storage_dir: Directory for experiment storage
        current_experiment: Active experiment

    Example:
        >>> manager = ABTestingManager()
        >>> manager.create_experiment("Hybrid vs Semantic")
        >>> results = manager.run_comparison(query, retriever, variants=[...])
        >>> comparison = manager.get_comparison_summary()
    """

    def __init__(self, storage_dir: str = "./ab_testing_results"):
        """
        Initialize A/B testing manager.

        Args:
            storage_dir: Directory for storing experiment results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[ABTestExperiment] = None

        logger.info(f"ABTestingManager initialized: storage={storage_dir}")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        variants: List[TestVariant] = None
    ) -> ABTestExperiment:
        """
        Create a new A/B testing experiment.

        Args:
            name: Human-readable experiment name
            description: Experiment description
            variants: List of variants to test (default: all)

        Returns:
            New ABTestExperiment instance
        """
        if variants is None:
            variants = list(TestVariant)

        variant_names = [v.value for v in variants]

        self.current_experiment = ABTestExperiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            variants=variant_names,
            results={v: [] for v in variant_names}
        )

        logger.info(f"Created experiment: {name} ({self.current_experiment.experiment_id})")
        return self.current_experiment

    def load_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """
        Load an existing experiment from storage.

        Args:
            experiment_id: Experiment ID to load

        Returns:
            Loaded experiment or None if not found
        """
        file_path = self.storage_dir / f"{experiment_id}.json"

        if not file_path.exists():
            logger.warning(f"Experiment not found: {experiment_id}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct experiment
            results = {}
            for variant, variant_results in data.pop('results', {}).items():
                results[variant] = []
                for r in variant_results:
                    metrics_data = r.pop('metrics')
                    metrics = RetrievalMetrics(**metrics_data)
                    results[variant].append(ABTestResult(**r, metrics=metrics))

            self.current_experiment = ABTestExperiment(**data, results=results)
            logger.info(f"Loaded experiment: {experiment_id}")
            return self.current_experiment

        except Exception as e:
            logger.error(f"Failed to load experiment: {e}")
            return None

    def save_experiment(self) -> bool:
        """
        Save current experiment to storage.

        Returns:
            True if saved successfully
        """
        if not self.current_experiment:
            logger.warning("No active experiment to save")
            return False

        file_path = self.storage_dir / f"{self.current_experiment.experiment_id}.json"

        try:
            data = self._experiment_to_dict(self.current_experiment)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved experiment: {self.current_experiment.experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            return False

    def _experiment_to_dict(self, experiment: ABTestExperiment) -> Dict[str, Any]:
        """Convert experiment to JSON-serializable dict."""
        results_dict = {}
        for variant, variant_results in experiment.results.items():
            results_dict[variant] = []
            for r in variant_results:
                r_dict = asdict(r)
                r_dict['metrics'] = asdict(r.metrics)
                results_dict[variant].append(r_dict)

        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "created_at": experiment.created_at,
            "variants": experiment.variants,
            "results": results_dict,
            "metadata": experiment.metadata
        }

    def record_result(
        self,
        query: str,
        variant: str,
        latency_ms: float,
        scores: List[float],
        retrieved_content: List[str],
        user_feedback: int = None
    ) -> ABTestResult:
        """
        Record a single A/B test result.

        Args:
            query: Test query
            variant: Retrieval variant used
            latency_ms: Retrieval latency in milliseconds
            scores: Relevance scores for retrieved documents
            retrieved_content: Preview of retrieved content
            user_feedback: Optional user rating (1-5)

        Returns:
            Recorded ABTestResult
        """
        if not self.current_experiment:
            self.create_experiment("Default Experiment")

        # Calculate metrics
        metrics = RetrievalMetrics(
            latency_ms=latency_ms,
            num_results=len(scores),
            top_score=max(scores) if scores else 0.0,
            avg_score=statistics.mean(scores) if scores else 0.0,
            score_variance=statistics.variance(scores) if len(scores) > 1 else 0.0,
            retrieval_method=variant
        )

        result = ABTestResult(
            test_id=str(uuid.uuid4()),
            query=query,
            variant=variant,
            metrics=metrics,
            retrieved_content_preview=[c[:200] for c in retrieved_content[:3]],
            user_feedback=user_feedback
        )

        # Add to experiment
        if variant not in self.current_experiment.results:
            self.current_experiment.results[variant] = []
        self.current_experiment.results[variant].append(result)

        # Auto-save
        self.save_experiment()

        logger.info(f"Recorded result for variant '{variant}': latency={latency_ms:.1f}ms")
        return result

    def run_single_test(
        self,
        query: str,
        retriever_func,
        variant: TestVariant,
        k: int = 5
    ) -> ABTestResult:
        """
        Run a single test for one variant.

        Args:
            query: Test query
            retriever_func: Function that takes (query, method, k) and returns results
            variant: Variant to test
            k: Number of results to retrieve

        Returns:
            Test result with metrics
        """
        start_time = time.perf_counter()

        # Run retrieval
        results = retriever_func(query, variant.value, k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract scores and content
        scores = [r.final_score for r in results] if results else []
        content = [r.document.page_content for r in results] if results else []

        return self.record_result(
            query=query,
            variant=variant.value,
            latency_ms=latency_ms,
            scores=scores,
            retrieved_content=content
        )

    def run_comparison(
        self,
        query: str,
        retriever_func,
        variants: List[TestVariant] = None,
        k: int = 5
    ) -> Dict[str, ABTestResult]:
        """
        Run comparison test across multiple variants.

        Args:
            query: Test query
            retriever_func: Retrieval function
            variants: Variants to compare (default: all)
            k: Number of results to retrieve

        Returns:
            Dictionary mapping variant names to results
        """
        if variants is None:
            variants = list(TestVariant)

        results = {}
        for variant in variants:
            try:
                results[variant.value] = self.run_single_test(
                    query, retriever_func, variant, k
                )
            except Exception as e:
                logger.error(f"Failed to run test for {variant.value}: {e}")

        return results

    def add_user_feedback(self, test_id: str, feedback: int) -> bool:
        """
        Add user feedback to a test result.

        Args:
            test_id: Test result ID
            feedback: User rating (1-5)

        Returns:
            True if feedback was added
        """
        if not self.current_experiment:
            return False

        for variant_results in self.current_experiment.results.values():
            for result in variant_results:
                if result.test_id == test_id:
                    result.user_feedback = feedback
                    self.save_experiment()
                    return True

        return False

    def get_variant_statistics(self, variant: str) -> Dict[str, Any]:
        """
        Calculate statistics for a specific variant.

        Args:
            variant: Variant name

        Returns:
            Dictionary with statistical measures
        """
        if not self.current_experiment or variant not in self.current_experiment.results:
            return {}

        results = self.current_experiment.results[variant]
        if not results:
            return {}

        latencies = [r.metrics.latency_ms for r in results]
        top_scores = [r.metrics.top_score for r in results]
        avg_scores = [r.metrics.avg_score for r in results]
        feedbacks = [r.user_feedback for r in results if r.user_feedback is not None]

        return {
            "variant": variant,
            "num_tests": len(results),
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies)
            },
            "top_score": {
                "mean": statistics.mean(top_scores),
                "median": statistics.median(top_scores),
                "std": statistics.stdev(top_scores) if len(top_scores) > 1 else 0
            },
            "avg_score": {
                "mean": statistics.mean(avg_scores),
                "median": statistics.median(avg_scores)
            },
            "user_feedback": {
                "count": len(feedbacks),
                "mean": statistics.mean(feedbacks) if feedbacks else None
            }
        }

    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Get comparison summary across all variants.

        Returns:
            Dictionary with comparison statistics
        """
        if not self.current_experiment:
            return {}

        summary = {
            "experiment_id": self.current_experiment.experiment_id,
            "experiment_name": self.current_experiment.name,
            "total_tests": sum(len(r) for r in self.current_experiment.results.values()),
            "variants": {}
        }

        for variant in self.current_experiment.variants:
            summary["variants"][variant] = self.get_variant_statistics(variant)

        # Determine best variant by average score
        best_variant = None
        best_score = -1

        for variant, stats in summary["variants"].items():
            if stats and stats.get("avg_score", {}).get("mean", 0) > best_score:
                best_score = stats["avg_score"]["mean"]
                best_variant = variant

        summary["recommendation"] = {
            "best_variant": best_variant,
            "best_avg_score": best_score
        }

        return summary

    def export_results(self, format: str = "json") -> Optional[str]:
        """
        Export experiment results.

        Args:
            format: Export format ("json" or "csv")

        Returns:
            Exported data as string
        """
        if not self.current_experiment:
            return None

        if format == "json":
            return json.dumps(self._experiment_to_dict(self.current_experiment), indent=2)

        elif format == "csv":
            lines = ["test_id,query,variant,latency_ms,num_results,top_score,avg_score,user_feedback,timestamp"]

            for variant, results in self.current_experiment.results.items():
                for r in results:
                    lines.append(
                        f'"{r.test_id}","{r.query}","{r.variant}",'
                        f'{r.metrics.latency_ms:.2f},{r.metrics.num_results},'
                        f'{r.metrics.top_score:.4f},{r.metrics.avg_score:.4f},'
                        f'{r.user_feedback or ""},'
                        f'"{r.timestamp}"'
                    )

            return "\n".join(lines)

        else:
            logger.error(f"Unsupported export format: {format}")
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all saved experiments.

        Returns:
            List of experiment summaries
        """
        experiments = []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                total_tests = sum(len(r) for r in data.get("results", {}).values())
                experiments.append({
                    "experiment_id": data.get("experiment_id"),
                    "name": data.get("name"),
                    "created_at": data.get("created_at"),
                    "variants": data.get("variants", []),
                    "total_tests": total_tests
                })
            except Exception as e:
                logger.warning(f"Failed to read experiment file {file_path}: {e}")

        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment from storage.

        Args:
            experiment_id: Experiment ID to delete

        Returns:
            True if deleted successfully
        """
        file_path = self.storage_dir / f"{experiment_id}.json"

        if not file_path.exists():
            return False

        try:
            file_path.unlink()

            if self.current_experiment and self.current_experiment.experiment_id == experiment_id:
                self.current_experiment = None

            logger.info(f"Deleted experiment: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}")
            return False
