"""
Tests for A/B Testing module.
"""

import pytest
import time
from unittest.mock import MagicMock
from core.ab_testing import (
    ABTestingManager,
    ABTestExperiment,
    ABTestResult,
    TestVariant,
    RetrievalMetrics
)


class TestRetrievalMetrics:
    """Test cases for RetrievalMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating retrieval metrics."""
        metrics = RetrievalMetrics(
            latency_ms=150.5,
            num_results=3,
            top_score=0.95,
            avg_score=0.85,
            score_variance=0.02,
            retrieval_method="hybrid"
        )

        assert metrics.latency_ms == 150.5
        assert metrics.num_results == 3
        assert metrics.top_score == 0.95
        assert metrics.avg_score == 0.85


class TestABTestResult:
    """Test cases for ABTestResult dataclass."""

    def test_result_creation(self):
        """Test creating A/B test result."""
        metrics = RetrievalMetrics(
            latency_ms=100,
            num_results=3,
            top_score=0.9,
            avg_score=0.8,
            score_variance=0.01,
            retrieval_method="semantic"
        )

        result = ABTestResult(
            test_id="test-123",
            query="What is AI?",
            variant="semantic",
            metrics=metrics
        )

        assert result.test_id == "test-123"
        assert result.query == "What is AI?"
        assert result.variant == "semantic"
        assert result.user_feedback is None


class TestTestVariant:
    """Test cases for TestVariant enum."""

    def test_variant_values(self):
        """Test variant enum values."""
        assert TestVariant.CONTROL.value == "semantic"
        assert TestVariant.VARIANT_A.value == "bm25"
        assert TestVariant.VARIANT_B.value == "hybrid"
        assert TestVariant.VARIANT_C.value == "hybrid_rerank"


class TestABTestingManager:
    """Test cases for ABTestingManager class."""

    def test_initialization(self, temp_ab_testing_dir):
        """Test A/B testing manager initialization."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)

        assert manager.storage_dir.exists()
        assert manager.current_experiment is None

    def test_create_experiment(self, temp_ab_testing_dir):
        """Test creating an experiment."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        experiment = manager.create_experiment(
            name="Test Experiment",
            description="Testing hybrid vs semantic"
        )

        assert experiment is not None
        assert experiment.name == "Test Experiment"
        assert len(experiment.variants) == 4  # All variants by default
        assert manager.current_experiment == experiment

    def test_create_experiment_with_specific_variants(self, temp_ab_testing_dir):
        """Test creating experiment with specific variants."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        experiment = manager.create_experiment(
            name="Limited Experiment",
            variants=[TestVariant.CONTROL, TestVariant.VARIANT_A]
        )

        assert len(experiment.variants) == 2
        assert "semantic" in experiment.variants
        assert "bm25" in experiment.variants

    def test_record_result(self, temp_ab_testing_dir):
        """Test recording a test result."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        result = manager.record_result(
            query="What is ML?",
            variant="semantic",
            latency_ms=100.5,
            scores=[0.9, 0.8, 0.7],
            retrieved_content=["Content 1", "Content 2", "Content 3"]
        )

        assert result is not None
        assert result.query == "What is ML?"
        assert result.variant == "semantic"
        assert result.metrics.latency_ms == 100.5
        assert result.metrics.num_results == 3

    def test_record_result_auto_creates_experiment(self, temp_ab_testing_dir):
        """Test that recording auto-creates experiment if none exists."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)

        # Don't create experiment first
        result = manager.record_result(
            query="Test?",
            variant="semantic",
            latency_ms=50,
            scores=[0.9],
            retrieved_content=["Content"]
        )

        assert manager.current_experiment is not None
        assert result is not None

    def test_save_and_load_experiment(self, temp_ab_testing_dir):
        """Test saving and loading experiment."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        experiment = manager.create_experiment("Test Experiment")
        exp_id = experiment.experiment_id

        manager.record_result("Q1?", "semantic", 100, [0.9], ["C1"])
        manager.record_result("Q2?", "hybrid", 150, [0.8, 0.7], ["C1", "C2"])
        manager.save_experiment()

        # Load in new manager
        manager2 = ABTestingManager(storage_dir=temp_ab_testing_dir)
        loaded = manager2.load_experiment(exp_id)

        assert loaded is not None
        assert loaded.experiment_id == exp_id
        assert "semantic" in loaded.results
        assert "hybrid" in loaded.results

    def test_load_nonexistent_experiment(self, temp_ab_testing_dir):
        """Test loading non-existent experiment returns None."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        result = manager.load_experiment("nonexistent-id")

        assert result is None

    def test_add_user_feedback(self, temp_ab_testing_dir):
        """Test adding user feedback to result."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        result = manager.record_result("Q?", "semantic", 100, [0.9], ["C"])
        test_id = result.test_id

        success = manager.add_user_feedback(test_id, feedback=5)

        assert success is True
        assert result.user_feedback == 5

    def test_add_feedback_invalid_id(self, temp_ab_testing_dir):
        """Test adding feedback with invalid ID returns False."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        success = manager.add_user_feedback("invalid-id", feedback=3)
        assert success is False

    def test_get_variant_statistics(self, temp_ab_testing_dir):
        """Test getting statistics for a variant."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        # Record multiple results
        manager.record_result("Q1?", "semantic", 100, [0.9], ["C1"])
        manager.record_result("Q2?", "semantic", 120, [0.85], ["C2"])
        manager.record_result("Q3?", "semantic", 110, [0.95], ["C3"])

        stats = manager.get_variant_statistics("semantic")

        assert stats["variant"] == "semantic"
        assert stats["num_tests"] == 3
        assert "mean" in stats["latency"]
        assert "mean" in stats["top_score"]

    def test_get_variant_statistics_empty(self, temp_ab_testing_dir):
        """Test getting statistics for empty variant."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        stats = manager.get_variant_statistics("nonexistent")
        assert stats == {}

    def test_get_comparison_summary(self, temp_ab_testing_dir):
        """Test getting comparison summary."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        # Record results for multiple variants
        manager.record_result("Q1?", "semantic", 100, [0.9], ["C1"])
        manager.record_result("Q2?", "hybrid", 150, [0.95], ["C2"])

        summary = manager.get_comparison_summary()

        assert summary["total_tests"] == 2
        assert "semantic" in summary["variants"]
        assert "hybrid" in summary["variants"]
        assert "recommendation" in summary

    def test_export_results_json(self, temp_ab_testing_dir):
        """Test exporting results as JSON."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")
        manager.record_result("Q?", "semantic", 100, [0.9], ["C"])

        json_export = manager.export_results(format="json")

        assert json_export is not None
        assert "experiment_id" in json_export
        assert "results" in json_export

    def test_export_results_csv(self, temp_ab_testing_dir):
        """Test exporting results as CSV."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")
        manager.record_result("Q1?", "semantic", 100, [0.9], ["C1"])
        manager.record_result("Q2?", "hybrid", 150, [0.8], ["C2"])

        csv_export = manager.export_results(format="csv")

        assert csv_export is not None
        assert "test_id" in csv_export
        assert "query" in csv_export
        assert "variant" in csv_export
        assert "latency_ms" in csv_export

    def test_export_results_invalid_format(self, temp_ab_testing_dir):
        """Test exporting with invalid format returns None."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Test")

        result = manager.export_results(format="invalid")
        assert result is None

    def test_list_experiments(self, temp_ab_testing_dir):
        """Test listing all experiments."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)

        # Create multiple experiments
        manager.create_experiment("Experiment 1")
        manager.record_result("Q?", "semantic", 100, [0.9], ["C"])

        manager.create_experiment("Experiment 2")
        manager.record_result("Q?", "hybrid", 150, [0.8], ["C"])

        experiments = manager.list_experiments()

        assert len(experiments) == 2

    def test_delete_experiment(self, temp_ab_testing_dir):
        """Test deleting an experiment."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        experiment = manager.create_experiment("To Delete")
        exp_id = experiment.experiment_id
        manager.save_experiment()

        result = manager.delete_experiment(exp_id)

        assert result is True
        assert manager.current_experiment is None

        # Verify deleted
        experiments = manager.list_experiments()
        assert len(experiments) == 0

    def test_delete_nonexistent_experiment(self, temp_ab_testing_dir):
        """Test deleting non-existent experiment returns False."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        result = manager.delete_experiment("nonexistent")

        assert result is False

    def test_run_single_test(self, temp_ab_testing_dir):
        """Test running a single test with mock retriever."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Single Test")

        # Create mock retriever function
        mock_result = MagicMock()
        mock_result.final_score = 0.9
        mock_result.document = MagicMock()
        mock_result.document.page_content = "Test content"

        def mock_retriever_func(query, method, k):
            time.sleep(0.01)  # Simulate latency
            return [mock_result] * k

        result = manager.run_single_test(
            query="Test query",
            retriever_func=mock_retriever_func,
            variant=TestVariant.CONTROL,
            k=3
        )

        assert result is not None
        assert result.metrics.latency_ms > 0
        assert result.metrics.num_results == 3

    def test_run_comparison(self, temp_ab_testing_dir):
        """Test running comparison across variants."""
        manager = ABTestingManager(storage_dir=temp_ab_testing_dir)
        manager.create_experiment("Comparison Test")

        mock_result = MagicMock()
        mock_result.final_score = 0.9
        mock_result.document = MagicMock()
        mock_result.document.page_content = "Content"

        def mock_retriever_func(query, method, k):
            return [mock_result] * k

        results = manager.run_comparison(
            query="Test query",
            retriever_func=mock_retriever_func,
            variants=[TestVariant.CONTROL, TestVariant.VARIANT_A],
            k=3
        )

        assert len(results) == 2
        assert "semantic" in results
        assert "bm25" in results
