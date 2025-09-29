"""
Metrics computation for evaluations.
"""
from typing import Dict, List, Any


def compute_metrics(results: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, float]:
    """
    Compute specified metrics from evaluation results.
    
    Args:
        results: List of result dicts with 'expected' and 'generated' fields
        metric_names: List of metric names to compute
        
    Returns:
        Dict mapping metric names to computed values
    """
    metrics = {}
    
    for metric_name in metric_names:
        if metric_name == "accuracy":
            metrics["accuracy"] = compute_accuracy(results)
        elif metric_name == "response_length":
            metrics["avg_response_length"] = compute_avg_response_length(results)
        else:
            metrics[metric_name] = None  # Unknown metric
    
    return metrics


def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Simple exact match accuracy.
    
    Note: This is a naive implementation. Replace with task-specific logic.
    """
    if not results:
        return 0.0
    
    correct = 0
    total = 0
    
    for result in results:
        if result.get("generated") is None:
            continue
        
        expected = str(result.get("expected", "")).strip().lower()
        generated = str(result.get("generated", "")).strip().lower()
        
        if expected in generated or generated in expected:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def compute_avg_response_length(results: List[Dict[str, Any]]) -> float:
    """Average character length of generated responses."""
    lengths = [
        len(result.get("generated", ""))
        for result in results
        if result.get("generated") is not None
    ]
    
    return sum(lengths) / len(lengths) if lengths else 0.0
