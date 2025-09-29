"""
Example evaluation implementation.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

from shared import get_project_root, logger
from shared.models import OpenRouterProvider

from metrics import compute_metrics


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset."""
    project_root = get_project_root()
    full_path = project_root / dataset_path
    
    logger.info(f"Loading dataset from {dataset_path}...")
    with open(full_path) as f:
        data = json.load(f)
    
    return data


def run_eval(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main eval entrypoint.
    
    Args:
        config: Evaluation configuration dict
        
    Returns:
        Dict containing evaluation results and metrics
    """
    eval_name = config["eval"]["name"]
    
    with logger.span(f"run_eval.{eval_name}"):
        # Load dataset
        dataset_path = config["eval"]["dataset"]["path"]
        dataset = load_dataset(dataset_path)
        logger.info(f"Loaded dataset", num_examples=len(dataset))
        
        # Initialize model provider
        model_config = config["model"]
        provider_type = model_config["provider"]
        
        if provider_type == "openrouter":
            provider = OpenRouterProvider(
                model_id=model_config["model_id"],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 512)
            )
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
        logger.info(
            f"Initialized provider",
            provider=provider_type,
            model=model_config['model_id']
        )
        
        # Run eval
        results = []
        for i, example in enumerate(dataset):
            with logger.span(f"process_example.{i}"):
                prompt = example.get("prompt", example.get("question", ""))
                expected = example.get("expected", example.get("answer", ""))
                
                try:
                    response_data = provider.generate(
                        prompt=prompt,
                        temperature=model_config.get("temperature", 0.7),
                        max_tokens=model_config.get("max_tokens", 512)
                    )
                    
                    results.append({
                        "prompt": prompt,
                        "expected": expected,
                        "generated": response_data["response"],
                        "metadata": response_data["metadata"]
                    })
                    
                    logger.info(
                        f"Generated response",
                        example_idx=i,
                        tokens_used=response_data["metadata"]["tokens_total"],
                        latency_ms=response_data["metadata"]["latency_ms"]
                    )
                    
                    if config.get("output", {}).get("verbose", False):
                        logger.debug(
                            "Response preview",
                            response=response_data['response'][:100]
                        )
                
                except Exception as e:
                    logger.error(f"Failed to generate response", example_idx=i, error=str(e))
                    results.append({
                        "prompt": prompt,
                        "expected": expected,
                        "generated": None,
                        "error": str(e)
                    })
        
        # Compute metrics
        metric_names = config.get("metrics", [])
        metrics = compute_metrics(results, metric_names)
        
        logger.info("Computed metrics", metrics=metrics)
        
        # Save results if output directory specified
        output_config = config.get("output", {})
        if results_dir := output_config.get("results_dir"):
            save_results(results, metrics, config, results_dir)
        
        return {
            "results": results,
            "metrics": metrics,
            "config": config
        }


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    config: Dict[str, Any],
    results_dir: str
):
    """Save evaluation results to disk."""
    project_root = get_project_root()
    output_dir = project_root / results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_name = config["eval"]["name"]
    output_file = output_dir / f"{eval_name}_results.json"
    
    output_data = {
        "config": config,
        "metrics": metrics,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
