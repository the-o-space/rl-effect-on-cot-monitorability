from reasoning_gym.graphs.quantum_lock import QuantumLockDataset, QuantumLockConfig
from collections import deque
from random import Random
from typing import Any, List, Dict

DATASET_NAME = "quantum_lock_extended"

class LongerQuantumLock(QuantumLockDataset):
    """Extended quantum lock with fixed difficulty and longer max_steps"""

    def __init__(self, config: QuantumLockConfig, fixed_difficulty: int = 15):
        """
        Args:
            config: QuantumLockConfig instance
            fixed_difficulty: Use this exact difficulty (no random sampling)
        """
        super().__init__(config)
        self.fixed_difficulty = fixed_difficulty

    def __getitem__(self, idx: int) -> dict:
        """Generate a single task with fixed difficulty"""
        rng = Random(self.seed + idx)

        difficulty = self.fixed_difficulty  # Use fixed difficulty

        puzzle_data = self.generate_quantum_puzzle(rng, difficulty)

        return {
            "question": self.format_puzzle(rng.choice(self._prompt_templates), puzzle=puzzle_data),
            "answer": " → ".join(puzzle_data["solution"]),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "solution_path": puzzle_data["solution"],
                "target_value": puzzle_data["target_value"],
                "buttons": puzzle_data["buttons"],
                "initial_state": puzzle_data["initial_state"],
                "initial_value": puzzle_data["initial_value"],
                "difficulty": {"difficulty": difficulty},
            },
        }


def generate_problems(
    difficulty: int,
    min_path_length: int,
    n_problems: int,
    seed: int = 42
) -> List[Dict]:
    """
    Generate quantum lock problems at a specific difficulty level.

    Args:
        difficulty: Problem difficulty level
        min_path_length: Minimum solution path length
        n_problems: Number of problems to generate
        seed: Random seed

    Returns:
        List of problem dicts meeting the criteria
    """
    problems = []
    attempts = 0
    max_attempts = n_problems * 10  # Allow rejection sampling

    attempt_idx = 0
    while len(problems) < n_problems and attempts < max_attempts:
        # Create dataset with this specific difficulty
        config = QuantumLockConfig(seed=seed + attempt_idx, size=1, difficulty=difficulty)
        dataset = LongerQuantumLock(config, fixed_difficulty=difficulty)

        # Get the problem
        problem = dataset[0]
        solution_length = len(problem['metadata']['solution_path'])

        # Filter by minimum path length
        if solution_length >= min_path_length:
            problems.append(problem)

        attempt_idx += 1
        attempts += 1

    if len(problems) < n_problems:
        print(f"  Warning: Only generated {len(problems)}/{n_problems} problems after {attempts} attempts")

    return problems