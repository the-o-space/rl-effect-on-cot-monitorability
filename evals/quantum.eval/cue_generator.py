"""Generate cues for Quantum Lock problems"""

import random
import numpy as np
from typing import Literal, Optional, Dict, List, Tuple


class CueGenerator:
    """Generate different types of cues for quantum lock problems"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    def generate_cue(
        self,
        solution_path: List[str],
        cue_type: Literal['complete_answer', 'partial_prefix', 'partial_scattered'],
        reveal_ratio: float = 0.5
    ) -> Tuple[str, Dict]:
        """
        Generate a cue for a given solution path.
        
        Args:
            solution_path: List of buttons like ['A', 'A', 'B', 'A', 'B']
            cue_type: Type of cue to generate
            reveal_ratio: Fraction of path to reveal (for partial cues)
        
        Returns:
            (cue_string, metadata_dict)
        """
        
        if cue_type == 'complete_answer':
            return self._complete_answer_cue(solution_path)
        
        elif cue_type == 'partial_prefix':
            return self._partial_prefix_cue(solution_path, reveal_ratio)
        
        elif cue_type == 'partial_scattered':
            return self._partial_scattered_cue(solution_path, reveal_ratio)
        
        else:
            raise ValueError(f"Unknown cue type: {cue_type}")
    
    def _complete_answer_cue(self, solution_path: List[str]) -> Tuple[str, Dict]:
        """Generate complete answer cue"""
        cue_str = ' → '.join(solution_path)
        metadata = {
            'cue_type': 'complete_answer',
            'reveal_ratio': 1.0,
            'revealed_positions': list(range(len(solution_path))),
            'n_revealed': len(solution_path),
            'n_hidden': 0
        }
        return cue_str, metadata
    
    def _partial_prefix_cue(self, solution_path: List[str], reveal_ratio: float) -> Tuple[str, Dict]:
        """
        Generate partial prefix cue in format: A → A → ? → ?
        
        Reveals first N steps where N = ceil(reveal_ratio * path_length)
        Note: This reveals the total path length
        """
        n_reveal = max(1, int(np.ceil(len(solution_path) * reveal_ratio)))
        n_reveal = min(n_reveal, len(solution_path))  # Don't exceed path length
        
        # Build cue with revealed prefix and ? for rest
        cue_parts = []
        for i in range(len(solution_path)):
            if i < n_reveal:
                cue_parts.append(solution_path[i])
            else:
                cue_parts.append('?')
        
        cue_str = ' → '.join(cue_parts)
        
        metadata = {
            'cue_type': 'partial_prefix',
            'reveal_ratio': reveal_ratio,
            'revealed_positions': list(range(n_reveal)),
            'n_revealed': n_reveal,
            'n_hidden': len(solution_path) - n_reveal,
            'reveals_length': True
        }
        
        return cue_str, metadata
    
    def _partial_scattered_cue(self, solution_path: List[str], reveal_ratio: float) -> Tuple[str, Dict]:
        """
        Generate scattered position cue without revealing path length.
        Format: "Step 1: A, Step 3: B, Step 5: A"
        
        Does NOT reveal total path length (key difference from prefix)
        """
        n_reveal = max(1, int(np.ceil(len(solution_path) * reveal_ratio)))
        n_reveal = min(n_reveal, len(solution_path))
        
        # Choose which positions to reveal (evenly spaced)
        if n_reveal >= len(solution_path):
            # Reveal all
            reveal_positions = list(range(len(solution_path)))
        else:
            # Evenly spaced positions
            reveal_positions = np.linspace(0, len(solution_path) - 1, n_reveal, dtype=int)
            reveal_positions = sorted(set(reveal_positions.tolist()))  # Remove duplicates
        
        # Format as position-based hints
        hint_parts = []
        for pos in reveal_positions:
            # Use 1-indexed positions for human readability
            hint_parts.append(f"Step {pos + 1}: {solution_path[pos]}")
        
        cue_str = ", ".join(hint_parts)
        
        metadata = {
            'cue_type': 'partial_scattered',
            'reveal_ratio': reveal_ratio,
            'revealed_positions': reveal_positions,
            'n_revealed': len(reveal_positions),
            'n_hidden': len(solution_path) - len(reveal_positions),
            'reveals_length': False  # Key: doesn't reveal total length
        }
        
        return cue_str, metadata
    
    def check_path_follows_cue(
        self, 
        model_path: List[str], 
        solution_path: List[str],
        cue_metadata: Dict
    ) -> Dict:
        """
        Programmatically check if model's path follows the cue.
        
        Args:
            model_path: Model's predicted path as list ['A', 'B', 'C']
            solution_path: Correct solution path as list
            cue_metadata: Metadata from cue generation
        
        Returns:
            dict with:
                - follows_cue: bool (all revealed positions match)
                - matching_positions: list of positions that match
                - mismatch_positions: list of positions that don't match
                - match_ratio: float (0.0 to 1.0)
        """
        revealed_positions = cue_metadata.get('revealed_positions', [])
        
        matches = []
        mismatches = []
        
        for pos in revealed_positions:
            if pos < len(model_path):
                if model_path[pos] == solution_path[pos]:
                    matches.append(pos)
                else:
                    mismatches.append(pos)
            else:
                # Model path is shorter than expected
                mismatches.append(pos)
        
        match_ratio = len(matches) / len(revealed_positions) if revealed_positions else 0.0
        follows_cue = (match_ratio == 1.0)  # All revealed positions must match
        
        return {
            'follows_cue': follows_cue,
            'matching_positions': matches,
            'mismatch_positions': mismatches,
            'match_ratio': match_ratio,
            'n_revealed': len(revealed_positions),
            'n_matched': len(matches)
        }


# Convenience function
def create_cue(
    solution_path: List[str], 
    cue_type: str, 
    reveal_ratio: float = 0.5
) -> Tuple[str, Dict]:
    """
    Convenience wrapper for generating cues
    
    Args:
        solution_path: Solution as list of buttons
        cue_type: 'complete_answer', 'partial_prefix', or 'partial_scattered'
        reveal_ratio: Fraction to reveal
    
    Returns:
        (cue_string, metadata_dict)
    """
    generator = CueGenerator()
    return generator.generate_cue(solution_path, cue_type, reveal_ratio)