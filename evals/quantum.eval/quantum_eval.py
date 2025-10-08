"""Progressive Complexity Quantum Reasoning Eval"""

import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
import reasoning_gym
import numpy as np

from utils import Config, PromptLoader, OpenRouterClient, extract_answer_from_response, normalize_path
from cue_generator import CueGenerator
from judge import Judge
from dataset import generate_problems


@dataclass
class ReasoningTrace:
    """Single reasoning trace"""
    problem_id: str  # e.g., "a1b2c3d4_initial_1" or "a1b2c3d4_cued_0_reveal0.4"
    base_problem_id: str  # Content hash for matching across traces (e.g., "a1b2c3d4")
    difficulty: int
    min_path_length: int
    actual_path_length: int
    task_description: str
    expected_answer: str

    # Model response
    output: Optional[str]  # May be None if no output
    reasoning: Optional[str]  # Reasoning trace if available
    predicted_answer: Optional[str]
    score: float

    # Metadata
    attempt_number: int  # 1, 2, 3 for retries; 0 for cued attempts
    cue_type: Optional[str] = None
    reveal_ratio: Optional[float] = None
    cue: Optional[str] = None

    # Flags
    has_output: bool = True
    answer_extracted: bool = True


@dataclass
class ComplexityLevel:
    """A single complexity level"""
    difficulty: int
    min_path_length: int
    
    def __str__(self):
        return f"diff{self.difficulty}_len{self.min_path_length}"


class QuantumReasoningEval:
    """Progressive complexity evaluation"""
    
    def __init__(self, config_path: str = "config.yaml", prompts_dir: str = "prompts"):
        self.config = Config(config_path)
        self.prompt_loader = PromptLoader(prompts_dir)
        self.client = OpenRouterClient(base_url=self.config.api_base_url)
        self.cue_generator = CueGenerator()
        self.judge = Judge(
            client=self.client,
            prompt_loader=self.prompt_loader,
            judge_model=self.config.get('models.judge_model', 'anthropic/claude-sonnet-4')
        )

        # Create output directories
        results_dir = self.config.get('output.results_dir', 'eval_results')
        self.output_dir = Path(results_dir)
        self.traces_dir = self.output_dir / "traces"
        self.failed_dir = self.output_dir / "failed_problems"
        self.cue_dir = self.output_dir / "cue_traces"
        self.judge_dir = self.output_dir / "judge_results"

        for d in [self.traces_dir, self.failed_dir, self.cue_dir, self.judge_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def generate_complexity_grid(self) -> List[ComplexityLevel]:
        """
        Generate progressive complexity levels from config.

        Returns list of ComplexityLevel objects
        """
        # Try to load from config first
        grid_config = self.config.get('complexity_grid')

        if grid_config:
            # Load from config (list of [difficulty, min_path_length] pairs)
            levels = [ComplexityLevel(diff, path_len) for diff, path_len in grid_config]
        else:
            # Default fallback
            levels = [
                ComplexityLevel(1, 3),
                ComplexityLevel(50, 10),
                ComplexityLevel(1000, 15),
            ]

        return levels
    
    def generate_problems_for_level(
        self,
        level: ComplexityLevel,
        n_problems: int,
        seed: int = 42
    ) -> List[Dict]:
        """Generate n problems at a specific complexity level"""

        problems = generate_problems(
            difficulty=level.difficulty,
            min_path_length=level.min_path_length,
            n_problems=n_problems,
            seed=seed
        )

        # Add complexity level metadata
        for problem in problems:
            problem['complexity_level'] = str(level)
            problem['min_path_length'] = level.min_path_length

        return problems
    
    def test_problem(
        self,
        problem: Dict,
        model: str,
        attempt_number: int = 1,
        cue_type: Optional[str] = None,
        reveal_ratio: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_config: Optional[Dict] = None
    ) -> ReasoningTrace:
        """
        Test model on a single problem.
        
        Returns ReasoningTrace with all info including flags for missing output/answer.
        """
        
        # Generate cue if needed
        cue_str = None
        if cue_type:
            solution_path = problem['metadata']['solution_path']
            cue_str, _ = self.cue_generator.generate_cue(
                solution_path=solution_path,
                cue_type=cue_type,
                reveal_ratio=reveal_ratio
            )
            prompt_type = cue_type
        else:
            prompt_type = 'no_cue'
        
        # Get prompt
        prompt = self.prompt_loader.get_actor_prompt(
            prompt_type,
            problem=problem['question'],
            cue=cue_str if cue_str else ''
        )
        
        # Use config defaults if not provided
        if temperature is None:
            temperature = self.config.get('progressive_eval.temperature', 1.0)
        if max_tokens is None:
            max_tokens = self.config.get('progressive_eval.max_tokens', 4000)
        if reasoning_config is None:
            # Build reasoning config from config file
            effort = self.config.get('reasoning.effort')
            reasoning_max_tokens = self.config.get('reasoning.max_tokens')
            exclude = self.config.get('reasoning.exclude')

            if effort or reasoning_max_tokens or exclude:
                reasoning_config = {}
                if effort:
                    reasoning_config['effort'] = effort
                if reasoning_max_tokens:
                    reasoning_config['max_tokens'] = reasoning_max_tokens
                if exclude:
                    reasoning_config['exclude'] = exclude

        # Get response
        messages = [{'role': 'user', 'content': prompt}]

        try:
            result = self.client.get_text_response(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_config=reasoning_config
            )
            output = result['output']
            reasoning = result['reasoning']
        except Exception as e:
            print(f"  Error getting response: {e}")
            output = None
            reasoning = None
        
        # Check if we have output
        has_output = (output is not None and len(output.strip()) > 0)
        
        # Extract answer
        answer = None
        answer_extracted = False
        if has_output:
            answer = extract_answer_from_response(output)
            answer_extracted = (answer is not None)
        
        # Score
        score = 0.0
        if answer_extracted:
            dataset = reasoning_gym.create_dataset('quantum_lock', size=1)
            score = dataset.score_answer(answer, problem)

        # Generate unique problem hash from task description
        task_description = problem['question']
        content_hash = hashlib.md5(task_description.encode()).hexdigest()[:8]
        base_problem_id = content_hash

        # Create structured problem_id
        if cue_type:
            # Cued attempt: hash_cued_attemptnum_reveal{ratio}
            problem_id = f"{content_hash}_cued_{attempt_number}_reveal{reveal_ratio}"
        elif attempt_number == 1:
            # Initial attempt: hash_initial_1
            problem_id = f"{content_hash}_initial_{attempt_number}"
        else:
            # Retry attempt: hash_retry_N
            problem_id = f"{content_hash}_retry_{attempt_number}"

        trace = ReasoningTrace(
            problem_id=problem_id,
            base_problem_id=base_problem_id,
            difficulty=problem['metadata']['difficulty']['difficulty'],
            min_path_length=problem.get('min_path_length', 0),
            actual_path_length=len(problem['metadata']['solution_path']),
            task_description=task_description,
            expected_answer=' → '.join(problem['metadata']['solution_path']),
            output=output,
            reasoning=reasoning,
            predicted_answer=answer,
            score=score,
            attempt_number=attempt_number,
            cue_type=cue_type,
            reveal_ratio=reveal_ratio,
            has_output=has_output,
            answer_extracted=answer_extracted,
            cue=cue_str
        )

        return trace
    
    def run_progressive_eval(
        self,
        model: Optional[str] = None,
        n_problems_per_level: Optional[int] = None,
        n_retry_attempts: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_config: Optional[Dict] = None,
        n_cue_examples: Optional[int] = None,
        complexity_grid: Optional[List[ComplexityLevel]] = None
    ):
        """
        Run full progressive complexity evaluation.

        Steps:
        1. For each complexity level, generate n problems
        2. Test on each problem, save trace
        3. Identify failed problems (score<=0.5)
        4. Retry failed problems n times
        5. For problems that failed ALL attempts, run cue evaluation
        6. Run judges on all traces

        All parameters are optional - defaults are loaded from config.yaml
        """

        # Load defaults from config
        if model is None:
            model = self.config.get('models.actor_model')
            if model is None:
                raise ValueError("model must be specified either as parameter or in config.yaml")

        if n_problems_per_level is None:
            n_problems_per_level = self.config.get('progressive_eval.n_problems_per_level', 2)

        if n_retry_attempts is None:
            n_retry_attempts = self.config.get('progressive_eval.n_retry_attempts', 2)

        if temperature is None:
            temperature = self.config.get('progressive_eval.temperature', 1.0)

        if max_tokens is None:
            max_tokens = self.config.get('progressive_eval.max_tokens', 4000)

        if n_cue_examples is None:
            n_cue_examples = self.config.get('progressive_eval.n_cue_examples', 4)

        if complexity_grid is None:
            complexity_grid = self.generate_complexity_grid()

        # Build reasoning config from config file if not provided
        if reasoning_config is None:
            effort = self.config.get('reasoning.effort')
            reasoning_max_tokens = self.config.get('reasoning.max_tokens')
            exclude = self.config.get('reasoning.exclude')

            if effort or reasoning_max_tokens or exclude:
                reasoning_config = {}
                if effort:
                    reasoning_config['effort'] = effort
                if reasoning_max_tokens:
                    reasoning_config['max_tokens'] = reasoning_max_tokens
                if exclude:
                    reasoning_config['exclude'] = exclude

        print("="*60)
        print(f"PROGRESSIVE COMPLEXITY EVAL: {model}")
        print("="*60)
        print(f"  Problems per level: {n_problems_per_level}")
        print(f"  Retry attempts: {n_retry_attempts}")
        print(f"  Temperature: {temperature}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Complexity levels: {len(complexity_grid)}")
        print("="*60)

        all_traces = []
        # Track all attempts per problem to determine if ALL failed
        problem_attempts = {}  # base_problem_id -> list of (problem, trace) tuples

        # Step 1-3: Test all complexity levels
        for level in complexity_grid:
            print(f"\n[{level}] Generating {n_problems_per_level} problems...")

            problems = self.generate_problems_for_level(
                level=level,
                n_problems=n_problems_per_level
            )

            print(f"[{level}] Testing {len(problems)} problems (attempt 1)...")

            for problem in problems:
                trace = self.test_problem(
                    problem=problem,
                    model=model,
                    attempt_number=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_config=reasoning_config
                )

                all_traces.append(trace)

                # Track attempts per problem using base_problem_id
                if trace.base_problem_id not in problem_attempts:
                    problem_attempts[trace.base_problem_id] = []
                problem_attempts[trace.base_problem_id].append((problem, trace))

                # Log issues
                if not trace.has_output:
                    print(f"  ⚠ {trace.problem_id}: No output (max_tokens issue?)")
                elif not trace.answer_extracted:
                    print(f"  ⚠ {trace.problem_id}: No answer extracted")
                elif trace.score == 0.0:
                    print(f"  ✗ {trace.problem_id}: Incorrect (score=0)")
                elif trace.score == 0.5:
                    print(f"  ✗ {trace.problem_id}: Target is reached, but path is not shortest (score=0.5)")
                else:
                    print(f"  ✓ {trace.problem_id}: Correct (score={trace.score})")

        # Save initial traces
        self._save_traces(all_traces, self.traces_dir / f"{model.replace('/', '_')}_initial.json")

        # Identify problems that failed first attempt
        initially_failed_problems = [
            (problem, trace)
            for base_problem_id, attempts in problem_attempts.items()
            for problem, trace in attempts
            if trace.attempt_number == 1 and trace.score <= 0.5 and trace.has_output and trace.answer_extracted
        ]

        print(f"\n{'='*60}")
        print(f"Initial run complete: {len(all_traces)} traces")
        print(f"Failed problems (first attempt): {len(initially_failed_problems)}")
        print(f"{'='*60}")

        # Step 4: Retry failed problems
        if initially_failed_problems:
            print(f"\nRetrying {len(initially_failed_problems)} failed problems...")

            retry_traces = []

            for problem, initial_trace in initially_failed_problems:
                for retry_num in range(2, n_retry_attempts + 2):  # attempts 2, 3, ...
                    print(f"  Retry {retry_num-1}: {initial_trace.problem_id}")

                    trace = self.test_problem(
                        problem=problem,
                        model=model,
                        attempt_number=retry_num,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning_config=reasoning_config
                    )

                    retry_traces.append(trace)

                    # Track this attempt using base_problem_id
                    problem_attempts[trace.base_problem_id].append((problem, trace))

            all_traces.extend(retry_traces)
            self._save_traces(retry_traces, self.traces_dir / f"{model.replace('/', '_')}_retries.json")

        # Step 5: Identify problems that failed ALL attempts
        always_failed_problems = []
        for base_problem_id, attempts in problem_attempts.items():
            # Get all traces for this problem
            traces = [trace for _, trace in attempts]

            # Check if ALL attempts failed (score <= 0.5)
            all_failed = all(
                trace.score <= 0.5 and trace.has_output and trace.answer_extracted
                for trace in traces
            )

            if all_failed:
                # Use the last trace and problem for cue eval
                problem, last_trace = attempts[-1]
                always_failed_problems.append((problem, last_trace))

        print(f"\nProblems that failed ALL {n_retry_attempts + 1} attempts: {len(always_failed_problems)}")

        # Step 6: Cue evaluation on always-failed problems
        if always_failed_problems:
            print(f"\nRunning cue evaluation on {len(always_failed_problems)} always-failed problems...")

            cue_traces = self._run_cue_evaluation(
                failed_problems=always_failed_problems,
                model=model,
                n_cue_examples=n_cue_examples,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_config=reasoning_config
            )

            all_traces.extend(cue_traces)

        # Step 7: Run judges
        print(f"\nRunning judges on {len(all_traces)} traces...")
        self._run_judges(all_traces, model)

        # Final summary
        self._print_summary(all_traces)
    
    def _run_cue_evaluation(
        self,
        failed_problems: List[Tuple[Dict, ReasoningTrace]],
        model: str,
        n_cue_examples: int = 4,
        temperature: float = 1.0,
        max_tokens: int = 4000,
        reasoning_config: Optional[Dict] = None
    ) -> List[ReasoningTrace]:
        """Run cue evaluation with different reveal ratios"""

        # Get cue settings from config
        reveal_ratios_config = self.config.get('cues.reveal_ratios')
        if reveal_ratios_config and len(reveal_ratios_config) > 0:
            # Use first n_cue_examples from config
            reveal_ratios = reveal_ratios_config[:n_cue_examples]
        else:
            # Default: evenly spaced from 0.1 to 1.0
            reveal_ratios = np.linspace(0.1, 1.0, n_cue_examples)

        cue_type = self.config.get('cues.default_cue_type', 'partial_prefix')
        
        cue_traces = []
        
        for problem, failed_trace in failed_problems:
            print(f"  Cue eval: {failed_trace.problem_id}")
            
            for reveal_ratio in reveal_ratios:
                trace = self.test_problem(
                    problem=problem,
                    model=model,
                    attempt_number=0,  # Special marker for cue attempts
                    cue_type=cue_type,
                    reveal_ratio=reveal_ratio,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_config=reasoning_config
                )
                
                cue_traces.append(trace)
        
        self._save_traces(
            cue_traces,
            self.cue_dir / f"{model.replace('/', '_')}_cue_traces.json"
        )
        
        return cue_traces
    
    def _run_judges(self, traces: List[ReasoningTrace], model: str):
        """Run judges on traces"""
        
        judge_results = []
        
        # Filter traces with output for judging
        judgeable_traces = [t for t in traces if t.has_output and t.output]
        
        print(f"  Judging {len(judgeable_traces)} traces with output...")
        
        for trace in judgeable_traces:
            result = {'problem_id': trace.problem_id}
            
            # Run correctness and verifiability judges on all traces
            if trace.cue_type is None:  # No cue = baseline
                result['correctness'] = self.judge.evaluate(
                    'reasoning_correctness_judge',
                    problem=trace.task_description,
                    reasoning=trace.reasoning,
                    answer=trace.predicted_answer or '',
                    correct_answer=trace.expected_answer
                )
                
                result['verifiability'] = self.judge.evaluate(
                    'cot_verifiability_judge',
                    problem=trace.task_description,
                    reasoning_without_answer=trace.reasoning  # Could strip answer here
                )
            
            # Run verbalization judge only on cue traces
            if trace.cue_type:
                # Reconstruct cue for judge
                cue_str, _ = self.cue_generator.generate_cue(
                    solution_path=trace.expected_answer.split(' → '),
                    cue_type=trace.cue_type,
                    reveal_ratio=trace.reveal_ratio
                )
                
                result['verbalization'] = self.judge.evaluate(
                    'verbalization_judge',
                    problem=trace.task_description,
                    cue=cue_str,
                    reasoning=trace.reasoning
                )
            
            judge_results.append(result)
        
        # Save judge results
        output_path = self.judge_dir / f"{model.replace('/', '_')}_judge_results.json"
        with open(output_path, 'w') as f:
            json.dump(judge_results, f, indent=2)
        
        print(f"  Saved judge results to {output_path}")
    
    def _save_traces(self, traces: List[ReasoningTrace], output_path: Path):
        """Save traces to JSON"""
        with open(output_path, 'w') as f:
            json.dump([asdict(t) for t in traces], f, indent=2)
        print(f"  Saved {len(traces)} traces to {output_path}")
    
    def _print_summary(self, traces: List[ReasoningTrace]):
        """Print evaluation summary"""
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Overall stats
        total = len(traces)
        with_output = sum(1 for t in traces if t.has_output)
        with_answer = sum(1 for t in traces if t.answer_extracted)
        correct = sum(1 for t in traces if t.score >= 1.0)
        
        print(f"Total traces: {total}")
        print(f"  With output: {with_output}/{total} ({with_output/total*100:.1f}%)")
        print(f"  Answer extracted: {with_answer}/{total} ({with_answer/total*100:.1f}%)")
        print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        
        # By attempt
        baseline_traces = [t for t in traces if t.attempt_number >= 1 and t.cue_type is None]
        cue_traces = [t for t in traces if t.cue_type is not None]
        
        if baseline_traces:
            correct_baseline = sum(1 for t in baseline_traces if t.score >= 1.0)
            print(f"\nBaseline (no cues): {correct_baseline}/{len(baseline_traces)} ({correct_baseline/len(baseline_traces)*100:.1f}%)")
        
        if cue_traces:
            correct_cue = sum(1 for t in cue_traces if t.score >= 1.0)
            print(f"With cues: {correct_cue}/{len(cue_traces)} ({correct_cue/len(cue_traces)*100:.1f}%)")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    evaluator = QuantumReasoningEval()
    
    # Run progressive eval
    evaluator.run_progressive_eval(
        model="deepseek/deepseek-r1",
        n_problems_per_level=2,
        n_retry_attempts=2,
        temperature=1.0
    )