"""Judge classes for evaluating model responses"""

import json
import re
import asyncio
from typing import Optional, Dict, List
from tqdm.asyncio import tqdm as atqdm


class Judge:
    """Generic judge that uses different prompts for different evaluations"""

    def __init__(self, client, prompt_loader, judge_model: str, evaluation_mode: str = "cot_with_fallback"):
        """
        Args:
            client: OpenRouterClient instance
            prompt_loader: PromptLoader instance
            judge_model: Model name to use for judging
            evaluation_mode: How to construct reasoning text for evaluation
                - "cot_only": Use only reasoning field (CoT)
                - "output_only": Use only output field (action)
                - "cot_with_fallback": Use reasoning if available, otherwise use output
                - "cot_and_output": Concatenate reasoning + output
        """
        self.client = client
        self.prompt_loader = prompt_loader
        self.judge_model = judge_model
        self.evaluation_mode = evaluation_mode

    def prepare_reasoning_text(self, reasoning: Optional[str], output: Optional[str]) -> str:
        """
        Prepare reasoning text based on evaluation mode.

        Args:
            reasoning: CoT/thinking trace (may be None)
            output: Model's final output/action (may be None)

        Returns:
            Combined text to evaluate
        """
        if self.evaluation_mode == "cot_only":
            return reasoning or ""

        elif self.evaluation_mode == "output_only":
            return output or ""

        elif self.evaluation_mode == "cot_with_fallback":
            # Use reasoning if available, otherwise fall back to output
            if reasoning and reasoning.strip():
                return reasoning
            return output or ""

        elif self.evaluation_mode == "cot_and_output":
            # Concatenate both with clear separation
            parts = []
            if reasoning and reasoning.strip():
                parts.append(f"=== REASONING/THINKING ===\n{reasoning}")
            if output and output.strip():
                parts.append(f"=== OUTPUT/ACTION ===\n{output}")
            return "\n\n".join(parts) if parts else ""

        else:
            # Default to fallback mode
            if reasoning and reasoning.strip():
                return reasoning
            return output or ""
    
    def evaluate(self, judge_type: str, **kwargs) -> Dict:
        """
        Run a judge evaluation.

        Args:
            judge_type: Which judge prompt to use ('verbalization_judge',
                       'reasoning_correctness_judge', 'cot_verifiability_judge')
            **kwargs: Arguments to format the prompt (problem, cue, reasoning, etc.)
                     Can include 'output' field for fallback/combination modes

        Returns:
            Parsed JSON response from judge
        """
        # Prepare reasoning text based on evaluation mode
        # Extract output if provided, prepare combined reasoning
        output = kwargs.pop('output', None)
        reasoning = kwargs.get('reasoning')

        if reasoning is not None or output is not None:
            # Prepare combined reasoning based on mode
            prepared_reasoning = self.prepare_reasoning_text(reasoning, output)
            kwargs['reasoning'] = prepared_reasoning

            # Also handle reasoning_without_answer for verifiability judge
            if 'reasoning_without_answer' in kwargs:
                reasoning_without_answer = kwargs['reasoning_without_answer']
                prepared_reasoning_no_answer = self.prepare_reasoning_text(reasoning_without_answer, output)
                kwargs['reasoning_without_answer'] = prepared_reasoning_no_answer

        # Get the appropriate prompt
        prompt = self.prompt_loader.get_judge_prompt(judge_type, **kwargs)

        # Call judge model
        messages = [{'role': 'user', 'content': prompt}]
        result = self.client.get_text_response(
            model=self.judge_model,
            messages=messages,
            temperature=0.0,  # Deterministic
            reasoning_config=None  # No reasoning for judges
        )

        # Parse JSON response
        response = result['output']
        parsed = self._parse_json_response(response)
        parsed['raw_judge_response'] = response

        return parsed
    
    def _parse_json_response(self, response: str) -> Dict:
        """Extract and parse JSON from response"""
        # Find JSON blocks
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[-1])  # Try last match
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {'error': 'Failed to parse JSON', 'raw_response': response}
    
    def evaluate_full(
        self,
        problem: str,
        cue: Optional[str],
        reasoning: str,
        answer: str,
        correct_answer: str
    ) -> Dict:
        """
        Run all judges on a response.
        
        Args:
            problem: Problem statement
            cue: Hint that was given (None if baseline)
            reasoning: Model's reasoning/response
            answer: Model's extracted answer
            correct_answer: Correct answer
        
        Returns:
            Dict with results from all applicable judges
        """
        results = {}
        
        # Verbalization (only if cue was given)
        if cue:
            results['verbalization'] = self.evaluate(
                'verbalization_judge',
                problem=problem,
                cue=cue,
                reasoning=reasoning
            )
        
        # Reasoning correctness
        results['correctness'] = self.evaluate(
            'reasoning_correctness_judge',
            problem=problem,
            reasoning=reasoning,
            answer=answer,
            correct_answer=correct_answer
        )
        
        # CoT verifiability
        reasoning_without_answer = self._strip_answer(reasoning, answer)
        results['verifiability'] = self.evaluate(
            'cot_verifiability_judge',
            problem=problem,
            reasoning_without_answer=reasoning_without_answer
        )
        
        return results
    
    def _strip_answer(self, reasoning: str, answer: str) -> str:
        """Remove the final answer from reasoning"""
        if not answer:
            return reasoning

        # Replace answer with redaction
        answer_escaped = re.escape(answer)
        reasoning_redacted = re.sub(
            answer_escaped + r'(?=[^→A-C]*$)',  # Match near end
            '[ANSWER REDACTED]',
            reasoning,
            count=1
        )

        return reasoning_redacted

    def batch_evaluate(
        self,
        judge_requests: List[Dict],
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Evaluate multiple judge requests in parallel using async batch processing.

        Args:
            judge_requests: List of dicts with format:
                {
                    'judge_type': 'verbalization_judge',
                    'kwargs': {problem: ..., cue: ..., reasoning: ..., output: ...}
                }
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of judge results in same order as requests
        """
        # Handle Jupyter notebook environment
        try:
            # Check if we're in a Jupyter environment
            get_ipython()  # This will raise NameError if not in IPython/Jupyter
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self._batch_evaluate_async(judge_requests, max_concurrent))
        except (NameError, ImportError):
            # Not in Jupyter or nest_asyncio not available
            pass

        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we get here, there's a loop running (notebook case without nest_asyncio)
            raise RuntimeError(
                "batch_evaluate called with a running event loop. "
                "Please install nest_asyncio: pip install nest_asyncio"
            )
        except RuntimeError:
            # No running loop (CLI case) - safe to use asyncio.run()
            return asyncio.run(self._batch_evaluate_async(judge_requests, max_concurrent))

    async def _batch_evaluate_async(
        self,
        judge_requests: List[Dict],
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Async implementation of batch evaluation with progress bar.

        Args:
            judge_requests: List of request dicts
            max_concurrent: Maximum concurrent requests

        Returns:
            List of results
        """
        # Build API requests for batch processing
        api_requests = []

        for req in judge_requests:
            judge_type = req['judge_type']
            kwargs = req['kwargs'].copy()

            # Prepare reasoning text based on evaluation mode
            output = kwargs.pop('output', None)
            reasoning = kwargs.get('reasoning')

            if reasoning is not None or output is not None:
                prepared_reasoning = self.prepare_reasoning_text(reasoning, output)
                kwargs['reasoning'] = prepared_reasoning

                # Also handle reasoning_without_answer for verifiability judge
                if 'reasoning_without_answer' in kwargs:
                    reasoning_without_answer = kwargs['reasoning_without_answer']
                    prepared_reasoning_no_answer = self.prepare_reasoning_text(reasoning_without_answer, output)
                    kwargs['reasoning_without_answer'] = prepared_reasoning_no_answer

            # Get the prompt
            prompt = self.prompt_loader.get_judge_prompt(judge_type, **kwargs)

            # Build API request dict
            api_requests.append({
                'model': self.judge_model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.0,
                'reasoning_config': None
            })

        # Execute batch requests with progress bar
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request(idx, req_dict):
            async with semaphore:
                result = await self.client.get_text_response_async(**req_dict)
                return idx, result

        # Create all tasks with index tracking
        tasks = [limited_request(i, req) for i, req in enumerate(api_requests)]

        # Execute with progress bar and collect results
        results_with_idx = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Judging", unit="eval"):
            idx, result = await coro
            results_with_idx.append((idx, result))

        # Sort by original index to maintain order
        results_with_idx.sort(key=lambda x: x[0])
        api_results = [result for _, result in results_with_idx]

        # Parse results
        parsed_results = []
        for result in api_results:
            if result.get('error'):
                parsed_results.append({'error': result['error']})
            else:
                response = result['output']
                parsed = self._parse_json_response(response)
                parsed['raw_judge_response'] = response
                parsed_results.append(parsed)

        return parsed_results