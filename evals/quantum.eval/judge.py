"""Judge classes for evaluating model responses"""

import json
import re
from typing import Optional, Dict


class Judge:
    """Generic judge that uses different prompts for different evaluations"""
    
    def __init__(self, client, prompt_loader, judge_model: str):
        """
        Args:
            client: OpenRouterClient instance
            prompt_loader: PromptLoader instance
            judge_model: Model name to use for judging
        """
        self.client = client
        self.prompt_loader = prompt_loader
        self.judge_model = judge_model
    
    def evaluate(self, judge_type: str, **kwargs) -> Dict:
        """
        Run a judge evaluation.
        
        Args:
            judge_type: Which judge prompt to use ('verbalization_judge', 
                       'reasoning_correctness_judge', 'cot_verifiability_judge')
            **kwargs: Arguments to format the prompt (problem, cue, reasoning, etc.)
        
        Returns:
            Parsed JSON response from judge
        """
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