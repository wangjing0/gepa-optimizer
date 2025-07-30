"""Optimized core GEPA optimizer."""

import os
import random
import time
import hashlib
import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from .models import Task, Candidate, CacheEntry

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class GEPAOptimizer:
    """Optimized GEPA optimization engine with caching, batching, and parallel processing."""
    
    def __init__(
        self, 
        model_name: str = "claude-sonnet-4-20250514",
        cache_size: int = 1000,
        max_workers: int = 10,
        cache_ttl: int = 3600
    ):
        """Initialize optimizer with advanced features.
        
        Args:
            model_name: Claude model to use for both target and reflection
            cache_size: Maximum number of cached responses
            max_workers: Maximum number of parallel API calls
            cache_ttl: Cache time-to-live in seconds
        """
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model_name = model_name
        self.rollout_count = 0
        self.candidate_pool: List[Candidate] = []
        self.best_candidate: Optional[Candidate] = None
        
        # Performance optimizations
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.api_call_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Optimization state
        self._pareto_cache: Optional[Set[int]] = None
        self._pareto_cache_generation = -1
    
    def _generate_cache_key(self, prompt: str, input_text: str, operation: str = "rollout") -> str:
        """Generate a cache key for API requests."""
        content = f"{operation}:{self.model_name}:{prompt}:{input_text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - entry.timestamp) < self.cache_ttl
    
    def _cleanup_cache(self):
        """Remove expired and least used cache entries."""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if (current_time - entry.timestamp) > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        
        # Remove least used entries if cache is full
        if len(self.cache) > self.cache_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].usage_count, x[1].timestamp)
            )
            num_to_remove = len(self.cache) - self.cache_size
            for key, _ in sorted_entries[:num_to_remove]:
                del self.cache[key]
    
    def log_message(self, message: str, msg_type: str = 'info') -> None:
        """Log messages with appropriate level and formatting."""
        if msg_type == 'success':
            logger.info("✅ SUCCESS: %s", message)
        elif msg_type == 'fail':
            logger.error("❌ FAIL: %s", message)
        elif msg_type == 'best':
            logger.info("⭐ BEST: %s", message)
        else:
            logger.info("ℹ️  INFO: %s", message)
    
    def run_claude_rollout(self, prompt: str, input_text: str, use_cache: bool = True) -> str:
        """Execute a single rollout using Claude API with caching.
        
        Args:
            prompt: The prompt to evaluate
            input_text: Input text for the task
            use_cache: Whether to use caching for this request
            
        Returns:
            Generated response from Claude
            
        Raises:
            Exception: If API call fails
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(prompt, input_text)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if self._is_cache_valid(entry):
                    entry.usage_count += 1
                    self.cache_hits += 1
                    return entry.response
                else:
                    del self.cache[cache_key]
        
        self.cache_misses += 1
        full_prompt = f"{prompt}\n\nText: \"{input_text}\"\n\nResponse:"
        
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": full_prompt}]
            )
            result = response.content[0].text
            
            # Cache the response
            if use_cache:
                self.cache[cache_key] = CacheEntry(
                    response=result,
                    timestamp=time.time(),
                    usage_count=1
                )
                self._cleanup_cache()
            
            # Track API performance
            api_time = time.time() - start_time
            self.api_call_times.append(api_time)
            
            return result
            
        except Exception as e:
            if "api_key" in str(e).lower():
                raise Exception("Claude API Error: Authorization failed. Check your Anthropic API key.")
            raise Exception(f"Claude API Error: {str(e)}")
    
    def batch_evaluate_candidate(self, candidate: Candidate, tasks: List[Task]) -> Tuple[List[float], float]:
        """Evaluate a candidate on multiple tasks using parallel processing.
        
        Args:
            candidate: Candidate to evaluate
            tasks: List of tasks to evaluate on
            
        Returns:
            Tuple of (scores, average_score)
        """
        def evaluate_single_task(task_idx_pair):
            task, idx = task_idx_pair
            try:
                output = self.run_claude_rollout(candidate.prompt, task.input)
                eval_result = self.evaluate_output(output, task)
                return idx, eval_result["score"]
            except Exception as e:
                self.log_message(f"Error evaluating task {idx+1}: {str(e)}", 'fail')
                return idx, 0.0
        
        # Create futures for parallel evaluation
        futures = []
        task_pairs = [(task, i) for i, task in enumerate(tasks)]
        
        # Submit tasks to thread pool
        for task_pair in task_pairs:
            if self.rollout_count >= getattr(self, '_budget', float('inf')):
                break
            future = self.executor.submit(evaluate_single_task, task_pair)
            futures.append(future)
            self.rollout_count += 1
        
        # Collect results
        scores = [0.0] * len(tasks)
        completed_tasks = 0
        
        for future in as_completed(futures):
            try:
                idx, score = future.result(timeout=30)  # 30 second timeout per task
                scores[idx] = score
                completed_tasks += 1
            except Exception as e:
                self.log_message(f"Task evaluation failed: {str(e)}", 'fail')
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return scores, avg_score
    
    def evaluate_output(self, output: str, task: Task) -> Dict[str, Any]:
        """Evaluate model output using LLM-based evaluation for better quality assessment."""
        if not output or not isinstance(output, str):
            return {"score": 0.0, "feedback": "No valid output generated."}
        
        if not task.output:
            return {"score": 0.0, "feedback": "No evaluation criteria found."}
        
        # Use LLM to evaluate the output quality
        try:
            score, feedback = self._llm_evaluate_output(task.input, output, task.output)
            
            # Validate the score
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                logger.warning("Invalid LLM evaluation score: %s, falling back to keyword matching", score)
                return self._keyword_evaluate_output(output, task)
                
            return {"score": float(score), "feedback": feedback}
            
        except Exception as e:
            logger.error("LLM evaluation failed, falling back to keyword matching: %s", str(e))
            # Fallback to keyword-based evaluation if LLM fails
            return self._keyword_evaluate_output(output, task)
    
    def _llm_evaluate_output(self, task_input: str, output: str, expected_criteria: List[str]) -> Tuple[float, str]:
        """Use LLM to evaluate output quality and relevance."""
        # Create evaluation prompt
        criteria_text = ", ".join(expected_criteria)
        
        evaluation_prompt = f"""You are an expert evaluator. Assess the response quality and provide scores in the EXACT format shown below.

INPUT: {task_input}
RESPONSE: {output}
EXPECTED: {criteria_text}

Rate each dimension from 0 to 10, then provide an overall score from 0.0 to 1.0.

Format your response EXACTLY like this (replace numbers with your scores):
RELEVANCE: 8/10
ACCURACY: 7/10
COMPLETENESS: 6/10
QUALITY: 9/10
OVERALL_SCORE: 0.75
FEEDBACK: Brief explanation here

Be precise with numbers. The OVERALL_SCORE must be a decimal between 0.0 and 1.0."""

        try:
            # Get evaluation from LLM
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=300,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            
            evaluation_text = response.content[0].text
            
            # Log the raw evaluation for debugging
            logger.debug("Raw LLM evaluation response: %s", evaluation_text)
            
            # Parse the evaluation
            score, feedback = self._parse_llm_evaluation(evaluation_text)
            
            # Log successful parsing
            logger.debug("Parsed LLM evaluation - Score: %.3f", score)
            
            return score, feedback
            
        except Exception as e:
            logger.error("Error in LLM evaluation: %s", str(e))
            raise e
    
    def _parse_llm_evaluation(self, evaluation_text: str) -> Tuple[float, str]:
        """Parse the LLM evaluation response to extract score and feedback with robust error handling."""
        lines = evaluation_text.strip().split('\n')
        
        relevance = accuracy = completeness = quality = 0
        overall_score = 0.0
        feedback = "Unable to parse evaluation."
        
        # Helper function to safely extract numeric values
        def safe_extract_number(text: str, default: float = 0.0) -> float:
            try:
                # Remove any non-numeric characters except dots and minus
                import re
                numeric_part = re.search(r'[-+]?(\d+\.?\d*|\.\d+)', text)
                if numeric_part:
                    return float(numeric_part.group(0))
                return default
            except (ValueError, AttributeError):
                return default
        
        try:
            for line in lines:
                line = line.strip()
                if line.startswith('RELEVANCE:'):
                    try:
                        score_part = line.split(':')[1].split('/')[0].strip()
                        relevance = int(safe_extract_number(score_part))
                    except (IndexError, ValueError):
                        relevance = 0
                        
                elif line.startswith('ACCURACY:'):
                    try:
                        score_part = line.split(':')[1].split('/')[0].strip()
                        accuracy = int(safe_extract_number(score_part))
                    except (IndexError, ValueError):
                        accuracy = 0
                        
                elif line.startswith('COMPLETENESS:'):
                    try:
                        score_part = line.split(':')[1].split('/')[0].strip()
                        completeness = int(safe_extract_number(score_part))
                    except (IndexError, ValueError):
                        completeness = 0
                        
                elif line.startswith('QUALITY:'):
                    try:
                        score_part = line.split(':')[1].split('/')[0].strip()
                        quality = int(safe_extract_number(score_part))
                    except (IndexError, ValueError):
                        quality = 0
                        
                elif line.startswith('OVERALL_SCORE:'):
                    try:
                        score_part = line.split(':')[1].strip()
                        overall_score = safe_extract_number(score_part)
                    except (IndexError, ValueError):
                        overall_score = 0.0
                        
                elif line.startswith('FEEDBACK:'):
                    try:
                        feedback = line.split(':', 1)[1].strip()
                    except IndexError:
                        feedback = "No feedback provided."
            
            # If overall score wasn't provided or parsed correctly, calculate from components
            if overall_score == 0.0 and (relevance > 0 or accuracy > 0 or completeness > 0 or quality > 0):
                overall_score = (relevance + accuracy + completeness + quality) / 40.0
            
            # If still no score, try to infer from text content
            if overall_score == 0.0:
                # Look for any decimal numbers in the text that could be the score
                import re
                decimal_matches = re.findall(r'\b[0-1]?\.\d+\b', evaluation_text)
                if decimal_matches:
                    # Use the first decimal that looks like a score (0.0-1.0)
                    for match in decimal_matches:
                        potential_score = float(match)
                        if 0.0 <= potential_score <= 1.0:
                            overall_score = potential_score
                            break
                
                # If still no score, provide a reasonable default based on content
                if overall_score == 0.0:
                    # Simple heuristic: count positive vs negative words
                    positive_words = ['good', 'excellent', 'correct', 'accurate', 'complete', 'relevant', 'clear']
                    negative_words = ['poor', 'bad', 'incorrect', 'inaccurate', 'incomplete', 'irrelevant', 'unclear']
                    
                    text_lower = evaluation_text.lower()
                    positive_count = sum(1 for word in positive_words if word in text_lower)
                    negative_count = sum(1 for word in negative_words if word in text_lower)
                    
                    if positive_count > negative_count:
                        overall_score = 0.7  # Good default
                    elif negative_count > positive_count:
                        overall_score = 0.3  # Poor default
                    else:
                        overall_score = 0.5  # Neutral default
            
            # Ensure score is within bounds
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Create detailed feedback
            detailed_feedback = f"""LLM Evaluation Results:
- Relevance: {relevance}/10
- Accuracy: {accuracy}/10  
- Completeness: {completeness}/10
- Quality: {quality}/10
- Overall Score: {overall_score:.2f}

Feedback: {feedback}

Raw LLM Response: {evaluation_text[:200]}..."""
            
            return overall_score, detailed_feedback
            
        except Exception as e:
            logger.error("Error parsing LLM evaluation: %s", str(e))
            logger.error("Raw evaluation text: %s", evaluation_text[:500])
            
            # Fallback: try to extract any decimal number as score
            try:
                import re
                decimal_matches = re.findall(r'\b[0-1]?\.\d+\b', evaluation_text)
                if decimal_matches:
                    for match in decimal_matches:
                        potential_score = float(match)
                        if 0.0 <= potential_score <= 1.0:
                            return potential_score, f"Fallback score extraction: {evaluation_text}"
            except:
                pass
            
            # Final fallback
            return 0.5, f"Parsing failed, using default score. Raw text: {evaluation_text}"
    
    def _keyword_evaluate_output(self, output: str, task: Task) -> Dict[str, Any]:
        """Fallback keyword-based evaluation method."""
        output_lower = output.lower()
        found_keywords = 0
        feedback_parts = []
        
        for keyword in task.output:
            if keyword.lower() in output_lower:
                found_keywords += 1
                feedback_parts.append(f"SUCCESS: Output contained '{keyword}'.")
            else:
                feedback_parts.append(f"FAILURE: Missing required keyword '{keyword}'.")
        
        score = found_keywords / len(task.output)
        feedback = "FALLBACK KEYWORD EVALUATION:\n" + '\n'.join(feedback_parts) + f"\nFinal Score: {score:.2f}"
        
        return {"score": score, "feedback": feedback}
    
    def reflect_and_mutate_prompt(self, current_prompt: str, examples: List[Dict]) -> str:
        """Generate improved prompt using Claude reflection with enhanced caching and optimization."""
        # Create optimized cache key using hash for better performance
        examples_hash = hashlib.md5(json.dumps(examples, sort_keys=True).encode()).hexdigest()
        cache_key = self._generate_cache_key(current_prompt, examples_hash, "reflection")
        
        # Check cache with early return
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if self._is_cache_valid(entry):
                entry.usage_count += 1
                self.cache_hits += 1
                return entry.response
            else:
                del self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Optimize example formatting for better token efficiency
        examples_text = self._format_examples_optimized(examples)
        
        # Use optimized reflection prompt template
        reflection_prompt = self._build_reflection_prompt(current_prompt, examples_text)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": reflection_prompt}]
            )
            result = response.content[0].text.strip()
            
            # Cache with optimized cleanup
            self.cache[cache_key] = CacheEntry(
                response=result,
                timestamp=time.time(),
                usage_count=1
            )
            
            # Only cleanup cache when necessary
            if len(self.cache) > self.cache_size * 0.9:
                self._cleanup_cache()
            
            return result
            
        except Exception as e:
            raise Exception(f"Claude API Error during reflection: {str(e)}")
    
    def _format_examples_optimized(self, examples: List[Dict]) -> str:
        """Optimized example formatting for better performance."""
        if not examples:
            return "No examples provided."
        
        # Format examples more efficiently
        formatted_examples = []
        for i, e in enumerate(examples, 1):
            example_text = f"Example {i}:\nInput: {e['input']}\nOutput: {e['output']}\nFeedback: {e['feedback']}"
            formatted_examples.append(example_text)
        
        return '\n\n'.join(formatted_examples)
    
    def _build_reflection_prompt(self, current_prompt: str, examples_text: str) -> str:
        """Build optimized reflection prompt template."""
        return f"""Expert prompt engineer task: Improve the given prompt based on performance feedback.

CURRENT PROMPT:
{current_prompt}

PERFORMANCE EXAMPLES:
{examples_text}

TASK: Analyze the feedback and create an improved prompt that addresses failures and enhances successful patterns. Return ONLY the new prompt text."""
    
    def select_candidate_for_mutation(self, num_tasks: int) -> Candidate:
        """Optimized candidate selection using cached Pareto front calculation."""
        if len(self.candidate_pool) == 1:
            return self.candidate_pool[0]
        
        # Check if Pareto front cache is valid
        current_generation = len(self.candidate_pool)
        if (self._pareto_cache is None or 
            self._pareto_cache_generation < current_generation):
            
            # Recalculate Pareto front
            self._calculate_pareto_front(num_tasks)
            self._pareto_cache_generation = current_generation
        
        if not self._pareto_cache:
            return max(self.candidate_pool, key=lambda c: c.avg_score)
        
        # Random selection from cached Pareto front
        selected_id = random.choice(list(self._pareto_cache))
        return next(c for c in self.candidate_pool if c.id == selected_id)
    
    def _calculate_pareto_front(self, num_tasks: int):
        """Calculate and cache the Pareto front."""
        # Vectorized calculation of best scores per task
        best_scores_per_task = [-1.0] * num_tasks
        for candidate in self.candidate_pool:
            for i in range(num_tasks):
                if candidate.scores[i] > best_scores_per_task[i]:
                    best_scores_per_task[i] = candidate.scores[i]
        
        # Identify Pareto front
        pareto_front_ids = set()
        for i, best_score in enumerate(best_scores_per_task):
            for candidate in self.candidate_pool:
                if abs(candidate.scores[i] - best_score) < 1e-6:
                    pareto_front_ids.add(candidate.id)
        
        self._pareto_cache = pareto_front_ids
    
    def test_model_connection(self) -> Tuple[bool, str]:
        """Test Claude API connection."""
        try:
            response = self.run_claude_rollout("Say hello", "World", use_cache=False)
            return True, response
        except Exception as e:
            return False, str(e)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_calls = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_calls * 100) if total_calls > 0 else 0
        avg_api_time = sum(self.api_call_times) / len(self.api_call_times) if self.api_call_times else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_api_time": f"{avg_api_time:.3f}s",
            "total_api_calls": len(self.api_call_times),
            "cache_size": len(self.cache)
        }
    
    def run_optimization(
        self, 
        seed_prompt: str, 
        training_data: List[Task], 
        budget: int,
        test_split: float = 0.5,
        random_seed: int = 42,
        early_stopping_patience: int = 3,
        min_improvement: float = 0.01
    ) -> Tuple[Candidate, Dict[str, Any]]:
        """Run the optimized GEPA optimization process with train/test split.
        
        Args:
            seed_prompt: Initial prompt to optimize
            training_data: List of training tasks
            budget: Maximum number of model rollouts
            test_split: Fraction of data to use for testing (default: 0.5)
            random_seed: Random seed for reproducible splits (default: 42)
            
        Returns:
            Tuple of (best_candidate, results_dict) where results_dict contains
            train scores, test scores, and performance metrics
        """
        self._budget = budget  # Store budget for batch evaluation
        self.log_message("Starting Optimized GEPA Process...")
        
        # Split data into train and test sets
        random.seed(random_seed)  # For reproducible splits
        shuffled_data = training_data.copy()
        random.shuffle(shuffled_data)
        
        split_idx = int(len(shuffled_data) * (1 - test_split))
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]
        
        self.log_message(f"Data split: {len(train_data)} train, {len(test_data)} test tasks")
        self.log_message(f"Train/Test split ratio: {len(train_data)/(len(train_data)+len(test_data)):.1%}/{len(test_data)/(len(train_data)+len(test_data)):.1%}")
        
        # Test model connection
        self.log_message(f"Testing connection to model: {self.model_name}")
        connection_ok, test_result = self.test_model_connection()
        if not connection_ok:
            self.log_message(f"Model connection failed: {test_result}", 'fail')
            raise Exception(f"Cannot connect to model '{self.model_name}': {test_result}")
        self.log_message("Model connection successful!", 'success')
        
        # Initialize with seed prompt evaluation on training set
        logger.info("\n" + "="*50)
        self.log_message("Phase 1: Evaluating Initial Seed Prompt on Training Set")
        
        initial_candidate = Candidate(
            id=0,
            prompt=seed_prompt,
            parent_id=None,
            scores=[0.0] * len(train_data),
            avg_score=0.0
        )
        
        # Use batch evaluation for initial candidate on training data
        scores, avg_score = self.batch_evaluate_candidate(initial_candidate, train_data)
        initial_candidate.scores = scores
        initial_candidate.avg_score = avg_score
        
        self.candidate_pool.append(initial_candidate)
        self.best_candidate = initial_candidate
        
        self.log_message(f"Seed prompt initial score: {initial_candidate.avg_score:.2f}", 'best')
        logger.info("Current Best Prompt:\n---\n%s\n---", self.best_candidate.prompt)
        
        # Main optimization loop with early stopping
        logger.info("\n" + "="*50)
        self.log_message(f"Phase 2: Starting Optimization Loop (Budget: {budget} rollouts)")
        self.log_message(f"Early stopping: patience={early_stopping_patience}, min_improvement={min_improvement}")
        
        iteration_count = 0
        no_improvement_count = 0
        best_score_history = [initial_candidate.avg_score]
        
        while self.rollout_count < budget:
            iteration_start_rollouts = self.rollout_count
            iteration_count += 1
            self.log_message(f"--- Iteration {iteration_count} (Rollouts: {self.rollout_count}/{budget}) ---")
            
            # Select parent candidate for mutation
            parent_candidate = self.select_candidate_for_mutation(len(train_data))
            self.log_message(f"Selected candidate #{parent_candidate.id} (Score: {parent_candidate.avg_score:.2f}) for mutation.")
            
            # Choose random task for reflection from training data
            task_index = random.randint(0, len(train_data) - 1)
            reflection_task = train_data[task_index]
            self.log_message(f"Performing reflective mutation using training task {task_index + 1}...")
            
            try:
                # Generate output for reflection
                rollout_output = self.run_claude_rollout(parent_candidate.prompt, reflection_task.input)
                self.rollout_count += 1
                eval_result = self.evaluate_output(rollout_output, reflection_task)
                
                # Generate new prompt via reflection
                new_prompt = self.reflect_and_mutate_prompt(parent_candidate.prompt, [{
                    "input": reflection_task.input,
                    "output": rollout_output,
                    "feedback": eval_result["feedback"]
                }])
                
                # Create new candidate
                new_candidate = Candidate(
                    id=len(self.candidate_pool),
                    prompt=new_prompt,
                    parent_id=parent_candidate.id,
                    scores=[0.0] * len(train_data),
                    avg_score=0.0
                )
                self.log_message(f"Generated new candidate prompt #{new_candidate.id}.")
                
                # Smart budget allocation for evaluation
                remaining_budget = budget - self.rollout_count
                min_required = len(train_data)
                
                if remaining_budget >= min_required:
                    # Full evaluation when budget allows
                    scores, avg_score = self.batch_evaluate_candidate(new_candidate, train_data)
                    new_candidate.scores = scores
                    new_candidate.avg_score = avg_score
                elif remaining_budget >= min_required // 2:
                    # Partial evaluation on subset when budget is limited
                    subset_size = min(remaining_budget, len(train_data) // 2)
                    train_subset = train_data[:subset_size]
                    self.log_message(f"Limited budget: evaluating on subset of {subset_size} tasks", 'fail')
                    scores, avg_score = self.batch_evaluate_candidate(new_candidate, train_subset)
                    # Scale scores to full dataset
                    new_candidate.scores = scores + [0.0] * (len(train_data) - len(scores))
                    new_candidate.avg_score = avg_score
                else:
                    self.log_message("Insufficient budget for meaningful evaluation.", 'fail')
                    break
                
                # Add to pool if improved
                if new_candidate.avg_score > parent_candidate.avg_score:
                    self.log_message(f"New candidate #{new_candidate.id} improved! Score: {new_candidate.avg_score:.2f} > {parent_candidate.avg_score:.2f}", 'success')
                    self.candidate_pool.append(new_candidate)
                    
                    # Check if this is a new best candidate
                    if new_candidate.avg_score > self.best_candidate.avg_score:
                        improvement = new_candidate.avg_score - self.best_candidate.avg_score
                        if improvement >= min_improvement:
                            self.best_candidate = new_candidate
                            no_improvement_count = 0  # Reset counter on significant improvement
                            self.log_message("NEW BEST PROMPT FOUND!", 'best')
                            self.log_message(f"Improvement: {improvement:.3f} (≥ {min_improvement})", 'success')
                            logger.info("Current Best Prompt:\n---\n%s\n---", self.best_candidate.prompt)
                        else:
                            no_improvement_count += 1
                            self.log_message(f"Minor improvement: {improvement:.3f} (< {min_improvement})", 'fail')
                    else:
                        no_improvement_count += 1
                else:
                    self.log_message(f"New candidate #{new_candidate.id} did not improve. Score: {new_candidate.avg_score:.2f}. Discarding.", 'fail')
                    no_improvement_count += 1
                
                # Track score history
                best_score_history.append(self.best_candidate.avg_score)
                
                # Early stopping check
                if no_improvement_count >= early_stopping_patience:
                    self.log_message(f"Early stopping triggered! No significant improvement for {no_improvement_count} iterations.", 'success')
                    break
                    
            except Exception as e:
                self.log_message(f"Error in optimization iteration: {str(e)}", 'fail')
                # Ensure rollout is counted even if iteration fails
                if iteration_start_rollouts == self.rollout_count:
                    self.rollout_count += 1
        
        # Phase 3: Evaluate best candidate on test set
        logger.info("\n" + "="*50)
        self.log_message("Phase 3: Evaluating Best Candidate on Test Set")
        
        test_scores, test_avg_score = self.batch_evaluate_candidate(self.best_candidate, test_data)
        
        self.log_message(f"Training set score: {self.best_candidate.avg_score:.2f}")
        self.log_message(f"Test set score: {test_avg_score:.2f}")
        
        # Calculate generalization gap
        generalization_gap = self.best_candidate.avg_score - test_avg_score
        self.log_message(f"Generalization gap: {generalization_gap:.2f}")
        
        if generalization_gap > 0.1:
            self.log_message("Warning: Large generalization gap detected. Model may be overfitting.", 'fail')
        elif generalization_gap < -0.05:
            self.log_message("Interesting: Test score exceeds training score.", 'success')
        else:
            self.log_message("Good generalization performance.", 'success')
        
        logger.info("\nFinal Best Prompt:")
        logger.info("---\n%s\n---", self.best_candidate.prompt)
        
        # Compile results
        results = {
            "best_candidate": self.best_candidate,
            "train_score": self.best_candidate.avg_score,
            "test_score": test_avg_score,
            "test_scores": test_scores,
            "generalization_gap": generalization_gap,
            "train_data_size": len(train_data),
            "test_data_size": len(test_data),
            "total_iterations": iteration_count,
            "early_stopped": no_improvement_count >= early_stopping_patience,
            "no_improvement_count": no_improvement_count,
            "score_history": best_score_history,
            "performance_stats": self.get_performance_stats()
        }
        
        # Print performance statistics
        stats = results["performance_stats"]
        logger.info("\nPerformance Statistics:")
        logger.info("  Cache hit rate: %s", stats['cache_hit_rate'])
        logger.info("  Average API time: %s", stats['avg_api_time'])
        logger.info("  Total iterations: %d", iteration_count)
        logger.info("  Final cache size: %d", stats['cache_size'])
        logger.info("  Train/Test sizes: %d/%d", len(train_data), len(test_data))
        logger.info("="*50)
        
        # Cleanup
        self.executor.shutdown(wait=True)
        
        return self.best_candidate, results
    
    async def reflect_and_mutate_prompt_async(self, current_prompt: str, examples: List[Dict]) -> str:
        """Async version of reflect_and_mutate_prompt for better performance."""
        # Create optimized cache key using hash for better performance
        examples_hash = hashlib.md5(json.dumps(examples, sort_keys=True).encode()).hexdigest()
        cache_key = self._generate_cache_key(current_prompt, examples_hash, "reflection")
        
        # Check cache with early return
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if self._is_cache_valid(entry):
                entry.usage_count += 1
                self.cache_hits += 1
                return entry.response
            else:
                del self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Optimize example formatting for better token efficiency
        examples_text = self._format_examples_optimized(examples)
        
        # Use optimized reflection prompt template
        reflection_prompt = self._build_reflection_prompt(current_prompt, examples_text)
        
        try:
            # Use asyncio.to_thread for async API call
            def make_api_call():
                return self.client.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    temperature=0.7,
                    messages=[{"role": "user", "content": reflection_prompt}]
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, make_api_call)
            result = response.content[0].text.strip()
            
            # Cache with optimized cleanup
            self.cache[cache_key] = CacheEntry(
                response=result,
                timestamp=time.time(),
                usage_count=1
            )
            
            # Only cleanup cache when necessary
            if len(self.cache) > self.cache_size * 0.9:
                self._cleanup_cache()
            
            return result
            
        except Exception as e:
            raise Exception(f"Claude API Error during reflection: {str(e)}")
    
    async def run_claude_rollout_async(self, prompt: str, input_text: str, use_cache: bool = True) -> str:
        """Async version of run_claude_rollout for better performance."""
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(prompt, input_text)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if self._is_cache_valid(entry):
                    entry.usage_count += 1
                    self.cache_hits += 1
                    return entry.response
                else:
                    del self.cache[cache_key]
        
        self.cache_misses += 1
        full_prompt = f"{prompt}\n\nText: \"{input_text}\"\n\nResponse:"
        
        start_time = time.time()
        try:
            def make_api_call():
                return self.client.messages.create(
                    model=self.model_name,
                    max_tokens=100,
                    temperature=0.7,
                    messages=[{"role": "user", "content": full_prompt}]
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, make_api_call)
            result = response.content[0].text
            
            # Cache the response
            if use_cache:
                self.cache[cache_key] = CacheEntry(
                    response=result,
                    timestamp=time.time(),
                    usage_count=1
                )
                if len(self.cache) > self.cache_size * 0.9:
                    self._cleanup_cache()
            
            # Track API performance
            api_time = time.time() - start_time
            self.api_call_times.append(api_time)
            
            return result
            
        except Exception as e:
            if "api_key" in str(e).lower():
                raise Exception("Claude API Error: Authorization failed. Check your Anthropic API key.")
            raise Exception(f"Claude API Error: {str(e)}")
    
    async def batch_evaluate_candidate_async(self, candidate: Candidate, tasks: List[Task]) -> Tuple[List[float], float]:
        """Async version of batch_evaluate_candidate for better performance."""
        async def evaluate_single_task_async(task_idx_pair):
            task, idx = task_idx_pair
            try:
                output = await self.run_claude_rollout_async(candidate.prompt, task.input)
                eval_result = self.evaluate_output(output, task)
                return idx, eval_result["score"]
            except Exception as e:
                self.log_message(f"Error evaluating task {idx+1}: {str(e)}", 'fail')
                return idx, 0.0
        
        # Create tasks for parallel evaluation
        task_pairs = [(task, i) for i, task in enumerate(tasks)]
        
        # Limit concurrent tasks to avoid overwhelming the API
        max_concurrent = min(self.max_workers, len(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_evaluate(task_pair):
            async with semaphore:
                if self.rollout_count >= getattr(self, '_budget', float('inf')):
                    return task_pair[1], 0.0
                self.rollout_count += 1
                return await evaluate_single_task_async(task_pair)
        
        # Execute all evaluations concurrently
        results = await asyncio.gather(*[bounded_evaluate(tp) for tp in task_pairs])
        
        # Collect results
        scores = [0.0] * len(tasks)
        for idx, score in results:
            scores[idx] = score
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return scores, avg_score
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)



def run_gepa_optimization(
    model_name: str,
    seed_prompt: str,
    training_data: List[Dict],
    budget: int,
    cache_size: int = 1000,
    max_workers: int = 10,
    test_split: float = 0.5,
    random_seed: int = 42,
    early_stopping_patience: int = 3,
    min_improvement: float = 0.01
) -> Tuple[Candidate, Dict[str, Any]]:
    """Convenience function to run optimized GEPA optimization with train/test split and early stopping.
    
    Args:
        model_name: Claude model name
        seed_prompt: Initial prompt to optimize
        training_data: List of training task dictionaries
        budget: Maximum number of model rollouts
        cache_size: Maximum number of cached responses
        max_workers: Maximum number of parallel API calls
        test_split: Fraction of data to use for testing (default: 0.5)
        random_seed: Random seed for reproducible splits (default: 42)
        early_stopping_patience: Stop if no improvement for N iterations (default: 3)
        min_improvement: Minimum improvement to reset patience counter (default: 0.01)
        
    Returns:
        Tuple of (best_candidate, results_dict) where results_dict contains
        train scores, test scores, generalization gap, early stopping info, and performance metrics
    """
    # Convert training data to Task objects
    tasks = [Task(input=item["input"], output=item["output"]) 
             for item in training_data]
    
    optimizer = GEPAOptimizer(
        model_name=model_name,
        cache_size=cache_size,
        max_workers=max_workers
    )
    return optimizer.run_optimization(
        seed_prompt=seed_prompt, 
        training_data=tasks, 
        budget=budget,
        test_split=test_split,
        random_seed=random_seed,
        early_stopping_patience=early_stopping_patience,
        min_improvement=min_improvement
    )


async def run_gepa_optimization_async(
    model_name: str,
    seed_prompt: str,
    training_data: List[Dict],
    budget: int,
    cache_size: int = 1000,
    max_workers: int = 10,
    test_split: float = 0.5,
    random_seed: int = 42,
    early_stopping_patience: int = 3,
    min_improvement: float = 0.01
) -> Tuple[Candidate, Dict[str, Any]]:
    """Async convenience function to run optimized GEPA optimization with enhanced performance.
    
    Args:
        model_name: Claude model name
        seed_prompt: Initial prompt to optimize
        training_data: List of training task dictionaries
        budget: Maximum number of model rollouts
        cache_size: Maximum number of cached responses
        max_workers: Maximum number of parallel API calls
        test_split: Fraction of data to use for testing (default: 0.5)
        random_seed: Random seed for reproducible splits (default: 42)
        early_stopping_patience: Stop if no improvement for N iterations (default: 3)
        min_improvement: Minimum improvement to reset patience counter (default: 0.01)
        
    Returns:
        Tuple of (best_candidate, results_dict) where results_dict contains
        train scores, test scores, generalization gap, early stopping info, and performance metrics
    """
    # Convert training data to Task objects
    tasks = [Task(input=item["input"], output=item["output"]) 
             for item in training_data]
    
    optimizer = GEPAOptimizer(
        model_name=model_name,
        cache_size=cache_size,
        max_workers=max_workers
    )
    
    # For now, run the sync version until we implement full async optimization
    # This provides the async interface while maintaining compatibility
    def run_sync():
        return optimizer.run_optimization(
            seed_prompt=seed_prompt, 
            training_data=tasks, 
            budget=budget,
            test_split=test_split,
            random_seed=random_seed,
            early_stopping_patience=early_stopping_patience,
            min_improvement=min_improvement
        )
    
    # Run in thread pool to avoid blocking
    return await asyncio.get_event_loop().run_in_executor(None, run_sync)

