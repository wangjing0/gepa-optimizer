#!/usr/bin/env python3
"""Test script for LLM evaluation parsing."""

import os
import logging
from src.gepa_optimizer.core import GEPAOptimizer
from src.gepa_optimizer.models import Task

# Configure logging to show debug info
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Turn off HTTP request logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def test_evaluation_parsing():
    """Test the LLM evaluation parsing with various formats."""
    
    optimizer = GEPAOptimizer(model_name="claude-sonnet-4-20250514")
    
    # Test cases with different response formats
    test_cases = [
        # Perfect format
        """RELEVANCE: 8/10
ACCURACY: 7/10
COMPLETENESS: 6/10
QUALITY: 9/10
OVERALL_SCORE: 0.75
FEEDBACK: Good response with clear structure""",
        
        # Missing some components
        """RELEVANCE: 5/10
OVERALL_SCORE: 0.5
FEEDBACK: Partial response""",
        
        # Different formatting
        """Relevance: 9 out of 10
Accuracy: 8/10
Completeness: 7/10
Quality: 6/10
Overall Score: 0.75
Feedback: Excellent work""",
        
        # Malformed numbers
        """RELEVANCE: high/10
ACCURACY: 7/10
COMPLETENESS: N/A/10
QUALITY: 9/10
OVERALL_SCORE: good
FEEDBACK: Mixed quality response""",
        
        # Completely different format
        """This is a good response. I would rate it 7 out of 10 for relevance and accuracy. Overall score: 0.7""",
        
        # Empty/minimal response
        """Score: 0.5""",
    ]
    
    logger.info("Testing LLM evaluation parsing with various formats...")
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n--- Test Case {i} ---")
        logger.info(f"Input: {repr(test_case[:50])}...")
        
        try:
            score, feedback = optimizer._parse_llm_evaluation(test_case)
            logger.info(f"✅ Parsed Score: {score:.3f}")
            logger.info(f"📝 Feedback length: {len(feedback)} chars")
            
            # Validate score is in correct range
            if 0.0 <= score <= 1.0:
                logger.info("✅ Score is in valid range")
            else:
                logger.error("❌ Score out of range!")
                
        except Exception as e:
            logger.error(f"❌ Parsing failed: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("LLM Evaluation parsing test completed!")

def test_full_evaluation():
    """Test full evaluation with real API call if API key is available."""
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.info("No API key found, skipping full evaluation test")
        return
    
    logger.info("\n--- Testing Full Evaluation with API ---")
    
    optimizer = GEPAOptimizer(model_name="claude-sonnet-4-20250514")
    
    # Create test task
    test_task = Task(
        input="What is the capital of France?",
        output=["Paris", "capital", "France"]
    )
    
    test_outputs = [
        "Paris is the capital of France.",
        "The capital city of France is Paris.",
        "France's capital is called Paris.",
        "I don't know the answer.",
        "The answer is London.",  # Wrong answer
    ]
    
    for i, output in enumerate(test_outputs, 1):
        logger.info(f"\nTest {i}: {output}")
        try:
            result = optimizer.evaluate_output(output, test_task)
            logger.info(f"Score: {result['score']:.3f}")
            logger.info(f"Feedback: {result['feedback'][:100]}...")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

if __name__ == '__main__':
    logger.info("🧪 LLM Evaluation Test Suite")
    logger.info("="*60)
    
    # Test parsing without API calls
    test_evaluation_parsing()
    
    # Test full evaluation with API calls
    test_full_evaluation()
    
    logger.info("\n🎉 All tests completed!")