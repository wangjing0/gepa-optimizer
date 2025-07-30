#!/usr/bin/env python3
"""Example usage of the GEPA optimizer with simple logging configuration."""

import os
import logging
from src.gepa_optimizer import run_gepa_optimization
import json

# Simple logging configuration - shows just the message
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simple format - just show the message
    handlers=[
        logging.StreamHandler(),  # Display in console
    ]
)

# Turn off HTTP request logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def main():
    """Run a simple GEPA optimization example."""
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("Please set your ANTHROPIC_API_KEY environment variable")
        logger.info("Example: export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Configuration
    MODEL_NAME = "claude-sonnet-4-20250514"
    SEED_PROMPT = "You are a helpful assistant."
    
    # Training data
    training_data = json.load(open('data/example_training_data.json'))
    
    BUDGET = 50  # Smaller budget for demo
    
    logger.info("🧬 GEPA Optimization Example")
    logger.info("=" * 50)
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Budget: %d rollouts", BUDGET)
    logger.info("Training examples: %d", len(training_data))
    logger.info("Initial prompt: \n%s\n%s\n%s", '-' * 20, SEED_PROMPT, '-' * 20)
    logger.info("=" * 50)
    
    # Run optimization
    try:
        final_result, results = run_gepa_optimization(
            model_name=MODEL_NAME,
            seed_prompt=SEED_PROMPT,
            training_data=training_data,
            budget=BUDGET
        )
        
        logger.info("\n🎉 Optimization completed successfully!")
        logger.info("📊 Final training score: %.2f", final_result.avg_score)
        logger.info("📊 Final test score: %.2f", results['test_score'])
        logger.info("📊 Generalization gap: %.2f", results['generalization_gap'])
        logger.info("🏆 Best prompt:\n%s", '-' * 20)
        logger.info("%s", final_result.prompt)
        logger.info("%s", "-" * 20)
        
    except Exception as e:
        logger.error("\n❌ An error occurred during execution: %s", e)


if __name__ == '__main__':
    main()