#!/usr/bin/env python3
"""Example usage of the GEPA optimizer."""

import os
import logging
import asyncio
from src.gepa_optimizer import run_gepa_optimization, run_gepa_optimization_async
import json

# Configure logging to show all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Display in console
    ]
)

# Turn off HTTP request logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def main():
    """Run a simple GEPA optimization example."""
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("Please set your ANTHROPIC_API_KEY environment variable")
        logger.info("Example: export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Configuration
    MODEL_NAME = "claude-3-5-haiku-latest"
    SEED_PROMPT = "You are a helpful assistant."
    
    # Training data
    training_data = json.load(open('example_training_data.json'))
    
    BUDGET = 100
    
    logger.info("🧬 GEPA Optimization Example")
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Budget: %d rollouts", BUDGET)
    logger.info("Training examples: %d", len(training_data))
    logger.info("Initial prompt: \n%s\n%s\n%s", '-' * 20, SEED_PROMPT, '-' * 20)
    logger.info("\n"+"=" * 50)
    
    # Run optimization
    try:
        import time
        
        # Run sync optimization
        logger.info("🔄 Running synchronous optimization...")
        start_time = time.time()
        final_result, results = run_gepa_optimization(
            model_name=MODEL_NAME,
            seed_prompt=SEED_PROMPT,
            training_data=training_data,
            budget=BUDGET
        )
        sync_duration = time.time() - start_time

        # Run async optimization
        logger.info("🚀 Running asynchronous optimization...")
        start_time = time.time()
        final_result_async, results_async = await run_gepa_optimization_async(
            model_name=MODEL_NAME,
            seed_prompt=SEED_PROMPT,
            training_data=training_data,
            budget=BUDGET
        )
        async_duration = time.time() - start_time
        
        logger.info("\n🎉 Both optimizations completed successfully!")
        logger.info("\n📊 SYNC RESULTS:")
        logger.info("   Final training score: %.2f", final_result.avg_score)
        logger.info("   Final test score: %.2f", results['test_score'])
        logger.info("   Generalization gap: %.2f", results['generalization_gap'])
        logger.info("   Duration: %.1f seconds", sync_duration)
        logger.info("\n🏆 Best prompt (from sync run):\n%s", '-' * 20)
        logger.info("%s", final_result.prompt)
        logger.info("%s", "-" * 20)
        
        logger.info("\n📊 ASYNC RESULTS:")
        logger.info("   Final training score: %.2f", final_result_async.avg_score)
        logger.info("   Final test score: %.2f", results_async['test_score'])
        logger.info("   Generalization gap: %.2f", results_async['generalization_gap'])
        logger.info("   Duration: %.1f seconds", async_duration)
        logger.info("\n🏆 Best prompt (from async run):\n%s", '-' * 20)
        logger.info("%s", final_result_async.prompt)
        logger.info("%s", "-" * 20)
        
        # Performance comparison
        if sync_duration > 0:
            speedup = (sync_duration - async_duration) / sync_duration * 100
            logger.info(f"\n⚡ Performance: Async was {speedup:.1f}% faster than sync")
        
        
        
    except Exception as e:
        logger.error("\n❌ An error occurred during execution: %s", e)


if __name__ == '__main__':
    asyncio.run(main())