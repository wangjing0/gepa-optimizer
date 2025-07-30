#!/usr/bin/env python3
"""Test script to demonstrate speed optimizations in GEPA optimization."""

import os
import logging
import json
import time
import asyncio
from src.gepa_optimizer import run_gepa_optimization, run_gepa_optimization_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Turn off HTTP request logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def main():
    """Test speed optimizations with different configurations."""
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("Please set your ANTHROPIC_API_KEY environment variable")
        return
    
    # Load small subset of training data for testing
    try:
        with open('data/pupa_training_data.json', 'r') as f:
            training_data = json.load(f)
        
        # Use first 5 examples for quick testing
        training_data = training_data[:5]
        logger.info(f"Loaded {len(training_data)} training examples")
        
    except FileNotFoundError:
        logger.error("Training data file not found. Using fallback data.")
        training_data = [
            {
                "input": "What is the capital of France?",
                "output": ["Paris", "capital", "France"]
            },
            {
                "input": "Explain photosynthesis briefly.",
                "output": ["photosynthesis", "plants", "sunlight"]
            },
            {
                "input": "What is 2+2?",
                "output": ["4", "four", "math"]
            },
            {
                "input": "Name a programming language.",
                "output": ["Python", "programming", "language"]
            },
            {
                "input": "What color is the sky?",
                "output": ["blue", "sky", "color"]
            }
        ]
    
    # Configuration
    MODEL_NAME = "claude-sonnet-4-20250514"
    SEED_PROMPT = "You are a helpful assistant."
    BUDGET = 30  # Small budget for testing
    
    logger.info("=" * 60)
    logger.info("🚀 GEPA Speed Optimization Test")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Budget: {BUDGET} rollouts")
    logger.info(f"Training examples: {len(training_data)}")
    logger.info(f"Seed prompt: {SEED_PROMPT}")
    
    # Test 1: Standard optimization without early stopping
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test 1: Standard Optimization (No Early Stopping)")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        final_result_1, results_1 = run_gepa_optimization(
            model_name=MODEL_NAME,
            seed_prompt=SEED_PROMPT,
            training_data=training_data,
            budget=BUDGET,
            early_stopping_patience=999,  # Disable early stopping
            min_improvement=0.0
        )
        
        duration_1 = time.time() - start_time
        logger.info(f"✅ Standard optimization completed in {duration_1:.1f}s")
        logger.info(f"📈 Final score: {final_result_1.avg_score:.3f}")
        logger.info(f"🔄 Total iterations: {results_1['total_iterations']}")
        logger.info(f"⚡ Cache hit rate: {results_1['performance_stats']['cache_hit_rate']}")
        
    except Exception as e:
        logger.error(f"❌ Standard optimization failed: {e}")
        return
    
    # Test 2: Optimization with early stopping
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test 2: Optimized with Early Stopping")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        final_result_2, results_2 = run_gepa_optimization(
            model_name=MODEL_NAME,
            seed_prompt=SEED_PROMPT,
            training_data=training_data,
            budget=BUDGET,
            early_stopping_patience=3,  # Enable early stopping
            min_improvement=0.01
        )
        
        duration_2 = time.time() - start_time
        logger.info(f"✅ Optimized optimization completed in {duration_2:.1f}s")
        logger.info(f"📈 Final score: {final_result_2.avg_score:.3f}")
        logger.info(f"🔄 Total iterations: {results_2['total_iterations']}")
        logger.info(f"🛑 Early stopped: {results_2['early_stopped']}")
        logger.info(f"⚡ Cache hit rate: {results_2['performance_stats']['cache_hit_rate']}")
        
        # Calculate speedup
        if duration_1 > 0:
            speedup = (duration_1 - duration_2) / duration_1 * 100
            logger.info(f"🚀 Speed improvement: {speedup:.1f}% faster")
        
    except Exception as e:
        logger.error(f"❌ Optimized optimization failed: {e}")
        return
    
    # Test 3: Async optimization (if available)
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test 3: Async Optimization")
    logger.info("=" * 60)
    
    async def test_async():
        start_time = time.time()
        try:
            final_result_3, results_3 = await run_gepa_optimization_async(
                model_name=MODEL_NAME,
                seed_prompt=SEED_PROMPT,
                training_data=training_data,
                budget=BUDGET,
                early_stopping_patience=3,
                min_improvement=0.01
            )
            
            duration_3 = time.time() - start_time
            logger.info(f"✅ Async optimization completed in {duration_3:.1f}s")
            logger.info(f"📈 Final score: {final_result_3.avg_score:.3f}")
            logger.info(f"🔄 Total iterations: {results_3['total_iterations']}")
            logger.info(f"🛑 Early stopped: {results_3['early_stopped']}")
            logger.info(f"⚡ Cache hit rate: {results_3['performance_stats']['cache_hit_rate']}")
            
            return duration_3
            
        except Exception as e:
            logger.error(f"❌ Async optimization failed: {e}")
            return None
    
    # Run async test
    duration_3 = asyncio.run(test_async())
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 OPTIMIZATION COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    if duration_3:
        durations = [duration_1, duration_2, duration_3]
        names = ["Standard", "Early Stopping", "Async"]
        
        for i, (name, duration) in enumerate(zip(names, durations)):
            logger.info(f"{i+1}. {name:<15}: {duration:>6.1f}s")
        
        fastest = min(durations)
        fastest_idx = durations.index(fastest)
        logger.info(f"\n🏆 Fastest method: {names[fastest_idx]} ({fastest:.1f}s)")
        
        # Calculate relative improvements
        baseline = durations[0]
        for i, (name, duration) in enumerate(zip(names[1:], durations[1:]), 1):
            improvement = (baseline - duration) / baseline * 100
            logger.info(f"   {name} is {improvement:.1f}% faster than baseline")
    
    logger.info("\n🎯 Speed optimization recommendations:")
    logger.info("   • Use early stopping (patience=3, min_improvement=0.01)")
    logger.info("   • Enable caching for repeated operations")
    logger.info("   • Use async functions for I/O-bound operations")
    logger.info("   • Consider smart budget allocation for large datasets")

if __name__ == '__main__':
    main()