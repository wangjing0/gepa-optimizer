#!/usr/bin/env python3
"""Test logging functionality for GEPA optimization."""

import logging
import os
import json
from src.gepa_optimizer import GEPAOptimizer

def setup_logging():
    """Set up logging configuration for testing."""
    # Ensure logs directory exists
    import os
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('logs/gepa_test.log')  # File output
        ]
    )

def main():
    """Test logging functionality with GEPA."""
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GEPA logging test...")
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        print("Please set your ANTHROPIC_API_KEY environment variable")
        return
    
    # Load a small subset of data for testing
    try:
        with open('data/pupa_training_data.json', 'r') as f:
            training_data = json.load(f)
        
        # Use only first 3 examples for quick test
        training_data = training_data[:3]
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
            }
        ]
    
    # Create optimizer
    logger.info("Creating GEPA optimizer...")
    optimizer = GEPAOptimizer(
        model_name="claude-sonnet-4-20250514",
        cache_size=100,
        max_workers=2
    )
    
    # Convert training data
    from src.gepa_optimizer.models import Task
    tasks = [Task(input=item["input"], output=item["output"]) 
             for item in training_data]
    
    # Test connection
    logger.info("Testing model connection...")
    connection_ok, test_result = optimizer.test_model_connection()
    
    if connection_ok:
        logger.info("✅ Model connection successful!")
        
        # Run a very small optimization test
        logger.info("Running small optimization test (budget: 5)...")
        try:
            final_result, results = optimizer.run_optimization(
                seed_prompt="You are a helpful assistant.",
                training_data=tasks,
                budget=5,  # Very small budget for testing
                test_split=0.5
            )
            
            logger.info("🎉 Optimization test completed!")
            logger.info(f"Final training score: {final_result.avg_score:.2f}")
            logger.info(f"Test score: {results['test_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            
    else:
        logger.error(f"❌ Model connection failed: {test_result}")
    
    logger.info("GEPA logging test completed!")
    logger.info("Check 'logs/gepa_test.log' file for logged output.")

if __name__ == '__main__':
    main()