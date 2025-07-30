"""Command-line interface for GEPA optimizer."""

import argparse
import json
import os
import sys

from .core import run_gepa_optimization


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GEPA: Genetic-Evolutionary Prompt Architecture"
    )
    
    parser.add_argument(
        "--model", 
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    
    parser.add_argument(
        "--seed-prompt",
        required=True,
        help="Initial prompt to optimize"
    )
    
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to JSON file with training data"
    )
    
    parser.add_argument(
        "--budget",
        type=int,
        default=15,
        help="Maximum number of model rollouts (default: 15)"
    )
    
    parser.add_argument(
        "--output",
        help="Path to save the optimized prompt"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    # Load training data
    try:
        with open(args.training_data, 'r') as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training data file not found: {args.training_data}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in training data file: {args.training_data}")
        sys.exit(1)
    
    # Validate training data format
    if not isinstance(training_data, list):
        print("Error: Training data must be a list of dictionaries")
        sys.exit(1)
    
    for i, item in enumerate(training_data):
        if not isinstance(item, dict):
            print(f"Error: Training data item {i} must be a dictionary")
            sys.exit(1)
        if "input" not in item or "output" not in item:
            print(f"Error: Training data item {i} must have 'input' and 'output' fields")
            sys.exit(1)
    
    print(f"Starting GEPA optimization...")
    print(f"Model: {args.model}")
    print(f"Seed prompt: {args.seed_prompt}")
    print(f"Training examples: {len(training_data)}")
    print(f"Budget: {args.budget}")
    print("-" * 50)
    
    try:
        # Run optimization
        result = run_gepa_optimization(
            model_name=args.model,
            seed_prompt=args.seed_prompt,
            training_data=training_data,
            budget=args.budget
        )
        
        # Output results
        print(f"\n🎉 Optimization completed successfully!")
        print(f"Final score: {result.avg_score:.2f}")
        print(f"Best prompt:\n{'-' * 20}")
        print(result.prompt)
        print("-" * 20)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "optimized_prompt": result.prompt,
                    "score": result.avg_score,
                    "candidate_id": result.id,
                    "parent_id": result.parent_id,
                    "scores": result.scores
                }, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()