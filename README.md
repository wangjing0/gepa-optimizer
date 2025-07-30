# GEPA: Genetic-Pareto Evolutionary Algorithm
https://arxiv.org/abs/2507.19457
GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning


## Features

- Genetic Evolution: Automatically evolves prompts using genetic algorithm principles
- Claude Integration: Uses Anthropic's Claude models for both target and reflection tasks
- Performance Tracking: Tracks and displays best-performing prompts during optimization
- Pareto Optimization: Selects candidates based on multi-objective optimization
- Reflective Mutation: Uses powerful LLM reflection to generate improved prompts
- Customizable Evaluation: Easy-to-customize evaluation functions for different tasks

## Quick Start

### 1. Set up your API key

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the example

```bash
uv run python example.py
```


## Installation

Using uv (recommended):

```bash
git clone https://github.com/mnemosyne/gepa-optimizer.git
cd gepa-optimizer
uv sync
```

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Or create a .env file:

```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Python API

```python
from src.gepa_optimizer import run_gepa_optimization
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset 'Columbia-NLP/PUPA'
training_data = pandas.read_table('data/pupa_tnb.parquet')

result = run_gepa_optimization(
    model_name="claude-sonnet-4-20250514",
    seed_prompt="Summarize the following text.",
    training_data=training_data,
    budget=15
)

print(f"Best prompt: {result.prompt}")
print(f"Score: {result.avg_score:.2f}")
```

### Command Line

```bash
uv run gepa --seed-prompt "Your initial prompt" --training-data data.json --budget 20 --output result.json
```

## Training Data Format

Training data should be a JSON file with this structure:

```json
[
  {
    "input": "Text to be processed by the prompt",
    "expected_keywords": ["keyword1", "keyword2", "keyword3"]
  }
]
```

## Development

```bash
# Install development dependencies
uv sync --dev

# Run the example
uv run python example.py

# Run with CLI
uv run gepa --help
```


## License

MIT License