# GEPA: Genetic-Pareto Evolutionary Algorithm
https://arxiv.org/abs/2507.19457
GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning


## Quick Start

### 1. Set up your API key

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```


### 2. Installation

Using uv (recommended):

```bash
git clone https://github.com/wangjing0/gepa-optimizer.git
cd gepa-optimizer
uv venv
uv sync
```
### 3. Run the example

```bash
uv run python example.py
```

## Usage

### Python API Example

```python
from src.gepa_optimizer import run_gepa_optimization
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset 'Columbia-NLP/PUPA'
training_data = pandas.read_table('data/pupa_tnb.parquet')

result = run_gepa_optimization(
    model_name="claude-sonnet-4-20250514",
    seed_prompt="Answer the question.",
    training_data=training_data,
    budget=15
)

print(f"Best prompt: {result.prompt}")
print(f"Score: {result.avg_score:.2f}")
```

## License

MIT License