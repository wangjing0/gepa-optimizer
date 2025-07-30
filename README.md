# GEPA: Genetic-Pareto Evolutionary Algorithm
Implementation of the GEPA optimizer.

Reference:
[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)

Before optimization, throw a randomprompt to the model.
```
You are a helpful assistant.
```

After optimization, you will get a prompt that is much improved from the seed prompt, with respect to your task and training data.

```
You are a helpful assistant that provides complete, well-structured responses. When analyzing or explaining concepts:

1. Always provide a full, complete response - never cut off mid-sentence or leave thoughts unfinished
2. Structure your response with clear sections using headers or bullet points when appropriate
3. Include relevant examples to illustrate key points and enhance understanding
4. Ensure accuracy while maintaining comprehensive coverage of the topic
5. Before responding, mentally outline your complete response to avoid incomplete answers

Always finish your thoughts completely and provide concrete examples when explaining concepts or definitions.
```


### 1. Set up your API key

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```
or create a .env file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 2. Installation

```bash
git clone https://github.com/wangjing0/gepa-optimizer.git
cd gepa-optimizer
uv init
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
import json

# Load training data (converted from PUPA dataset or your own format)
training_data = json.load(open('data/pupa_training_data.json'))

final_result, results = run_gepa_optimization(
    model_name="claude-sonnet-4-20250514",
    seed_prompt="You are a helpful assistant.",
    training_data=training_data,
    budget=200,
    early_stopping_patience=3,
    min_improvement=0.01
)

print(f"Final training score: {results['train_score']:.2f}")
print(f"Final test score: {results['test_score']:.2f}")  
print(f"Generalization gap: {results['generalization_gap']:.2f}")
print(f"Best prompt:\n{'-' * 20}")
print(final_result.prompt)
print(f"{'-' * 20}")
```
## License

MIT License