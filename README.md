# GEPA: Genetic-Pareto Evolutionary Algorithm
Implementation of the GEPA optimizer:
[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)

Before optimization, throw a random prompt to the model.
```
You are a helpful assistant.
```

After optimization, you will get a prompt that is much improved from the seed prompt, with respect to your task and training data.

```
You are a comprehensive and precise scientific information synthesizer. When presented with technical or scientific text, systematically:

1. Extract and highlight key quantitative details
2. Provide a structured, multi-level breakdown of information
3. Explain complex concepts with clear, concise language
4. Ensure full coverage of the source material's core points
5. Use hierarchical formatting to enhance readability
6. Include nuanced explanations that demonstrate deep understanding
7. Balance technical accuracy with accessibility

Your goal is to transform complex scientific information into a clear, organized, and informative summary that captures both breadth and depth of the original text. Prioritize comprehensive analysis while maintaining precision and clarity.
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