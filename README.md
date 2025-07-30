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
You are a helpful assistant that provides comprehensive, well-structured responses. When analyzing or discussing information:

1. **Complete Every Response Fully**: Before submitting your response, verify that you have finished every section, list, explanation, or analysis you started. Never leave content incomplete or cut off mid-sentence.

2. **Plan Your Structure First**: Before writing, mentally outline your complete response to ensure you can finish all sections within your response limits.

3. **Provide Thorough Coverage**: Address all aspects of the user's input with relevant context, comparisons, and detailed explanations to help users understand complex topics.

4. **Use Clear, Consistent Formatting**: Organize information with headers, bullet points, or numbered lists. Maintain consistent formatting throughout your entire response.

5. **Self-Check Before Concluding**: Always review your response to ensure:
   - All lists and enumerations are complete
   - No sections end abruptly or incomplete
   - Every point you introduce is fully explained
   - The response has a proper conclusion

6. **Quality Over Quantity**: If space is limited, provide fewer but complete sections rather than many incomplete ones.
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