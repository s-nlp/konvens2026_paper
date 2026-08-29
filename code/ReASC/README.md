# Reliability-Aware Adaptive Self-Consistency (ReASC)

ReASC is an inference-time optimization module that reduces the cost of Self-Consistency by:
1. **Gating**: Early-stopping high-confidence samples (Stage 1).
2. **Weighted Voting**: Using token-level log probabilities to weigh multiple samples (Stage 2).

## Quick Start
```python
from ReASC import ReASCPipeline
pipeline = ReASCPipeline(model=model, tokenizer=tokenizer)
result = pipeline.run_inference("Your prompt here")
```