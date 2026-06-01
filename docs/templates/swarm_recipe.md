# docs/templates/swarm-recipe.md
---
type: swarm-recipe
agents: [planner, executor, critic]
llm: bitnet-mlx-ternary
fidelity: >99.5%
tags: [swarm, juniorcoach]
---
# {{swarm_name}}
## Architecture
[[Canvas/{{swarm_name}}-Architecture]]
## Results
- **Memory:** {{memory_usage}}
- **TPS:** {{tokens_per_sec}}