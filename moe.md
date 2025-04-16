### Changes introduced in `model_training_cost_analysis_deepseek()`
1. Changes to attention module: Deepseek uses Multi-head latent attention - this includes additional parameter introduced by latent vectors for q k v
2. Changes to FFN: We need to consider parameters introduced by router, routed experts, and shared expert.
3. RMSNorm: Deepseek uses RMSNorm instead of LayerNorm. RMSNorm is faster and requires less learnable parameters.
   
### Advantages of MoE
Mixture of Experts (MoE) models offer significant advantages in scalability, efficiency, and specialization. By activating only a subset of experts per input, they reduce computational costs while maintaining high model capacity, enabling efficient training and inference for large-scale models. MoE models distribute computations across multiple devices, reducing memory overhead and improving parallelism, making them ideal for handling massive datasets. Their expert specialization enhances performance in multi-task and multi-modal learning, adapting dynamically to input complexity.

### My Model Config
Model designed to fit provided budget of $5 million.

Run: 
```
python3 model_training_cost_analysis.py --model_config my_model_config.json
```