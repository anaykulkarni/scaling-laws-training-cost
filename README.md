# Scaling Laws and Training Cost Analysis

### Model Training Cost Analysis

We attempt to estimate the training cost of Llama-7B. We begin with counting the number of model parameters and the amount of flops and memory required for training. In the file `llama_7b_config.json`, the model configurations of Llama-7B is provided. We implement the `model_training_cost_analysis_llama` function in the `model_training_cost_analysis.py` file. This function takes the path to the Llama-7B model configuration file and outputs the number of trainable model parameters, number of TFLOPs and the peak memory cost required during training. 

The total model parameters should include:
* the parameters for word embedding
* the parameters for positional embedding
* the parameters for the transformer layers (Attention, MLP, and Layernorm)

The number of TFLOPs refers to the amount of computation required for the forward pass of a single transformer layer in the Llama-7B model. The peak memory cost is the amount of GPU memory required for the forward pass of a single transformer layer using fixed `fp16` precision training. We assume that we do checkpoint rematerialization at each of the transformer boundary.

We will use the following command to check the output of the implementation:
```bash
python3 model_training_cost_analysis.py --model_config <path-to-llama_7b-config.json>
```

### Exercise: Design your own model training schema
The challenge is to determine the optimal model size and amount of training tokens given a fixed training budget and a scaling law as follows:

$$
L (N, D) = \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.29}} + 1.69
$$

Assume we have a training budget of 5 million dollars and the following three GPU options:

- NVIDIA A100: cost per hour = $4.0, peak FP16 performance = 312 TFLOPs
- NVIDIA V100: cost per hour = $2.5, peak FP16 performance = 125 TFLOPs
- NVIDIA T4:  cost per hour = $1.0, peak FP16 performance = 65 TFLOPs

Assume MFU = 40% for all 3 types of GPUs. 

We implement the `get_optimal_N_D_from_cost` function in the file `model_training_cost_analysis.py`. You need to select the best GPU type, compute the best total effective training FLOPs, and get optimal value of N and D using the scaling law. 

We will use the following command to check the output of your implementation:
```bash
python3 model_training_cost_analysis.py --training_budget <training_budget>
```

After getting optimal model size, we design our own model architecture and create a configuration file named `my_model_config.json`, following similar format of `llama_7b_config.json`.

### MoE Model Cost Analysis

Now let's work on a new popular model: `DeepSeek-V3`. The configuration of this model is provided in the file `deepseek_v3_config.json`. We write a new function `model_training_cost_analysis_deepseek` to analyze the cost of `DeepSeek-V3`. Deepseek *claims* that they can train such a good model using 5 million dollar. 

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