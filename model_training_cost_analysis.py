import argparse
import json
import math
import numpy as np

# Llama

def calc_attn_flops(s, h, n):
    # QKV generation
    qkv_flops = 6 * s * h * h
    # Rope
    rope = 3 * s * h
    # attention :     QK^T +  softmax 
    attn_score_flops = (2 * s * s * h) + (3 * s * s * n)
    # PV
    pv_flops = 2 * s * s * h
    # Total attn
    attn_flops = rope + attn_score_flops + pv_flops
    # output layer
    out = 2 * s * h * h 
    return qkv_flops + attn_flops + out

def llama_ff_flops(s, h, i):
    # Gate and up
    gate_up = 2 * 2 * s * h * i
    # Swish and elementwise
    swish_elem = 5 * s * i
    # Down
    down = 2 * s * h * i
    return gate_up + swish_elem + down

def TFLOPs_per_layer(s, h, i, n):
    # 2 layer norms
    # 1 attention 
    # one llama feed forward 
    # two skip connections
    return (2 * calc_rmsnorm_flops(h) 
            + calc_attn_flops(s, h, n) 
            + llama_ff_flops(s, h, i) 
            + 2 * s * h) # skip connections

# Deepseek v3

def calc_rmsnorm_flops(h):
    # rms : square and mean = 2h. Then, sqrt O(1)
    # divide: h
    # scale (gamma) : h
    return 4 * h

def calc_mla_attn_flops(s, h, c_q, c_kv, n):
    # QKV generation
    qkv_flops = 6 * s * h * h
    # Up and Down Projections for q
    updown_q = 2 * 2 * s * h * c_q
    # Up and Down Projections for k, v
    updown_kv = (2 * 2 * s * h * c_kv) * 2 # k and v
    # ignoring rope for simplicity
    # attention :     QK^T  + softmax 
    attn_score_flops = (2 * s * s * h) + (3 * s * s * n)
    # multiply by V
    attn_flops = attn_score_flops + 2 * s * s * h
    # Output linear layer
    output_layer = 2 * s * h * h 

    return qkv_flops + (updown_q + updown_kv) + attn_flops + output_layer

def moe_flops(s, h, int_r, int_s):
    # gating
    gating_flops = 2 * s * h * 256

    # experts
    routed_expert_flops = 8 * (2 * s * h  * int_r) * 2 # Top 8 experts, up-down
    shared_expert_flops = 1 * (2 * s * h * int_s) * 2 # one shared expert up-down

    # multiply accumulate
    gating_weights = s * h * 9 # multiply weights to (shared + routed) expert outputs
    add = 9 * s * h # combine shared and routed outputs

    return gating_flops + routed_expert_flops + shared_expert_flops + (gating_weights + add)

def TFLOPs_per_layer_deepseek(s, h, c_q, c_kv, int_r, int_s, n):
    # 2  layer norms
    # MLA attention 
    # MoE feed forward 
    # two skip connections in transformer layer
    return (2 * calc_rmsnorm_flops(h) 
            + calc_mla_attn_flops(s, h, c_q, c_kv, n) 
            + moe_flops(s, h, int_r, int_s) 
            + 2 * s * h)

# Cost Analysis Llama
def model_training_cost_analysis_llama(model_config_path):
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Fixing batch size = 1, and sequence length = max_sequence_length
    max_sequence_length = model_config['max_sequence_length']
    batch_size = 1

    # Get Config
    vocab_size = model_config['vocab_size']
    hidden_size = model_config['hidden_size']
    intermediate_size = model_config['intermediate_size']
    num_attention_heads = model_config['num_attention_heads']  
    num_hidden_layers = model_config['num_hidden_layers']
    

    # now we have to get the token embeddings  ( 2 is for the up projection and down projection) 
    embedding_params = vocab_size * hidden_size * 2
    # Parameters for the transformer layers 
    # attention: Q, K, V, projections and output projection
    attention_params = 4 * hidden_size * hidden_size 
    # MLP up projection and down projection
    mlp_params = 3 * hidden_size * intermediate_size 
    # layer norm, 2 per layer - pre attention and pre mlp (gamma only)
    layer_norm_params = 2 * hidden_size
    # Params in a single layer
    params_per_layer = attention_params + mlp_params + layer_norm_params
    # Params in all layers
    transformer_params = num_hidden_layers * params_per_layer
    # Total params
    total_params = embedding_params + transformer_params + hidden_size  
    
    # Calulcate FLOPs
    # Assuming batch_size = 1
    flops_layer_TF = TFLOPs_per_layer(
        max_sequence_length, hidden_size, intermediate_size, num_attention_heads
    )

    # Calculate Memory for single layer
    weights_memory = params_per_layer * 2 # FP16
    activations_memory =  max_sequence_length * hidden_size * 2 # FP16
    peak_memory_GB = (weights_memory + activations_memory) / (1024**3)

    return total_params, batch_size * flops_layer_TF/(10**12), batch_size * peak_memory_GB


# Cost analysis deepseek
def model_training_cost_analysis_deepseek(model_config_path):
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    # fixing batch = 1, and sequence length = max_sequence_length
    max_sequence_length = 128000
    batch_size = 1

    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    intermediate_size = config["intermediate_size"]
    moe_intermediate_size = config["moe_intermediate_size"]
    n_routed_experts = config["n_routed_experts"]
    n_shared_experts = config["n_shared_experts"]
    q_lora_rank = config["q_lora_rank"]
    kv_lora_rank = config["kv_lora_rank"]
    num_attention_heads = config["num_attention_heads"]
    
    # Embeddings + LM Head
    token_embed = vocab_size * hidden_size
    lm_head = vocab_size * hidden_size  # tie_word_embeddings=False
    embeddings_total = token_embed + lm_head

    # Transformer decoder layers
    # Attention Layers
    # Base QKV projections
    qkv_base = 3 * hidden_size**2
    # LoRA adapters (Q: 2 matrices, K/V: 2 matrices each)
    q_lora = 2 * hidden_size * q_lora_rank
    kv_lora = 2 * 2 * hidden_size * kv_lora_rank  # K and V
    lora_total = q_lora + kv_lora
    # Output projection
    output_proj = hidden_size**2
    # Layer Norms (gamma only, RMSNorm)
    layer_norms = hidden_size * 2 * num_layers  # 2 norms per layer
    # Total per attention layer
    attention_per_layer = qkv_base + lora_total + output_proj + layer_norms

    # MoE FFN Layers
    # Gating
    gating_params = hidden_size * n_routed_experts
    # Routed experts (256 per layer)
    routed_expert_params = 2 * hidden_size * moe_intermediate_size
    routed_experts_per_layer = n_routed_experts * routed_expert_params
    # Shared expert (1 per layer)
    shared_expert_params = 2 * hidden_size * intermediate_size
    # Total per MoE layer
    moe_per_layer = gating_params + routed_experts_per_layer + shared_expert_params

    # Total params per decoder layer (Attention + MoE)
    decoder_total_per_layer = attention_per_layer + moe_per_layer

    # Total params in all decoder layers
    decoder_params = num_layers * decoder_total_per_layer

    # Total Parameters
    total_params = (
        embeddings_total +
        decoder_params
    )

    # Calulcate FLOPs
    # Assuming batch_size = 1
    # s, h, c_q, c_kv, int_r, int_s
    flops_layer_TF = TFLOPs_per_layer_deepseek(
        max_sequence_length, hidden_size, 
        q_lora_rank, kv_lora_rank,
        moe_intermediate_size, intermediate_size,
        num_attention_heads
    )

    # Calculate Memory for single layer
    weights_memory = decoder_total_per_layer * 2 # FP16
    activations_memory =  max_sequence_length * hidden_size * 2 # FP16
    peak_memory_GB = (weights_memory + activations_memory) / (1024**3)

    return total_params, batch_size * flops_layer_TF/(10**12), batch_size * peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    # 1) Define GPU info: (cost_per_hour, peak_tf32_tflops)
    gpu_options = {
        "A100": {"cost_hr": 4.0, "peak_fp16": 312},
        "V100": {"cost_hr": 2.5, "peak_fp16": 125},
        "T4":   {"cost_hr": 1.0, "peak_fp16":  65},
    }
    mfu = 0.40
    best_gpu = None
    best_flops_per_dollar = 0.0

    # 2) Find best FLOPs/$ ratio
    for gpu_name, spec in gpu_options.items():
        cost_hr = spec["cost_hr"]
        peak = spec["peak_fp16"]
        # Effective TFLOPs/s
        effective_tflops_s = peak * mfu
        # FLOPs per hour
        flops_per_hour = effective_tflops_s * (3600.0) * 1e12  # TF -> actual FLOPs
        # FLOPs per dollar
        flops_per_dollar = flops_per_hour / cost_hr
        if flops_per_dollar > best_flops_per_dollar:
            best_flops_per_dollar = flops_per_dollar
            best_gpu = gpu_name

    # 3) Once we know the best GPU, compute total FLOPs = flops_per_dollar * budget
    spec = gpu_options[best_gpu]
    cost_hr = spec["cost_hr"]
    peak = spec["peak_fp16"]
    effective_tflops_s = peak * mfu
    flops_per_hour = effective_tflops_s * 3600.0 * 1e12
    flops_per_dollar = flops_per_hour / cost_hr
    training_budget_flops = F_max = flops_per_dollar * cost_budget # training budget flops

    # 4) Our constraint is 6*N*D = F_max  => N*D = K
    K = F_max / 6.0

    # 5) Solve for N that minimizes L(N) = A/N^a + B/(K/N)^b + C
    A, a = 406.4, 0.34
    B, b = 410.7, 0.29
    C = 1.69

    # After calculating derivative of L(N, K/N) and setting it to zero we get
    # N^(a+b) = (aA/bB) * K^b
    N_ab = (a*A) * np.power(K, b) / (b*B)
    N = np.power(N_ab, 1/(a+b))
    D = K / N

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        elif 'my_model' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    