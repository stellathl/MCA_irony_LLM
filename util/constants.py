# MODELS = {
#     "Gemma-3-4B":          "google/gemma-3-4b-it",
#     "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
#     "OLMo-2-7B":           "allenai/OLMo-2-1124-7B-Instruct",
#     "Qwen3-8B":            "Qwen/Qwen3-8B",
#     "Gpt2-1.5B":            "openai-community/gpt2-xl",
#     "Llama-3-8B":          "meta-llama/Llama-3.1-8B-Instruct"
# }

MODELS = {
    "Gemma-3-1B": "google/gemma-3-1b-it"
 }

PROMPT_FILES = {
    "rsa": "rsa_prompt.yaml",
    #  "general": "general_prompt.yaml",
    #  "general_reasoning": "general_prompt_with_reasoning.yaml"
}

SHARED_SEEDS = 1

SEEDS = {
    "Gemma-3-1B"         : SHARED_SEEDS,
    "Gemma-3-4B"         : SHARED_SEEDS,
    "Mistral-7B-Instruct": SHARED_SEEDS,
    "OLMo-2-7B"          : SHARED_SEEDS,
    "Qwen3-8B"           : SHARED_SEEDS,
    "Gpt2-1.5B"      : SHARED_SEEDS,
    "Llama-3-8B"         : SHARED_SEEDS,
}

# ─────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────
MAX_CONTEXT = {
    "Gemma-3-1B":           32_768,
    "Gemma-3-4B":          131_072,
    "Mistral-7B-Instruct":  32_768,
    "OLMo-2-7B":           131_072,
    "Qwen3-8B":            131_072,
    "Gpt2-1.5B":            1_024,
    "Llama-3-8B":          131_072,
}