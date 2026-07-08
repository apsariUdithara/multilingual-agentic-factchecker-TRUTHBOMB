# 05: Models and Extraction Logic

## Custom Fine-Tuning Pipeline

The heart of the system is a specialized model for extracting verifiable claims from Sri Lankan social media posts. This model was created through a Supervised Fine-Tuning (SFT) process.

### 1. Training Data (`data/extraction_train.jsonl`)
The model was trained on a curated dataset of instruction-input-output triplets.
- **Instruction**: "You are an expert linguistic AI. Extract the core verifiable claim... Translate it to professional English..."
- **Input**: Raw social media posts in English, Sinhala, or code-mixed Singlish (e.g., "Abe mata dakka! Police maama eka hadala gana drug trade kiyala!").
- **Output**: A normalized, declarative English claim (e.g., "A police officer was arrested for drug trafficking...").

### 2. Fine-Tuning Process (Unsloth + Google Colab)
The model was fine-tuned using the **Unsloth** library, which provides 2x-3x faster training and 70% less VRAM usage compared to standard methods.
- **Base Model**: `unsloth/Phi-3.5-mini-instruct` (3.8B parameters).
- **Technique**: Low-Rank Adaptation (LoRA) using PEFT.
- **Hardware**: Trained on a Tesla T4/A100 in Google Colab.
- **Optimizations**:
    -   4-bit quantization during training.
    -   Gradient checkpointing for VRAM efficiency.
    -   Alpaca-style prompt templates.

### 3. Model Deployment (GGUF + llama.cpp)
After fine-tuning, the adapters were merged into the 16-bit base model and converted to the **GGUF** format for local deployment.
- **Config Variable**: `GGUF_MODEL_PATH` in `src/config.py`.
- **Default Path**: `models/phi3-extractor-unsloth.Q4_K_M.gguf`
- **Quantization**: `Q4_K_M` (4-bit quantization) to ensure high-speed inference on consumer-grade GPUs (like GTX 1650) without sacrificing extraction accuracy.

## The `LinguisticExtractor` Class

Defined in `src/models/extractor.py`, this class wraps the native `llama.cpp` engine and initializes using the path defined in the system configuration.

### `extract_claim(raw_post: str) -> str`
This function transforms raw social media noise into verifyable facts:
1.  **Prompt Construction**: Wraps the raw post in an Alpaca-style instruction block.
2.  **Deterministic Inference**: Sets `temperature=0.0` and `top_p=1.0` for strict, non-random extraction.
3.  **Halt Tokens**: Uses `stop=["<|end|>", "### Instruction:"]` to prevent the model from hallucinating additional context beyond the claim.
4.  **Cleaning**: Strips the "### Response:" header and returns the clean claim.

### Advantages of Generalization
Unlike the previous traffic-only iteration, this model now understands a wide range of domains including crime, economics, disasters, and political events, allowing the system to act as a general-purpose fact-checker for the Sri Lankan digital landscape.
