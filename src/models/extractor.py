# # # # # # # # # import torch
# # # # # # # # # # Monkeypatch for torch.utils._pytree.register_constant which is missing in torch 2.6.0 
# # # # # # # # # # but expected by torchao/transformers 5.5.0
# # # # # # # # # import torch.utils._pytree
# # # # # # # # # if not hasattr(torch.utils._pytree, "register_constant"):
# # # # # # # # #     torch.utils._pytree.register_constant = lambda x: x

# # # # # # # # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # # # # # # # import os

# # # # # # # # # class LinguisticExtractor:
# # # # # # # # #     """
# # # # # # # # #     Wrapper for the fine-tuned Phi-3.5-mini-instruct model to extract 
# # # # # # # # #     verifiable claims from code-mixed Sri Lankan social media posts.
# # # # # # # # #     """
# # # # # # # # #     def __init__(self, base_model_name="unsloth/Phi-3.5-mini-instruct", adapter_path="models/phi3-extractor-lora/"):
# # # # # # # # #         self.base_model_name = base_model_name
# # # # # # # # #         self.adapter_path = adapter_path
        
# # # # # # # # #         try:
# # # # # # # # #             # Load tokenizer from adapter path as it contains fine-tuned configs
# # # # # # # # #             self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            
# # # # # # # # #             # Load base model with 4-bit quantization
# # # # # # # # #             self.model = AutoModelForCausalLM.from_pretrained(
# # # # # # # # #                 self.base_model_name,
# # # # # # # # #                 load_in_4bit=True,
# # # # # # # # #                 device_map="auto",
# # # # # # # # #                 trust_remote_code=True
# # # # # # # # #             )
            
# # # # # # # # #             # Load the local adapters using PEFT
# # # # # # # # #             from peft import PeftModel
# # # # # # # # #             self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
# # # # # # # # #             self.model.eval()
            
# # # # # # # # #         except Exception as e:
# # # # # # # # #             raise RuntimeError(f"Initialization of LinguisticExtractor failed: {str(e)}")

# # # # # # # # #     def extract_claim(self, raw_post: str) -> str:
# # # # # # # # #         """
# # # # # # # # #         Formats the input post into the Alpaca prompt structure and runs inference.
# # # # # # # # #         """
# # # # # # # # #         prompt = (
# # # # # # # # #             "Below is an instruction that describes a task, paired with an input that provides further context. "
# # # # # # # # #             "Write a response that appropriately completes the request.\n\n"
# # # # # # # # #             "### Instruction:\n"
# # # # # # # # #             "You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. "
# # # # # # # # #             "Translate it to professional English, remove all emotion and slang, and output only the declarative claim.\n\n"
# # # # # # # # #             f"### Input:\n{raw_post}\n\n"
# # # # # # # # #             "### Response:"
# # # # # # # # #         )
        
# # # # # # # # #         try:
# # # # # # # # #             # Tokenize input
# # # # # # # # #             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
# # # # # # # # #             # Generate response
# # # # # # # # #             with torch.no_grad():
# # # # # # # # #                 generated_ids = self.model.generate(
# # # # # # # # #                     **inputs,
# # # # # # # # #                     max_new_tokens=150,  # Sufficient for a single claim
# # # # # # # # #                     pad_token_id=self.tokenizer.eos_token_id,
# # # # # # # # #                     do_sample=False     # Deterministic for extraction tasks
# # # # # # # # #                 )
            
# # # # # # # # #             # Decode the output
# # # # # # # # #             full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
# # # # # # # # #             # Slicing out everything before and including "### Response:"
# # # # # # # # #             if "### Response:" in full_response:
# # # # # # # # #                 claim = full_response.split("### Response:")[-1].strip()
# # # # # # # # #             else:
# # # # # # # # #                 # Fallback if the model didn't repeat the header
# # # # # # # # #                 claim = full_response.strip()
                
# # # # # # # # #             return claim
            
# # # # # # # # #         except torch.cuda.OutOfMemoryError:
# # # # # # # # #             return "Error: Local VRAM limit exceeded during inference."
# # # # # # # # #         except Exception as e:
# # # # # # # # #             return f"Error during extraction process: {str(e)}"

# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     # Quick sanity check if run directly
# # # # # # # # #     print("LinguisticExtractor module loaded.")





# # # # # # # # import torch
# # # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # # # # # # # from peft import PeftModel

# # # # # # # # class LinguisticExtractor:
# # # # # # # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # # # # # # #         # 1. Strict 4-bit config
# # # # # # # #         quantization_config = BitsAndBytesConfig(
# # # # # # # #             load_in_4bit=True,
# # # # # # # #             bnb_4bit_compute_dtype=torch.float16,
# # # # # # # #             bnb_4bit_quant_type="nf4",
# # # # # # # #             bnb_4bit_use_double_quant=True,
# # # # # # # #         )

# # # # # # # #         # 2. Load the base model
# # # # # # # #         self.base_model = AutoModelForCausalLM.from_pretrained(
# # # # # # # #             "unsloth/Phi-3.5-mini-instruct",
# # # # # # # #             quantization_config=quantization_config,
# # # # # # # #             device_map="auto",
# # # # # # # #             trust_remote_code=True,
# # # # # # # #             low_cpu_mem_usage=True
# # # # # # # #         )

# # # # # # # #         # 3. Load tokenizer
# # # # # # # #         self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")

# # # # # # # #         # 4. Wrap with PeftModel to inject custom adapters
# # # # # # # #         self.model = PeftModel.from_pretrained(self.base_model, model_path)
# # # # # # # #         self.model.eval()

# # # # # # # #     def extract_claim(self, raw_post: str) -> str:
# # # # # # # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # # # # # # ### Instruction:
# # # # # # # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # # # # # # ### Input:
# # # # # # # # {input}

# # # # # # # # ### Response:
# # # # # # # # """
# # # # # # # #         prompt = prompt_template.format(input=raw_post)
# # # # # # # #         inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

# # # # # # # #         with torch.no_grad():
# # # # # # # #             outputs = self.model.generate(
# # # # # # # #                 **inputs, 
# # # # # # # #                 max_new_tokens=128, 
# # # # # # # #                 use_cache=True,
# # # # # # # #                 pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
# # # # # # # #             )
        
# # # # # # # #         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
# # # # # # # #         try:
# # # # # # # #             claim = response.split("### Response:\n")[1].strip()
# # # # # # # #             return claim
# # # # # # # #         except IndexError:
# # # # # # # #             return "Error: Could not parse model response."

# # # # # # # import torch
# # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # # # # # # from peft import PeftModel

# # # # # # # class LinguisticExtractor:
# # # # # # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # # # # # #         # 1. Tweak 4-bit config for GTX 1650 hardware stability
# # # # # # #         quantization_config = BitsAndBytesConfig(
# # # # # # #             load_in_4bit=True,
# # # # # # #             bnb_4bit_compute_dtype=torch.float16,
# # # # # # #             bnb_4bit_quant_type="fp4",        # Changed from nf4 for GTX 1650 stability
# # # # # # #             bnb_4bit_use_double_quant=False,  # Simplified for older architectures
# # # # # # #         )

# # # # # # #         # 2. Load the base model
# # # # # # #         self.base_model = AutoModelForCausalLM.from_pretrained(
# # # # # # #             "unsloth/Phi-3.5-mini-instruct",
# # # # # # #             quantization_config=quantization_config,
# # # # # # #             device_map="auto",
# # # # # # #             trust_remote_code=True,
# # # # # # #             low_cpu_mem_usage=True
# # # # # # #         )

# # # # # # #         # 3. Load tokenizer from the ADAPTER path (contains Unsloth's special tokens)
# # # # # # #         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

# # # # # # #         # 4. Wrap with PeftModel to inject custom adapters
# # # # # # #         self.model = PeftModel.from_pretrained(self.base_model, model_path)
# # # # # # #         self.model.eval()

# # # # # # #     def extract_claim(self, raw_post: str) -> str:
# # # # # # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # # # # # ### Instruction:
# # # # # # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # # # # # ### Input:
# # # # # # # {input}

# # # # # # # ### Response:
# # # # # # # """
# # # # # # #         prompt = prompt_template.format(input=raw_post)
# # # # # # #         inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

# # # # # # #         with torch.no_grad():
# # # # # # #             outputs = self.model.generate(
# # # # # # #                 **inputs, 
# # # # # # #                 max_new_tokens=128, 
# # # # # # #                 use_cache=True,
# # # # # # #                 do_sample=False,          # FORCE greedy decoding (strict, no randomness)
# # # # # # #                 temperature=None,
# # # # # # #                 top_p=None,
# # # # # # #                 repetition_penalty=1.1,   # Prevents infinite character loops
# # # # # # #                 pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
# # # # # # #             )
        
# # # # # # #         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
# # # # # # #         try:
# # # # # # #             claim = response.split("### Response:\n")[1].strip()
# # # # # # #             return claim
# # # # # # #         except IndexError:
# # # # # # #             return "Error: Could not parse model response."

# # # # # # import torch
# # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # # # # # from peft import PeftModel

# # # # # # class LinguisticExtractor:
# # # # # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # # # # #         # 1. Hardware-Specific Fix: Force compute to float32 for GTX 1650 stability
# # # # # #         quantization_config = BitsAndBytesConfig(
# # # # # #             load_in_4bit=True,
# # # # # #             bnb_4bit_compute_dtype=torch.float32,  # <-- CHANGED: Fixes the gibberish math bug
# # # # # #             bnb_4bit_quant_type="fp4",
# # # # # #             bnb_4bit_use_double_quant=False,
# # # # # #         )

# # # # # #         # 2. Load the base model
# # # # # #         self.base_model = AutoModelForCausalLM.from_pretrained(
# # # # # #             "unsloth/Phi-3.5-mini-instruct",
# # # # # #             quantization_config=quantization_config,
# # # # # #             device_map="auto",
# # # # # #             trust_remote_code=True,
# # # # # #             low_cpu_mem_usage=True
# # # # # #         )

# # # # # #         # 3. Load tokenizer
# # # # # #         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

# # # # # #         # 4. Wrap with PeftModel to inject custom adapters
# # # # # #         self.model = PeftModel.from_pretrained(self.base_model, model_path)
# # # # # #         self.model.eval()

# # # # # #     def extract_claim(self, raw_post: str) -> str:
# # # # # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # # # # ### Instruction:
# # # # # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # # # # ### Input:
# # # # # # {input}

# # # # # # ### Response:
# # # # # # """
# # # # # #         prompt = prompt_template.format(input=raw_post)
        
# # # # # #         # Ensure special tokens (like BOS) are added correctly
# # # # # #         inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

# # # # # #         with torch.no_grad():
# # # # # #             outputs = self.model.generate(
# # # # # #                 **inputs, 
# # # # # #                 max_new_tokens=128, 
# # # # # #                 use_cache=True,
# # # # # #                 do_sample=False,
# # # # # #                 pad_token_id=self.tokenizer.eos_token_id
# # # # # #             )
        
# # # # # #         # We only want to decode the *new* tokens, not the prompt again
# # # # # #         input_length = inputs["input_ids"].shape[1]
# # # # # #         generated_tokens = outputs[0][input_length:]
        
# # # # # #         response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
# # # # # #         return response.strip()




# # # # # from unsloth import FastLanguageModel
# # # # # import torch

# # # # # class LinguisticExtractor:
# # # # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # # # #         """
# # # # #         Initializes the model using Unsloth's highly optimized, bug-free inference engine.
# # # # #         """
# # # # #         print("Loading model via Unsloth (Optimized for GTX 1650)...")
        
# # # # #         # Unsloth automatically reads the adapters, pulls the base model, and handles the 4-bit math safely
# # # # #         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
# # # # #             model_name = model_path,
# # # # #             max_seq_length = 2048,
# # # # #             load_in_4bit = True,
# # # # #         )
        
# # # # #         # This is the magic command: It applies custom C++ optimizations 
# # # # #         # to prevent gibberish and doubles the inference speed.
# # # # #         FastLanguageModel.for_inference(self.model)

# # # # #     def extract_claim(self, raw_post: str) -> str:
# # # # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # # # ### Instruction:
# # # # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # # # ### Input:
# # # # # {input}

# # # # # ### Response:
# # # # # """
# # # # #         prompt = prompt_template.format(input=raw_post)
# # # # #         inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

# # # # #         with torch.no_grad():
# # # # #             outputs = self.model.generate(
# # # # #                 **inputs, 
# # # # #                 max_new_tokens=128, 
# # # # #                 use_cache=True,
# # # # #                 do_sample=False, # Strict facts, no random hallucination
# # # # #                 pad_token_id=self.tokenizer.eos_token_id
# # # # #             )
        
# # # # #         # Mathematically slice out only the new tokens the model generated
# # # # #         input_length = inputs["input_ids"].shape[1]
# # # # #         generated_tokens = outputs[0][input_length:]
        
# # # # #         response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
# # # # #         return response.strip()


# # # # import torch
# # # # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # # # from peft import PeftModel

# # # # class LinguisticExtractor:
# # # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # # #         print("Initializing Base Model...")
# # # #         quantization_config = BitsAndBytesConfig(
# # # #             load_in_4bit=True,
# # # #             bnb_4bit_compute_dtype=torch.float16,
# # # #             bnb_4bit_quant_type="nf4",
# # # #             bnb_4bit_use_double_quant=True,
# # # #         )

# # # #         self.base_model = AutoModelForCausalLM.from_pretrained(
# # # #             "unsloth/Phi-3.5-mini-instruct",
# # # #             quantization_config=quantization_config,
# # # #             device_map="auto",
# # # #             trust_remote_code=True,
# # # #         )

# # # #         print("Loading Tokenizer & Adapters...")
# # # #         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
# # # #         self.model = PeftModel.from_pretrained(self.base_model, model_path)
# # # #         self.model.eval()

# # # #     def extract_claim(self, raw_post: str) -> str:
# # # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # # ### Instruction:
# # # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # # ### Input:
# # # # {input}

# # # # ### Response:
# # # # """
# # # #         prompt = prompt_template.format(input=raw_post)
# # # #         inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

# # # #         with torch.no_grad():
# # # #             outputs = self.model.generate(
# # # #                 **inputs, 
# # # #                 max_new_tokens=128, 
# # # #                 use_cache=False,  # CRITICAL: Fixes the GTX 1650 gibberish corruption
# # # #                 do_sample=False,  # Forces factual, non-random outputs
# # # #                 pad_token_id=self.tokenizer.eos_token_id
# # # #             )
        
# # # #         input_length = inputs["input_ids"].shape[1]
# # # #         generated_tokens = outputs[0][input_length:]
# # # #         response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
# # # #         return response.strip()



# # # import torch
# # # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # # from peft import PeftModel

# # # class LinguisticExtractor:
# # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # #         print("Initializing Base Model...")
# # #         quantization_config = BitsAndBytesConfig(
# # #             load_in_4bit=True,
# # #             bnb_4bit_compute_dtype=torch.float16,
# # #             bnb_4bit_quant_type="nf4",
# # #             bnb_4bit_use_double_quant=True,
# # #         )

# # #         # CHANGED THIS LINE: Pulling the pre-compressed 4-bit model directly!
# # #         self.base_model = AutoModelForCausalLM.from_pretrained(
# # #             "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
# # #             quantization_config=quantization_config,
# # #             device_map="auto",
# # #             trust_remote_code=True,
# # #         )

# # #         print("Loading Tokenizer & Adapters...")
# # #         # NOTE: Also updated tokenizer to match the bnb-4bit path
# # #         self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct-bnb-4bit")
# # #         self.model = PeftModel.from_pretrained(self.base_model, model_path)
# # #         self.model.eval()

# # #     def extract_claim(self, raw_post: str) -> str:
# # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # ### Instruction:
# # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # ### Input:
# # # {input}

# # # ### Response:
# # # """
# # #         prompt = prompt_template.format(input=raw_post)
# # #         inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

# # #         with torch.no_grad():
# # #             outputs = self.model.generate(
# # #                 **inputs, 
# # #                 max_new_tokens=128, 
# # #                 use_cache=False,  
# # #                 do_sample=False,  
# # #                 pad_token_id=self.tokenizer.eos_token_id
# # #             )
        
# # #         input_length = inputs["input_ids"].shape[1]
# # #         generated_tokens = outputs[0][input_length:]
# # #         response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
# # #         return response.strip()


# # import torch
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from peft import PeftModel

# # class LinguisticExtractor:
# #     def __init__(self, model_path="models/phi3-extractor-lora"):
# #         print("Initializing Base Model (Strict GPU Pinning)...")
        
# #         # 1. Load pre-compressed model. 
# #         # device_map={"": 0} FORCES it onto the GPU. No CPU spilling allowed.
# #         # torch_dtype=torch.float16 forces math compatible with GTX 1650.
# #         self.base_model = AutoModelForCausalLM.from_pretrained(
# #             "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
# #             device_map={"": 0},
# #             torch_dtype=torch.float16,
# #             trust_remote_code=True,
# #         )

# #         print("Loading Tokenizer & Adapters...")
# #         self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct-bnb-4bit")
# #         self.model = PeftModel.from_pretrained(self.base_model, model_path)
        
# #         # Ensure the adapters also strictly use float16
# #         self.model.to(torch.float16)
# #         self.model.eval()

# #     def extract_claim(self, raw_post: str) -> str:
# #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # ### Instruction:
# # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # ### Input:
# # {input}

# # ### Response:
# # """
# #         prompt = prompt_template.format(input=raw_post)
# #         inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda:0")

# #         with torch.no_grad():
# #             outputs = self.model.generate(
# #                 **inputs, 
# #                 max_new_tokens=128, 
# #                 use_cache=True,   # Turned back on for speed
# #                 do_sample=False,  
# #                 pad_token_id=self.tokenizer.eos_token_id
# #             )
        
# #         input_length = inputs["input_ids"].shape[1]
# #         generated_tokens = outputs[0][input_length:]
# #         response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
# #         return response.strip()


# import os
# from llama_cpp import Llama

# class LinguisticExtractor:
#     def __init__(self, model_path="models/phi3-extractor-unsloth.Q4_K_M.gguf"):
#         """
#         Initializes the model using the high-performance native C++ llama.cpp engine.
#         """
#         print("Initializing LinguisticExtractor via llama.cpp (GPU Accelerated)...")
        
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(
#                 f"GGUF model file not found at '{model_path}'. "
#                 f"Please ensure the downloaded file from Colab is named correctly and placed in the models/ folder."
#             )
            
#         # Initialize the native engine
#         # n_gpu_layers=-1 completely offloads all network layers to your GPU VRAM
#         self.llm = Llama(
#             model_path=model_path,
#             n_gpu_layers=-1, 
#             n_ctx=2048,
#             verbose=False
#         )

#     def extract_claim(self, raw_post: str) -> str:
#         prompt = (
#             "Below is an instruction that describes a task, paired with an input that provides further context. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n"
#             "You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. "
#             "Translate it to professional English, remove all emotion and slang, and output only the declarative claim.\n\n"
#             f"### Input:\n{raw_post}\n\n"
#             "### Response:"
#         )

#         # Execute high-speed greedy inference directly inside your GPU
#         response = self.llm(
#             prompt,
#             max_tokens=128,
#             temperature=0.0,  # Strict factual deduction
#             top_p=1.0,
#             echo=False
#         )
        
#         # Parse the output out of the returned OpenAI-format data dictionary
#         extracted_text = response["choices"][0]["text"].strip()
#         return extracted_text


import os
from llama_cpp import Llama

class LinguisticExtractor:
    def __init__(self, model_path="models/phi3-extractor-unsloth.Q4_K_M.gguf"):
        print("Initializing LinguisticExtractor via llama.cpp (GPU Accelerated)...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at '{model_path}'.")
            
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=2048,
            verbose=False
        )

    def extract_claim(self, raw_post: str) -> str:
        # 1. FIXED PROMPT SPACING (Added \n after Response:)
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. "
            "Translate it to professional English, remove all emotion and slang, and output only the declarative claim.\n\n"
            f"### Input:\n{raw_post}\n\n"
            "### Response:\n"
        )

        # 2. INJECTED STOP TOKENS
        response = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.0, 
            top_p=1.0,
            echo=False,
            stop=["<|end|>", "<|endoftext|>", "### Instruction:"] # Forces the model to stop hallucinating extra sentences
        )
        
        extracted_text = response["choices"][0]["text"].strip()
        return extracted_text