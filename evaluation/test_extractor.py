# # # # # # import sys
# # # # # # import os
# # # # # # import time

# # # # # # # Ensure the root directory is in the path so we can import src
# # # # # # sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# # # # # # try:
# # # # # #     from src.models.extractor import LinguisticExtractor
# # # # # #     import torch
# # # # # # except ImportError as e:
# # # # # #     print(f"Import Error: {e}")
# # # # # #     sys.exit(1)

# # # # # # def run_validation():
# # # # # #     print("--- Local Inference Validation ---")
    
# # # # # #     # 1. Initialization
# # # # # #     try:
# # # # # #         print("[1/3] Initializing LinguisticExtractor (Native Transformers + BitsAndBytes)...")
# # # # # #         start_init = time.time()
# # # # # #         # Note: We use the default paths defined in the class:
# # # # # #         # base_model: unsloth/Phi-3.5-mini-instruct
# # # # # #         # adapter_path: models/phi3-extractor-lora/
# # # # # #         extractor = LinguisticExtractor()
# # # # # #         print(f"Model and Adapters loaded in {time.time() - start_init:.2f}s.")
# # # # # #     except Exception as e:
# # # # # #         print(f"CRITICAL: Failed to initialize extractor: {e}")
# # # # # #         return

# # # # # #     # 2. Test Input (Sinhala-English Code-Mixed)
# # # # # #     # Translation: "Is the highway closed today? I heard there's a big accident near Kadawatha."
# # # # # #     sample_post = "Highway eka ada wahalada? Kadawatha gawa loku accident ekak wela kiyala aranchiyak awa. Kauda danne?"
# # # # # #     print(f"\n[2/3] Input Post:\n\"{sample_post}\"")

# # # # # #     # 3. Inference
# # # # # #     print("\n[3/3] Running extraction...")
# # # # # #     try:
# # # # # #         start_inf = time.time()
# # # # # #         claim = extractor.extract_claim(sample_post)
# # # # # #         duration = time.time() - start_inf
        
# # # # # #         print("\n--- Result ---")
# # # # # #         print(f"Extracted Claim: {claim}")
# # # # # #         print(f"Inference Time: {duration:.2f}s")
# # # # # #         print("--------------")
        
# # # # # #         if claim and "Error" not in claim:
# # # # # #             print("\nSUCCESS: End-to-end local inference verified.")
# # # # # #         else:
# # # # # #             print("\nWARNING: Extraction returned an unexpected result or error.")
            
# # # # # #     except Exception as e:
# # # # # #         print(f"ERROR: Inference failed: {e}")

# # # # # # if __name__ == "__main__":
# # # # # #     run_validation()



# # # # # import torch
# # # # # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# # # # # from peft import PeftModel

# # # # # class LinguisticExtractor:
# # # # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # # # #         """
# # # # #         Initializes the fine-tuned model for local inference using modern Transformers API.
# # # # #         """
# # # # #         # 1. Define the 4-bit quantization configuration explicitly
# # # # #         bnb_config = BitsAndBytesConfig(
# # # # #             load_in_4bit=True,
# # # # #             bnb_4bit_use_double_quant=True,
# # # # #             bnb_4bit_quant_type="nf4",
# # # # #             bnb_4bit_compute_dtype=torch.float16
# # # # #         )
        
# # # # #         # 2. Load the base model with the quantization config
# # # # #         base_model = AutoModelForCausalLM.from_pretrained(
# # # # #             "unsloth/Phi-3.5-mini-instruct",
# # # # #             quantization_config=bnb_config,
# # # # #             device_map="auto",
# # # # #             trust_remote_code=True
# # # # #         )
        
# # # # #         # 3. Load the tokenizer
# # # # #         self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")
        
# # # # #         # 4. Merge our custom LoRA adapters onto the base model
# # # # #         self.model = PeftModel.from_pretrained(base_model, model_path)
        
# # # # #         # Set to evaluation mode for deterministic outputs
# # # # #         self.model.eval()

# # # # #     def extract_claim(self, raw_post: str) -> str:
# # # # #         prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# # # # # ### Instruction:
# # # # # You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. Translate it to professional English, remove all emotion and slang, and output only the declarative claim.

# # # # # ### Input:
# # # # # {input}

# # # # # ### Response:
# # # # # """
# # # # #         # Format the prompt
# # # # #         prompt = prompt_template.format(input=raw_post)
# # # # #         inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

# # # # #         # Run inference
# # # # #         with torch.no_grad():
# # # # #             outputs = self.model.generate(
# # # # #                 **inputs, 
# # # # #                 max_new_tokens=128, 
# # # # #                 use_cache=True,
# # # # #                 pad_token_id=self.tokenizer.eos_token_id # Use eos_token_id for padding
# # # # #             )
        
# # # # #         # Decode and slice out the generated text
# # # # #         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
# # # # #         try:
# # # # #             claim = response.split("### Response:\n")[1].strip()
# # # # #             return claim
# # # # #         except IndexError:
# # # # #             return "Error: Could not parse model response."






# # # # import sys
# # # # import os
# # # # import time

# # # # # Ensure the root directory is in the path so we can import src
# # # # sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# # # # try:
# # # #     from src.models.extractor import LinguisticExtractor
# # # #     import torch
# # # # except ImportError as e:
# # # #     print(f"Import Error: {e}")
# # # #     sys.exit(1)

# # # # def run_validation():
# # # #     print("--- Local Inference Validation ---")
    
# # # #     # 1. Initialization
# # # #     try:
# # # #         print("[1/3] Initializing LinguisticExtractor (Native Transformers + BitsAndBytes)...")
# # # #         start_init = time.time()
# # # #         # Note: We use the default paths defined in the class
# # # #         extractor = LinguisticExtractor()
# # # #         print(f"Model and Adapters loaded in {time.time() - start_init:.2f}s.")
# # # #     except Exception as e:
# # # #         print(f"CRITICAL: Failed to initialize extractor: {e}")
# # # #         return

# # # #     # 2. Test Input (Sinhala-English Code-Mixed)
# # # #     # Translation: "Is the highway closed today? I heard there's a big accident near Kadawatha."
# # # #     sample_post = "Highway eka ada wahalada? Kadawatha gawa loku accident ekak wela kiyala aranchiyak awa. Kauda danne?"
# # # #     print(f"\n[2/3] Input Post:\n\"{sample_post}\"")

# # # #     # 3. Inference
# # # #     print("\n[3/3] Running extraction...")
# # # #     try:
# # # #         start_inf = time.time()
# # # #         claim = extractor.extract_claim(sample_post)
# # # #         duration = time.time() - start_inf
        
# # # #         print("\n--- Result ---")
# # # #         print(f"Extracted Claim: {claim}")
# # # #         print(f"Inference Time: {duration:.2f}s")
# # # #         print("--------------")
        
# # # #         if claim and "Error" not in claim:
# # # #             print("\nSUCCESS: End-to-end local inference verified.")
# # # #         else:
# # # #             print("\nWARNING: Extraction returned an unexpected result or error.")
            
# # # #     except Exception as e:
# # # #         print(f"ERROR: Inference failed: {e}")

# # # # if __name__ == "__main__":
# # # #     run_validation()



# # # import torch
# # # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # # from peft import PeftModel

# # # class LinguisticExtractor:
# # #     def __init__(self, model_path="models/phi3-extractor-lora"):
# # #         """
# # #         Initializes the fine-tuned model for local inference using strict BitsAndBytes configuration.
# # #         """
# # #         # 1. Define strict 4-bit config to avoid 'unexpected keyword' errors down the line
# # #         quantization_config = BitsAndBytesConfig(
# # #             load_in_4bit=True,
# # #             bnb_4bit_compute_dtype=torch.float16,
# # #             bnb_4bit_quant_type="nf4",
# # #             bnb_4bit_use_double_quant=True,
# # #         )

# # #         # 2. Load the base model explicitly forcing the quantization config
# # #         print("Loading base model...")
# # #         base_model = AutoModelForCausalLM.from_pretrained(
# # #             "unsloth/Phi-3.5-mini-instruct",
# # #             quantization_config=quantization_config,
# # #             device_map="auto",
# # #             trust_remote_code=True,
# # #             low_cpu_mem_usage=True
# # #         )

# # #         # 3. Load tokenizer
# # #         print("Loading tokenizer...")
# # #         self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")

# # #         # 4. Wrap with PeftModel to inject your custom adapters
# # #         print("Injecting custom LoRA adapters...")
# # #         self.model = PeftModel.from_pretrained(base_model, model_path)
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
# # #         inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

# # #         with torch.no_grad():
# # #             outputs = self.model.generate(
# # #                 **inputs, 
# # #                 max_new_tokens=128, 
# # #                 use_cache=True,
# # #                 pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
# # #             )
        
# # #         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
# # #         try:
# # #             claim = response.split("### Response:\n")[1].strip()
# # #             return claim
# # #         except IndexError:
# # #             return "Error: Could not parse model response."





# # import torch
# # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # from peft import PeftModel

# # class LinguisticExtractor:
# #     def __init__(self, model_path="models/phi3-extractor-lora"):
# #         """
# #         Initializes the fine-tuned model for local inference using strict BitsAndBytes configuration.
# #         """
# #         quantization_config = BitsAndBytesConfig(
# #             load_in_4bit=True,
# #             bnb_4bit_compute_dtype=torch.float16,
# #             bnb_4bit_quant_type="nf4",
# #             bnb_4bit_use_double_quant=True,
# #         )

# #         print("Loading base model...")
# #         base_model = AutoModelForCausalLM.from_pretrained(
# #             "unsloth/Phi-3.5-mini-instruct",
# #             quantization_config=quantization_config,
# #             device_map="auto",
# #             trust_remote_code=True,
# #             low_cpu_mem_usage=True
# #         )

# #         print("Loading tokenizer...")
# #         self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")

# #         print("Injecting custom LoRA adapters...")
# #         self.model = PeftModel.from_pretrained(base_model, model_path)
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
# #         inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

# #         with torch.no_grad():
# #             outputs = self.model.generate(
# #                 **inputs, 
# #                 max_new_tokens=128, 
# #                 use_cache=True,
# #                 pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
# #             )
        
# #         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
# #         try:
# #             claim = response.split("### Response:\n")[1].strip()
# #             return claim
# #         except IndexError:
# #             return "Error: Could not parse model response."




# import sys
# import os
# import time

# # Ensure python can find the src folder
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# from src.models.extractor import LinguisticExtractor

# def run_validation():
#     print("--- Local Inference Validation ---")
    
#     print("[1/3] Initializing LinguisticExtractor (Loading model into VRAM)...")
#     start_init = time.time()
    
#     # This creates the object and triggers the download/loading
#     extractor = LinguisticExtractor()
    
#     print(f"Model and Adapters loaded successfully in {time.time() - start_init:.2f}s.")

#     # Translation: "Is the highway closed today? I heard there's a big accident near Kadawatha."
#     sample_post = "Highway eka ada wahalada? Kadawatha gawa loku accident ekak wela kiyala aranchiyak awa. Kauda danne?"
#     print(f"\n[2/3] Input Post:\n\"{sample_post}\"")

#     print("\n[3/3] Running extraction...")
#     start_inf = time.time()
    
#     # This actually calls the model to do the work
#     claim = extractor.extract_claim(sample_post)
    
#     duration = time.time() - start_inf
    
#     print("\n--- Result ---")
#     print(f"Extracted Claim: {claim}")
#     print(f"Inference Time: {duration:.2f}s")
#     print("--------------")

# if __name__ == "__main__":
#     run_validation()




import sys
import os
import time

# ==============================================================================
# AUTOMATED WINDOWS CUDA DLL LINKER
# ==============================================================================
if sys.platform == "win32":
    # Gather every folder Python touches, including global and user namespaces
    search_paths = sys.path + [
        os.path.join(os.environ.get("APPDATA", ""), "Python", "Python312", "site-packages"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Python312", "Lib", "site-packages"),
    ]
    
    registered_directories = set()
    for base_folder in search_paths:
        if base_folder and os.path.exists(base_folder):
            nvidia_entry = os.path.join(base_folder, "nvidia")
            if os.path.exists(nvidia_entry):
                # Walk recursively through the nvidia folder to find the DLL binaries
                for root, folders, files in os.walk(nvidia_entry):
                    if any(file.lower().endswith(".dll") for file in files):
                        if root not in registered_directories:
                            try:
                                os.add_dll_directory(root)
                                registered_directories.add(root)
                            except Exception:
                                pass
# ==============================================================================

# Ensure the root directory is in the path so we can cleanly import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.models.extractor import LinguisticExtractor
except Exception as e:
    print(f"CRITICAL: Failed to import LinguisticExtractor module: {e}")
    sys.exit(1)

def run_validation():
    print("--- Local Inference Validation ---")
    
    print("[1/3] Initializing LinguisticExtractor (Loading GGUF into GPU)...")
    start_init = time.time()
    
    try:
        extractor = LinguisticExtractor()
        print(f"Model loaded natively via llama.cpp in {time.time() - start_init:.2f}s.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # Test Input (Sinhala-English Code-Mixed Singlish post)
    sample_post = "Highway eka ada wahalada? Kadawatha gawa loku accident ekak wela kiyala aranchiyak awa. Kauda danne?"
    print(f"\n[2/3] Input Post:\n\"{sample_post}\"")

    print("\n[3/3] Running extraction...")
    start_inf = time.time()
    
    try:
        claim = extractor.extract_claim(sample_post)
        duration = time.time() - start_inf
        
        print("\n--- Result ---")
        print(f"Extracted Claim: {claim}")
        print(f"Inference Time: {duration:.2f}s")
        print("--------------")
        
        if claim and "Error" not in claim and len(claim.strip()) > 5:
            print("\nSUCCESS: End-to-end local inference verified.")
        else:
            print("\nWARNING: Extraction returned an empty or unexpected result.")
            
    except Exception as e:
        print(f"ERROR: Inference processing loop failed: {e}")

if __name__ == "__main__":
    run_validation()