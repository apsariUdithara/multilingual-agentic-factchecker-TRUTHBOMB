import json
import time
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

# 1. Setup your API Key (Get a free one at aistudio.google.com)
genai.configure(api_key="AIzaSyC12neSHoodZqkOvwJgIj-LYZKDJKaXWns")
model = genai.GenerativeModel('gemini-2.5-flash-lite')

PDF_PATH = "../data/Motor-Traffic-Consolidated-2024.pdf"
OUTPUT_PATH = "../data/traffic_law_dataset.jsonl"

# 2. The prompt that forces the AI to generate our specific format
generation_prompt = """
You are a legal data generation expert. I will give you a chunk of the Sri Lankan Motor Traffic Act. 
Generate 3 distinct fact-checking scenarios based strictly on this text.

For each scenario, create a JSON object with these exact keys:
"instruction": "You are an expert multilingual fact-checker. Analyze the claim, think step-by-step, and output a JSON-like format with Verdict, Uncertainty, and Rationale."
"input": "[A realistic claim a citizen might make about this law, either true or false]"
"output": "<think>\n[Step-by-step logical reasoning based on the text]\n</think>\nVerdict: [True/False]\nUncertainty: [0-5%]\nRationale: [Clear final explanation]"

Output ONLY a valid JSON array of these 3 objects. Do not use markdown blocks like ```json. Just raw JSON.

Legal Text:
{text}
"""

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Loaded {len(pages)} pages. Generating dataset...")

with open(OUTPUT_PATH, "w") as f:
    # Loop through a subset of pages (e.g., pages 5 to 50 to skip the index)
    for i, page in enumerate(pages[5:50]):
        try:
            print(f"Processing page {i+5}...")
            prompt = generation_prompt.format(text=page.page_content)
            
            response = model.generate_content(prompt)
            
            # Clean and parse the response into Python dictionaries
            raw_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            scenarios = json.loads(raw_json)
            
            # Write each scenario as a single line in the JSONL file
            for scenario in scenarios:
                f.write(json.dumps(scenario) + "\n")
                
            time.sleep(4) # Pause to respect free-tier API rate limits
            
        except Exception as e:
            print(f"Error on page {i+5}: {e}")

print(f"✓ Dataset successfully generated at {OUTPUT_PATH}")