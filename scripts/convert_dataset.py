import json
import os

# Configuration
INPUT_FILE = 'data/dataset.json'
OUTPUT_FILE = 'data/extraction_train.jsonl'
INSTRUCTION = (
    "You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. "
    "Translate it to professional English, remove all emotion and slang, and output only the declarative claim."
)

def convert_dataset():
    print(f"Reading {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return

    jsonl_records = []
    
    for entry in data:
        # Some entries might have variations, some might be flat. 
        # Based on your provided dataset.json structure:
        variations = entry.get('variations', [])
        
        for var in variations:
            raw_post = var.get('raw_post', '')
            normalized_claim = var.get('normalized_claim', '')
            
            if raw_post and normalized_claim:
                record = {
                    "instruction": INSTRUCTION,
                    "input": raw_post,
                    "output": normalized_claim
                }
                jsonl_records.append(record)

    print(f"Converted {len(jsonl_records)} variations into JSONL format.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Successfully wrote new training data to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_dataset()
