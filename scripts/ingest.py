import json
import requests
import concurrent.futures
import time

# Configuration
ENDPOINT_URL = "http://localhost:8000/extract"
INPUT_FILE = "data/test_data.json"
MAX_WORKERS = 4  # Adjust based on your server's capacity

def send_request(post_text):
    """
    Sends a single post to the FastAPI endpoint and returns the result.
    """
    try:
        response = requests.post(
            ENDPOINT_URL,
            json={"post_text": post_text},
            timeout=30  # Local LLM might take some time
        )
        if response.status_code == 200:
            return post_text, response.json().get("claim", "No claim returned")
        else:
            return post_text, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return post_text, f"Request Failed: {str(e)}"

def main():
    # Load test data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode {INPUT_FILE}.")
        return

    print(f"Starting ingestion of {len(posts)} posts...\n")
    print(f"{'RAW POST':<60} | {'EXTRACTED CLAIM'}")
    print("-" * 120)

    # Execute requests concurrently using a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the send_request function to the posts
        future_to_post = {executor.submit(send_request, post): post for post in posts}
        
        for future in concurrent.futures.as_completed(future_to_post):
            raw_post, extracted_claim = future.result()
            
            # Formatting for console display
            # Truncate long raw posts for better side-by-side view
            display_post = (raw_post[:57] + '...') if len(raw_post) > 60 else raw_post
            print(f"{display_post:<60} | {extracted_claim}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nFinished in {time.time() - start_time:.2f} seconds.")
