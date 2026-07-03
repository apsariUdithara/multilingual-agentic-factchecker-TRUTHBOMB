import json
import logging
from sklearn.metrics import f1_score, classification_report
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/api/v1/verify"
DATASET_PATH = "data/dataset.json"

def main():
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    y_true = []
    y_pred = []

    total_claims = sum(len(item["variations"]) for item in dataset)
    processed = 0

    for item in dataset:
        # Prefer an explicit label field; fall back to a suffix convention on the
        # event title. (The dataset key is ``core_event``, not ``core_claim``.)
        label = str(item.get("label", "")).upper()
        core_event = item.get("core_event", "")
        if label in {"TRUE", "FALSE", "UNVERIFIED"}:
            true_verdict = label
        elif core_event.endswith("- FALSE"):
            true_verdict = "FALSE"
        elif core_event.endswith("- UNVERIFIED"):
            true_verdict = "UNVERIFIED"
        else:
            true_verdict = "TRUE"

        for variation in item["variations"]:
            raw_post = variation["raw_post"]
            
            try:
                response = requests.post(API_URL, json={"text": raw_post}, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    pred_verdict = result.get("verdict", "UNVERIFIED")
                else:
                    logger.warning(f"API Error {response.status_code} for text: {raw_post[:30]}")
                    pred_verdict = "UNVERIFIED"
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                pred_verdict = "UNVERIFIED"

            y_true.append(true_verdict)
            y_pred.append(pred_verdict)
            
            processed += 1
            logger.info(f"Processed {processed}/{total_claims}: True={true_verdict}, Pred={pred_verdict}")

    labels = ["TRUE", "FALSE", "UNVERIFIED"]
    logger.info("\n" + classification_report(y_true, y_pred, labels=labels, zero_division=0))
    f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    logger.info(f"Weighted F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
