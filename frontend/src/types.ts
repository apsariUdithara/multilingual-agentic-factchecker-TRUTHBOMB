export interface VerificationRequest {
  text: string;
  metadata_hint?: string;
}

export interface ContextSource {
  content: string;
  distance_score: number;
  source: string;
  timestamp: string;
}

export interface VerificationResponse {
  raw_input: string;
  extracted_claim_en: string;
  verdict: 'TRUE' | 'FALSE' | 'UNVERIFIED';
  confidence_score: number;
  retrieved_context: ContextSource[];
  reasoning: string;
  processed_at: string;
}
