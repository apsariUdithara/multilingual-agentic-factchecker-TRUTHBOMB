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
  verdict: 'TRUE' | 'FALSE' | 'UNVERIFIED' | 'MISLEADING';
  confidence_score: number;
  retrieved_context: ContextSource[];
  reasoning: string;
  processed_at: string;
  routing_destination?: string;
  summaries?: {
    english: string;
    sinhala: string;
    tamil: string;
  };
  debater_a_claim?: string;
  debater_b_claim?: string;
  consensus_status?: string;
}
