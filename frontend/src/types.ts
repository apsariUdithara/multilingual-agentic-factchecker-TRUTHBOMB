export interface VerificationRequest {
  text: string;
  metadata_hint?: string;
}

export interface ContextSource {
  content: string;
  source: string;
  tier: string;
  timestamp: string;
  distance_score: number; // dense similarity (higher = closer)
  final_score: number;    // fused RRF x tier x recency score
}

export interface AuditEntry {
  node: string;
  status: string;
  detail: string;
}

export type Verdict = 'TRUE' | 'FALSE' | 'MISLEADING' | 'UNVERIFIED';

export interface VerificationResponse {
  raw_input: string;
  extracted_claim_en: string;
  verdict: 'TRUE' | 'FALSE' | 'UNVERIFIED' | 'MISLEADING';
  confidence_score: number;
  reasoning: string;     // English
  reasoning_si: string;  // Sinhala
  reasoning_ta: string;  // Tamil
  retrieved_context: ContextSource[];
  audit_trail: AuditEntry[];
  processed_at: string;
}
