import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { VerificationResponse } from '@/types';
import { cn } from '@/lib/utils';

interface ResultCardProps {
  data: VerificationResponse;
}

export const ResultCard: React.FC<ResultCardProps> = ({ data }) => {
  const [activeLang, setActiveLang] = useState<'english' | 'sinhala' | 'tamil'>('english');

  const getVerdictColor = (verdict: string) => {
    switch (verdict.toUpperCase()) {
      case 'TRUE':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'FALSE':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'MISLEADING':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const confidencePct = (data.confidence_score * 100).toFixed(1);

  return (
    <Card className="w-full max-w-2xl mx-auto mt-8 overflow-hidden shadow-md">
      <CardHeader className="border-b bg-muted/50">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Verification Result</CardTitle>
            {data.routing_destination && (
              <span className="text-xs text-muted-foreground block mt-1">
                Agentic Route: <span className="font-semibold text-primary">{data.routing_destination === 'local_researcher' ? 'Local OSINT DB' : 'Trusted Web Search'}</span>
              </span>
            )}
          </div>
          <Badge className={cn("px-3 py-1 text-sm font-bold uppercase", getVerdictColor(data.verdict))}>
            {data.verdict}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        <div>
          <h4 className="text-sm font-semibold text-muted-foreground mb-1">Extracted Claim (English)</h4>
          <p className="text-lg font-medium">{data.extracted_claim_en}</p>
        </div>

        {/* Dual-Debater Consensus Panel */}
        {(data.debater_a_claim || data.debater_b_claim) && (
          <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Multi-LLM Debate & Arbitration
              </span>
              <Badge 
                className={cn(
                  "text-[10px] font-bold px-2 py-0.5 uppercase",
                  data.consensus_status === 'AGREEMENT' 
                    ? "bg-emerald-100 text-emerald-800 border-emerald-200" 
                    : "bg-amber-100 text-amber-800 border-amber-200"
                )}
              >
                {data.consensus_status === 'AGREEMENT' ? '✓ Consensus Reached' : '⚡ Arbitrated Divergence'}
              </Badge>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="bg-white p-3 rounded-lg border border-slate-100 shadow-2xs">
                <span className="font-semibold text-slate-500 block mb-1">Debater A (Fine-Tuned GGUF):</span>
                <p className="text-slate-700 italic">"{data.debater_a_claim || data.extracted_claim_en}"</p>
              </div>
              <div className="bg-white p-3 rounded-lg border border-slate-100 shadow-2xs">
                <span className="font-semibold text-slate-500 block mb-1">Debater B (Cloud Gemini):</span>
                <p className="text-slate-700 italic">"{data.debater_b_claim || data.extracted_claim_en}"</p>
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center gap-8">
          <div>
            <h4 className="text-sm font-semibold text-muted-foreground mb-1">Confidence</h4>
            <div className="flex items-end gap-1">
              <span className="text-3xl font-bold">{confidencePct}%</span>
            </div>
          </div>
          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <div 
              className={cn(
                "h-full transition-all duration-500",
                data.verdict === 'TRUE' ? "bg-green-500" : data.verdict === 'FALSE' ? "bg-red-500" : data.verdict === 'MISLEADING' ? "bg-amber-500" : "bg-gray-500"
              )}
              style={{ width: `${confidencePct}%` }}
            />
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-muted-foreground mb-1">Reasoning</h4>
          <p className="text-base leading-relaxed text-slate-700">{data.reasoning}</p>
        </div>

        {data.summaries && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-semibold text-muted-foreground mb-3">Multilingual Verdict Summaries</h4>
            
            {/* Tab Buttons */}
            <div className="flex space-x-2 mb-3 bg-slate-100 p-1 rounded-lg w-max">
              <button
                type="button"
                onClick={() => setActiveLang('english')}
                className={cn(
                  "px-3 py-1 text-xs font-semibold rounded-md transition-all",
                  activeLang === 'english' ? "bg-white shadow text-primary" : "text-muted-foreground hover:text-slate-800"
                )}
              >
                English
              </button>
              <button
                type="button"
                onClick={() => setActiveLang('sinhala')}
                className={cn(
                  "px-3 py-1 text-xs font-semibold rounded-md transition-all",
                  activeLang === 'sinhala' ? "bg-white shadow text-primary" : "text-muted-foreground hover:text-slate-800"
                )}
              >
                සිංහල
              </button>
              <button
                type="button"
                onClick={() => setActiveLang('tamil')}
                className={cn(
                  "px-3 py-1 text-xs font-semibold rounded-md transition-all",
                  activeLang === 'tamil' ? "bg-white shadow text-primary" : "text-muted-foreground hover:text-slate-800"
                )}
              >
                தமிழ்
              </button>
            </div>

            {/* Tab Contents */}
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-100 italic text-slate-700 min-h-[60px] flex items-center">
              {activeLang === 'english' && <p>{data.summaries.english}</p>}
              {activeLang === 'sinhala' && <p className="font-semibold text-slate-800">{data.summaries.sinhala}</p>}
              {activeLang === 'tamil' && <p>{data.summaries.tamil}</p>}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
