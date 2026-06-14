import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { VerificationResponse } from '@/types';
import { cn } from '@/lib/utils';

interface ResultCardProps {
  data: VerificationResponse;
}

export const ResultCard: React.FC<ResultCardProps> = ({ data }) => {
  const getVerdictColor = (verdict: string) => {
    switch (verdict.toUpperCase()) {
      case 'TRUE':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'FALSE':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const confidencePct = (data.confidence_score * 100).toFixed(1);

  return (
    <Card className="w-full max-w-2xl mx-auto mt-8 overflow-hidden">
      <CardHeader className="border-b bg-muted/50">
        <div className="flex items-center justify-between">
          <CardTitle>Verification Result</CardTitle>
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
                data.verdict === 'TRUE' ? "bg-green-500" : data.verdict === 'FALSE' ? "bg-red-500" : "bg-gray-500"
              )}
              style={{ width: `${confidencePct}%` }}
            />
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-muted-foreground mb-1">Reasoning</h4>
          <p className="text-base leading-relaxed">{data.reasoning}</p>
        </div>
      </CardContent>
    </Card>
  );
};
