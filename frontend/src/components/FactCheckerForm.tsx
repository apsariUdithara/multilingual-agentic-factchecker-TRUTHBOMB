import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

interface FactCheckerFormProps {
  onVerify: (text: string) => void;
  isLoading: boolean;
}

export const FactCheckerForm: React.FC<FactCheckerFormProps> = ({ onVerify, isLoading }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onVerify(input);
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Social Media Fact-Checker</CardTitle>
        <CardDescription>
          Paste a Singlish, Sinhala or English social media claim and verify it against retrieved context.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Textarea
            placeholder="e.g., Colombo wala ada loku protest ekak thiyenawa kiyala news ekak yanawa"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="min-h-[150px]"
            disabled={isLoading}
          />
          <Button type="submit" className="w-full" disabled={isLoading || !input.trim()}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Verifying...
              </>
            ) : (
              'Verify'
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};
