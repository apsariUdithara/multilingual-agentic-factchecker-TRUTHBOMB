import React from 'react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ContextSource } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ContextSourcesProps {
  sources: ContextSource[];
}

export const ContextSources: React.FC<ContextSourcesProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return (
      <Card className="w-full max-w-2xl mx-auto mt-8">
        <CardContent className="pt-6">
          <p className="text-center text-muted-foreground">No context documents were returned.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-2xl mx-auto mt-8 mb-12">
      <CardHeader>
        <CardTitle className="text-lg">Retrieved Context Sources</CardTitle>
      </CardHeader>
      <CardContent>
        <Accordion type="single" collapsible className="w-full">
          {sources.map((source, index) => (
            <AccordionItem key={index} value={`item-${index}`}>
              <AccordionTrigger className="text-left">
                <div className="flex flex-col">
                  <span className="font-semibold">{source.source}</span>
                  <span className="text-xs text-muted-foreground">{source.timestamp}</span>
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-4 pt-2">
                  <div className="flex items-center justify-between text-xs text-muted-foreground border-b pb-2">
                    <span>Distance Score: {source.distance_score.toFixed(4)}</span>
                  </div>
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">
                    {source.content}
                  </p>
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardContent>
    </Card>
  );
};
