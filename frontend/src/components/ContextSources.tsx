import React from 'react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ContextSource } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Globe } from "lucide-react";

interface ContextSourcesProps {
  sources: ContextSource[];
}

export const ContextSources: React.FC<ContextSourcesProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return (
      <Card className="w-full max-w-2xl mx-auto mt-8 border-slate-200 shadow-sm rounded-xl">
        <CardContent className="pt-6">
          <p className="text-center text-sm text-slate-500 font-medium">
            No context documents retrieved for verification.
          </p>
        </CardContent>
      </Card>
    );
  }

  // Segment sources into web-search findings vs. local database documents
  const webSources = sources.filter(s => 
    s.source.toLowerCase().includes('web_researcher') || 
    s.source.toLowerCase().includes('web_search') ||
    s.source.toLowerCase().includes('web')
  );
  
  const localSources = sources.filter(s => 
    !s.source.toLowerCase().includes('web_researcher') && 
    !s.source.toLowerCase().includes('web_search') &&
    !s.source.toLowerCase().includes('web')
  );

  return (
    <div className="w-full max-w-2xl mx-auto mt-8 mb-16 space-y-6">
      {/* 1. Local Database Context (RAG) */}
      {localSources.length > 0 && (
        <Card className="border-slate-200/80 shadow-sm rounded-xl bg-white">
          <CardHeader className="py-4 border-b border-slate-100">
            <CardTitle className="text-sm font-semibold tracking-wide uppercase text-slate-500 flex items-center gap-2">
              <Database className="h-4 w-4 text-slate-400" />
              Retrieved Local RAG Context
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Accordion type="single" collapsible className="w-full">
              {localSources.map((source, index) => (
                <AccordionItem key={`local-${index}`} value={`local-item-${index}`} className="border-slate-100 last:border-0">
                  <AccordionTrigger className="text-left py-3 hover:no-underline">
                    <div className="flex flex-col gap-0.5">
                      <span className="font-semibold text-slate-800 text-sm">{source.source}</span>
                      <span className="text-xs text-slate-400 font-normal">{source.timestamp}</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4">
                    <div className="space-y-3 pt-1 text-slate-600">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {source.content}
                      </p>
                      {source.distance_score > 0 && (
                        <div className="text-[11px] font-mono text-slate-400 border-t border-slate-50 pt-2 flex items-center justify-between">
                          <span>Retrieve score: {source.distance_score.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      )}

      {/* 2. Web Search Context */}
      {webSources.length > 0 && (
        <Card className="border-slate-200/80 shadow-sm rounded-xl bg-white">
          <CardHeader className="py-4 border-b border-slate-100">
            <CardTitle className="text-sm font-semibold tracking-wide uppercase text-slate-500 flex items-center gap-2">
              <Globe className="h-4 w-4 text-slate-400" />
              External Web Search Findings
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Accordion type="single" collapsible className="w-full">
              {webSources.map((source, index) => (
                <AccordionItem key={`web-${index}`} value={`web-item-${index}`} className="border-slate-100 last:border-0">
                  <AccordionTrigger className="text-left py-3 hover:no-underline">
                    <div className="flex flex-col gap-0.5">
                      <span className="font-semibold text-slate-800 text-sm">{source.source}</span>
                      <span className="text-xs text-slate-400 font-normal">{source.timestamp}</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4">
                    <div className="pt-1 text-slate-600">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {source.content}
                      </p>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
