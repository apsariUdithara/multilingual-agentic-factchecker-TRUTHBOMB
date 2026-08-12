import React from 'react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { ContextSource } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Globe, Newspaper, ExternalLink } from "lucide-react";

interface ContextSourcesProps {
  sources: ContextSource[];
}

export const ContextSources: React.FC<ContextSourcesProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return (
      <Card className="w-full max-w-2xl mx-auto mt-8 border-slate-200 shadow-sm rounded-xl">
        <CardContent className="pt-6">
          <p className="text-center text-sm text-slate-500 font-medium">
            No context documents or proofs retrieved for verification.
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
      {/* 1. Local Database Proofs (News & Telegram Feeds) */}
      {localSources.length > 0 && (
        <Card className="border-slate-200/80 shadow-sm rounded-xl bg-white overflow-hidden">
          <CardHeader className="py-4 border-b border-slate-100 bg-slate-50/50">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold tracking-wide uppercase text-slate-700 flex items-center gap-2">
                <Database className="h-4 w-4 text-slate-500" />
                Retrieved Local Database Proofs (OSINT & News Feeds)
              </CardTitle>
              <Badge variant="outline" className="text-[10px] bg-white border-slate-200 text-slate-600 font-mono">
                {localSources.length} Proof {localSources.length === 1 ? 'Record' : 'Records'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="pt-2">
            <Accordion type="single" collapsible defaultValue="local-item-0" className="w-full">
              {localSources.map((source, index) => (
                <AccordionItem key={`local-${index}`} value={`local-item-${index}`} className="border-slate-100 last:border-0">
                  <AccordionTrigger className="text-left py-3 hover:no-underline">
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <Newspaper className="h-3.5 w-3.5 text-slate-400" />
                        <span className="font-semibold text-slate-800 text-sm">
                          News Feed Proof #{index + 1}
                        </span>
                        <Badge className="bg-slate-100 text-slate-700 hover:bg-slate-100 text-[10px] border-0 font-normal">
                          Verified DB Record
                        </Badge>
                      </div>
                      <span className="text-xs text-slate-400 font-normal ml-5">
                        Source: {source.source} | Indexed: {source.timestamp}
                      </span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4 ml-5">
                    <div className="space-y-3 pt-1 text-slate-700 bg-slate-50/60 p-4 rounded-xl border border-slate-200/80">
                      {/* Structured Metadata Bar */}
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px] bg-white p-2.5 rounded-lg border border-slate-100 font-mono text-slate-500 mb-2">
                        <div><span className="font-semibold text-slate-700">Source:</span> {source.source}</div>
                        <div><span className="font-semibold text-slate-700">Date:</span> {source.timestamp}</div>
                        <div><span className="font-semibold text-slate-700">Type:</span> Verified DB Record</div>
                      </div>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap font-sans">
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

      {/* 2. External Web Search Evidence */}
      {webSources.length > 0 && (
        <Card className="border-slate-200/80 shadow-sm rounded-xl bg-white overflow-hidden">
          <CardHeader className="py-4 border-b border-slate-100 bg-slate-50/50">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold tracking-wide uppercase text-slate-700 flex items-center gap-2">
                <Globe className="h-4 w-4 text-blue-600" />
                External Web Search Evidence (Live Internet Research)
              </CardTitle>
              <Badge className="bg-blue-50 text-blue-700 border-blue-200 text-[10px] font-semibold">
                Live Search Evidence
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="pt-2">
            <Accordion type="single" collapsible defaultValue="web-item-0" className="w-full">
              {webSources.map((source, index) => (
                <AccordionItem key={`web-${index}`} value={`web-item-${index}`} className="border-slate-100 last:border-0">
                  <AccordionTrigger className="text-left py-3 hover:no-underline">
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <ExternalLink className="h-3.5 w-3.5 text-blue-500" />
                        <span className="font-semibold text-slate-800 text-sm">
                          Live Web Evidence Item #{index + 1}
                        </span>
                        <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100 text-[10px] border-0 font-normal">
                          External Web Findings
                        </Badge>
                      </div>
                      <span className="text-xs text-slate-400 font-normal ml-5">
                        Source: {source.source} | Date: {source.timestamp}
                      </span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4 ml-5">
                    <div className="space-y-3 pt-1 text-slate-700 bg-blue-50/30 p-4 rounded-xl border border-blue-100/60">
                      {/* Structured Metadata Bar */}
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px] bg-white p-2.5 rounded-lg border border-blue-100 font-mono text-slate-500 mb-2">
                        <div><span className="font-semibold text-blue-800">Source:</span> {source.source}</div>
                        <div><span className="font-semibold text-blue-800">Date:</span> {source.timestamp}</div>
                        <div><span className="font-semibold text-blue-800">Type:</span> Live Web Evidence</div>
                      </div>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap font-sans">
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
