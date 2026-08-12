import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Radio, RefreshCw, Newspaper, ArrowRight } from "lucide-react";

interface NewsItem {
  id: string;
  source: string;
  timestamp: string;
  content: string;
}

interface LatestNewsTickerProps {
  onSelectNews?: (text: string) => void;
}

export const LatestNewsTicker: React.FC<LatestNewsTickerProps> = ({ onSelectNews }) => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchLatestNews = async () => {
    try {
      const response = await fetch('/api/v1/news/latest');
      if (response.ok) {
        const data = await response.json();
        setNews(data.news || []);
        setTotalCount(data.total_count || 0);
      }
    } catch (error) {
      console.error("Failed to fetch latest news feed:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLatestNews();
    // Poll every 15 seconds for live news stream updates
    const interval = setInterval(fetchLatestNews, 15000);
    return () => clearInterval(interval);
  }, []);

  const getSourceBadgeStyle = (source: string) => {
    if (source.toLowerCase().includes('adaderana')) {
      return 'bg-amber-100 text-amber-800 border-amber-200';
    } else if (source.toLowerCase().includes('googlenews')) {
      return 'bg-blue-100 text-blue-800 border-blue-200';
    } else if (source.toLowerCase().includes('newswire')) {
      return 'bg-indigo-100 text-indigo-800 border-indigo-200';
    } else if (source.toLowerCase().includes('economynext')) {
      return 'bg-emerald-100 text-emerald-800 border-emerald-200';
    }
    return 'bg-slate-100 text-slate-800 border-slate-200';
  };

  if (loading) {
    return (
      <div className="w-full max-w-2xl mx-auto mb-8 p-4 bg-slate-50 border border-slate-200 rounded-xl animate-pulse">
        <div className="h-4 bg-slate-200 rounded w-1/3 mb-2"></div>
        <div className="h-3 bg-slate-200 rounded w-2/3"></div>
      </div>
    );
  }

  if (news.length === 0) return null;

  return (
    <Card className="w-full max-w-2xl mx-auto mb-8 border-slate-200/80 shadow-sm rounded-xl bg-white overflow-hidden">
      <CardHeader className="py-3 px-4 border-b border-slate-100 bg-slate-50/60">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Radio className="h-3.5 w-3.5 text-emerald-600" />
              Live Confirmed Sri Lankan News Stream
            </CardTitle>
          </div>
          
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px] bg-white border-slate-200 text-slate-500 font-mono">
              {totalCount.toLocaleString()} Articles Stored
            </Badge>
            <button 
              onClick={fetchLatestNews} 
              className="text-slate-400 hover:text-slate-600 transition-colors p-1"
              title="Refresh News Feed"
            >
              <RefreshCw className="h-3 w-3" />
            </button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="divide-y divide-slate-100 max-h-64 overflow-y-auto">
          {news.map((item, index) => (
            <div 
              key={item.id || index}
              className="p-3.5 hover:bg-slate-50/80 transition-colors group cursor-pointer"
              onClick={() => onSelectNews && onSelectNews(item.content)}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1 flex-1">
                  <div className="flex items-center gap-2">
                    <Badge className={`text-[10px] font-semibold px-2 py-0.2 border ${getSourceBadgeStyle(item.source)}`}>
                      {item.source}
                    </Badge>
                    <span className="text-[11px] text-slate-400 font-mono">
                      {item.timestamp}
                    </span>
                  </div>
                  <p className="text-xs text-slate-700 font-medium leading-snug line-clamp-2">
                    {item.content}
                  </p>
                </div>
                
                {onSelectNews && (
                  <div className="text-slate-300 group-hover:text-primary transition-colors self-center">
                    <ArrowRight className="h-3.5 w-3.5" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
