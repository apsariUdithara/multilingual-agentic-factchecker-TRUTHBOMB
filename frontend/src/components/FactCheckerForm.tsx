import React, { useState, useRef } from 'react';
import { createWorker } from 'tesseract.js';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, ImageUp, X } from "lucide-react";

interface FactCheckerFormProps {
  onVerify: (text: string) => void;
  isLoading: boolean;
  initialText?: string;
}

export const FactCheckerForm: React.FC<FactCheckerFormProps> = ({ onVerify, isLoading, initialText = '' }) => {
  const [input, setInput] = useState(initialText);
  const [isOcrLoading, setIsOcrLoading] = useState(false);
  const [ocrProgress, setOcrProgress] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (initialText) {
      setInput(initialText);
    }
  }, [initialText]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onVerify(input);
    }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    setIsOcrLoading(true);
    setOcrProgress(0);

    try {
      const worker = await createWorker('eng+sin+tam', 1, {
        logger: (m) => {
          if (m.status === 'recognizing text') {
            setOcrProgress(Math.round(m.progress * 100));
          }
        },
      });
      const { data: { text } } = await worker.recognize(file);
      await worker.terminate();
      setInput(text.trim());
    } catch (err) {
      console.error('OCR failed:', err);
    } finally {
      setIsOcrLoading(false);
      setOcrProgress(0);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const clearImage = () => {
    setPreviewUrl(null);
    setInput('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Social Media Fact-Checker</CardTitle>
        <CardDescription>
          Paste a Singlish, Sinhala or English social media claim, or upload a screenshot to extract text automatically.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">

          {previewUrl && (
            <div className="relative">
              <img src={previewUrl} alt="Uploaded" className="w-full max-h-48 object-contain rounded border" />
              <button
                type="button"
                onClick={clearImage}
                className="absolute top-1 right-1 bg-white rounded-full p-0.5 shadow"
              >
                <X className="h-4 w-4 text-gray-600" />
              </button>
            </div>
          )}

          {isOcrLoading && (
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Extracting text… {ocrProgress}%
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div
                  className="bg-blue-500 h-1.5 rounded-full transition-all"
                  style={{ width: `${ocrProgress}%` }}
                />
              </div>
            </div>
          )}

          <Textarea
            placeholder="e.g., Colombo wala ada loku protest ekak thiyenawa kiyala news ekak yanawa"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="min-h-[150px]"
            disabled={isLoading || isOcrLoading}
          />

          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              className="flex-1"
              disabled={isLoading || isOcrLoading}
              onClick={() => fileInputRef.current?.click()}
            >
              <ImageUp className="mr-2 h-4 w-4" />
              {isOcrLoading ? 'Reading image...' : 'Upload Image'}
            </Button>
            <Button type="submit" className="flex-1" disabled={isLoading || isOcrLoading || !input.trim()}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Verifying...
                </>
              ) : (
                'Verify'
              )}
            </Button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />
        </form>
      </CardContent>
    </Card>
  );
};
