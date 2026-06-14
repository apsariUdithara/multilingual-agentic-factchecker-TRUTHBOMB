import { useState } from 'react'
import { FactCheckerForm } from './components/FactCheckerForm'
import { ResultCard } from './components/ResultCard'
import { ContextSources } from './components/ContextSources'
import { VerificationResponse } from './types'
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<VerificationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleVerify = async (text: string) => {
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/v1/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Verification failed (${response.status})`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      console.error('Verification error:', err)
      setError(err.message || 'An unexpected error occurred. Please check if the backend is running.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full bg-slate-50 relative overflow-auto py-12 px-4">
      {/* Background Gradient */}
      <div 
        className="fixed inset-0 pointer-events-none" 
        style={{ 
          background: 'radial-gradient(circle at 15% 15%, #eef4ff 0%, #f7fafc 45%, #edf6f1 100%)',
          zIndex: 0
        }} 
      />

      <div className="relative z-10 max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-extrabold tracking-tight lg:text-5xl mb-4">
            Social Media Fact-Checker
          </h1>
          <p className="text-xl text-muted-foreground">
            Multilingual Social Media Claim Verification Engine
          </p>
        </header>

        <main className="space-y-8">
          <FactCheckerForm onVerify={handleVerify} isLoading={isLoading} />

          {error && (
            <Alert variant="destructive" className="max-w-2xl mx-auto">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {result && (
            <>
              <ResultCard data={result} />
              <ContextSources sources={result.retrieved_context} />
            </>
          )}
        </main>

        <footer className="mt-20 text-center text-sm text-muted-foreground">
          <p>© 2026 Social Media Claim Fact-Checking Engine. All rights reserved.</p>
        </footer>
      </div>
    </div>
  )
}

export default App
