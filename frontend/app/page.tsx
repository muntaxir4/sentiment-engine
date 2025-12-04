import { SentimentAnalyzer } from "@/components/sentiment-analyzer"

export default function Home() {
  return (
    <main className="min-h-screen bg-background relative">
      {/* Subtle grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.015]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fillRule='evenodd'%3E%3Cg fill='%23000000' fillOpacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }}
      />

      <div className="relative container mx-auto px-4 py-16 max-w-3xl">
        <header className="mb-14">
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground mb-3">AI-Powered Analysis</p>
          <h1 className="font-serif text-5xl text-foreground mb-4 text-balance italic">Sentiment Extraction</h1>
          <p className="text-muted-foreground text-lg leading-relaxed max-w-xl text-pretty">
            Uncover the emotional undercurrents within text. Powered by language models to identify polarity, emotion,
            and reasoning.
          </p>
        </header>
        <SentimentAnalyzer />

        {/* Footer */}
        <footer className="mt-20 pt-8 border-t border-border">
          <p className="text-xs text-muted-foreground">Final Year Project — Sentiment Analysis</p>
        </footer>
      </div>
    </main>
  )
}
