import { cn } from "@/lib/utils"
import type { SentimentResult } from "@/components/sentiment-analyzer"

interface SentimentCardProps {
  sentiment: SentimentResult
  index: number
}

const polarityConfig = {
  Positive: {
    accent: "bg-positive",
    bg: "bg-positive-light",
    text: "text-positive",
  },
  Negative: {
    accent: "bg-negative",
    bg: "bg-negative-light",
    text: "text-negative",
  },
  Neutral: {
    accent: "bg-neutral-sentiment",
    bg: "bg-neutral-light",
    text: "text-neutral-sentiment",
  },
}

export function SentimentCard({ sentiment, index }: SentimentCardProps) {
  const config = polarityConfig[sentiment.polarity]
  const confidencePercent = Math.round(sentiment.confidence_score * 100)

  return (
    <div
      className={cn(
        "group relative flex items-stretch border border-border rounded-lg overflow-hidden transition-all duration-300 animate-in fade-in slide-in-from-bottom-2",
        "hover:border-foreground/20",
      )}
      style={{ animationDelay: `${index * 80}ms` }}
    >
      {/* Accent bar */}
      <div className={cn("w-1 shrink-0", config.accent)} />

      <div className="flex-1 p-5">
        {/* Top row: polarity + emotion */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <span className={cn("text-sm font-medium", config.text)}>{sentiment.polarity}</span>
            <span className="text-muted-foreground text-sm">·</span>
            <span className="text-sm text-foreground/70">{sentiment.emotion}</span>
          </div>

          {/* Confidence */}
          <div className="flex items-center gap-2">
            <div className="w-16 h-1 bg-border rounded-full overflow-hidden">
              <div
                className={cn("h-full rounded-full transition-all duration-700", config.accent)}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground tabular-nums w-8 text-right">{confidencePercent}%</span>
          </div>
        </div>

        {/* Reasoning */}
        <p className="text-sm text-foreground/80 leading-relaxed">{sentiment.reasoning}</p>
      </div>
    </div>
  )
}
