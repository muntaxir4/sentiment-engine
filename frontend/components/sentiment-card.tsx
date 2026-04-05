import { cn } from "@/lib/utils";
import type { SentimentResult } from "@/components/sentiment-analyzer";

interface SentimentCardProps {
  sentiment: SentimentResult;
  index: number;
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
};

export function SentimentCard({ sentiment, index }: SentimentCardProps) {
  const config = polarityConfig[sentiment.polarity];
  const asAny = sentiment as unknown as {
    emotion?: string;
    confidence_score?: number;
    emotions?: Array<{ emotion?: string; confidence_score?: number } | string>;
  };

  const emotionEntries =
    Array.isArray(asAny.emotions) && asAny.emotions.length > 0
      ? asAny.emotions
          .map((item) => {
            if (typeof item === "string") {
              return {
                emotion: item,
                confidence_score: asAny.confidence_score,
              };
            }
            return {
              emotion: item?.emotion ?? "",
              confidence_score: item?.confidence_score,
            };
          })
          .filter((item) => item.emotion)
      : asAny.emotion
        ? [{ emotion: asAny.emotion, confidence_score: asAny.confidence_score }]
        : [];

  const topConfidence =
    emotionEntries.length > 0
      ? Math.max(
          ...emotionEntries.map((e) =>
            typeof e.confidence_score === "number" ? e.confidence_score : 0,
          ),
        )
      : typeof asAny.confidence_score === "number"
        ? asAny.confidence_score
        : 0;

  const confidencePercent = Math.round(topConfidence * 100);

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
        {/* Top row: polarity + emotions */}
        <div className="flex items-start justify-between mb-3 gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className={cn("text-sm font-medium", config.text)}>
              {sentiment.polarity}
            </span>
            {emotionEntries.length > 0 && (
              <span className="text-muted-foreground text-sm">·</span>
            )}
            {emotionEntries.length > 0 ? (
              emotionEntries.map((entry, emotionIndex) => {
                const score =
                  typeof entry.confidence_score === "number"
                    ? Math.round(entry.confidence_score * 100)
                    : null;
                return (
                  <span
                    key={`${entry.emotion}-${emotionIndex}`}
                    className={cn(
                      "inline-flex items-center rounded-full border shadow-xs px-2 py-0.5 text-xs",
                      config.bg,
                      config.text,
                    )}
                  >
                    {entry.emotion}
                    {score !== null ? ` ${score}%` : ""}
                  </span>
                );
              })
            ) : (
              <span className="text-sm text-foreground/70">N/A</span>
            )}
          </div>

          {/* Confidence */}
          <div className="flex items-center gap-2">
            <div className="w-16 h-1 bg-border rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-700",
                  config.accent,
                )}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground tabular-nums w-8 text-right">
              {confidencePercent}%
            </span>
          </div>
        </div>

        {/* Reasoning */}
        <p className="text-sm text-foreground/80 leading-relaxed">
          {sentiment.reasoning}
        </p>
      </div>
    </div>
  );
}
