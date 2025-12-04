interface ReasoningDisplayProps {
  reasoning: string
  isLoading: boolean
}

export function ReasoningDisplay({ reasoning, isLoading }: ReasoningDisplayProps) {
  return (
    <div className="py-6 border-y border-border">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-xs uppercase tracking-[0.15em] text-muted-foreground font-medium">Reasoning</span>
        {isLoading && <span className="text-xs text-primary animate-pulse">thinking...</span>}
      </div>
      <div className="text-foreground/80 leading-[1.75] text-[15px]">
        {reasoning}
        {isLoading && (
          <span className="inline-block w-[3px] h-[18px] bg-primary ml-0.5 animate-pulse align-text-bottom" />
        )}
      </div>
    </div>
  )
}
