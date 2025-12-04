"use client"

import { useState, useMemo } from "react"
import { useChat, useCompletion } from "@ai-sdk/react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { SentimentCard } from "@/components/sentiment-card"
import { ReasoningDisplay } from "@/components/reasoning-display"
import { ArrowRight, Loader2 } from "lucide-react"
import { DefaultChatTransport } from "ai"

export interface SentimentResult {
    polarity: "Positive" | "Negative" | "Neutral"
    emotion: "Joy" | "Sadness" | "Anger" | "Fear" | "Surprise" | "Disgust" | "Neutral"
    confidence_score: number
    reasoning: string
}

export function SentimentAnalyzer() {
    const [inputText, setInputText] = useState("")
    const [sentiments, setSentiments] = useState<SentimentResult[]>([])

    const { messages, error, sendMessage, status, setMessages } = useChat({
        transport: new DefaultChatTransport({
            api: "/api/analyze"
        }),
        onFinish: (data) => {
            setMessages([]);

            if (data?.message && data?.message.role === "assistant") {
                console.log("Final message:", data.message);
                data?.message.parts.forEach(part => {
                    if (part.type === "text") {
                        try {
                            const extractedText = part.text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
                            const parsedSentiment = JSON.parse(extractedText);
                            if (Array.isArray(parsedSentiment)) {
                                setSentiments(parsedSentiment);
                            } else {
                                setSentiments([parsedSentiment]);
                            }
                        } catch (error) {
                            console.error("Error parsing sentiment JSON:", error);
                        }
                    }
                })

            }
        }
    })


    const handleAnalyze = () => {
        if (!inputText.trim()) return
        sendMessage({ text: inputText.trim() })
    }

    return (
        <div className="space-y-10">
            {/* Input Section */}
            <div className="space-y-4">
                <div className="flex items-baseline justify-between">
                    <label htmlFor="text-input" className="text-sm font-medium text-foreground">
                        Input Text
                    </label>
                    <span className="text-xs text-muted-foreground tabular-nums">{inputText.length} chars</span>
                </div>
                <Textarea
                    id="text-input"
                    placeholder="Paste or type text to analyze..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    className="min-h-[160px] resize-none bg-card border-border text-foreground placeholder:text-muted-foreground/60 text-base leading-relaxed"
                />
                <Button onClick={handleAnalyze} disabled={status == "streaming" || status == "submitted" || !inputText.trim()} className="gap-2 h-11 px-6">
                    {status == "streaming" || status == "submitted" ? (
                        <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span>Analyzing</span>
                        </>
                    ) : (
                        <>
                            <span>Analyze</span>
                            <ArrowRight className="h-4 w-4" />
                        </>
                    )}
                </Button>
            </div>
            {
                messages?.map(message => {
                    if (message?.role != "assistant") return null;

                    return (
                        message.parts.map((part, index) =>
                            part.type == "reasoning" && <ReasoningDisplay key={index} reasoning={part.text} isLoading={status == "submitted"} />
                        )
                    )

                })
            }
            {/* Reasoning Display */}
            {/* {reasoning && <ReasoningDisplay reasoning={reasoning} isLoading={status == "streaming" || status == "submitted"} />} */}

            {/* Results Section */}
            {sentiments.length > 0 && (
                <div className="space-y-5">
                    <div className="flex items-baseline gap-3">
                        <h2 className="font-serif text-2xl text-foreground italic">Results</h2>
                        <span className="text-sm text-muted-foreground">
                            {sentiments.length} sentiment{sentiments.length > 1 ? "s" : ""} detected
                        </span>
                    </div>
                    <div className="grid gap-4">
                        {sentiments.map((sentiment, index) => (
                            <SentimentCard key={index} sentiment={sentiment} index={index} />
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
