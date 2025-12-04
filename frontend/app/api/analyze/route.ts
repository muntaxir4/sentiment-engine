import { convertToModelMessages, streamText, UIMessage } from 'ai';
import { createOllama } from 'ollama-ai-provider-v2';
// Allow streaming responses up to 30 seconds

const ollama = createOllama({
    baseURL: 'http://localhost:11434/api',
})
export const maxDuration = 30;

export async function POST(req: Request) {
    const { messages }: { messages: UIMessage[] } = await req.json();

    const result = streamText({
        model: ollama('sentiment-engine'),
        messages: convertToModelMessages(messages),
        providerOptions: {
            ollama: {
                think: true
            }
        }
    });

    return result.toUIMessageStreamResponse();
}