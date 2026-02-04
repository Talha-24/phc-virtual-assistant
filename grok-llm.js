import { Groq } from 'groq-sdk';
import dotenv from "dotenv"
dotenv.config();

export const groq = new Groq({apiKey:process.env.GROQ_API_KEY});

const chatCompletion = await groq.chat.completions.create({
  "messages": [
    {
      "role": "user",
      "content": "How are you doing man\n"
    },
    {
      "role": "assistant",
      "content": "Hey there! I’m doing great, thanks for asking. How about you? Anything interesting on your mind today?"
    },
    {
      "role": "user",
      "content": ""
    }
  ],
  "model": "openai/gpt-oss-120b",
  "temperature": 1,
  "max_completion_tokens": 8192,
  "top_p": 1,
  "stream": true,
  "reasoning_effort": "medium",
  "stop": null
});

for await (const chunk of chatCompletion) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}

