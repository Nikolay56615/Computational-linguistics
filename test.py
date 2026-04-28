from openrouter import OpenRouter
import os

with OpenRouter(
  api_key=os.getenv("OPENROUTER_API_KEY", "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"),
) as client:
  response = client.chat.send(
    model="qwen/qwen3-next-80b-a3b-instruct:free",
    messages=[
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  )

  print(response.choices[0].message.content)