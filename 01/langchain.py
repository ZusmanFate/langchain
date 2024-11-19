import os
from openai import OpenAI

client = OpenAI(
    api_key = os.environ.get("ARK_API_KEY"),
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

response = client.chat.completions.create(
  model="ep-20241101111259-bhp5f",
  temperature=0.5,
  max_tokens=4000,
  messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "大语言模型除了文本生成式模型，还有哪些类别的模型？比如说有名的Bert模型，是不是文本生成式的模型？"},
    ],
    )
print(response.choices[0].message.content)
