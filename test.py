from openai import OpenAI
import json
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(

  extra_body={},
  model="meta-llama/llama-3.3-70b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "你知道什么是llmfactor嘛?"
    }
  ]
)
print(completion.choices[0].message.content)

# 使用 model_dump() 将 pydantic 对象转换为 Python 字典
result_dict = completion.model_dump()

# 转换成 JSON 格式并打印
print(json.dumps(result_dict, indent=2, ensure_ascii=False))