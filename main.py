import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ---- CORS 設定を追加 ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら本番で制限する
    allow_credentials=True,
    allow_methods=["*"],  # ← OPTIONS を含む全メソッド許可
    allow_headers=["*"],
)

# -------------------------

class ChatReq(BaseModel):
    user_message: str
    context: str = ""

system_prompt = """
あなたは採用面接官です。
・質問は1つずつ
・回答に対して簡単なフィードバックを返す
・必要に応じて次の質問を出す
"""

@app.post("/interview")
def interview_chat(req: ChatReq):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.user_message},
        ]
    )

    # ← SDK 仕様変更に対応した書き方
    return {"reply": response.choices[0].message.content}
