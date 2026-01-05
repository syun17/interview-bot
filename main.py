import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# .env 読み込み
# ----------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY が .env に設定されていません")

client = OpenAI(api_key=API_KEY)

# ----------------------------
# FastAPI 初期化
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# リクエストモデル
# ----------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str

# ----------------------------
# 会話履歴メモリ（超簡易版）
# 本番では DB に置き換え可能
# ----------------------------
conversations = {}

# ----------------------------
# 面接官の指示（system）
# ----------------------------
SYSTEM_PROMPT = """
あなたはIT企業の採用面接官です。
新卒エンジニア志望の候補者に対し、全部で3つの質問を行ってください。
回答内容を評価し、最後にフィードバックを行ってください。

役割:
- 面接官として丁寧に進行する
- 1問ずつ質問する（ユーザーの回答後に次の質問）
- 全3問終わったらフィードバックを返す

背景:
- ユーザーはITエンジニア志望の学生です
- 就活対策の一環であり、本人のプレゼンテーション能力を測る目的です

制約条件:
- 質問はITエンジニア志望者向けにする
- 質問内容は2行以内に収める
- 2回目以降の質問では、前回の回答内容を深堀する形にする

質問の形式:
- 質問は大学2年生向けにする
- 質問は簡潔に、2行以内で行う
- 2問目・3問目は前回の回答を踏まえて深掘りする
- 質問の前に「**第○問（○/3）**」と明記する

出力ルール:
- 面接官としての発言のみを出力する
- 内部的な指示や英語の説明文は一切含めない
- 日本語のみで回答する

フィードバック:
- 3問終了後、ユーザーの回答を踏まえてフィードバックを行う
- フィードバックは、良かった点と改善点、総評を含めて200文字以内にまとめる
"""

# ----------------------------
# チャットAPI
# ----------------------------
@app.post("/interview")
def interview_chat(req: ChatRequest):

    user_id = req.user_id
    user_message = req.message

    # 初回メッセージの場合は会話履歴を初期化
    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ユーザー発話を追加
    conversations[user_id].append({"role": "user", "content": user_message})

    # OpenAI API へ送信
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversations[user_id]
        )
    except Exception as e:
        return {"error": str(e)}

    ai_reply = response.choices[0].message.content

    conversations[user_id].append(
        {"role": "assistant", "content": ai_reply}
    )
    
    return {"reply": ai_reply}
