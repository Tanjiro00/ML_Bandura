from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from assistant import Assistant
from langchain_community.chat_models import ChatOpenAI
import os
import pickle
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
import requests
import httpx
import shutil

load_dotenv()  # Загружаем переменные из .env файла

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


client = ChatOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model=os.getenv("MODEL_NAME")
    )

with open("data/question_new.pickle", "rb") as f:
    questions = pickle.load(f)

assistant = Assistant(
    client=client,
    questions=questions,
    embedding_model_path=os.getenv("MODEL_BERT"),
    faiss_path="data/vecstore"
)


chatEndpoint = os.getenv("CHAT_ENDPOINT")

async def getHistory(user_id: str, k: int = 8) -> str:
    print("Начал ждать /chat")

    async with httpx.AsyncClient() as client:
        response = await client.get(chatEndpoint, params={
            "user_id": user_id,
            "offset": 0,
            "limit": k
        })

    print("Закончил ждать /chat")
    data = response.json()
    print(data)

    history = ""
    for msg in data:
        sender = msg.get("sender_type", "unknown")
        content = msg.get("content", {})

        if "text" in content:
            history += f"{sender}: {content['text']}\n"
        elif "button" in content:
            history += f"{sender}: {content['button'].get('question', '')}\n"
        elif "response" in content and "output" in content["response"]:
            history += f"{sender}: {content['response']['output']}\n"

    return history


@app.get("/")
async def root():
  return {"message": "Hello World"}


class inputQuery(BaseModel):
    query: str
    user_id: int


class inputRelQues(BaseModel):
    query: str


@app.post("/get_relevant_questions")
async def getRelevantQuestions(request: inputRelQues) -> List[str]:
    return assistant.getQuestions(request.query)


class outputQuery(BaseModel):
    output: str
    category: str


@app.post("/predictOnQuery")
async def predictOnQuery(request: inputQuery) -> outputQuery:
    return {
            "output": assistant.getAnswerOnStage1(request.query, await getHistory(request.user_id)),
            "category": assistant.clf_input(request.query)
    }


class Button(BaseModel):
    button: str
    question: str

class outputStage2(BaseModel):
    flag: int
    response: Union[List[Button], str]
    category: Union[str, None]

@app.post("/predictStage2")
async def predictStage2(request: inputQuery) -> outputStage2:
    """
    Возвращает 
    flag - 0 (если инфы нет, надо предложить поискать в инете), 1 (в response простой ответ LLM текстом), 2 (возвращаются buttons в response)
    response - str или Button
    """
    ans = assistant.getAnswerOnStage2(request.query, await getHistory(request.user_id))
    if ans == "Не нашел информацию в своей базе знаний, стоит ли поискать данные в интернете?":
        return {
            "flag": 0,
            "response": ans,
            "category": None
        }
    elif type(ans) is str: 
        print("ANS", ans)
        return {
            "flag": 1,
            "response": ans,
            "category": assistant.clf_input(ans)
        }
    else:
        return {
            "flag": 2,
            "response": ans,
            "category": None
        }


class webPage(BaseModel):
    url: str
    text: str

class websearchOutput(BaseModel):
    parse_web: List[webPage]
    final_output: str
    category: str


@app.post("/websearch")
async def websearch(request: inputQuery) -> websearchOutput:
    return assistant.getAnswerFromInternet(request.query, await getHistory(request.user_id))


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Папка для сохранения загруженных файлов
    upload_dir = "./uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Сохраняем файл на диск
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Пересоздаем базу знаний с новым PDF
    assistant._add_pdf(pdf_path=file_location)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
