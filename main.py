from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from assistant import Assistant
from langchain_community.chat_models import ChatOpenAI
import os
import pickle
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union

load_dotenv()  # Загружаем переменные из .env файла

app = FastAPI()

client = ChatOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model=os.getenv("MODEL_NAME")
    )

with open("data/question.pickle", "rb") as f:
    questions = pickle.load(f)

assistant = Assistant(
    client=client,
    questions=questions,
    embedding_model_path=os.getenv("MODEL_BERT"),
    faiss_path="data/vecstore"
)

@app.get("/")
async def root():
  return {"message": "Hello World"}


class inputQuery(BaseModel):
    query: str


@app.post("/get_relevant_questions")
async def getRelevantQuestions(request: inputQuery) -> List[str]:
    return assistant.getQuestions(request.query)


@app.post("/predictOnQuery")
async def predictOnQuery(request: inputQuery) -> str:
    return assistant.getAnswerOnStage1(request.query)


class Button(BaseModel):
    button: str
    question: str

class outputStage2(BaseModel):
    flag: int
    response: Union[List[Button], str]

@app.post("/predictStage2")
async def predictStage2(request: inputQuery) -> outputStage2:
    """
    Возвращает 
    flag - 0 (если инфы нет, надо предложить поискать в инете), 1 (в response простой ответ LLM текстом), 2 (возвращаются buttons в response)
    response - str или Button
    """
    ans = assistant.getAnswerOnStage2(request.query)
    if ans == "Не нашел информацию в своей базе знаний, стоит ли поискать данные в интернете?":
        return {
            "flag": 0,
            "response": ans
        }
    elif type(ans) == "str": 
        return {
            "flag": 1,
            "response": ans
        }
    else:
        return {
            "flag": 2,
            "response": ans
        }


class webPage(BaseModel):
    url: str
    text: str

class websearchOutput(BaseModel):
    parse_web: List[webPage]
    final_output: str


@app.post("/websearch")
async def websearch(request: inputQuery) -> websearchOutput:
    return assistant.getAnswerFromInternet(request.query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
