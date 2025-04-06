from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatOpenAI
import json
from langchain.vectorstores import FAISS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from preprocessing import make_query_list
from websearch import search_text


class MyEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sergeyzh/rubert-tiny-turbo", device: str = "cpu"):
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Принимает список текстов и возвращает список эмбеддингов.
        """
        # normalize_embeddings=True делает эмбеддинги пригодными для cosine similarity
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Принимает один текст и возвращает его эмбеддинг.
        """
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


class Assistant:
    def __init__(self, client: ChatOpenAI, questions: List, embedding_model_path: str, faiss_path: str):
        self.llm = client


        self.questions = questions
        self.embedder = self._getEmbeddingModel(embedding_model_path)
        # Создаем FAISS индекс на основе вопросов
        self.index, self.question_embeddings = self._create_faiss_index(questions)

        self.indexKnowledge = FAISS.load_local(
                                          faiss_path,
                                          embeddings=self.embedder,
                                          distance_strategy=DistanceStrategy.COSINE,
                                          allow_dangerous_deserialization=True
                                      )

        self.mainAgent = self._getMainAgent()

        self.buttonAgent = self._getButtonAgent()

        self.internetAgent = self._getInternetAgent()


    def preprocessing(self, query) -> str:
        """
        Транслитерация + эндпоинт на сервис с синонимами и опечатками
        """
        return make_query_list(query)

    def _getEmbeddingModel(self, model_path, device="cpu"):
        embedding_model = MyEmbeddings(
            model_name=model_path
        )
        return embedding_model

    def _create_faiss_index(self, questions: list):
        """
        Вычисляет embedding для каждого вопроса и создает FAISS индекс с использованием inner product (IP).
        При нормализации векторов inner product эквивалентен cosine similarity.
        """
        embeddings = []
        for q in questions:
            emb = self.embedder.embed_query(q)  # предположим, что возвращается список float
            embeddings.append(emb)
        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, embeddings

    def retrive(self, query, k=10, treshold=0.65):
        results = self.indexKnowledge.similarity_search_with_score(
            query,
            k
        )
        return [res.page_content for res, score in results if score > treshold]


    def _getMainAgent(self):
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["context", "query"],
                template="""Ты эксперт портала поставщиков. Твоя задача — помогать пользователям находить информацию о тендерах, закупках и требованиях к поставщикам.
                            При формировании ответа обязательно приводи источник информации (например, название документа или идентификатор) и процитируй релевантные фрагменты контекста, в которых содержится ответ на вопрос.
                            Если в предоставленном контексте нет информации, явно укажи, что релевантных данных не обнаружено.

                            Контекст:
                            {context}

                            Вопрос:
                            {query}

                            Пожалуйста, дай подробный и исчерпывающий ответ на запрос, указывая источник(ы) информации и цитируя ключевые фрагменты контекста.
                """),
            output_key="response"
        )

    def _getButtonAgent(self):
          return LLMChain(
              llm=self.llm,
              prompt=PromptTemplate(
                input_variables=["context", "query"],
                template="""Ты помощник, который помогает уточнить вопрос пользователя, чтобы найти подходящий документ из некоего набора документов.

                            У тебя есть:
                            question: {query}
                            docks: {context}

                            Твоя задача:

                            На основе question и docks придумать 5 вариантов уточнения вопроса (которые помогут точнее найти правильный документ или раздел документа).

                            Каждый вариант содержит:

                            button: короткая фраза (1–3 слова), которая описывает вариант уточнения.

                            question: переформулированный вопрос пользователя, включающий дополнительную конкретику (чтобы найти правильный документ).

                            Выводить результат строго в формате JSON-списка без оформления в Markdown. Никаких дополнительных ключей и текста не добавляй.

                            Пример структуры вывода (строго следуй этому формату, без лишних пробелов и переносов):

                            [{{"button": "Текст кнопки", "question": "Переформулированный вопрос"}}, {{"button": "Текст кнопки", "question": "Переформулированный вопрос"}}]
                            Важно:

                            Не добавляй никаких заголовков, приписок, пояснений или текстов вне JSON.

                            Итоговый ответ — это ровно 5 объектов внутри списка JSON.
                """),
            output_key="response"
        )

    def _getInternetAgent(self):
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query", "web_info"],
                template="""Ты ассистент, который принимает на вход запрос query и web_info.
                Ты должен найти информацию в web_info который релевантен для ответа на запрос query.
                Если ты не нашел информацию в web_info выводи 'No info'
                query: {query} и web_info: {web_info}
                """),
            output_key="response"
        )

    def getQuestions(self, query: str, k: int = 3):
        # Предварительная обработка запроса
        processed_query = self.preprocessing(query)
        # Получаем embedding запроса
        query_embedding = self.embedder.embed_query(processed_query)
        query_embedding = np.array([query_embedding]).astype('float32')
        # Поиск k ближайших векторов
        distances, indices = self.index.search(query_embedding, k)
        similar_questions = [self.questions[i] for i in indices[0]]
        return similar_questions


    def getAnswerOnStage1(self, query):
        chunks = self.retrive(query, k=5)
        context = ""
        for idx, chunk in enumerate(chunks):
          context += f"{idx + 1}. {chunk}\n\n"
        print(context)
        response = self.mainAgent({
            "context": context,
            "query": query
        })
        return response["response"]


    def getAnswerOnStage2(self, query):
        query = self.preprocessing(query)
        chunks = self.retrive(query)
        n_chunks = len(chunks)
        if n_chunks == 0:
            return "Не нашел информацию в своей базе знаний, стоит ли поискать данные в интернете?"

        if n_chunks <= 5:
            context = ""
            for idx, chunk in enumerate(chunks):
                context += f"{idx + 1}. {chunk}\n\n"

            response = self.mainAgent({
                "context": context,
                "query": query
            })
            return response["response"]

        if n_chunks > 5:
            return self.getButtons(chunks, query)


    def getAnswerFromInternet(self, query):
        results = search_text(query)
        all_model_results = ''
        final_json = {
            'parse_web': [],
            'final_output' : ''
        }
        for idx, result in enumerate(results, 1):
            if result['status'] == 'success':
                output = self.internetAgent({
                    "query": query,
                    "web_info": result["text"]
                })["response"]
                all_model_results = all_model_results + ' ' + output
                final_json['parse_web'].append({
                    'url': result['url'],
                    'text': output,
                })
        response = self.internetAgent({
                          "query": query,
                          "web_info": all_model_results
                        })["response"]

        final_json['final_output'] = response
        return final_json


    def getButtons(self, chunks, query):
        context = ""
        for idx, chunk in enumerate(chunks):
            context += f"{idx + 1}. {chunk}\n\n"

        response = self.buttonAgent({
            "context": context,
            "query": query
        })
        output = response["response"]
        output = output.replace("`", "").replace("json", "")

        # Попытка парсинга JSON-ответа
        data = json.loads(output)

        return data