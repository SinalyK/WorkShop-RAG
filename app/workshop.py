from logging import Logger
import os

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder, SentenceTransformer
from app.agent import ReActAgent
import requests
from langchain_core.tools import Tool

load_dotenv(override=True)
logger = Logger(__name__)


model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "../chroma_store"))
collection = chroma_client.get_or_create_collection(name="agentic_collection")


def transforme_docs(candidate_docs):
    candidate_docs_c = []
    for doc in zip(
        candidate_docs["ids"][0],
        candidate_docs["documents"][0],
        candidate_docs["metadatas"][0],
    ):

        candidate_docs_c.append({"id": doc[0], "page_content": doc[1], "metadata": doc[2]})

    return candidate_docs_c


def reranker_rag(query: str):

    query_embedding = model.encode([query]).tolist()

    candidate_docs = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas"],
    )

    pairs = [(query, doc) for doc in candidate_docs["documents"][0]]

    scores = cross_model.predict(pairs)

    # Sort candidates by score
    docs = transforme_docs(candidate_docs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return reranked


def fillful_data(query):

    if not query:
        return "query must be a string"

    context = reranker_rag(query)
    return context


def get_tavily_engine():
    from langchain_tavily import TavilySearch

    try:
        return TavilySearch(
            max_results=5,
            topic="general",
        )
    except Exception as e:
        print(e)
        return object()


"""### Weather API from RapidAPI"""


def get_weather_by_city(city: str):

    try:
        url = "https://open-weather13.p.rapidapi.com/city"

        querystring = {"city": city, "lang": "EN"}

        headers = {
            "x-rapidapi-key": "ac0480af20msh9d1cb0e36f13761p1a3064jsnd45e9104368d",
            "x-rapidapi-host": "open-weather13.p.rapidapi.com",
        }

        response = requests.get(url, headers=headers, params=querystring)

        return response.json()
    except Exception as e:
        return e


retriever_tool = Tool(
    name="Retriever",
    func=fillful_data,
    description="Retrieve book action relevant to the query from Chroma vectorstore. requires a {'query': 'your search query'}",
)


search_tool = Tool(
    name="TavilySearch",
    func=get_tavily_engine().invoke,
    description="Web search engine for retrieving information on internet. requires a {'query': 'your search query'}",
)


weather_tool = Tool(
    name="WeatherAction",
    func=get_weather_by_city,
    description="Weather API call. requires a {'city': 'city name'}",
)


load_dotenv(override=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_groq = ChatGroq(model="openai/gpt-oss-120b")

tools = [retriever_tool, search_tool, weather_tool]

agent = ReActAgent(llm=llm_groq, tools=tools)
