import json
import os
import re
import textwrap
import traceback
from logging import Logger
from typing import Any, Dict, List, Optional, TypedDict

import backoff
import chromadb
import numpy as np
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from IPython.display import Markdown, display
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer
from termcolor import colored  # pip install termcolor
from app.utils import verbose_context,fillful_data
from app.prompts import system_cot,system_tot
from app.agent import ReActAgent
load_dotenv(override = True)
logger = Logger(__name__)


path = "docs/Building_Agentic_AI_Systems_Create_intelligent,_autonomous_AI_agents.pdf"
loader = PyPDFLoader(path)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

chunks = []

for idx, document in enumerate(documents):

    chunk_lines = text_splitter.split_text(document.page_content)

    for index, chunk in  enumerate(chunk_lines):

        chunks.append({
            "id" : f"page{idx}_chunk{index}",
            "content": chunk,
            "metadata": {
            "title" : "Building Agentic AI Systems Create_intelligent, autonomous AI agents",
            "source": "Building_Agentic_AI_Systems_Create_intelligent,_autonomous_AI_agents.pdf",
            "page": idx
        }})


model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="agentic_collection")

collection.count()

for chunk in chunks:

    collection.add(
        ids=chunk["id"],
        documents=chunk["content"],
        embeddings=model.encode(chunk["content"]),
        metadatas = chunk["metadata"]
    )


cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# self-retrieving

document_content_description = (
    "Agentic System Book which explains different archetures of AI systems"
)

embedding_func = embedding_function = HuggingFaceEmbeddings(
    model_name="distiluse-base-multilingual-cased-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding_function,
    collection_name="agentic_collection",
)

metadata_field_info = [
    AttributeInfo(
        name="content",
        description="Chunk content",
        type="string",
    ),
]

llm_groq = ChatGroq(model = "openai/gpt-oss-20b")
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")


self_retriever = SelfQueryRetriever.from_llm(
    llm_groq,
    vectorstore,
    document_content_description,
    metadata_field_info,
)



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

tavily = get_tavily_engine()


"""### Weather API from RapidAPI"""

import requests

def get_weather_by_city(city: str):

    try:
        url = "https://open-weather13.p.rapidapi.com/city"

        querystring = {"city":city,"lang":"EN"}

        headers = {
        	"x-rapidapi-key": "172266d2e5mshdc26fd674ea88fdp1c18cdjsn941dfcc7a085",
        	"x-rapidapi-host": "open-weather13.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)

        return response.json()
    except Exception as e:
        return e



### Tools
from langchain_core.tools import Tool

retriever_tool = Tool(
    name="Retriever",
    func=self_retriever.invoke,
    description="Retrieve book action relevant to the query from Chroma vectorstore.",
)



search_tool = Tool(
    name = "TavilySearch",
    func = get_tavily_engine().invoke,
    description = "Web search engine for retrieving information on internet."
)


weather_tool = Tool(
    name = "WeatherAction",
    func = get_weather_by_city,
    description = ""
)


tools = [retriever_tool, search_tool, weather_tool]

agent = ReActAgent(llm = llm_groq , tools = tools)



