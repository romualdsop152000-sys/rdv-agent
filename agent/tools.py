from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
