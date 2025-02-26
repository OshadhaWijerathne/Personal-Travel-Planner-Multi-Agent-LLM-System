# config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
API_KEY = os.getenv("API_KEY")

# Initialize the LLM once
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=API_KEY)
