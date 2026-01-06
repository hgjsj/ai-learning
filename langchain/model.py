import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv("../.env")

# print env GOOGLE_API_KEY here
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
def get_gemini_model():
    return  ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def get_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")