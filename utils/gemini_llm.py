import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables (.env)
load_dotenv()


# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


def generate(prompt: str) -> str:
    """
    Send prompt to Gemini and return response text.
    """

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        print("[Gemini LLM ERROR]", e)

        return ""