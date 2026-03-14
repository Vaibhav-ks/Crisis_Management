from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3
)


def generate_dispatch_message(route_plan, rescue_plan):

    prompt = f"""
    You are an emergency response communication officer.

    Convert the following structured rescue plan into a clear
    and human-readable instruction for rescue teams.

    Rescue resources:
    {rescue_plan}

    Routes:
    {route_plan}

    Write concise instructions responders can follow immediately.
    """

    response = llm.invoke(prompt)

    return response.content