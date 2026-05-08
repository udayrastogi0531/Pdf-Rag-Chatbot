from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.7,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Senior Research Analyst with 20 years of experience.
Your job is to provide clear, concise, and accurate information.

STRICT RULES:
- Return ONLY the answer without any preamble or introduction
- Do NOT start with "Sure", "Of course", "Great question" etc.
- Be direct and factual when listing multiple items
- Use bullet points
- Keep the response under 100 words"""
    ),
    (
        "human",
        "Explain the following topic: {topic}"
    )
])

# rag chain

chain = prompt | llm | StrOutputParser()
topic = input('Enter the topic : ')
response = chain.invoke({"topic": topic})

print(response)