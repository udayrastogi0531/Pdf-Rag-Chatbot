from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0
)

ques = input('Enter your question: ')

response = llm.invoke([HumanMessage(content=ques)])

print(f'AI Response: {response.content}')