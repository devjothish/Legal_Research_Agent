# app.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from retriever import ingest_documents, retrieve_documents
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import LangChainTracer
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

# Initialize LangSmith client
client = Client()

# Setup LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

# Prompt Template
prompt_template = """
You are a legal research assistant. Based on the following documents, answer the user's question clearly, cite cases directly, and always disclaim that this is not formal legal advice.

Documents:
{documents}

Question:
{question}
"""

def main():
    action = input("Do you want to (1) Ingest Docs or (2) Ask a Question? ")
    if action == "1":
        ingest_documents()
        print("Documents ingested into vectorstore.")
    elif action == "2":
        query = input("Enter your legal question: ")
        docs = retrieve_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        
        # Run the chain with LangSmith tracing
        tracer = LangChainTracer()
        answer = chain.invoke(
            {"documents": context, "question": query},
            config={"callbacks": [tracer]}
        )

        print("\nGenerated Answer:\n")
        print(answer.content)
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()
