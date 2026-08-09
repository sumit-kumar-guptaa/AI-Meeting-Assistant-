from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, StructuredOutputParser
from langchain_text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest", mistral_api_key=os.environ.get("MISTRAL_API_KEY"), temperature=0.3
    )

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. combine these partial summaries into a single, coherent summary. Ensure the final summary is concise and captures all key points.",
                "into one final professional meeting summary in bullet points format.",
            ),
            ("human", "{text}"),
        ]
    )