from importlib import metadata
import os
from typing import Type
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub

from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore

from langchain.prompts import PromptTemplate,ChatPromptTemplate,AIMessagePromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate, BasePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import StrOutputParser

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

from langchain_core.runnables import RunnableLambda, chain as as_runnable

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END

from langchain_core.runnables import RunnableConfig, RunnableParallel, RunnablePassthrough

from langchain.chains import load_summarize_chain

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langchain_core.prompts import MessagesPlaceholder

from .fns import cleanup_name

# ================================
# Caching
# ================================
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import tiktoken

def setup_inmemory_cache():
    set_llm_cache(InMemoryCache())

def setup_sqlite_cache(db_cache_name=".langchain.db"):
    set_llm_cache(SQLiteCache(database_path=db_cache_name))


# ================================================
# LLM tools
# ================================================

def get_openai_llms(regular_model: str = "gpt-3.5-turbo", long_context_model: str = "gpt-3.5-turbo") -> tuple[ChatOpenAI, ChatOpenAI]:
    return ChatOpenAI(model=regular_model), ChatOpenAI(model=long_context_model)

def get_chat_prompt_from_prompt_templates(messages: list) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(messages)

def get_anthropic_llms() -> tuple[ChatAnthropic, ChatAnthropic]:
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")
    ANTHROPIC_MODEL = ANTHROPIC_MODEL if ANTHROPIC_MODEL else "claude-3-haiku-20240307"
    llm = ChatAnthropic(model_name=ANTHROPIC_MODEL)
    return llm, llm

def get_ollama_llms(regular_ollama_base_url = "http://192.168.68.99:11434", regular_model = "mistral:instruct", 
                   long_context_ollama_base_url = "http://192.168.68.99:11434", long_context_model = "mistral:instruct"
                   ) -> tuple[ChatOllama, ChatOllama]:
    llm = ChatOllama(base_url=regular_ollama_base_url, 
                    model=regular_model,
                    temperature=0, 
                    verbose=True)
    
    long_llm = ChatOllama(base_url=long_context_ollama_base_url, 
                    model=long_context_model,
                    temperature=0, 
                    verbose=True)
    
    return llm, long_llm


# GPT4ALL Embeddings
def get_gpt4all_embeddings() -> Type[Embeddings]:
    return GPT4AllEmbeddings()

def get_openai_embeddings(model: str|None = None) -> Type[Embeddings]:
    embeddings = OpenAIEmbeddings(model=model) if model else OpenAIEmbeddings()
    return embeddings
    
    

# ================================================
# Vector DB
# ================================================
def get_inmemory_db(reference_docs, embeddings) -> SKLearnVectorStore:
        vectorstore = SKLearnVectorStore.from_documents(
        reference_docs,
        embedding=embeddings,
    )
        
        return vectorstore

# ================================================
# Prompt and Message tools
# ================================================

def generate_chat_prompt(system_template: str, human_template: str) -> ChatPromptTemplate:
    
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    return get_chat_prompt_from_prompt_templates([system_prompt, human_prompt])

def generate_system_chat_prompt(system_template: str) -> SystemMessagePromptTemplate:
    return SystemMessagePromptTemplate.from_template(system_template)

def generate_human_chat_prompt(human_template: str) -> HumanMessagePromptTemplate:
    return HumanMessagePromptTemplate.from_template(human_template)

def generate_human_message(human_template: str, name: str|None = None) -> HumanMessage:
    return HumanMessage(content=human_template, name=name)

def get_ai_message(content: str, name: str = "AI") -> AIMessage:
    return AIMessage(content=content, name=name)

def tag_with_name(ai_message: BaseMessage, name: str) -> BaseMessage:
    # Clean up name
    name = cleanup_name(name)
    
    ai_message.name = name
    return ai_message


# ================================================
# Output Tools
# ================================================
def get_pydantic_parser(pydantic_object: Type[BaseModel]) -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=pydantic_object)


def get_prompt_with_outputparser_partial(promt: ChatPromptTemplate, output_parser) -> ChatPromptTemplate:
    return promt.partial(format_instructions=output_parser.get_format_instructions())


def get_chain_with_outputparser(chat_prompt: ChatPromptTemplate, llm, output_parser):
    # Check chat prompt has field `format_instructions`
    if "format_instructions" not in chat_prompt.input_variables:
        raise ValueError(f"Chat prompt must have field `format_instructions`. Current prompt variables: {chat_prompt.input_variables}")
    
    fixer_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
    return get_prompt_with_outputparser_partial(chat_prompt, output_parser) | llm | fixer_parser


# ===============================================
# Vectorstore Tools
# ===============================================

def get_notes_from_vectorstore(query:str, vectorstore: VectorStore, notes_token_limit=1000, doc_token_limit=250, k=20):
    notes: str = ''
    tiktokens = tiktoken.get_encoding("cl100k_base")
    docs = vectorstore.similarity_search(query, k=k)
    
    # compile the notes from the docs
    for doc in docs:
        title = doc.metadata['title'] if 'title' in doc.metadata else ''
        if title is not None:
            notes += f"Title: {title}\n\n"
        
        source = doc.metadata['source'] if 'source' in doc.metadata else ''
        if source is not None:
            notes += f"Source: {source}\n\n"
        
        
        summary = doc.metadata['summary'] if 'summary' in doc.metadata else ''
        if summary is not None:
            notes += f"Summary: {summary}\n\n"
        

        text = doc.page_content
        # trim text to doc_token_limit
        if len(tiktokens.encode(text)) > doc_token_limit:
            # split by tokens
            text = ' '.join(tiktokens.decode(tiktokens.encode(text)[:doc_token_limit]))
        
        notes += text
        notes += '\n=================\n'
        
        # Check length of notes
        tokens = tiktokens.encode(notes)
        if len(tokens) > notes_token_limit:
            print(f"Notes has reached token limit: {len(tokens)} > {notes_token_limit}")
            break
    
    return notes
    