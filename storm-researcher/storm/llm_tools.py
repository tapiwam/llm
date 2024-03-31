from typing import Type
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate,AIMessagePromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate, BasePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel

from langchain_core.runnables import RunnableLambda, chain as as_runnable

from langgraph.graph import StateGraph, END

from langchain_core.runnables import RunnableConfig

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from sympy import N


from .fns import cleanup_name



def get_openai_llms(regular_model: str = "gpt-3.5-turbo", long_context_model: str = "gpt-4-turbo") -> tuple[ChatOpenAI, ChatOpenAI]:
    return ChatOpenAI(model=regular_model), ChatOpenAI(model=long_context_model)

def get_chat_prompt_from_prompt_templates(messages: list) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(messages)

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

def get_pydantic_parser(pydantic_object: Type[BaseModel]) -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=pydantic_object)


def get_prompt_with_outputparser_partial(promt: ChatPromptTemplate, output_parser) -> ChatPromptTemplate:
    return promt.partial(format_instructions=output_parser.get_format_instructions())


def get_chain_with_outputparser(chat_prompt: ChatPromptTemplate, llm, output_parser):
    # Check chat prompt has field `format_instructions`
    if "format_instructions" not in chat_prompt.input_variables:
        raise ValueError(f"Chat prompt must have field `format_instructions`. Current prompt variables: {chat_prompt.input_variables}")
    
    return get_prompt_with_outputparser_partial(chat_prompt, output_parser) | llm | output_parser


def tag_with_name(ai_message: AIMessage, name: str) -> AIMessage:
    # Clean up name
    name = cleanup_name(name)
    
    ai_message.name = name
    return ai_message

def get_wikipedia_retriever() -> WikipediaRetriever:
    return WikipediaRetriever(load_all_available_meta=True, top_k_results=1)


@tool
async def search_engine(query: str):
    """Search engine to the internet."""

    print(f"Searching DuckDuckGo for [{query}]")

    results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)

    print(f"Got search engine results: {len(results)} for [{query}]")
    
    return [{"content": r["body"], "url": r["href"]} for r in results]

