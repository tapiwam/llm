from importlib import metadata
import os
from typing import Type
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub

from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma

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
from numpy import where

from .models import *
from .fns import cleanup_name

# ================================
# Caching
# ================================
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

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

# ================================================
# Search tools
# ================================================

def get_wikipedia_retriever(k: int = 3, content_chars_max: int = 8000) -> WikipediaRetriever:
    return WikipediaRetriever(load_all_available_meta=True, top_k_results=k, doc_content_chars_max=content_chars_max)


@tool
async def search_engine(query: str) -> list[dict]:
    """Search engine to the internet."""

    print(f"Searching DuckDuckGo for [{query}]")

    results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)

    print(f"Got search engine results: {len(results)} for [{query}]")
    
    for r in results:    
        print(f"- {r}")
    
    return [{"content": r["body"], "url": r["href"]} for r in results]


@tool
async def get_web_page_docs_from_url1(url: str, tags_to_extract=["span", "p", "h1", "h2", "h3", "div", "li", "ul", "ol", "a"]) -> list[Document]|None:
    """
    Get web page contents from url in the form of Documents
    
    Args:
        url (str): The url to fetch the web page from
        tags_to_extract (list[str]): The tags to extract from the web page

    Returns:
        docs (list[Document]): The web page contents in the form of Documents
    """
    
    docs_transformed = None
    print(f"Loading web page from url: {url}")

    try:
        loader = AsyncHtmlLoader([url])
        html =  loader.load()
        
        # Transform
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=tags_to_extract)

    except Exception as e:
        print(f"Error loading web page from url: {url}, error: {e}")
        return None
   

    return docs_transformed

# async def fetch_page_content_from_refs(references: list[Document], limit: int|None = None) -> list[Document]:
#     pages = []
    
#     for ref in references:
#         try:
#             url = ref.metadata['source'] if 'source' in ref.metadata else None
#             if not url:
#                 continue
#             else:
#                 print(f"Fetching web page from url: {url}")
#                 docs1 = await get_web_page_docs_from_url1(url, tags_to_extract=[ "p", "h1", "h2", "h3"])
#                 if docs1 is not None:
#                     if limit is not None:
#                         for doc in docs1:
#                             doc.page_content = doc.page_content[:limit]
#                     pages.extend(docs1)
                    
#         except Exception as e:
#             print(f"Error fetching web page from url: {url}, error: {e}")
#             continue
        
#     return pages

async def fetch_pages_from_refs(references: list[Document], limit: int|None = None) -> dict[str, list[Document]]:
    page_map = {}
    urls = [ref.metadata['source'] if 'source' in ref.metadata else None for ref in references]

    # Get web pages
    docs: list[list[Document]] = await get_web_page_docs_from_url1.abatch(urls, tags_to_extract=[ "p", "h1", "h2", "h3"], limit=limit)
    
    # filter None docs
    
    for doc in docs:
        if doc is not None and len(doc) > 0:
            key = doc[0].metadata['source'] if 'source' in doc[0].metadata else None
            if key is not None:
                if key not in page_map:
                    page_map[key] = []
                page_map[key].extend(doc)
    
    return page_map


def summarize_single_doc(llm, topic, docs: list[Document]) -> Document:
    
    prompt_template = """Gieven the provided context, write a concise summary of the text provided. 
    Include as many details as possible from the gathered information. If the context is not useful, return no summary.
    
    Context:
    ```
    {topic}
    ```
    
    Content:
    {text}
    
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    prompt = prompt.partial(topic=topic)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text"
    )
    result = chain({"input_documents": split_docs}, return_only_outputs=True)
    doc = Document(page_content=result['output_text'], metadata = docs[0].metadata)
    return doc

def summarize_full_docs(llm, topic, docs: dict[str, list[Document]]) -> dict[str, list[Document]]:
    
    summaries = {}
    if docs is not None and len(docs) > 0:
        for i, (key, doc) in enumerate(docs.items()):
            try:
                print(f"Summarizing {i+1}/{len(docs)} : {key}")
                summary = summarize_single_doc(llm, topic, doc)
                summaries[key] = summary
            except Exception as e:
                print(f"Error summarizing [{key}], error: {e}")
            
    return summaries



# ===========================================
# Vectorstore tools
# ===========================================

def store_wiki_docs_to_vectorstore(logger, interview_config: InterviewConfig, docs: list[Document], 
                                   chunk_size: int = 1000, chunk_overlap: int = 0) -> int:
    
    chunks_stored = 0
    vectorstore = interview_config.vectorstore
    
    # Initialize the vectorstore if it doesn't exist
    if vectorstore is None:
        embeddings = interview_config.embeddings
        if embeddings is None:
            embeddings = get_gpt4all_embeddings()
            interview_config.embeddings = embeddings
        
        vectorstore = Chroma(embedding=interview_config.embeddings, persist_directory=interview_config.vectorstore_dir)
        interview_config.vectorstore = vectorstore
        
    # Recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    for sdoc in docs:
        # Check if vectorstore already has this doc
        has_doc = False
        where_clause = {"source": sdoc.metadata['source']}
        search_res: dict[str, Any] = vectorstore.get(where=where_clause)
        logger.info(f'Search result: {search_res}')
        if  search_res is not None and 'ids' in search_res and len(search_res['ids']) > 0:
            logger.info(f'Vectorstore already has doc: {where_clause} ')
            has_doc = True
            continue
        
        logger.info(f'Storing doc: {sdoc.metadata["source"]}')
    
        if not has_doc:
            
            # stringify all metadata
            for key in sdoc.metadata:
                sdoc.metadata[key] = str(sdoc.metadata[key])
            
            sub_docs = text_splitter.split_documents([sdoc])
            vectorstore.add_documents(documents=sub_docs)
            chunks_stored += len(sub_docs)
        logger.info(f'Done storing doc: {sdoc.metadata["source"]}')
    
    vectorstore.persist()
    logger.info(f'Data stored in vector store. Chunks: {len(docs)}')
    
    return chunks_stored

def store_docs_to_vectorstore(logger, interview_config: InterviewConfig, docs: list[list[dict[str, str]]], 
                              chunk_size: int = 1000, chunk_overlap: int = 0) -> int:
    
    chunks_stored = 0
    vectorstore = interview_config.vectorstore
    
    # Initialize the vectorstore if it doesn't exist
    if vectorstore is None:
        embeddings = interview_config.embeddings
        if embeddings is None:
            embeddings = get_gpt4all_embeddings()
            interview_config.embeddings = embeddings
        
        vectorstore = Chroma(embedding=interview_config.embeddings, persist_directory=interview_config.vectorstore_dir)
        interview_config.vectorstore = vectorstore
        
    # Recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    for sdoc in docs:
        # Check if vectorstore already has this doc
        has_doc = False
        where_clause = {"source": sdoc[0]['url']}
        search_res: dict[str, Any] = vectorstore.get(where=where_clause)
        logger.info(f'Search result: {search_res}')
        if  search_res is not None and 'ids' in search_res and len(search_res['ids']) > 0:
            logger.info(f'Vectorstore already has doc: {where_clause} ')
            has_doc = True
            continue
        
        logger.info(f'Storing doc: {sdoc[0]["url"]} with query: {sdoc[0]["query"]}')
        for doc in sdoc:
            url = doc['url']
            content = doc['content']
            query = doc['query']
            metadata = {
                "source": url, 
                "query": query
            }
            
            d = Document(page_content=content, metadata=metadata)
            
            if not has_doc:
                logger.debug(f'Storing doc chunk for: {url}')
                sub_docs = text_splitter.split_documents([d])
                vectorstore.add_documents(documents=sub_docs)
                chunks_stored += len(sub_docs)
        logger.info(f'Done storing doc: {sdoc[0]["url"]}')
    
    vectorstore.persist()
    logger.info(f'Data stored in vector store. Chunks: {len(docs)}')
    
    return chunks_stored



