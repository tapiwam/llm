from langgraph.graph import END, StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema.document import Document

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import  DuckDuckGoSearchRun, DuckDuckGoSearchResults


# State for every node

from typing import Dict, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


# ==========================================
# ==========================================
# ==========================================
# Node definitions
# ==========================================
# ==========================================
# ==========================================


# ==========================================
# Retrieve documents
# ==========================================

def retrieve_documents(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.info('--- Retrieving documents ---')
    
    # Get question
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)

    # Add to state
    return {
        "keys": {
            "documents": documents, 
            "question": question
        }
    }

# ==========================================
# Generate answer
# ==========================================
def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer, that contains generated answer
    """
    logger.info('--- Generating answer ---')
    
    # Get question
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Chain
    rag_chain = rag_prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {
            "documents": documents, 
            "question": question, 
            "generation": generation
       }
    }

# ==========================================
# Grade documents
# ==========================================
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, grade, that contains grade
    """
    logger.info('--- Grading documents - Checking relevance ---')
    
    # Get question
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = PromptTemplate(
        template=prompt_templates["grade_documents"],
        input_variables=["question", "context"],
    )

    chain = prompt | llm | JsonOutputParser() 

    # Run through documents and compile filtered documents
    filtered_docs = []
    search = "No"
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score["score"]
        if grade == "yes":
            logger.info(f"Document is relevant to the question")
            filtered_docs.append(d)
        else:
            logger.info(f"Document is not relevant to the question")
            search = "Yes" # Perform web search
            continue

    # Return filtered documents
    return {
        "keys": {
            "documents": filtered_docs, 
            "question": question, 
            "run_web_search": search
        }
    }

# ==========================================
# Transform Query to produce a better question
# ==========================================
def transform_query(state):
    """
    Transform query to produce a better question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    logger.info('--- Transforming query ---')
    
    # Get question
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = PromptTemplate(
        template=prompt_templates["question_transform"], 
        input_variables=["question"]
    )

    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    return {
        "keys": {
            "question": better_question,
            "documents": documents
        }
    }

# ==========================================
# Search web for relevant documents
# ==========================================
def extract_duckduckgo_docs(docs):
    final_docs = []
    for doc in docs:
        metadata = {
            "source": "DuckDuckGo",
            "title": doc.title,
            "url": doc.url
        }

        new_doc = Document(page_content=doc.content, metadata=metadata)
        final_docs.append(new_doc)

    return final_docs


def web_search(state):
    """
    Search web for relevant documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    logger.info('--- Performing web search ---')
    
    # Get question
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = PromptTemplate(
        template=prompt_templates["question_transform"], 
        input_variables=["question"]
    )

    # Use tavily search to get documents and add as a new document to the documents list 
    # tool = TavilySearchResults()
    # docs = tool.invoke({"query": question})
    # web_results = "\n".join([d["content"] for d in docs])
    # web_results = Document(page_content=web_results)
    # documents.append(web_results)

    # Use DuckDuckGo to search
    tool = DuckDuckGoSearchResults()
    docs = tool.run(question)
    documents.extend(extract_duckduckgo_docs(docs))

    logger.info(f"Extracted {len(docs)} documents from web search")
    
    return {
        "keys": {
            "question": question,
            "documents": documents
        }
    }


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generate, that contains generate
    """
    logger.info('--- Deciding to generate new question ---')
    
    # Get question
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    search = state_dict["run_web_search"]

    # Prompt
    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info("--- DECISION: TRANSFORM QUERY and RUN WEB SEARCH")
        return "transform_query"
    else:
        # We have relevant documents, skip web search and go straight to generate answer
        logger.info("--- DECISION: GENERATE")
        return "generate"



workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve_documents)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate_answer)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()