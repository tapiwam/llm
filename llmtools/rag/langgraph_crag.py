
from venv import logger
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
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
import traceback
import os
import shutil


# ==========================================
# Graph state
# ==========================================
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

# ==========================================
# Prompt template dictionary
# ==========================================
prompt_templates = {
    "question_transform": """You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question: 
        """,
    "rag_prompt": """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:
        """,
    "grade_documents": """You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        """
}

# ==========================================
# Loader Functions
# ==========================================

# Function to load page from URL
def load_page(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data


# Function to split text into chunks
def split_text(logger, text, chunk_size=1000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(text)
    logger.info(f'Split text into chunks: {len(chunks)}')
    return chunks

# Function to load data into a vector store. Pass in url, embedding model and documents
def load_data_into_vector_store(logger, collection_name, embedding_model, documents):
    # Code to load data into a vector store based on the provided parameters
    vector_store = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=collection_name)
    vector_store.persist()
    logger.info(f'Data loaded into vector store. Collection name: {collection_name}, Chunks: {len(documents)}')
    return vector_store

# Function that takes a url and loads the page and splits the text into chunks, then loads the data into a vector store
def load_data(logger, url, collection_name, embedding_model, chunk_size=500, chunk_overlap=25):
    data = load_page(url)
    chunks = split_text(logger,data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = load_data_into_vector_store(logger, collection_name, embedding_model, chunks)
    return vector_store

# Function to format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to extract docs from duckduckgo
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


class ChromaDbWebLoader:
    def __init__(self, logger, db_name, embeddings, chunk_size=500, chunk_overlap=25, reset_db=False):
        self.urls = []
        self.db_name = db_name
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embeddings = embeddings
        self.logger = logger
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # if reset then delete the db
        if reset_db:
            self.logger.warn(f'Reset db set to true: {self.db_name}')
            if os.path.exists(self.db_name):
                self.logger.warn(f'Deleting db: {self.db_name}')
                shutil.rmtree(self.db_name)
        
        # Initialize the vectorstore
        self.vectorstore = Chroma(embedding_function=embeddings, persist_directory=self.db_name)
    
    def load(self, url):
        try:
            self.logger.info(f'Loading data from url: {url}')
            vectorstore = load_data(
                logger=self.logger,
                url=url,
                collection_name=self.db_name,
                embedding_model=self.embeddings,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
                )
            vectorstore.persist()
            self.vectorstore = vectorstore
            self.urls.append(url)
            self.logger.info(f'Data loaded into vector store. URL: {url}, Collection name: {self.db_name}')
        except Exception as e:
            self.logger.error(f'Error loading data into vector store: url: {url}, error: {e}')
            traceback.print_exc()
        
        return self.vectorstore

class LangGraphContextualRag:
    def __init__(self, logger, llm, vectorstore):
        self.logger = logger
        self.llm = llm
        self.vectorstore = vectorstore
        self.state = GraphState(keys={})
        self.build_graph()
    
    # Format the sate object
    def format_state(self, stage_name, state):
        keys = state["keys"]

        # Extract relevant information. Set empty strings if not available
        documentCount = len(keys["documents"]) if "documents" in keys else 0
        question = keys["question"] if "question" in keys else ""
        generation = "Done" if "generation" in keys else "Pending"

        # Compile final str
        final_str = "\n ====== Stage Name: " + str(stage_name) + " ======\n"
        final_str += "Question: " + str(question) + "\n"
        final_str += "Document Count: " + str(documentCount) + "\n"
        if generation != "":
            final_str += "Generation: " + str(generation)

        return final_str

    # ==========================================
    # Retrieve documents
    # ==========================================

    def retrieve_documents(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        logger = self.logger
        logger.info('--- Retrieving documents ---')
        
        # Get question
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self.vectorstore.as_retriever().get_relevant_documents(question)
        logger.info(f'Retrieved {len(documents)} documents from vector store')

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
    def generate_answer(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, answer, that contains generated answer
        """
        logger = self.logger
        llm = self.llm

        logger.info('--- Generating answer ---')
        
        # Get question
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        # Prompt
        rag_prompt = ChatPromptTemplate.from_template(prompt_templates["rag_prompt"])

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
    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, grade, that contains grade
        """
        logger = self.logger
        llm = self.llm

        logger.info('--- Grading documents - Checking relevance ---')
        logger.info (f'STATE: \n\n{str(state)}')
        
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
        logger.info(f'Grading {len(documents)} documents for relevance')
        filtered_docs = []
        search = "No"
        for idx, d in enumerate(documents):

            logger.info(f'{idx} Grading document: {str(d.metadata)}')

            score = chain.invoke({"question": question, "context": d.page_content})
            grade = score["score"]
            if grade == "yes":
                logger.info(f"{idx} Document is relevant to the question")
                filtered_docs.append(d)
            else:
                logger.info(f"{idx} Document is not relevant to the question")
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
    def transform_query(self, state):
        """
        Transform query to produce a better question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        logger = self.logger
        llm = self.llm

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
    def web_search(self, state):
        """
        Search web for relevant documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        logger = self.logger
        llm = self.llm

        logger.info('--- Performing web search ---')
        
        # Get question
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

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

    # ==========================================
    # Decide whether to generate an answer or re-generate a question
    # ==========================================
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer or re-generate a question for web search.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generate, that contains generate
        """
        logger = self.logger
        
        logger.info('--- Deciding to generate new question ---')
        
        # Get question
        state_dict = state["keys"]
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


    # ==========================================
    # Build Graph
    # ==========================================
    def build_graph(self):
        """
        Build Graph

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, graph, that contains the graph
        """
        logger = self.logger

        logger.info('--- Building Graph ---')
        
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve_documents )  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate_answer)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search", self.web_search)  # web search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
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
        self.app = app
        logger.info('--- Graph Built ---')

    def invoke(self, question):
        logger = self.logger
        state = self.state
        app = self.app

        logger.info('--- Invoking Graph ---')
        
        self.logger.info(f'Question: {question}')

        # Reset state with new question
        state = {"keys": {"question": question}}

        # Run
        for output in app.stream(state):
            print(output)
            for key, value in output.items():
                self.state = value
                self.stage_name = key
                logger.info("Currently processing node: " + key)
                logger.info(self.format_state(key, value))
        