import tiktoken
from .base import *
from .fns import *
from .llm_core import *
from .models import *
from . import prompts

logger = get_logger(__name__)

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


# ==========================================
# Chains
# ==========================================

# Outline chain
def get_chain_outline(fast_llm):
    outline_system_prompt = prompts.outline_system_wiki_writer
    outline_human_prompt = prompts.outline_user_topic_formatinstructions
    direct_gen_outline_prompt = get_chat_prompt_from_prompt_templates([outline_system_prompt, outline_human_prompt])

    outline_parser = get_pydantic_parser(pydantic_object=Outline)
    return get_chain_with_outputparser(direct_gen_outline_prompt, fast_llm, outline_parser)

def get_chain_expand_related_topics(fast_llm):
    related_subjects_prompt = get_chat_prompt_from_prompt_templates([prompts.related_subjects_human_wiki_writer])
    related_topics_parser = get_pydantic_parser(RelatedSubjects)
    return get_chain_with_outputparser(related_subjects_prompt, fast_llm, related_topics_parser)

def get_chain_perspective_generator(fast_llm):
    perspective_prompt = get_chat_prompt_from_prompt_templates([prompts.perspective_system_generator])
    perspective_parser = get_pydantic_parser(Perspectives)
    return get_chain_with_outputparser(perspective_prompt, fast_llm, perspective_parser)

def get_chain_queries(fast_llm):
    gen_queries_prompt = get_chat_prompt_from_prompt_templates([prompts.gen_queries_system_generator, prompts.generate_messages_placeholder()])
    queries_parser = get_pydantic_parser(Queries)
    return get_chain_with_outputparser(gen_queries_prompt, fast_llm, queries_parser)


def get_chain_answer(fast_llm):
    gen_answer_prompt = get_chat_prompt_from_prompt_templates([prompts.generate_answer_system_generator, prompts.generate_messages_placeholder()])
    ac_parser = get_pydantic_parser(pydantic_object=AnswerWithCitations)

    return get_chain_with_outputparser(gen_answer_prompt, fast_llm, ac_parser)\
        .with_config(run_name="GenerateAnswer")


def get_chain_question_generator(fast_llm):
    gen_qn_prompt = get_chat_prompt_from_prompt_templates([prompts.gen_question_system_generator, prompts.generate_messages_placeholder()])
    
    gn_chain = (
            gen_qn_prompt
            | fast_llm
        )
    
    
    return gn_chain

def get_chain_refine_outline(fast_llm):
    refine_outline_prompt = get_chat_prompt_from_prompt_templates([prompts.pmt_s_refine_outline, prompts.pmt_h_refine_outline])

    outline_parser = get_pydantic_parser(pydantic_object=Outline)
    refine_cahin = get_chain_with_outputparser(refine_outline_prompt, fast_llm, outline_parser)\
        .with_config(run_name="Refine Outline")
    
    return refine_cahin

def get_qa_rag_chain(llm, embeddings, persistent_location):
    prompt = hub.pull("rlm/rag-prompt")
    
    # rEFRESH VECTORSTORE
    vectorstore = Chroma(persist_directory=persistent_location, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_basic(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source


async def retrieve_fn(inputs: dict):
    retriever = inputs["retriever"]
    reserach = inputs["research"]
    docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
    formatted = "\n".join(
        [
            f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    
    formatted = f"<Document title=\"Expert Research Notes\">\n{reserach}\n<Document>\n{formatted}"
    return {"docs": formatted, **inputs}


def get_section_writer_chain(long_context_llm):
    section_writer_prompt = get_chat_prompt_from_prompt_templates([prompts.pmt_s_section_writer, prompts.pmt_h_section_writer])
    wiki_parser = get_pydantic_parser(WikiSection)
    fixer_parser = OutputFixingParser.from_llm(parser=wiki_parser, llm=long_context_llm)    
    
    section_writer = (
        retrieve_fn
        | section_writer_prompt.partial(format_instructions=wiki_parser.get_format_instructions())
        | long_context_llm
        | fixer_parser
    )
        
    return section_writer

def get_article_writer_chain(long_context_llm):
    writer_prompt = get_chat_prompt_from_prompt_templates([prompts.pmt_s_writer, prompts.pmt_h_writer])
    return writer_prompt | long_context_llm | StrOutputParser()



    