from .llm_tools import *
from .models import *
from .fns import *
from . import prompts

logger = get_logger(__name__)


# Question node 
@as_runnable
async def node_generate_question(state: InterviewState) -> dict[str, Any]:
    """
    Generates a question for the editor in the interview.

    Args:
        state (InterviewState): The interview state.

    Returns:
        InterviewState: The updated interview state with the generated question added as a message.
    """
    context = state.context
    editor: Editor = state.editor
    editor = editor if isinstance(editor, Editor) else Editor.from_dict(editor)

    interview_config = state.interview_config
    interview_config = state.interview_config if isinstance(interview_config, InterviewConfig) else InterviewConfig.from_dict(interview_config)
    fast_llm = interview_config.fast_llm

    # Normalize name
    name = cleanup_name(editor.name)
    editor.name = name
    
    messages = state.messages


    logger.info(f'Generating question for {name}')
    gn_chain = c = get_chain_question_generator(fast_llm)
    input = {"persona": editor.persona, "context": context, "messages": messages}
    
    ai_response = await gn_chain.ainvoke(input)
    
    # Convert AI response to HumanMessage to simulate human conversation
    tag_with_name(ai_response, name)
    message = HumanMessage(**ai_response.dict(exclude={"type"}))
    
    state.messages.append(message)

    logger.info(f'Generated question for {name}: {message.content}')
    return state.as_dict()


# Answer node
@as_runnable
async def node_generate_answer(state: InterviewState) -> dict[str, Any]:
    """
    Generates an answer for the editor's question in the interview.
    
    Args:
        state (InterviewState): The interview state.
    
    Returns:
        InterviewState: The updated interview state with the generated answer added as a message.
    """
    
    editor: Editor = state.editor
    editor = editor if isinstance(editor, Editor) else Editor.from_dict(editor)
    name = cleanup_name(editor.name)
    
    config = state.interview_config
    config = config if isinstance(config, InterviewConfig) else InterviewConfig.from_dict(config)
    fast_llm = config.fast_llm
    
    # last message from state.messages
    last_message = state.messages[-1] if len(state.messages) > 0 else {}
    last_message = dict_to_message(last_message)
    
    # Chain definitions
    gen_answer_chain = get_chain_answer(fast_llm)
    queries_chain = get_chain_queries(fast_llm)
    
    logger.info(f'START - Generate answer for [{name}] - Question: [{last_message.content}]')
    
    # Generate search engine queries
    
    q_in = {"messages": state.messages}
    queries:Queries = await queries_chain.ainvoke(q_in)
    logger.info(f"Got {len(queries.queries)} search engine queries for [{name}] -\n\t {queries.queries}")


    # Run search engine on all generated queries using tool and add to vector store
    query_results = await search_engine.abatch(queries.queries, config.runnable_config, return_exceptions=True)
    # pprint.pprint(f"\n\nQuery Results: \n{query_results}\n\n")

    # zip query with results
    for idx, q in enumerate(queries.queries):
        if isinstance(query_results[idx], Exception):
            logger.error(f"Error running search engine for [{name}]: {q} - {query_results[idx]}")
        else:
            for res in query_results[idx]:
                res["query"] = q
    
    successful_results = [res for res in query_results if not isinstance(res, Exception)]
    
    stored_chunks = store_docs_to_vectorstore(logger, config, docs=successful_results, chunk_size=1500, chunk_overlap=50)
    logger.info(f"Got {len(successful_results)} search engine results for [{name}] - stored_chunks={stored_chunks}")
    

    # QA Chain
    qa_chain = get_qa_rag_chain(fast_llm, config.embeddings, config.vectorstore_dir)
    answer_raw = await qa_chain.ainvoke(last_message.content)
    
    logger.info(f"Got answer from QA chain for [{name}]: \n\tQuestion: {last_message.content} \n\tRaw Answer: {answer_raw}")
    
    # Add answer to shared state
        
    # Only update the shared state with the final answer to avoid polluting the dialogue history with intermediate messages
    try:
        logger.info(f"Generating final answer for [{name}] - \n\t {answer_raw}")
        answer_msg = AIMessage(name="SearchEngine", content=format_qa_response(answer_raw))
        qa_messages:list[BaseMessage] = [last_message, answer_msg]
        qa_state = InterviewState(
            context=state.context,
            interview_config=config,
            editor=editor,
            messages=qa_messages,
            references=state.references)
        
        generated: AnswerWithCitations = await gen_answer_chain.ainvoke(qa_state.as_dict())
        logger.info(f"Genreted final answer {generated} for [{name}] - \n\t {generated.as_str}")

    except Exception as e:
        logger.error(f"Error generating answer for [{name}] - {e}")
        logger.exception(traceback.format_exc())
        
        generated = AnswerWithCitations(answer="", cited_urls=[])
    
    cited_urls = set(generated.cited_urls)
    
    # Update references with cited references - Save the retrieved information to a the shared state for future reference
    raw_answer_docs: list[Document] = answer_raw['context']
    
    cited_references = {doc.metadata['source']: doc.metadata['query'] if 'query' in doc.metadata else '' for doc in raw_answer_docs if doc.metadata['source'] in cited_urls}
    state.references = {**state.references, **cited_references}
    
    
    # # Add message to shared state
    formatted_message = AIMessage(name=name, content=generated.as_str)
    state.messages.append(formatted_message)
    
    logger.info(f'END - generate answer for [{name}]')    
    return state.as_dict()


# Route messages
def node_route_messages(state_dict: dict):
    
    # print(f'Routing messages: {state_dict}')
    
    state = InterviewState.from_dict(state_dict)

    editor = state.editor
    editor = editor if isinstance(editor, Editor) else Editor.from_dict(editor)

    config = state.interview_config
    config = config if isinstance(config, InterviewConfig) else InterviewConfig.from_dict(config)

    name = cleanup_name(editor.name)

    logger.info(f'Routing messages for [{name}]')

    messages = state.messages
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    if num_responses >= config.max_conversations:
        logger.info(f'Reached max number of responses for [{name}] - ResponseCount: {num_responses}')
        return END
    
    last_question = messages[-2]
    last_question_content = str(last_question.content if last_question.content else "")
    if "thank you so much" in last_question_content.lower():
        logger.info(f'Last question for [{name}] was a thank you - ResponseCount: {num_responses}')
        return END
    
    logger.info(f'Continue asking question for [{name}] as this is not the last end of the conversation - ResponseCount: {num_responses} of {config.max_conversations}')
    return "ask_question"


class StormInterviewGraph:
    def __init__(self, interview_config: InterviewConfig):
        self.interview_config = interview_config
        self.graph = self.build_graph()
        
    def build_graph(self):
        builder = StateGraph(InterviewState)

        builder.add_node("ask_question", node_generate_question)
        builder.add_node("answer_question",node_generate_answer)
        builder.add_conditional_edges("answer_question", node_route_messages)
        builder.add_edge("ask_question", "answer_question")

        builder.set_entry_point("ask_question")
        return builder.compile().with_config(run_name="Conduct Interviews")
    
    async def run_single_interview(self, state: InterviewState) -> dict[str, Any]:
        return await self.graph.ainvoke(state.as_dict())

    

    async def stream_and_return_results(self, state):
        async for step in self.graph.astream(state):
            name = next(iter(step))
            print(name)
            print(f"Processing step: {name}")
            print("-- ", str(step[name]["messages"])[:300])
            if END in step:
                final_step = step
                
        final_state = next(iter(final_step.values()))
        return final_state



