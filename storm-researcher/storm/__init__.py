import os, logging
from datetime import datetime
from distro import name
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file by default
load_dotenv(find_dotenv())

os.environ["LANGCHAIN_PROJECT"] = 'STORM_RESEARCHER'

from .models import *
from .fns import *
from .llm_tools import *

from . import prompts

# ==========================================
# Global tools and variables
# ==========================================

MAX_INTERVIEW_QUESTIONS = 3
TAGS_TO_EXTRACT = [ "p", "h1", "h2", "h3"]

wikipedia_retriever = get_wikipedia_retriever()


# ==========================================
# Setup logging
# ==========================================

# Log file with current date
log_file = f'./logs/storm_log_{datetime.now().strftime("%Y%m%d")}.log'

# Create parent directory for log file and create an empty file if it doesn't exist
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    open(log_file, 'a').close()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)

# Console logger
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# Colsole level to WARN ONLY
consoleHandler.setLevel(logging.INFO)


# Add file logger
fileHandler = logging.FileHandler(log_file)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

# Set level
logger.setLevel(logging.INFO)

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

def get_refine_outline_chain(fast_llm):
    refine_outline_prompt = get_chat_prompt_from_prompt_templates([prompts.pmt_s_refine_outline, prompts.pmt_h_refine_outline])
    outline_parser = get_pydantic_parser(pydantic_object=Outline)
    
    return get_chain_with_outputparser(refine_outline_prompt, fast_llm, outline_parser)\
        .with_config(run_name="RefineOutline")
    
    

def get_chain_question_generator(fast_llm):
    gen_qn_prompt = get_chat_prompt_from_prompt_templates([prompts.gen_question_system_generator, prompts.generate_messages_placeholder()])
    
    gn_chain = (
            gen_qn_prompt
            | fast_llm
        )
    
    
    return gn_chain



# ==========================================
# Interview Graph
# ==========================================

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
    editor: Editor = state.editor
    interview_config = state.interview_config
    fast_llm = interview_config.fast_llm


    logger.info(f'Generating question for {name}')
    gn_chain = c = get_chain_question_generator(fast_llm)
    input = {"persona": editor.persona}
    
    ai_response = await gn_chain.ainvoke(input)
    
    # Convert AI response to HumanMessage to simulate human conversation
    tag_with_name(ai_response, 'SubjectEditor')
    message = HumanMessage(**ai_response.dict(exclude={"type"}))
    
    state.messages.append(message)
    state.messagesQA.append(message)

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
    config = state.interview_config
    fast_llm = config.fast_llm
    
    # Chain definitions
    gen_answer_chain = get_chain_answer(fast_llm)
    queries_chain = get_chain_queries(fast_llm)
    
    # Normalize name
    name = cleanup_name(editor.name)
    # editor.name = name

    logger.info(f'START - Generate answers for [{name}]')
    
    # Generate search engine queries
    
    q_in = {"messages": state.messagesQA}
    queries:Queries = await queries_chain.ainvoke(q_in)
    logger.info(f"Got {len(queries.queries)} search engine queries for [{name}] -\n\t {queries.queries}")


    # Run search engine on all generated queries using tool
    query_results = await search_engine.abatch(queries.queries, config.runnable_config, return_exceptions=True)
    successful_results = [res for res in query_results if not isinstance(res, Exception)]
    all_query_results = {res["url"]: res["content"] for results in successful_results for res in results}
    dumped_successful_results = json.dumps(all_query_results)
    
    # Extract refence objects
    raw_references:list[Reference] = []
    for url, content in all_query_results.items():
        r = Reference(url=url, title=content)
        raw_references.append(r)

    # Fill in reference objects
    # full_references = await fetch_pages_from_refs(raw_references)
    # full_references_map = {ref.url: ref for ref in full_references}
    
    
    logger.info(f"Got {len(successful_results)} search engine results for [{name}] - \n\t {all_query_results}")
    logger.info(f"Dumped {len(dumped_successful_results)} characters for [{name}] - \n\t {dumped_successful_results}")

    # # Append Questions from Wikipedia and Answers from the search engine
    ai_message_for_queries: AIMessage = get_ai_message(json.dumps(queries.as_dict()))    
    tool_results_message = generate_human_message(dumped_successful_results)
    
    logger.debug(f"QUERY_AI_MSG: {ai_message_for_queries} for [{name}]")
    logger.debug(f"RESULTS_H_MSG: {tool_results_message} for [{name}]")
    state.messages.append(ai_message_for_queries)
    state.messages.append(tool_results_message)
    
    # Only update the shared state with the final answer to avoid polluting the dialogue history with intermediate messages
    try:
        generated: AnswerWithCitations = await gen_answer_chain.ainvoke(state.as_dict())
        logger.info(f"Genreted final answer {generated} for [{name}] - \n\t {generated.as_str}")

    except Exception as e:
        logger.error(f"Error generating answer for [{name}] - {e}")
        generated = AnswerWithCitations(answer="", cited_urls=[])
    
    cited_urls = set(generated.cited_urls)
    
    # Update references with cited references - Save the retrieved information to a the shared state for future reference
    cited_references = {r.url: r for r in raw_references if r.url in cited_urls}
    state.references = {**state.references, **cited_references}
    
    
    # # Add message to shared state
    formatted_message = AIMessage(name=name, content=generated.as_str)
    state.messages.append(formatted_message)
    state.messagesQA.append(formatted_message)
    
    
    logger.info(f'END - generate answer for [{name}]')    
    return state.as_dict()


# Route messages node for interview
def node_route_messages(state_dict: dict):
    
    print(f'Routing messages: {state_dict}')
    
    state = InterviewState.from_dict(state_dict)

    editor = state.editor
    config = state.interview_config
    name = cleanup_name(editor.name)

    print(f'Routing messages for [{name}]')

    messages = state.messages
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    if num_responses >= config.max_conversations:
        return END
    
    last_question = messages[-2]
    last_question_content = str(last_question.content if last_question.content else "")
    if last_question_content.endswith("Thank you so much for your help!"):
        return END
    
    print(f'Continue asking question for [{name}] as this is not the last end of the conversation')
    return "ask_question"

class StormInterviewGraph1:
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
    
    async def run_single_interview(self, state: InterviewState):
        return await self.graph.ainvoke(state)
    
    async def stream_and_return_results(self, state) -> dict[str, Any]:
        async for step in self.graph.astream(state):
            name = next(iter(step))
            print(name)
            print(f"Processing step: {name}")
            print("-- ", str(step[name]["messages"])[:300])
            if END in step:
                final_step = step
                
        final_state = next(iter(final_step.values()))
        return final_state
    

# ==========================================
# Research Graph
# ==========================================

# Outline node
@as_runnable
async def node_generate_outline(interview: ResearchState) -> dict[str, Any]:
    
    
    interview_config = interview.interview_config
    fast_llm = interview_config.fast_llm
    topic = interview.topic
    
    logger.info(f"Generating outline for topic: {topic}")
    
    # Get outline chain
    outline_chain = get_chain_outline(fast_llm)
    
    # Generate Outline
    outline = await outline_chain.ainvoke({"topic": topic})
    
    # Add outline to interview
    interview.outline = outline
    interview.initial_outline = outline
    
    return interview.to_dict()


# Perspectives node
@as_runnable
async def node_generate_perspectives(interview: ResearchState) -> dict[str, Any]:
    
    logger.info(f"Generating perspectives for topic: {interview.topic}")
    
    # Setup
    interview_config = interview.interview_config
    fast_llm = interview_config.fast_llm
    topic = interview.topic
    
    expand_chain = get_chain_expand_related_topics(fast_llm)
    wikipedia_retriever = get_wikipedia_retriever()
    gen_perspectives_chain = get_chain_perspective_generator(fast_llm)
    
    # Generate perspectives
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    interview.related_subjects = related_subjects
    logger.info(f"Related Subjects for [{topic}]: {related_subjects}")
    
    
    retrieved_docs = await wikipedia_retriever.abatch(related_subjects.topics, return_exceptions=True)

    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)

    logger.debug(f"Retrieved {len(all_docs)} docs for Topic: {topic}")
    
    formatted = format_docs(all_docs)
    perspectives = await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})
    
    # Cleanup Names on perspectives
    for editor in perspectives.editors:
        editor.name = cleanup_name(editor.name)
    
    interview.perspectives = perspectives
    logger.info(f"Generated {len(perspectives.editors)} perspectives for topic: [{topic}] \n\t- {[e.name for e in perspectives.editors]}")
    
    
    # Initialize conversations
    interview.initialize_conversations()
    logger.info(f"Initialized {len(interview.conversations)} conversations.")
    
    return interview.to_dict()


# Refine outline
@as_runnable
async def node_refine_outline(interview: ResearchState) -> dict[str, Any]:
    
    # Setup
    interview_config = interview.interview_config
    fast_llm = interview_config.fast_llm
    topic = interview.topic
    initial_outline: Outline = interview.initial_outline # type: ignore
    
    refine_outline_chain = get_refine_outline_chain(fast_llm)
    
    # Compile messages
    messages = []
    for conversation in interview.conversations.values():
        messages.extend(conversation.messages)
    
    refine_inpute = {
        "topic": topic,
        "old_outline": initial_outline.as_str,
        "conversations": "\n\n".join(
            f"### {m.name}\n\n{m.content}" for m in messages
        ),
    }
    
    refined_outline = await refine_outline_chain.ainvoke(refine_inpute)
    interview.outline = refined_outline
    
    return interview.to_dict()

# Interviews node
@as_runnable
async def node_interviews(interview: ResearchState) -> dict[str, Any]:
    
    logger.info(f"Running interviews for topic: {interview.topic}")
    
    # Setup
    interview_config = interview.interview_config
    
    interview_graph = StormInterviewGraph1(interview_config)
    
    # Run convos
    input = [i.as_dict() for i in interview.conversations.values()]
    
    logger.info(f"Running {len(input)} conversations.\n\t Sample - {input[0]}")
    
    results: list[dict[str, Any]] = await interview_graph.graph.abatch(input)
    
    for result in results:
        i = InterviewState.from_dict(result)        
        interview.conversations[i.editor.name] = i
        
    logger.info(f"Finished running interviews for topic: {interview.topic}")
    
    return interview.to_dict()

class StormResearchGraph:
    def __init__(self, interview_config: InterviewConfig, topic: str):
        self.topic = topic
        self.interview_config = interview_config
        self.graph = self.build_graph()
        
    def build_graph(self):
        builder = StateGraph(ResearchState)

        builder.add_node("generate_outline", node_generate_outline)
        builder.add_node("generate_perspectives",node_generate_perspectives)
        builder.add_node("run_interviews", node_interviews)
        
        # Refine outline
        # Index_references
        # Write sections
        # Write_article
        
        builder.set_entry_point("generate_outline")
        builder.add_edge("generate_outline", "generate_perspectives")
        builder.add_edge("generate_perspectives", "run_interviews")
        builder.set_finish_point("run_interviews")
        return builder.compile().with_config(run_name="Research")
    
    async def stream_and_return_results(self, state):
        async for step in self.graph.astream(state):
            name = next(iter(step))
            print(name)
            print(f"Processing step: {name}")
            # print("-- ", str(step[name]["messages"])[:300])
            if END in step:
                final_step = step
                
        final_state = next(iter(final_step.values()))
        return final_state