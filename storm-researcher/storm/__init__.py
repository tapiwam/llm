from logging import config
import os, logging
from datetime import datetime
import pprint
from dotenv import load_dotenv, find_dotenv
from pydantic import InstanceOf

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


def get_chain_question_generator(fast_llm):
    gen_qn_prompt = get_chat_prompt_from_prompt_templates([prompts.gen_question_system_generator, prompts.generate_messages_placeholder()])
    
    gn_chain = (
            gen_qn_prompt
            | fast_llm
        )
    
    
    return gn_chain

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


# ==========================================
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
    
    stored_chunks = store_docs_to_vectorstore(logger, config, docs=successful_results)
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


# ==========================================
# ==========================================
# Main Interviews Chain

@as_runnable
async def node_survey_subjects(state: Interviews)-> dict[str, Any]:

    topic = state.topic
    interview_config = state.interview_config
    fast_llm = interview_config.fast_llm

    print(f"\n-- Survey Subjects for Topic: [{topic}] --\n")

    # Define chains
    expand_chain = get_chain_expand_related_topics(fast_llm=fast_llm)
    gen_perspectives_chain = get_chain_perspective_generator(fast_llm)
    outline_chain = get_chain_outline(fast_llm)
    wikipedia_retriever = get_wikipedia_retriever()
    
    # Generate initial outline
    o_input = {"topic": topic}
    outline: Outline = await outline_chain.ainvoke({"topic": topic})
    print(f"Initial Outline: {outline.page_title} - {outline.sections}")
    
    state.outline = outline

    # Get related topics
    related_subjects: RelatedSubjects = await expand_chain.ainvoke(o_input)
    print(f"Related Subjects: {related_subjects.topics}")


    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs: list[Document] = []

    print(f"Retrieved {len(retrieved_docs)} wiki batches for Topic: {topic}:\n")
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        
        for doc in docs:
            # Add summary to doc page content
            doc.page_content = f"{doc.metadata['title']}\n\n{doc.metadata['summary']}\n\n{doc.page_content}"
            all_docs.append(doc)
            print(f"\tRetrieved doc: {doc.metadata['title']} - {doc.metadata['source']}")

    print(f"Retrieved {len(all_docs)} docs for Topic: {topic}\n")
    
    formatted = format_docs(all_docs, max_length=1000)

    # Generate perspectives
    perspectives: Perspectives = await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})
    print(f"Generated {len(perspectives.editors)} perspectives for Topic: [{topic}]\n")

    # Generate conversations
    conversations = list()
    if perspectives is not None and perspectives.editors is not None and len(perspectives.editors) > 0:
        for editor in perspectives.editors:
            convo = InterviewState(context=topic,interview_config=interview_config, editor=editor, messages=[], references={})
            conversations.append(convo)

            print(f">> Generated perspective for: {editor.name} \nAffiliation: - {editor.affiliation}\nPersona: - {editor.persona}\nTopic: - {topic}\n")
    
    # Update state
    state.related_subjects = related_subjects
    state.related_subjects_formatted = formatted
    state.perspectives = perspectives
    state.conversations = conversations

    save_json_to_file('interviews.json', state.as_dict(with_config=False))
    print(f"Interviews saved to interviews.json")

    print(f"\n-- Generated {len(perspectives.editors)} perspectives for Topic: [{topic}] --\n")

    return state.as_dict()


@as_runnable
async def node_run_interviews(state: Interviews)-> dict[str, Any]:

    # return state.as_dict()

    conversations = state.conversations or {}
    interview_config = state.interview_config

    # print(f"\n\n***\n{state.conversations}\n***\n\n")

    # Define interview grapgh
    graph = StormInterviewGraph1(interview_config)
    interview_config.interview_graph = graph

    # Run interviews
    responses = []
    for idx, c in enumerate(conversations):
        convo = InterviewState.from_dict(c) if isinstance(c, dict) else c
        editor = convo.editor
        editor = Editor.from_dict(editor) if isinstance(editor, dict) else editor

        print(f"\n\n===============\nRunning interview [{idx+1}/{len(conversations)}] for {editor.name} - {editor.persona}\n\n")
        try:
            response = await graph.run_single_interview(convo)
            # responses.append((editor, response))

            r = InterviewState.from_dict(response)
            r.interview_config = interview_config

            conversations[idx] = r
            print(f"{r}")
        except Exception as e:
            print(f"Error running interview for {editor.name}: {e}")
            print(traceback.format_exc())
            continue

        print(f"===============\nFinished interview for {editor.name} - {editor.persona}\n\n")

    # Save responses to file
    save_json_to_file('interviews.json', state.as_dict(with_config=False))
    print(f"Interviews saved to interviews.json")

    return state.as_dict()


async def node_generate_outline(state: Interviews) -> dict[str, Any]:
    """
    Generates an outline for the given topic.

    Args:
        state (Interviews): The interviews state.

    Returns:
        Interviews: The updated interviews state with the generated outline added
    """
    topic = state.topic
    interview_config = state.interview_config
    interview_config = interview_config if isinstance(interview_config, InterviewConfig) else InterviewConfig.from_dict(interview_config)
    
    fast_llm = interview_config.fast_llm
    
    logger.info(f'Generating outline for topic {topic}')
    
    outline_chain = get_chain_outline(interview_config.fast_llm)
    o_input = {"topic": topic}
    
    
    
    

class StormGraph:
    def __init__(self, interview_config: InterviewConfig, topic: str):
        self.interview_config = interview_config
        self.interviews = Interviews(interview_config=interview_config, topic=topic)
        self.graph = self.build_graph()
    
    def build_graph(self):
        builder = StateGraph(Interviews)

        builder.add_node("survey_subjects", node_survey_subjects)
        builder.add_node("run_interviews",node_run_interviews)

        builder.add_edge("survey_subjects", "run_interviews")

        builder.set_entry_point("survey_subjects")
        builder.set_finish_point("run_interviews")
        return builder.compile().with_config(run_name="Storm Researcher")
    

# ==========================================
# ==========================================


# def route_messages(state: InterviewState, name: str = "SubjectMatterExpert"):

#     name = cleanup_name(name)

#     logger.info(f'Routing messages for [{name}]')

#     messages = state["messages"]
#     num_responses = len(
#         [m for m in messages if isinstance(m, AIMessage) and m.name == name]
#     )

#     if num_responses >= MAX_INTERVIEW_QUESTIONS:
#         return END
    
#     last_question = messages[-2]
#     if last_question.content.endswith("Thank you so much for your help!"):
#         return END
    
#     logger.info(f'Continue asking question for [{name}] as this is not the last end of the conversation')
#     return "ask_question"


# def swap_roles(state: InterviewState, name: str) -> InterviewState:

#     # Normalize name
#     name = cleanup_name(name)

#     print(f'Swapping roles for {name}')

#     converted = []
#     for message in state["messages"]:
#         if isinstance(message, AIMessage) and message.name != name:
#             message = HumanMessage(**message.dict(exclude={"type"}))
#         converted.append(message)
    
#     print(f'Converted messages for {name} while swapping roles: {len(converted)} messages')
#     state['messages'] = converted
    
#     return state



# class StormInterviewGraph:
#     def __init__(self, fast_llm):
#         self.fast_llm = fast_llm
#         self.outline = get_chain_outline(fast_llm)
#         self.related_topics = get_chain_expand_related_topics(fast_llm)
#         self.perspective = get_chain_perspective_generator(fast_llm)
#         self.queries = get_chain_queries(fast_llm)
#         self.answer = get_chain_answer(fast_llm)
        
#         # new runnable from swap_roles
#         self.tag_with_name = RunnableLambda(lambda state: tag_with_name(state)).with_types(input_type=InterviewState, output_type=InterviewState)
#         self.survey_subjects = RunnableLambda(lambda topic: self.asurvey_subjects(topic)).with_types(input_type=str, output_type=Perspectives)
#         self.generate_question = RunnableLambda(lambda state: self.agenerate_question(state)).with_types(input_type=InterviewState, output_type=InterviewState)

#         self.graph = self.build_graph()
        
#     async def agenerate_question(self, state: InterviewState) -> InterviewState:
#         gen_qn_prompt = get_chat_prompt_from_prompt_templates([prompts.gen_question_system_generator, prompts.generate_messages_placeholder()])

#         editor: Editor = state["editor"]

#         name = cleanup_name(editor.name)
#         inputs = {"name": name, "state": state}

#         logger.info(f'Generating question for {name}')

#         gn_chain = (
#             RunnableLambda(swap_roles).bind(name=name)
#             | gen_qn_prompt.partial(persona=editor.persona)
#             | self.fast_llm
#             | self.tag_with_name.bind(name=name)
#         )
#         result:AIMessage = await gn_chain.ainvoke(state)
#         state["messages"] = ([result])

#         logger.info(f'Generated question for {name}')
#         return state
    
#     async def asurvey_subjects(self, topic: str)-> Perspectives:
#         logger.info(f"Survey Subjects for Topic: {topic}")
#         related_subjects = await self.related_topics.ainvoke({"topic": topic})
#         retrieved_docs = await wikipedia_retriever.abatch(
#             related_subjects.topics, return_exceptions=True
#         )
#         all_docs = []
#         for docs in retrieved_docs:
#             if isinstance(docs, BaseException):
#                 continue
#             all_docs.extend(docs)
#         logger.info(f"Retrieved {len(all_docs)} docs for Topic: {topic}")
        
#         formatted = format_docs(all_docs)
#         return await self.perspective.ainvoke({"examples": formatted, "topic": topic})

#     def build_graph(self):
#         builder = StateGraph(InterviewState)

#         builder.add_node("ask_question", self.generate_question)
#         builder.add_node("answer_question", self.answer)
#         builder.add_conditional_edges("answer_question", route_messages)
#         builder.add_edge("ask_question", "answer_question")

#         builder.set_entry_point("ask_question")
#         return builder.compile().with_config(run_name="Conduct Interviews")
    
