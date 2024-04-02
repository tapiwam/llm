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

def route_messages(state: InterviewState, name: str = "SubjectMatterExpert"):

    name = cleanup_name(name)

    logger.info(f'Routing messages for [{name}]')

    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    if num_responses >= MAX_INTERVIEW_QUESTIONS:
        return END
    
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    
    logger.info(f'Continue asking question for [{name}] as this is not the last end of the conversation')
    return "ask_question"


def swap_roles(state: InterviewState, name: str) -> InterviewState:

    # Normalize name
    name = cleanup_name(name)

    print(f'Swapping roles for {name}')

    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.dict(exclude={"type"}))
        converted.append(message)
    
    print(f'Converted messages for {name} while swapping roles: {len(converted)} messages')
    state['messages'] = converted
    
    return state



class StormInterviewGraph:
    def __init__(self, fast_llm):
        self.fast_llm = fast_llm
        self.outline = get_chain_outline(fast_llm)
        self.related_topics = get_chain_expand_related_topics(fast_llm)
        self.perspective = get_chain_perspective_generator(fast_llm)
        self.queries = get_chain_queries(fast_llm)
        self.answer = get_chain_answer(fast_llm)
        
        # new runnable from swap_roles
        self.tag_with_name = RunnableLambda(lambda state: tag_with_name(state)).with_types(input_type=InterviewState, output_type=InterviewState)
        self.survey_subjects = RunnableLambda(lambda topic: self.asurvey_subjects(topic)).with_types(input_type=str, output_type=Perspectives)
        self.generate_question = RunnableLambda(lambda state: self.agenerate_question(state)).with_types(input_type=InterviewState, output_type=InterviewState)

        self.graph = self.build_graph()
        
    async def agenerate_question(self, state: InterviewState) -> InterviewState:
        gen_qn_prompt = get_chat_prompt_from_prompt_templates([prompts.gen_question_system_generator, prompts.generate_messages_placeholder()])

        editor: Editor = state["editor"]

        name = cleanup_name(editor.name)
        inputs = {"name": name, "state": state}

        logger.info(f'Generating question for {name}')

        gn_chain = (
            RunnableLambda(swap_roles).bind(name=name)
            | gen_qn_prompt.partial(persona=editor.persona)
            | self.fast_llm
            | self.tag_with_name.bind(name=name)
        )
        result:AIMessage = await gn_chain.ainvoke(state)
        state["messages"] = ([result])

        logger.info(f'Generated question for {name}')
        return state
    
    async def asurvey_subjects(self, topic: str)-> Perspectives:
        logger.info(f"Survey Subjects for Topic: {topic}")
        related_subjects = await self.related_topics.ainvoke({"topic": topic})
        retrieved_docs = await wikipedia_retriever.abatch(
            related_subjects.topics, return_exceptions=True
        )
        all_docs = []
        for docs in retrieved_docs:
            if isinstance(docs, BaseException):
                continue
            all_docs.extend(docs)
        logger.info(f"Retrieved {len(all_docs)} docs for Topic: {topic}")
        
        formatted = format_docs(all_docs)
        return await self.perspective.ainvoke({"examples": formatted, "topic": topic})

    def build_graph(self):
        builder = StateGraph(InterviewState)

        builder.add_node("ask_question", self.generate_question)
        builder.add_node("answer_question", self.answer)
        builder.add_conditional_edges("answer_question", route_messages)
        builder.add_edge("ask_question", "answer_question")

        builder.set_entry_point("ask_question")
        return builder.compile().with_config(run_name="Conduct Interviews")