
from .llm_tools import *
from .models import *
from .fns import *
from .storm_interview_graph import StormInterviewGraph


logger = get_logger(__name__)


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
    
    stored_chunks = store_wiki_docs_to_vectorstore(logger, interview_config=interview_config, docs=all_docs)
    logger.info(f"Stored wiki results for related topics - stored_chunks={stored_chunks}")
    
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
    graph = StormInterviewGraph(interview_config)
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


# refine outline node
@as_runnable
async def node_refine_outline(state: Interviews)-> dict[str, Any]:
    """
    Refine the outline for each editor in the interview.

    Args:
        state (Interviews): The current interviews state.

    Returns:
        Interviews: The updated interviews state with the refined outline added as a message.
    """
    print(f"\n\n***\nRefining outline for {state.topic}\n***\n\n")

    interview_config = state.interview_config

    refine_chain = get_chain_refine_outline(interview_config.fast_llm) 

    convo_str = '==========\n'
    for convo in state.conversations or []:
        if isinstance(convo, dict):
            convo = InterviewState.from_dict(convo)
            
        c = ''
        msgs = convo.messages or []
        for msg in msgs:
            c += f"{msg.name}: {msg.content}\n"
        
        convo_str += c + '\n==========\n'
    
    logger.info(f"Compiled Conversations:\n{convo_str}")

    new_outline = await refine_chain.ainvoke({"old_outline": state.outline, "conversations": convo_str, "topic": state.topic})
    state.outline = new_outline

    logger.info(f"Refined outline:\n{state.outline}")

    return state.as_dict()

async def node_generate_sections(state: Interviews) -> dict[str, Any]:
    
    state = Interviews.from_dict(state)
    section_writer_chain = get_section_writer_chain(state.interview_config.long_llm)
    outline: Outline | None = state.outline
    
    cvs = state.extract_conversations()
    logger.info(f"Extracted conversations:\n{cvs}")
    
    ol = ''
    if outline is not None:
        if isinstance(outline, dict):
            outline = Outline.from_dict(outline)
        else:
            outline = Outline.from_dict(outline.as_dict())
    
        ol = outline.as_str
        # print(f"Outline: {ol}\n\n")
    
        sections: list[Section] = outline.sections or []
        compiled_sections: List[WikiSection] = []
        
        for s in sections:
            try:
                if isinstance(s, dict):
                    s = Section.from_dict(s)
                
                logger.info(f'Generating section: {s.section_title}')
                    
                writer_input = {
                    "outline": ol, 
                    "topic": state.topic, 
                    "section": s.section_title,
                    'retriever': state.interview_config.vectorstore.as_retriever(search_kwargs={'k': 8}),
                    'research': cvs
                }
                
                section: WikiSection = await section_writer_chain.ainvoke(writer_input)
                
                logger.info(f'\n==========\nGenerated section: {section.section_title} \n{section.as_str}\n==========')
                compiled_sections += [section]
                
            except Exception as e:
                logger.error(f'Error generating section: {s.section_title}')
                logger.error(e)
    else:
        logger.warn(f'No outline found for topic: {state.topic}')
    
    state.wiki_sections = compiled_sections
    
    save_json_to_file('interviews.json', state.as_dict(with_config=False))
    
    return state.as_dict()

async def node_write_article(state: Interviews) -> dict[str, Any]:
    state = Interviews.from_dict(state)
    article_writer_chain = get_article_writer_chain(state.interview_config.long_llm)
    
    logger.info(f"Writing article for topic: {state.topic}")
    
    notes = f"{state.extract_conversations()}\n\n"
    if state.outline is not None:
        for section in state.outline.sections:
            q1 = section.description
            q2 = section.section_title
            
            try:
                n = get_notes_from_vectorstore(query=f"{q1} {q2}", vectorstore=state.interview_config.vectorstore, notes_token_limit=1000, doc_token_limit=200, k=20) + '\n===========\n'
                print(f"SectionTitle: {q2}, Notes: {n[:200]}")
                notes += n
            except Exception as e:
                logger.error(f'Error getting notes from vectorstore: SectionTitle: {q2}, error: {e}')    
    
    article_in = {
        "topic": state.topic,
        "outline": state.outline.as_str if state.outline is not None else '',
        "notes": notes,
        "draft": state.extract_draft()
    }
    
    article = await article_writer_chain.ainvoke(article_in)
    state.article = article
    
    logger.info(f"Done writing Article: {article[:200]}")
    save_json_to_file('interviews1.json', state.as_dict(with_config=False))
    
    return state.as_dict()

class StormGraph:
    def __init__(self, interview_config: InterviewConfig, topic: str):
        self.interview_config = interview_config
        self.interviews = Interviews(interview_config=interview_config, topic=topic)
        self.graph = self.build_graph()
    
    def build_graph(self):
        builder = StateGraph(Interviews)

        builder.add_node("survey_subjects", node_survey_subjects)
        builder.add_node("run_interviews",node_run_interviews)
        builder.add_node("refine_outline",node_refine_outline)
        builder.add_node("generate_sections",node_generate_sections)
        builder.add_node("write_article",node_write_article)


        builder.set_entry_point("survey_subjects")
        builder.add_edge("survey_subjects", "run_interviews")
        builder.add_edge("run_interviews", "refine_outline")
        builder.add_edge("refine_outline", "generate_sections")
        builder.add_edge("generate_sections", "write_article")
        builder.set_finish_point("write_article")
        return builder.compile().with_config(run_name="Storm Researcher")
    
