from .llm_tools import generate_chat_prompt, generate_human_chat_prompt, generate_system_chat_prompt, generate_human_message
from langchain_core.prompts import MessagesPlaceholder


def generate_messages_placeholder():
    return MessagesPlaceholder(variable_name="messages", optional=True)

# ===========================================

outline_system_wiki_writer = generate_system_chat_prompt("""
You are a Wiki writer. Write an outline for a Wiki page about a user-provided topic. Be comprehensive and specific.
""")

outline_user_topic_formatinstructions = generate_human_chat_prompt("Topic of Interest: {topic}\n\n{format_instructions}")

# ===========================================

related_subjects_human_wiki_writer = generate_human_chat_prompt("""
I'm writing a Wiki page for a topic mentioned below. Please identify and recommend some Wiki pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

Please list the as many subjects and urls as you can.

Topic of interest: {topic}

{format_instructions}
""")

# ===========================================


perspective_system_generator = generate_system_chat_prompt("""
You need to select a diverse (and distinct) group of Wiki editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\
You can use other Wiki pages of related topics for inspiration. For each editor, add a description of what they will focus on.

Wiki page outlines of related topics for inspiration:
{examples}
""")

# ===========================================


gen_question_system_generator = generate_system_chat_prompt("""
You are an experienced Wikipedia writer and want to edit a specific page. \
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
Please only ask one question at a time and don't ask what you have asked before.\
Your questions should be related to the topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

Stay true to your specific perspective:

{persona}
""")

# ===========================================

gen_queries_system_generator = generate_system_chat_prompt("""
You are a helpful research assistant. Query the search engine to answer the user's questions.

{format_instructions}
""")

# ===========================================

initial_question = generate_human_message("So you said you were writing an article on {example_topic}?")

# ===========================================

generate_answer_system_generator = generate_system_chat_prompt("""
You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants 
to write a Wikipedia page on the topic you know. You have gathered the related information and will now use the information to form a response.

Make your response as informative as possible and make sure every sentence is supported by the gathered information.
Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.

{format_instructions}
""")