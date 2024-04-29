from .llm_core import *



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

Topic of interest: ```{topic}```

Wiki page outlines of related topics for inspiration:
```
{examples}
```

{format_instructions}
""")

# ===========================================


gen_question_system_generator = generate_system_chat_prompt("""
You are an experienced Wikipedia writer and want to edit a specific page. \
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
Please only ask one question at a time and don't ask what you have asked before.\
Your questions should be related to the context and topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

Context of interest: 
```{context}```


Stay true to your specific perspective. Expert perspective:
```
{persona}
```
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

Make your response as informative as possible with as many details as possible and make sure every sentence is supported by the gathered information.
Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.

{format_instructions}
""")

# ===========================================

pmt_s_refine_outline = generate_system_chat_prompt("""
You are a Wiki writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
You need to make sure that the outline is comprehensive and specific. \
Topic you are writing about: {topic} 

Topic of interest: ```{topic}```
                                                   
Old outline:
```{old_outline}```
""")

pmt_h_refine_outline = generate_human_chat_prompt("""
Please refine the outline based on your conversations with subject-matter experts. 
Ensure descriptions are as clear and concise as possible and include as many details as possible from the gathered information:

Conversations begining with '[CONVO]' and '[END_CONVO]':

[CONVO]
{conversations}
[END_CONVO]


{format_instructions}
""")

# ===========================================

pmt_h_section_writer = generate_human_chat_prompt("""
Write the full WikiSection for the `{section}` section. Include as many details as possible from the gathered information.

Outline:
```
{outline}
```

Documents:
```
<Documents>
{docs}
<Documents>
```

Topic of interest: {topic}

Please return only the required results and include as many details as possible from the gathered information:

{format_instructions}
""")

pmt_s_section_writer = generate_system_chat_prompt("""
You are an expert Wikipedia writer. Complete your assigned WikiSection from the provided outline and include as many details as possible from the gathered information.
""")

# ===========================================

pmt_s_writer = generate_system_chat_prompt("""
You are an expert Wikipedia author. Write the complete wiki article on the provided topic using the provided notes and drafts. 
Include as many details as possible from the gathered information.
Organize citations using footnotes like "[1]","" avoiding duplicates in the footer. Include URLs in the footer.
Strictly follow Wikipedia format guidelines.
""")

pmt_h_writer = generate_human_chat_prompt("""
Write the complete Wiki article using markdown format. 

Topic of interest: ```{topic}```

Outline:
```
{outline}
```

Drafts:
```
{draft}
```

Notes:
```
{notes}
```
""")

# ===========================================

pmt_s_page_summarizer = generate_system_chat_prompt("""
You are an expert wiki writer. Write a summary of the Wikipedia page on {topic} using the following section drafts:

{draft}

Strictly follow Wikipedia format guidelines.
""")

# pmt_h_page_summarizer = generate_human_chat_prompt("""
# Write a summary of the Wikipedia page on {topic} using the following section drafts: