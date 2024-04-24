import re
import json
from typing_extensions import TypedDict
from typing import Annotated, Sequence, List, Optional
from itertools import chain

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage, AIMessage, ChatMessage, FunctionMessage, ToolMessage
from langchain.schema import Document

from .models import *


def cleanup_name(name: str) -> str:

    # Remove all non-alphanumeric characters
    name = re.sub(r"[^a-zA-Z0-9_-]", "", name)

    return name

def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references: dict[str, Any], new_references: dict[str, Any]) -> dict[str, Any]:
    print(f"Updating references: \n\t{references} \nwith new references: \n\t{new_references}")
    if not references or isinstance(references, dict):
        references = {}
        
    # if instance of dict, update
    if new_references is not None and isinstance(new_references, dict):
        references.update(new_references)
    
    return references


def update_editor(editor, new_editor):
    # Can only set at the outset
    if not editor:
        return new_editor
    return editor


def format_doc(doc: Document, max_length=1000)-> str:
    related = "- ".join(doc.metadata["categories"]) if "categories" in doc.metadata else ""
    return f"### {doc.metadata['title']}\n\nSummary: {doc.metadata['summary']}\n\nRelated\n{related}"[
        :max_length
    ]


def format_docs(docs: List[Document], max_length=1000) -> str:
    return "\n\n".join(format_doc(doc, max_length=max_length) for doc in docs)

def format_docs_basic(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_qa_response(answer: dict[str, Any]) -> str:
    context: list[Document] = answer["context"]
    ans: str = answer["answer"]
    
    doc_context = ''
    for doc in context:
        doc_context += f"###\nSource: {doc.metadata['source']}\nQuestion: {doc.metadata['query'] if 'query' in doc.metadata else ''}\nContent: {doc.page_content}\n\n"
    
    resp = f"Answer: {ans}\n\n"
    resp += f"Context: \n{doc_context}"
    return resp
    
    

def save_json_to_file(file_path: str, data: dict[str, Any]) -> None:

    print(f"==================\nSaving data to {file_path}\n\n{data}\n\n=======================")

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved data to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}, error: {e}")
        print(traceback.print_exc())

def load_json_from_file(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)

def message_to_dict(message: BaseMessage) -> dict[str, Any]:

    if isinstance(message, BaseMessage):
        return dict(
            name=message.name,
            content=message.content,
            type=message.type,
            id=message.id,
            response_metadata=message.response_metadata
        )

    print(f"Unexpected message type: {type(message)}")
    return dict()

def dict_to_message(data: dict[str, Any]|BaseMessage) -> BaseMessage:
    if isinstance(data, BaseMessage):
        return data
    
    if isinstance(data, dict):
        if type(data["type"]) == "human":
            return HumanMessage(**data)
        if type(data["type"]) == "system":
            return SystemMessage(**data)
        if type(data["type"]) == "ai":
            return AIMessage(**data)
        if type(data["type"]) == "chat":
            return ChatMessage(**data)
        if type(data["type"]) == "function":
            return FunctionMessage(**data)
        if type(data["type"]) == "tool":
            return ToolMessage(**data)
        else:
            return BaseMessage(**data)
