import re
import json
from typing_extensions import TypedDict
from typing import Annotated, Sequence, List, Optional
from itertools import chain

from langchain_core.messages import AnyMessage
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
    related = "- ".join(doc.metadata["categories"])
    return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[
        :max_length
    ]


def format_docs(docs: List[Document], max_length=500) -> str:
    return "\n\n".join(format_doc(doc, max_length=max_length) for doc in docs)


def save_json_to_file(file_path: str, data: dict[str, Any]) -> None:

    print(f"==================\nSaving data to {file_path}\n\n{data}\n\n=======================")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {file_path}")

def load_json_from_file(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)
    
