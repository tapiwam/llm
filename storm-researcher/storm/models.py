from dataclasses import field
import traceback
from attr import dataclass
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Any, Dict, List, Optional, Type
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig

import json, re

from prometheus_client import Summary

from .fns import add_messages, update_references, update_editor

# ==============================================================================
# ==============================================================================

class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wiki page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wiki page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wiki page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()

# ==============================================================================
# ==============================================================================

class RelatedSubjects(BaseModel):
    topics: List[str] = Field(
        description="Comprehensive list of related subjects as background research.",
    )

    def as_dict(self) -> dict:
        return {"topics": self.topics}
    
    # from dict
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelatedSubjects":
        return cls(**data)


# ==============================================================================
# ==============================================================================

class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(
        description="Name of the editor.",
    )
    role: str = Field(
        description="Role of the editor in the context of the topic.",
    )
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )
    
    #hash
    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "affiliation": self.affiliation,
            "description": self.description,
        }

    # from dict
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Editor":
        return Editor(**data)
    

class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )

    def as_dict(self) -> dict:
        eds = [ed.as_dict() for ed in self.editors] if self.editors else []
        return {"editors": eds}

    # from dict
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Perspectives":
        editors: list[Editor] = [Editor.from_dict(ed) for ed in data["editors"]] if "editors" in data else []
        return Perspectives(editors=editors)


# ==============================================================================
# ==============================================================================

@dataclass
class Reference:
    url: str
    title: str|None = None
    summary: str|None = None
    html: str|None = None
    md: str|None = None
    txt: str|None = None

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "html": self.html,
            "md": self.md,
            "txt": self.txt
        }

    # from dict
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reference":
        return Reference(**data)


@dataclass
class InterviewConfig:
    long_llm: BaseChatModel
    fast_llm: BaseChatModel
    max_conversations: int = 5
    max_reference_length: int = 10000
    tags_to_extract: List[str] = []
    embeddings: Any = None
    vectorstore_dir: str = "./data/storm/vectorstore/"
    vectorstore: Any = None,
    interview_graph: Any = None
    runnable_config: RunnableConfig|None = None
    
    def as_dict(self) -> dict:
        return {
            "long_llm": self.long_llm,
            "fast_llm": self.fast_llm,
            "max_conversations": self.max_conversations,
            "max_reference_length": self.max_reference_length,
            "tags_to_extract": self.tags_to_extract,
            "embeddings": self.embeddings,
            "vectorstore_dir": self.vectorstore_dir,
            "vectorstore": self.vectorstore,
            "interview_graph": self.interview_graph,
            "runnable_config": self.runnable_config
        }
    
    # from dict
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterviewConfig":
        return InterviewConfig(**data)
    

@dataclass
class InterviewState:
    interview_config: InterviewConfig
    editor: Editor
    messages: list[AnyMessage] = []
    references: dict[str, Reference] = {}
    
    # as dict
    def as_dict(self, with_config: bool = True) -> dict:

        e = self.editor if self.editor else None
        e1 = e.as_dict() if isinstance(e, Editor) else e

        ic = self.interview_config if self.interview_config else None
        ic1 = ic.as_dict() if isinstance(ic, InterviewConfig) else ic

        refs = self.references if self.references else None
        refs1 = dict()
        if refs:
            for k, v in refs.items():
                refs1[k] = v.as_dict() if isinstance(v, Reference) else v

        return {
            "interview_config": ic1,
            "editor": e1,
            "messages": self.messages,
            "references": self.references
        }
    
    # from dict
    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        if isinstance(data, InterviewState):
            print("InterviewState.from_dict: data is an instance of InterviewState")
            return data

        # if instance of dict
        try:
            if isinstance(data, dict): 
                print("InterviewState.from_dict: data is an instance of dict")

                return cls(
                    interview_config=data["interview_config"],
                    editor=data["editor"],
                    messages=data["messages"] if "messages" in data else [],
                    references=data["references"] if "references" in data else {}
                )
        except Exception as e:
            print(f"InterviewState.from_dict: error: {e}")
            traceback.print_exc()
            
        raise ValueError(f"InterviewState.from_dict: \n data: {data}")
        
    
    def trim_messages(self, max_messages: int|None = None, max_characters: int|None = None) -> None:
        # trim messages to max_messages
        if max_messages is not None and len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
            print(f"Truncated messages to {max_messages} for msgName:{self.editor.name}")
        
        # trim characters to max_characters
        if max_characters is not None:
            for i,message in enumerate(self.messages):
                if len(message.content) > max_characters:
                    message.content = message.content[-max_characters:]
                    print(f"Truncated message {i}/{len(self.messages)} to {max_characters} characters for msgName:{self.editor.name}")
            
    
@dataclass
class Interviews:
    topic: str
    interview_config: InterviewConfig
    related_subjects: RelatedSubjects|None = None
    related_subjects_formatted: str|None = None
    perspectives: Perspectives|None = None
    conversations: dict[str, tuple[Editor, InterviewState]]|None = None
        
    def as_dict(self, with_config: bool = True) -> dict:
        conversations = {}
        for k, v in self.conversations.items() if self.conversations else {}:
            e: Editor = v[0]
            e1 = e.as_dict() if isinstance(e, Editor) else e
            s: InterviewState = v[1]
            s1 = s.as_dict() if isinstance(s, InterviewState) else s
            
            if not with_config:
                s1["interview_config"] = None
            
            conversations[k] = (e1, s1)


        rs = self.related_subjects if self.related_subjects else None 
        rs1 = rs.as_dict() if isinstance(rs, RelatedSubjects) else rs

        ps = self.perspectives if self.perspectives else None
        ps1 = ps.as_dict() if isinstance(ps, Perspectives) else ps

        return {
            "topic": self.topic,
            "related_subjects": rs1,
            "related_subjects_formatted": self.related_subjects_formatted,
            "interview_config": self.interview_config if with_config else None,
            "perspectives": ps1,
            "conversations": conversations
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        if isinstance(data, Interviews):
            print("Interviews.from_dict: data is an instance of Interviews")
            return data

        # if instance of dict
        try:
            if isinstance(data, dict): 
                print("Interviews.from_dict: data is an instance of dict")

                cv = {
                    k: (Editor.from_dict(v[0]), InterviewState.from_dict(v[1])) for k, v in data["conversations"].items()
                } if "conversations" in data else None

                return cls(
                    topic=data["topic"],
                    related_subjects=RelatedSubjects.from_dict(data["related_subjects"]),
                        # data["related_subjects"].as_dict() if "related_subjects" in data else None,
                    related_subjects_formatted=data["related_subjects_formatted"],
                    interview_config=data["interview_config"],
                    perspectives=data["perspectives"],
                    conversations=cv
                )
        except Exception as e:
            print(f"Interviews.from_dict: error: {e}")
            traceback.print_exc()
            
        raise ValueError(f"Interviews.from_dict:\n data: {data}")
    
    # to string
    def __str__(self) -> str:
        return f"Interviews\n\ttopic={self.topic}, \n\trelated_subjects={self.related_subjects}, \n\tperspectives={self.perspectives}, \n\tconversations={self.conversations}"
    


# ==============================================================================
# ==============================================================================

class Queries(BaseModel):
    queries: List[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's questions.",
    )
    
    # To dict
    def as_dict(self) -> dict:
        return {
            "queries": self.queries
        }

# ==============================================================================
# ==============================================================================

class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="Comprehensive answer to the user's question with citations.",
    )
    cited_urls: List[str] = Field(
        description="List of urls cited in the answer.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )

# ==============================================================================
# ==============================================================================

class SubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class WikiSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: List[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


# ==============================================================================
# ==============================================================================

class ResearchState(TypedDict):
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[InterviewState]
    # The final sections output
    sections: List[WikiSection]
    article: str
    

