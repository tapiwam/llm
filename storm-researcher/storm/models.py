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


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()

# ==============================================================================
# ==============================================================================

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


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )


# ==============================================================================
# ==============================================================================

@dataclass
class Reference:
    url: str = Field(..., description="URL of the reference")
    title: str|None = Field(None, description="Title of the reference")
    summary: str|None = Field(None, description="Summary of the reference")
    html: str|None = Field(None, description="HTML content of the reference")
    md: str|None = Field(None, description="Markdown content of the reference")
    txt: str|None = Field(None, description="Text content of the reference")


@dataclass
class InterviewConfig:
    long_llm: BaseChatModel = Field(..., description="Long context language model")
    fast_llm: BaseChatModel = Field(..., description="Fast context language model")
    max_conversations: int = 5
    max_reference_length: int = 10000
    tags_to_extract: List[str] = Field(default_factory=list, description="List of tags to extract from the web page")
    embeddings: Any = None
    vectorstore_dir: str = Field("./data/storm/vectorstore/", description="Directory to store the vector store")
    vectorstore: Any = None,
    runnable_config: Optional[RunnableConfig] = None
    
    def as_dict(self) -> dict:
        return {
            "long_llm": self.long_llm,
            "fast_llm": self.fast_llm,
            "max_conversations": self.max_conversations,
            "max_reference_length": self.max_reference_length,
            "tags_to_extract": self.tags_to_extract
        }
    

@dataclass
class InterviewState:
    interview_config: InterviewConfig = Field(..., description="Configuration for the interview")
    editor: Editor = Field(..., description="Editor for the interview")
    messages: list[AnyMessage] = Field(default_factory=list, description="List of messages for the conversation")
    references: dict[str, Reference] = Field(default_factory=dict, description="List of references for the interview") # Annotated[Optional[dict], update_references]    
    summary: str = Field("", description="Summary of the interview")
    
    # as dict
    def as_dict(self) -> dict:
        return {
            "interview_config": self.interview_config,
            "editor": self.editor,
            "messages": self.messages,
            "references": self.references,
            "summary": self.summary
        }
    
    # from dict
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            interview_config=data["interview_config"],
            editor=data["editor"],
            messages=data["messages"],
            references=data["references"],
            summary=data["summary"]
        )
    
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
    interview_config: InterviewConfig = Field(..., description="Configuration for the interview")
    perspectives: Perspectives|None = Field(..., description="List of perspectives for the interviews")
    conversations: dict[Editor, InterviewState] = Field(default_factory=dict, description="List of conversations for the interview") #Annotated[Dict[Editor, List[AnyMessage]], Field(default_factory=dict)]
    


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
    

