from dataclasses import field
import pprint
import traceback
from attr import dataclass
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Any, Dict, List, Optional, Type
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, BaseMessage
from typing import Annotated, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig

import json, re

from prometheus_client import Summary

from .fns import add_messages, update_references, update_editor, message_to_dict, dict_to_message

# ==============================================================================
# ==============================================================================

class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()
    
    def as_dict(self) -> dict:
        return {"subsection_title": self.subsection_title, "description": self.description}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subsection":
        return cls(**data)


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
    
    def as_dict(self) -> dict:
        ss = []
        for s in self.subsections or []:
            ss.append({"subsection_title": s.subsection_title, "description": s.description})
        return {"section_title": self.section_title, "description": self.description, "subsections": ss}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Section":
        if isinstance(data, Section):
            data = data.as_dict()
        
        ss = []
        for s in data["subsections"]:
            ss.append(Subsection.from_dict(s))
        return Section(section_title=data["section_title"], description=data["description"], subsections=ss)


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
    
    def as_dict(self) -> dict:
        ss = []
        for s in self.sections:
            if isinstance(s, Section):
                ss.append(s.as_dict())
            else:
                ss.append(s)
        return {"page_title": self.page_title, "sections": self.sections}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Outline":
        if isinstance(data, Outline):
            data = data.as_dict()
            
        ss = []
        for s in data["sections"]:
            ss.append(Section.from_dict(s))
        return Outline(page_title=data["page_title"], sections=ss)
    

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
        if isinstance(data, RelatedSubjects):
            data = data.as_dict()
        
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
    context: str
    interview_config: InterviewConfig
    editor: Editor
    messages: list[BaseMessage] = []
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

        m = self.messages if self.messages else None
        m1 = []
        if m:
            for v in m:
                if isinstance(v, BaseMessage):
                    m1.append(message_to_dict(v))
                else:
                    m1.append(v)

        return {
            "context": self.context,
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

                m = data["messages"] if "messages" in data else None
                m1: list[BaseMessage] = []
                if m and isinstance(m, list):
                    m1 = [dict_to_message(m1) for m1 in m]

                e = data["editor"] if "editor" in data else None
                e1: Editor = Editor.from_dict(e) if isinstance(e, dict) else e

                cf = data["interview_config"] if "interview_config" in data else None
                cf1: InterviewConfig = InterviewConfig.from_dict(cf) if isinstance(cf, dict) else cf

                return cls(
                    context=data["context"],
                    interview_config=cf1,
                    editor=e1,
                    messages=m1,
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

class WikiSubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()
    
    def as_dict(self) -> dict:
        return {"subsection_title": self.subsection_title, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        if isinstance(data, WikiSubSection):
            data = data.as_dict()
            
        return cls(
            subsection_title=data["subsection_title"],
            content=data["content"],
        )


class WikiSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[WikiSubSection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: List[str] = Field(default_factory=list, description="Citations to the sources in the content of the section.")

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        
        # pretty print subsections
        subsections = pprint.pprint(subsections)
        
        
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )
    
    def as_dict(self) -> dict:
        ss = []
        for s in self.subsections or []:            
            ss.append(WikiSubSection.from_dict(s).as_dict())

        return {
            "section_title": self.section_title,
            "content": self.content,
            "subsections": ss,
            "citations": self.citations
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WikiSection":
        if isinstance(data, WikiSection):
            data = data.as_dict()
        
        return cls(**data)



# ==============================================================================
# ==============================================================================

# class ResearchState(TypedDict):
#     topic: str
#     outline: Outline
#     editors: List[Editor]
#     interview_results: List[InterviewState]
#     # The final sections output
#     sections: List[WikiSection]
#     article: str
    
@dataclass
class Interviews:
    topic: str
    interview_config: InterviewConfig
    outline: Outline|None = None
    wiki_sections: list[WikiSection]| None = None
    related_subjects: RelatedSubjects|None = None
    related_subjects_formatted: str|None = None
    perspectives: Perspectives|None = None
    conversations: list[InterviewState]|None = None
        
    def as_dict(self, with_config: bool = True) -> dict:
        conversations = []
        for v in self.conversations if self.conversations else []:
            print(f"Handling conversation:\n\t{v}")
            s1 = v.as_dict() if isinstance(v, InterviewState) else v
            
            if not with_config and "interview_config" in s1:
                s1["interview_config"] = None

            # Check messages are converted
            for idx, m in enumerate(s1["messages"]):
                if isinstance(m, BaseMessage):
                    s1["messages"][idx] = message_to_dict(m)
            
            conversations.append(s1)


        rs = self.related_subjects if self.related_subjects else None 
        rs1 = rs.as_dict() if isinstance(rs, RelatedSubjects) else rs

        ps = self.perspectives if self.perspectives else None
        ps1 = ps.as_dict() if isinstance(ps, Perspectives) else ps
        
        o = self.outline if self.outline else None
        o1 = o.as_dict() if isinstance(o, Outline) else o
        if o1 is not None:
            for idx, s in enumerate(o1["sections"]):
                if isinstance(s, Section):
                    o1["sections"][idx] = s.as_dict()
        
        ws = self.wiki_sections if self.wiki_sections else None
        ws1 = []
        if ws:
            for v in ws:
                ws1.append(WikiSection.from_dict(v).as_dict())
                
        return {
            "topic": self.topic,
            "outline": o1,
            "wiki_sections": ws1,
            "related_subjects": rs1,
            "related_subjects_formatted": self.related_subjects_formatted,
            "interview_config": self.interview_config if with_config else None,
            "perspectives": ps1,
            "conversations": conversations
        }
        
    # GetItem
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    # in
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Interviews":
        if isinstance(data, Interviews):
            print("Interviews.from_dict: data is an instance of Interviews")
            
            # Make sure data is in correct format
            data.wiki_sections = [WikiSection.from_dict(v) for v in data["wiki_sections"]] if "wiki_sections" in data else None
            data.outline = Outline.from_dict(data["outline"]) if "outline" in data else None
            data.perspectives = Perspectives.from_dict(data["perspectives"]) if "perspectives" in data else None
            data.related_subjects = RelatedSubjects.from_dict(data["related_subjects"]) if "related_subjects" in data else None
            return data

        # if instance of dict
        try:
            if isinstance(data, dict): 
                print("Interviews.from_dict: data is an instance of dict")

                cv = [
                    InterviewState.from_dict(v) for v in data["conversations"]
                ] if "conversations" in data else []
                
                ws = data["wiki_sections"] if "wiki_sections" in data else None
                if ws and isinstance(ws, list):
                    data["wiki_sections"] = [WikiSection.from_dict(v) for v in ws]

                return cls(
                    topic=data["topic"],
                    outline=Outline.from_dict(data["outline"]),
                    related_subjects=RelatedSubjects.from_dict(data["related_subjects"]),
                        # data["related_subjects"].as_dict() if "related_subjects" in data else None,
                    related_subjects_formatted=data["related_subjects_formatted"],
                    interview_config=data["interview_config"],
                    perspectives=data["perspectives"],
                    conversations=cv,
                    wiki_sections=ws
                )
        except Exception as e:
            print(f"Interviews.from_dict: error: {e}")
            traceback.print_exc()
            
        raise ValueError(f"Interviews.from_dict:\n data: {data}")
    
    # to string
    def __str__(self) -> str:
        return f"Interviews\n\ttopic={self.topic}, \n\trelated_subjects={self.related_subjects}, \n\tperspectives={self.perspectives}, \n\tconversations={self.conversations}"
    

    def extract_conversations(self) -> str:
        c = ''
        
        for convo in self.conversations or []:
            try:
                if isinstance(convo, dict):
                    convo = InterviewState.from_dict(convo)
                
                convo.editor.description
                
                msgs = ''
                for msg in convo.messages or []:
                    if not isinstance(msg, BaseMessage):
                        mgs = dict_to_message(msg)
                        
                    msgs += f"{msg.name}: {msg.content}\n"
                
                c += f"\n\n########\nConversation between Subject Matter Expert {convo.editor.name} - {convo.editor.description}\n{msgs}"
            except Exception as e:
                print(f"Error extracting conversation: {convo}\n{e}")
                traceback.print_exc()
        return c