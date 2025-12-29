from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """
    Request model for the chat endpoint.
    """
    query: str = Field(..., min_length=1, description="The research question to ask the RAG system.")

class Source(BaseModel):
    """
    Represents a source document cited in the answer.
    """
    filename: str = Field(..., description="The name of the source file.")
    page_number: Optional[int] = Field(None, description="The page number in the source file, if available.")

class AnswerResponse(BaseModel):
    """
    Response model containing the generated answer and citations.
    """
    answer: str = Field(..., description="The generated answer based on the context.")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer.")

class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    detail: str = Field(..., description="Error message details.")
