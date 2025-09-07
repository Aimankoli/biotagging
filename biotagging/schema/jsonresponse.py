from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Dict, Any

class JSONBioResponse(BaseModel):
    sentence_id: Optional[int] = Field(None, description="Optional ID for the sentence")
    sentence: Optional[str] = Field(None, description="Original sentence if provided")
    tokens: List[str] = Field(..., description="List of tokens in the sentence")
    tags: List[str] = Field(..., description="List of BIO tags corresponding to each token")

    @field_validator('tokens', 'tags')
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("Cannot be empty")
        return v

    @model_validator(mode='after')
    def check_lengths(self):
        if len(self.tokens) != len(self.tags):
            raise ValueError(f"Length of tokens ({len(self.tokens)}) and tags ({len(self.tags)}) must be the same")
        return self

class JSONBioInput(BaseModel):
    sentence_id: Optional[int] = Field(None, description="Optional ID for the sentence")
    sentence: Union[str, List[str]] = Field(..., description="Sentence as string or list of tokens")
    tags: Union[str, List[str]] = Field(..., description="Tags as space-delimited string or list")

def validate_bio_output(data: Dict[str, Any]) -> JSONBioResponse:
    """Validate output data against BIO response schema"""
    return JSONBioResponse(**data)

def validate_bio_input(data: Dict[str, Any]) -> JSONBioInput:
    """Validate input data against BIO input schema"""  
    return JSONBioInput(**data)