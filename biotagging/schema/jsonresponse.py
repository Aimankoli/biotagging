from pydantic import BaseModel, Field
from typing import List, Optional

class JSONBioResponse(BaseModel):
    sentence_id: Optional[int] = Field(None, description="Optional ID for the sentence")
    sentence: Optional[str] = Field(None, description="Original sentence if provided")
    tokens: List[str] = Field(..., description="List of tokens in the sentence")
    tags: List[str] = Field(..., description="List of BIO tags corresponding to each token")


   # Validator is deprecated 
    
    # @validator('tokens', 'tags')
    # def not_empty(cls, v, field):
    #     if not v:
    #         raise ValueError(f"{field.name} cannot be empty")
    #     return v

    # @root_validator
    # def check_lengths(cls, values):
    #     tokens = values.get('tokens')
    #     tags = values.get('tags')
    #     if tokens and tags and len(tokens) != len(tags):
    #         raise ValueError("Length of tokens and tags must be the same")
    #     return values