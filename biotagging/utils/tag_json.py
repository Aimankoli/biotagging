from typing import List, Dict, Any, Union
from ..services.api import tag_sentence

def tag_from_json(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert JSON input format to tagged output format.
    
    Input format:
    [
        {"sentence_id": int, "sentence": str or [str], "tags": [str]},
        ...
    ]
    
    Output format:
    [
        {"sentence_id": int, "sentence": str, "tokens": [str], "tags": [str]},
        ...
    ]
    """
    results = []
    
    for item in data:
        sentence_id = item.get("sentence_id")
        sentence = item["sentence"]
        tags = item["tags"]
        
        result = tag_sentence(
            sentence=sentence,
            tags=tags,
            sentence_id=sentence_id,
            repair_illegal=True
        )
        results.append(result)
    
    return results