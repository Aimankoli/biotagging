from typing import List, Dict, Any
import pandas as pd
from ..services.api import tag_sentence

def tag_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert pandas DataFrame format to tagged output format.
    
    Input format:
    DataFrame with columns: sentence_id, word, tag
    
    Example:
        sentence_id    word      tag
        0             John      PER
        0             lives     O
        0             in        O
        0             New       LOC
        0             York      LOC
        1             IBM       ORG
        1             hired     O
        1             Mary      PER
    
    Output format:
    [
        {"sentence_id": int, "sentence": str, "tokens": [str], "tags": [str]},
        ...
    ]
    """
    results = []
    
    # Group by sentence_id
    for sentence_id, group in df.groupby('sentence_id'):
        words = group['word'].tolist()
        tags = group['tag'].tolist()
        
        result = tag_sentence(
            sentence=words,  # Pass as list of tokens
            tags=tags,
            sentence_id=sentence_id,
            repair_illegal=True
        )
        results.append(result)
    
    return results