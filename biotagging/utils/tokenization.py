from typing import List

def whitespace_tokenize(text: str) -> List[str]:
    if text is None:
        raise ValueError("sentence cannot be None")
    # Basic whitespace tokenizer for baseline. No punctuation splitting.
    tokens = text.split()
    if any(t == "" for t in tokens):
        raise ValueError("Sentence contains empty tokens after split()")
    return tokens

def align_or_error(tokens: List[str], tags: List[str]) -> None:
    if len(tokens) != len(tags):
        raise ValueError(
            f"Token/tag length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
        )

def delimeter_tokenize(text: str, delimeter: str) -> List[str]:
    if text is None:
        raise ValueError("sentence cannot be None")
    # Basic whitespace tokenizer for baseline. No punctuation splitting.
    tokens = text.split(f"{delimeter}")
    if any(t == "" for t in tokens):
        raise ValueError("Sentence contains empty tokens after split()")
    return tokens