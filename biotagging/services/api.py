from typing import Any, Dict, List, Tuple, Union

from ..utils.tokenization import whitespace_tokenize, align_or_error
from ..utils.bio import ensure_bio

SentenceInput = Union[str, List[str]]
TagsInput = Union[str, List[str]]
Item = Tuple[SentenceInput, List[str]]

def _normalize_sentence_and_tags(
    sentence: SentenceInput, tags: TagsInput
) -> tuple[list[str], list[str], str]:
    # Normalize tags
    if isinstance(tags, str):
        tags_list = tags.split()
    else:
        tags_list = list(tags)

    # Normalize sentence to tokens + recover original string
    if isinstance(sentence, str):
        tokens = whitespace_tokenize(sentence)
        original = sentence
    elif isinstance(sentence, list):
        tokens = [str(t) for t in sentence]
        original = " ".join(tokens)
    else:
        raise TypeError("`sentence` must be str or list[str]")

    align_or_error(tokens, tags_list)
    return tokens, tags_list, original

def tag_sentence(
    sentence: SentenceInput,
    tags: TagsInput,
    sentence_id: Union[int, str, None] = None,
    *,
    repair_illegal: bool = True,
) -> Dict[str, Any]:
    """
    Tag a single sentence. Supports sentence as str or list[str].
    Tags may be space-delimited str or list[str]. Returns a canonical dict.
    """
    tokens, raw_tags, original = _normalize_sentence_and_tags(sentence, tags)
    bio_tags, entities = ensure_bio(raw_tags, repair_illegal=repair_illegal)
    return {
        "sentence_id": sentence_id,
        "sentence": original,
        "tokens": tokens,
        "tags": bio_tags,
    }

def tag_sentences(
    items: List[Item],
    *,
    start_id: int = 0,
    repair_illegal: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch tag sentences. Each item is:
      (sentence: str | list[str], tags: list[str])
    Returns list of canonical dicts with sentence_id starting at start_id.
    """
    results: List[Dict[str, Any]] = []
    for i, (sentence, tags) in enumerate(items, start=start_id):
        try:
            results.append(
                tag_sentence(sentence, tags, sentence_id=i, repair_illegal=repair_illegal)
            )
        except Exception as e:
            # Re-raise with context; don't silently swallow bad rows
            raise type(e)(f"[index {i}] {e}") from e
    return results
