'''
json structure:

[
{sentence_id: int, sentence: str or [str], tags: [str] },
{sentence_id: int, sentence: str or [str], tags: [str] },
...
]

convert to:
[

{"sentence_id": int, "sentence": str, "tokens": [str], "tags": [str]},

...
]


'''