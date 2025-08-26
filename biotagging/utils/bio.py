from typing import List, Set, Tuple, Optional

def ensure_bio(
    tags: List[Optional[str]],
    repair_illegal: bool = True,
) -> Tuple[List[str], Set[str]]:
    """
    Normalizes tags to BIO with these rules:

    - 'O'            -> append 'O' and reset state.
    - '' or None     -> continuation of previous tag:
                        if inside previous entity: append 'I-<prev_cls>';
                        else: if repair_illegal: append 'O' (graceful), else raise.
    - startswith('B-') or startswith('I-'):
                      -> treat as explicit BIO (class may contain '-'! we only cut the first 2 chars).
                         'I-<cls>' mismatches or starts without a prior chunk -> B-repair if repair_illegal else raise.
    - any other non-empty string (raw class label like 'PER', 'LOC', 'GPE-STATE'):
                      -> start a new chunk with 'B-<tag>'.

    Returns (bio_tags, unique_entity_types)
    """
    bio: List[str] = []
    entities: Set[str] = set()

    prev_cls: Optional[str] = None   # last active class (without prefix)
    inside = False                   # are we inside a chunk?

    for raw in tags:
        # normalize element to a trimmed string or empty
        if raw is None:
            t = ""
        else:
            t = str(raw).strip()

        # outside marker
        if t == "O":
            bio.append("O")
            prev_cls, inside = None, False
            continue

        # continuation marker (empty)
        if t == "":
            if inside and prev_cls:
                bio.append(f"I-{prev_cls}")
            else:
                if repair_illegal:
                    bio.append("O")
                else:
                    raise ValueError("Empty continuation tag with no preceding entity")
            continue

        # explicit BIO prefixes — only the *leading* 'B-'/'I-' is special
        if t.startswith("B-"):
            cls = t[2:]
            if not cls:
                if repair_illegal:
                    bio.append("O")
                    prev_cls, inside = None, False
                else:
                    raise ValueError("Empty class after 'B-'")
            else:
                bio.append(f"B-{cls}")
                entities.add(cls)
                prev_cls, inside = cls, True
            continue

        if t.startswith("I-"):
            cls = t[2:]
            if not cls:
                if repair_illegal:
                    bio.append("O")
                    prev_cls, inside = None, False
                else:
                    raise ValueError("Empty class after 'I-'")
            else:
                if not inside or prev_cls != cls:
                    if repair_illegal:
                        bio.append(f"B-{cls}")
                        entities.add(cls)
                        prev_cls, inside = cls, True
                    else:
                        raise ValueError(
                            f"Illegal I- tag '{t}' without matching preceding chunk"
                        )
                else:
                    bio.append(f"I-{cls}")
                    entities.add(cls)
            continue

        # raw class label -> start a new chunk
        cls = t
        bio.append(f"B-{cls}")
        entities.add(cls)
        prev_cls, inside = cls, True

    return bio, entities
