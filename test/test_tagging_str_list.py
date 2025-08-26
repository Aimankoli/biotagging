from biotagging.services.api import tag_sentence, tag_sentences

# 1) sentence as string
print("hello world")
print(tag_sentence("John lives in New York", ["PER","O","O","LOC","LOC"]))

# 2) sentence already tokenized
print(tag_sentence(["John","lives","in","New","York"], ["PER","O","O","LOC","LOC"]))

# 3) batch
batch = [
    ("John lives in New York", ["PER","O","O","LOC","LOC"]),
    (["IBM","hired","Mary"], ["ORG","O","PER"]),
]
print(tag_sentences(batch, start_id=100))
