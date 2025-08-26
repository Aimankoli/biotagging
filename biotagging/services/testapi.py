from biotagging import tag_sentence, tag_sentences

# 1) sentence as string
print(tag_sentence("John Cena, Bill Gates lives in New York", ["PER","","PER", "", "O", "O", "LOC",""]))

# 2) sentence already tokenized
print(tag_sentence(["John","lives","in","New","York"], ["PER","O","O","LOC","LOC"]))

# 3) batch
batch = [
    ("John lives in New York", ["PER","O","O","LOC",""]),
    (["IBM","hired","Mary"], ["ORG","O","PER"]),
]
print(tag_sentences(batch))
