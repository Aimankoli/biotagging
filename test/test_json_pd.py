import pandas as pd
from biotagging import tag_from_json, tag_from_dataframe

# Test JSON input
print("Testing JSON input:")
json_data = [
    {
        "sentence_id": 0,
        "sentence": "John lives in New York",
        "tags": ["PER", "O", "O", "LOC", "LOC"]
    },
    {
        "sentence_id": 1,
        "sentence": ["IBM", "hired", "Mary"],
        "tags": ["ORG", "O", "PER"]
    }
]

json_results = tag_from_json(json_data)
for result in json_results:
    print(result)

print("\nTesting DataFrame input:")
# Test DataFrame input
df_data = pd.DataFrame([
    {"sentence_id": 0, "word": "John", "tag": "PER"},
    {"sentence_id": 0, "word": "lives", "tag": "O"},
    {"sentence_id": 0, "word": "in", "tag": "O"},
    {"sentence_id": 0, "word": "New", "tag": "LOC"},
    {"sentence_id": 0, "word": "York", "tag": "LOC"},
    {"sentence_id": 1, "word": "IBM", "tag": "ORG"},
    {"sentence_id": 1, "word": "hired", "tag": "O"},
    {"sentence_id": 1, "word": "Mary", "tag": "PER"},
])

print("DataFrame:")
print(df_data)
print("\nResults:")
df_results = tag_from_dataframe(df_data)
for result in df_results:
    print(result)