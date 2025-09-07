from biotagging.schema.jsonresponse import validate_bio_output, validate_bio_input, JSONBioResponse
from pydantic import ValidationError

print("Testing validation...")

# Test valid output
valid_output = {
    "sentence_id": 0,
    "sentence": "John lives in New York", 
    "tokens": ["John", "lives", "in", "New", "York"],
    "tags": ["B-PER", "O", "O", "B-LOC", "I-LOC"]
}

try:
    validated = validate_bio_output(valid_output)
    print("✓ Valid output passed validation")
    print(f"  Type: {type(validated)}")
    print(f"  Data: {validated.model_dump()}")
except ValidationError as e:
    print(f"✗ Valid output failed: {e}")

# Test mismatched lengths (should fail)
invalid_output = {
    "sentence_id": 0,
    "sentence": "John lives in New York",
    "tokens": ["John", "lives", "in"],  # Too short
    "tags": ["B-PER", "O", "O", "B-LOC", "I-LOC"]
}

try:
    validate_bio_output(invalid_output)
    print("✗ Invalid output should have failed but didn't")
except ValidationError as e:
    print("✓ Invalid output correctly rejected")
    print(f"  Error: {e}")

# Test valid input
valid_input = {
    "sentence_id": 0,
    "sentence": "John lives in New York",
    "tags": ["PER", "O", "O", "LOC", "LOC"]
}

try:
    validated = validate_bio_input(valid_input)
    print("✓ Valid input passed validation")
    print(f"  Data: {validated.model_dump()}")
except ValidationError as e:
    print(f"✗ Valid input failed: {e}")