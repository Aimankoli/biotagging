# Biotagging

A robust Python package for BIO entity tagging with automatic sequence repair and validation.

## Overview

Biotagging provides a comprehensive solution for processing and validating BIO (Beginning-Inside-Outside) entity tags commonly used in Named Entity Recognition (NER) tasks. The package offers automatic repair of malformed tag sequences, support for multiple input formats, and a professional command-line interface.

## Features

- **Automatic BIO Sequence Repair**: Intelligently fixes malformed tag sequences (I-tags without B-tags, mismatched classes, empty tags)
- **Multiple Input Formats**: Support for strings, token lists, JSON files, and pandas DataFrames
- **Robust Validation**: Built-in Pydantic schema validation with comprehensive error reporting
- **Unicode Support**: Full support for international text and mixed writing systems
- **High Performance**: Efficiently processes large datasets with 100k+ tokens
- **Command Line Interface**: Professional CLI for batch processing and file conversion
- **Multiple Output Formats**: JSON, CSV, and CoNLL output formats
- **Comprehensive Testing**: Over 117 tests covering edge cases and error conditions

## Installation

```bash
pip install biotagging
```

## Quick Start

### Python API

```python
from biotagging import tag_sentence, tag_sentences

# Process a single sentence
result = tag_sentence("John works at IBM", ["PER", "O", "O", "ORG"])
print(result)
# Output: {
#     "sentence_id": None,
#     "sentence": "John works at IBM", 
#     "tokens": ["John", "works", "at", "IBM"],
#     "tags": ["B-PER", "O", "O", "B-ORG"]
# }

# Process multiple sentences
batch = [
    ("John works at IBM", ["PER", "O", "O", "ORG"]),
    ("Apple hired Mary", ["ORG", "O", "PER"])
]
results = tag_sentences(batch)
```

### JSON Processing

```python
from biotagging import tag_from_json

json_data = [
    {
        "sentence": "John works at IBM",
        "tags": ["PER", "O", "O", "ORG"]
    }
]
results = tag_from_json(json_data)
```

### DataFrame Processing

```python
import pandas as pd
from biotagging import tag_from_dataframe

df = pd.DataFrame([
    {"sentence_id": 0, "word": "John", "tag": "PER"},
    {"sentence_id": 0, "word": "works", "tag": "O"},
    {"sentence_id": 0, "word": "at", "tag": "O"},
    {"sentence_id": 0, "word": "IBM", "tag": "ORG"}
])
results = tag_from_dataframe(df)
```

## Command Line Interface

### Process Single Sentence

```bash
biotagging sentence "John works at IBM" "PER O O ORG"
```

### Process JSON File

```bash
biotagging json input.json --output results.json --validate
```

### Process CSV File

```bash
biotagging csv data.csv --format conll --validate
```

### Validate Tagged Data

```bash
biotagging validate results.json
```

### Get Help

```bash
biotagging --help
biotagging sentence --help
```

## BIO Tag Repair

The package automatically repairs common BIO tagging issues:

```python
# I-tag without B-tag gets converted to B-tag
tag_sentence("John Smith", ["I-PER", "I-PER"])
# Result: ["B-PER", "I-PER"]

# Empty tags continue the previous entity
tag_sentence("New York City", ["LOC", "", ""])
# Result: ["B-LOC", "I-LOC", "I-LOC"]

# Mismatched I-tag classes get converted to B-tags
tag_sentence("John Smith", ["B-PER", "I-ORG"])
# Result: ["B-PER", "B-ORG"]
```

## Input Formats

### String Input
```python
tag_sentence("John works", ["PER", "O"])
```

### Token List Input
```python
tag_sentence(["John", "works"], ["PER", "O"])
```

### JSON Format
```python
[
    {
        "sentence_id": 1,
        "sentence": "John works at IBM",
        "tags": ["PER", "O", "O", "ORG"]
    }
]
```

### DataFrame Format
```python
# Required columns: sentence_id, word, tag
pd.DataFrame([
    {"sentence_id": 0, "word": "John", "tag": "PER"},
    {"sentence_id": 0, "word": "works", "tag": "O"}
])
```

## Output Formats

### JSON Output
```json
{
    "sentence_id": 0,
    "sentence": "John works at IBM",
    "tokens": ["John", "works", "at", "IBM"],
    "tags": ["B-PER", "O", "O", "B-ORG"]
}
```

### CoNLL Format
```
# Sentence 0
John    B-PER
works   O
at      O
IBM     B-ORG
```

## Validation

The package includes comprehensive validation using Pydantic schemas:

```python
from biotagging.schema.jsonresponse import validate_bio_output

# Validates token/tag length matching and schema compliance
try:
    validated = validate_bio_output(result)
    print("Validation passed")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Error Handling

The package provides clear error messages for common issues:

- Token/tag length mismatches
- Empty or whitespace-only sentences
- Malformed BIO sequences (when repair is disabled)
- Missing required fields in input data

## Configuration Options

### Repair Mode
```python
# Enable automatic repair (default)
tag_sentence("test", ["I-PER"], repair_illegal=True)
# Result: ["B-PER"]

# Strict mode - raise errors for illegal sequences
tag_sentence("test", ["I-PER"], repair_illegal=False)
# Raises: ValueError
```

## Performance

- Handles sentences with 100,000+ tokens efficiently
- Processes large batches with minimal memory overhead
- Unicode text processing with international character support
- Thread-safe for concurrent processing

## Requirements

- Python 3.9 or higher
- pandas >= 2.0.0
- pydantic >= 2.0.0

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/biotagging.git
cd biotagging

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest test/ -v

# Run linting
flake8 biotagging/ test/

# Build package
python -m build
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:

1. Add tests for new features
2. Follow the existing code style
3. Update documentation as needed
4. Ensure all tests pass

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- Saroop Makhija
- Aiman Koli  
- Meet Patel

## Support

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/Aimankoli/biotagging) or open an issue.

## Changelog

### Version 0.1.0
- Initial release
- Core BIO tagging functionality
- JSON and DataFrame processing
- Command line interface
- Comprehensive test suite
- Automatic sequence repair
- Multiple output formats