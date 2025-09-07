import pytest
import pandas as pd
from biotagging import tag_sentence, tag_sentences, tag_from_json, tag_from_dataframe
from biotagging.schema.jsonresponse import validate_bio_output, validate_bio_input
from pydantic import ValidationError

class TestCoreFunctions:
    """Test core tagging functions"""
    
    def test_tag_sentence_string_input(self):
        result = tag_sentence("John lives in New York", ["PER", "O", "O", "LOC", "LOC"])
        expected = {
            'sentence_id': None,
            'sentence': 'John lives in New York',
            'tokens': ['John', 'lives', 'in', 'New', 'York'],
            'tags': ['B-PER', 'O', 'O', 'B-LOC', 'B-LOC']
        }
        assert result == expected
    
    def test_tag_sentence_list_input(self):
        result = tag_sentence(["John", "lives", "in", "New", "York"], ["PER", "O", "O", "LOC", "LOC"])
        expected = {
            'sentence_id': None,
            'sentence': 'John lives in New York',
            'tokens': ['John', 'lives', 'in', 'New', 'York'],
            'tags': ['B-PER', 'O', 'O', 'B-LOC', 'B-LOC']
        }
        assert result == expected
        
    def test_tag_sentence_with_id(self):
        result = tag_sentence("John works", ["PER", "O"], sentence_id=42)
        assert result['sentence_id'] == 42
        
    def test_tag_sentences_batch(self):
        batch = [
            ("John lives in New York", ["PER", "O", "O", "LOC", "LOC"]),
            (["IBM", "hired", "Mary"], ["ORG", "O", "PER"])
        ]
        results = tag_sentences(batch, start_id=100)
        
        assert len(results) == 2
        assert results[0]['sentence_id'] == 100
        assert results[1]['sentence_id'] == 101
        assert results[0]['tags'] == ['B-PER', 'O', 'O', 'B-LOC', 'B-LOC']
        assert results[1]['tags'] == ['B-ORG', 'O', 'B-PER']

class TestBIOTagNormalization:
    """Test BIO tag normalization logic"""
    
    def test_basic_bio_conversion(self):
        result = tag_sentence("John Mary", ["PER", "PER"])
        assert result['tags'] == ['B-PER', 'B-PER']
        
    def test_continuation_tags(self):
        result = tag_sentence("New York City", ["LOC", "", ""])
        assert result['tags'] == ['B-LOC', 'I-LOC', 'I-LOC']
        
    def test_explicit_bio_tags(self):
        result = tag_sentence("John Smith", ["B-PER", "I-PER"])
        assert result['tags'] == ['B-PER', 'I-PER']
        
    def test_mixed_tag_formats(self):
        result = tag_sentence("John Smith works at IBM", ["B-PER", "I-PER", "O", "O", "ORG"])
        assert result['tags'] == ['B-PER', 'I-PER', 'O', 'O', 'B-ORG']

class TestJSONConversion:
    """Test JSON input/output conversion"""
    
    def test_json_conversion_basic(self):
        json_data = [{
            "sentence_id": 0,
            "sentence": "John lives in New York",
            "tags": ["PER", "O", "O", "LOC", "LOC"]
        }]
        
        results = tag_from_json(json_data)
        assert len(results) == 1
        assert results[0]['sentence_id'] == 0
        assert results[0]['tags'] == ['B-PER', 'O', 'O', 'B-LOC', 'B-LOC']
        
    def test_json_conversion_with_token_list(self):
        json_data = [{
            "sentence_id": 1,
            "sentence": ["IBM", "hired", "Mary"],
            "tags": ["ORG", "O", "PER"]
        }]
        
        results = tag_from_json(json_data)
        assert results[0]['sentence'] == "IBM hired Mary"
        assert results[0]['tokens'] == ["IBM", "hired", "Mary"]

class TestDataFrameConversion:
    """Test pandas DataFrame input/output conversion"""
    
    def test_dataframe_conversion_basic(self):
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "lives", "tag": "O"},
            {"sentence_id": 0, "word": "in", "tag": "O"},
            {"sentence_id": 0, "word": "New", "tag": "LOC"},
            {"sentence_id": 0, "word": "York", "tag": "LOC"}
        ])
        
        results = tag_from_dataframe(df)
        assert len(results) == 1
        assert results[0]['sentence_id'] == 0
        assert results[0]['tokens'] == ["John", "lives", "in", "New", "York"]
        assert results[0]['tags'] == ['B-PER', 'O', 'O', 'B-LOC', 'B-LOC']
        
    def test_dataframe_multiple_sentences(self):
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "works", "tag": "O"},
            {"sentence_id": 1, "word": "IBM", "tag": "ORG"},
            {"sentence_id": 1, "word": "hired", "tag": "O"},
            {"sentence_id": 1, "word": "Mary", "tag": "PER"}
        ])
        
        results = tag_from_dataframe(df)
        assert len(results) == 2
        assert results[0]['sentence_id'] == 0
        assert results[1]['sentence_id'] == 1
        assert results[0]['tags'] == ['B-PER', 'O']
        assert results[1]['tags'] == ['B-ORG', 'O', 'B-PER']

class TestValidation:
    """Test Pydantic validation"""
    
    def test_valid_output_validation(self):
        valid_output = {
            "sentence_id": 0,
            "sentence": "John works",
            "tokens": ["John", "works"],
            "tags": ["B-PER", "O"]
        }
        
        validated = validate_bio_output(valid_output)
        assert validated.sentence_id == 0
        assert len(validated.tokens) == len(validated.tags)
        
    def test_invalid_output_validation(self):
        invalid_output = {
            "sentence_id": 0,
            "sentence": "John works",
            "tokens": ["John"],  # Mismatched length
            "tags": ["B-PER", "O"]
        }
        
        with pytest.raises(ValidationError):
            validate_bio_output(invalid_output)
            
    def test_valid_input_validation(self):
        valid_input = {
            "sentence": "John works",
            "tags": ["PER", "O"]
        }
        
        validated = validate_bio_input(valid_input)
        assert validated.sentence == "John works"

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="Token/tag length mismatch"):
            tag_sentence("John works", ["PER"])  # Too few tags
            
    def test_empty_sentence(self):
        with pytest.raises(ValueError):
            tag_sentence("", ["O"])
            
    def test_none_sentence(self):
        with pytest.raises(TypeError, match="must be str or list"):
            tag_sentence(None, ["O"])

if __name__ == "__main__":
    # Run basic smoke test without pytest
    print("Running basic smoke tests...")
    
    # Test core functionality
    result = tag_sentence("John works at IBM", ["PER", "O", "O", "ORG"])
    assert result['tags'] == ['B-PER', 'O', 'O', 'B-ORG']
    print("✓ Core tagging works")
    
    # Test JSON conversion
    json_data = [{"sentence": "John works", "tags": ["PER", "O"]}]
    json_result = tag_from_json(json_data)
    assert len(json_result) == 1
    print("✓ JSON conversion works")
    
    # Test DataFrame conversion
    df = pd.DataFrame([
        {"sentence_id": 0, "word": "John", "tag": "PER"},
        {"sentence_id": 0, "word": "works", "tag": "O"}
    ])
    df_result = tag_from_dataframe(df)
    assert len(df_result) == 1
    print("✓ DataFrame conversion works")
    
    print("All smoke tests passed! ✓")