import pytest
import pandas as pd
import numpy as np
from biotagging import tag_sentence, tag_sentences, tag_from_json, tag_from_dataframe
from biotagging.schema.jsonresponse import validate_bio_output, validate_bio_input
from pydantic import ValidationError

class TestBIOTagEdgeCases:
    """Comprehensive BIO tag edge case testing"""
    
    def test_illegal_i_tag_without_b_repair_on(self):
        """I-tag without preceding B-tag should convert to B-tag when repair_illegal=True"""
        result = tag_sentence("John works at IBM", ["I-PER", "O", "O", "I-ORG"], repair_illegal=True)
        assert result['tags'] == ['B-PER', 'O', 'O', 'B-ORG']
    
    def test_illegal_i_tag_without_b_repair_off(self):
        """I-tag without preceding B-tag should raise error when repair_illegal=False"""
        with pytest.raises(ValueError, match="Illegal I- tag"):
            tag_sentence("John works", ["I-PER", "O"], repair_illegal=False)
    
    def test_empty_tags_in_sequence(self):
        """Empty tags should continue the previous entity"""
        result = tag_sentence("New York City", ["LOC", "", ""], repair_illegal=True)
        assert result['tags'] == ['B-LOC', 'I-LOC', 'I-LOC']
    
    def test_empty_tags_at_start(self):
        """Empty tag at start should become O when repair_illegal=True"""
        result = tag_sentence("The big city", ["", "O", "LOC"], repair_illegal=True)
        assert result['tags'] == ['O', 'O', 'B-LOC']
    
    def test_empty_tags_at_start_no_repair(self):
        """Empty tag at start should raise error when repair_illegal=False"""
        with pytest.raises(ValueError, match="Empty continuation tag"):
            tag_sentence("The city", ["", "LOC"], repair_illegal=False)
    
    def test_malformed_b_tag_empty_class(self):
        """B- tag with empty class"""
        result = tag_sentence("John works", ["B-", "O"], repair_illegal=True)
        assert result['tags'] == ['O', 'O']
    
    def test_malformed_i_tag_empty_class(self):
        """I- tag with empty class"""
        result = tag_sentence("John works", ["B-PER", "I-"], repair_illegal=True)
        assert result['tags'] == ['B-PER', 'O']
    
    def test_malformed_tags_no_repair(self):
        """Malformed tags should raise error when repair_illegal=False"""
        with pytest.raises(ValueError, match="Empty class"):
            tag_sentence("John", ["B-"], repair_illegal=False)
    
    def test_mismatched_i_tag_class(self):
        """I-tag with different class than preceding B-tag"""
        result = tag_sentence("John Smith", ["B-PER", "I-ORG"], repair_illegal=True)
        assert result['tags'] == ['B-PER', 'B-ORG']  # Should repair to B-ORG
    
    def test_complex_entity_names(self):
        """Entity names with hyphens, underscores, numbers"""
        result = tag_sentence("X-123 GPE-STATE company", ["PRODUCT-TYPE", "GPE-STATE", "ORG-COMPANY"])
        assert result['tags'] == ['B-PRODUCT-TYPE', 'B-GPE-STATE', 'B-ORG-COMPANY']
    
    def test_unicode_entity_names(self):
        """Unicode characters in entity names"""
        result = tag_sentence("Café München", ["PLACE-名前", "PLACE-名前"])
        assert result['tags'] == ['B-PLACE-名前', 'B-PLACE-名前']
    
    def test_all_o_tags(self):
        """All O tags"""
        result = tag_sentence("The quick brown fox", ["O", "O", "O", "O"])
        assert result['tags'] == ['O', 'O', 'O', 'O']
    
    def test_all_same_entity_separate(self):
        """All same entity type but separate instances"""
        result = tag_sentence("John Mary Bob", ["PER", "PER", "PER"])
        assert result['tags'] == ['B-PER', 'B-PER', 'B-PER']
    
    def test_all_same_entity_continuous(self):
        """All same entity type as one continuous entity"""
        result = tag_sentence("John Mary Bob", ["PER", "", ""])
        assert result['tags'] == ['B-PER', 'I-PER', 'I-PER']
    
    def test_very_long_entity_sequence(self):
        """Very long entity sequence"""
        tokens = ["Token"] * 100
        tags = ["ENT"] + [""] * 99  # First B-ENT, then 99 continuations
        result = tag_sentence(tokens, tags)
        expected = ['B-ENT'] + ['I-ENT'] * 99
        assert result['tags'] == expected

class TestInputFormatEdgeCases:
    """Edge cases for different input formats"""
    
    def test_single_character_tokens(self):
        """Single character tokens"""
        result = tag_sentence("J. R. R. Tolkien", ["PER", "PER", "PER", "PER"])  # 4 tokens = 4 tags
        assert len(result['tokens']) == 4
        assert result['tokens'] == ["J.", "R.", "R.", "Tolkien"]
    
    def test_empty_string_input(self):
        """Empty string should raise error"""
        with pytest.raises(ValueError):
            tag_sentence("", [])
    
    def test_whitespace_only_string(self):
        """String with only whitespace should raise error"""
        with pytest.raises(ValueError):
            tag_sentence("   ", [])
    
    def test_multiple_consecutive_spaces(self):
        """Multiple spaces between tokens"""
        result = tag_sentence("John    works     at    IBM", ["PER", "O", "O", "ORG"])
        assert result['tokens'] == ["John", "works", "at", "IBM"]
        assert result['sentence'] == "John    works     at    IBM"  # Original preserved
    
    def test_leading_trailing_whitespace(self):
        """Leading and trailing whitespace"""
        result = tag_sentence("  John works  ", ["PER", "O"])
        assert result['tokens'] == ["John", "works"]
        assert result['sentence'] == "  John works  "  # Original preserved
    
    def test_unicode_tokens(self):
        """Unicode characters in tokens"""
        result = tag_sentence("Café in München", ["PLACE", "O", "PLACE"])
        assert result['tokens'] == ["Café", "in", "München"]
    
    def test_numbers_as_tokens(self):
        """Numeric tokens"""
        result = tag_sentence("Room 123 on Floor 45", ["O", "NUM", "O", "O", "NUM"])
        assert result['tokens'] == ["Room", "123", "on", "Floor", "45"]
    
    def test_mixed_punctuation(self):
        """Tokens with punctuation"""
        result = tag_sentence("U.S.A. vs. U.K.", ["GPE", "O", "GPE"])
        assert result['tokens'] == ["U.S.A.", "vs.", "U.K."]
    
    def test_very_long_sentence(self):
        """Very long sentence (1000+ tokens)"""
        long_tokens = [f"token{i}" for i in range(1000)]
        long_tags = ["O"] * 1000
        long_sentence = " ".join(long_tokens)
        
        result = tag_sentence(long_sentence, long_tags)
        assert len(result['tokens']) == 1000
        assert len(result['tags']) == 1000
    
    def test_extreme_length_mismatch(self):
        """Extreme length mismatch should give clear error"""
        with pytest.raises(ValueError, match="Token/tag length mismatch: 3 tokens vs 1000 tags"):
            tag_sentence("just three tokens", ["O"] * 1000)

class TestJSONEdgeCases:
    """JSON format edge cases"""
    
    def test_missing_required_sentence_field(self):
        """Missing sentence field should raise KeyError"""
        json_data = [{"sentence_id": 0, "tags": ["O"]}]  # Missing sentence
        with pytest.raises(KeyError):
            tag_from_json(json_data)
    
    def test_missing_required_tags_field(self):
        """Missing tags field should raise KeyError"""
        json_data = [{"sentence_id": 0, "sentence": "test"}]  # Missing tags
        with pytest.raises(KeyError):
            tag_from_json(json_data)
    
    def test_missing_optional_sentence_id(self):
        """Missing sentence_id should work (it's optional)"""
        json_data = [{"sentence": "John works", "tags": ["PER", "O"]}]
        results = tag_from_json(json_data)
        assert results[0]['sentence_id'] is None
    
    def test_mixed_sentence_formats(self):
        """Mix of string and list sentence formats"""
        json_data = [
            {"sentence": "John works", "tags": ["PER", "O"]},  # String
            {"sentence": ["IBM", "hired", "Mary"], "tags": ["ORG", "O", "PER"]}  # List
        ]
        results = tag_from_json(json_data)
        assert len(results) == 2
        assert results[0]['sentence'] == "John works"
        assert results[1]['sentence'] == "IBM hired Mary"
    
    def test_empty_json_list(self):
        """Empty JSON list"""
        results = tag_from_json([])
        assert results == []
    
    def test_single_item_json(self):
        """Single item JSON"""
        json_data = [{"sentence": "test", "tags": ["O"]}]
        results = tag_from_json(json_data)
        assert len(results) == 1
    
    def test_large_json_batch(self):
        """Large JSON batch (1000 items)"""
        json_data = []
        for i in range(1000):
            json_data.append({
                "sentence_id": i,
                "sentence": f"Token{i} works",
                "tags": ["ENT", "O"]
            })
        
        results = tag_from_json(json_data)
        assert len(results) == 1000
        assert results[0]['sentence_id'] == 0
        assert results[999]['sentence_id'] == 999
    
    def test_json_with_none_values(self):
        """JSON with None values in non-required fields"""
        json_data = [{"sentence_id": None, "sentence": "test", "tags": ["O"]}]
        results = tag_from_json(json_data)
        assert results[0]['sentence_id'] is None

class TestDataFrameEdgeCases:
    """DataFrame format edge cases"""
    
    def test_empty_dataframe(self):
        """Empty DataFrame"""
        df = pd.DataFrame(columns=['sentence_id', 'word', 'tag'])
        results = tag_from_dataframe(df)
        assert results == []
    
    def test_single_word_dataframe(self):
        """Single word DataFrame"""
        df = pd.DataFrame([{"sentence_id": 0, "word": "test", "tag": "O"}])
        results = tag_from_dataframe(df)
        assert len(results) == 1
        assert results[0]['tokens'] == ["test"]
    
    def test_missing_sentence_id_column(self):
        """Missing sentence_id column should raise KeyError"""
        df = pd.DataFrame([{"word": "test", "tag": "O"}])
        with pytest.raises(KeyError):
            tag_from_dataframe(df)
    
    def test_missing_word_column(self):
        """Missing word column should raise KeyError"""
        df = pd.DataFrame([{"sentence_id": 0, "tag": "O"}])
        with pytest.raises(KeyError):
            tag_from_dataframe(df)
    
    def test_missing_tag_column(self):
        """Missing tag column should raise KeyError"""  
        df = pd.DataFrame([{"sentence_id": 0, "word": "test"}])
        with pytest.raises(KeyError):
            tag_from_dataframe(df)
    
    def test_non_contiguous_sentence_ids(self):
        """Non-contiguous sentence IDs"""
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "works", "tag": "O"},
            {"sentence_id": 5, "word": "IBM", "tag": "ORG"},  # Jump from 0 to 5
            {"sentence_id": 5, "word": "hired", "tag": "O"},
            {"sentence_id": 10, "word": "Mary", "tag": "PER"}  # Jump to 10
        ])
        results = tag_from_dataframe(df)
        assert len(results) == 3  # Should have 3 groups
        sentence_ids = [r['sentence_id'] for r in results]
        assert set(sentence_ids) == {0, 5, 10}
    
    def test_unordered_sentence_ids(self):
        """Unordered sentence IDs in DataFrame"""
        df = pd.DataFrame([
            {"sentence_id": 1, "word": "IBM", "tag": "ORG"},
            {"sentence_id": 0, "word": "John", "tag": "PER"},  # Out of order
            {"sentence_id": 1, "word": "hired", "tag": "O"},
            {"sentence_id": 0, "word": "works", "tag": "O"}   # Out of order
        ])
        results = tag_from_dataframe(df)
        assert len(results) == 2
        # Results should be ordered by sentence_id due to groupby
        assert results[0]['sentence_id'] == 0
        assert results[1]['sentence_id'] == 1
    
    def test_duplicate_sentence_ids_same_position(self):
        """Duplicate rows (same sentence_id, word, tag)"""
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "John", "tag": "PER"},  # Duplicate
            {"sentence_id": 0, "word": "works", "tag": "O"}
        ])
        results = tag_from_dataframe(df)
        # Should process duplicates as-is
        assert results[0]['tokens'] == ["John", "John", "works"]
    
    def test_nan_values_in_dataframe(self):
        """NaN values in DataFrame"""
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": np.nan, "tag": "O"},  # NaN word
            {"sentence_id": 0, "word": "works", "tag": np.nan}  # NaN tag
        ])
        # Should handle NaN by converting to string
        results = tag_from_dataframe(df)
        assert "nan" in results[0]['tokens']  # NaN becomes "nan"
    
    def test_mixed_data_types_in_columns(self):
        """Mixed data types in columns"""
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "Room", "tag": "O"},
            {"sentence_id": 0, "word": 123, "tag": "NUM"},  # Integer as word
            {"sentence_id": 0, "word": "Floor", "tag": 45.5}  # Float as tag
        ])
        results = tag_from_dataframe(df)
        # Should convert to strings and process as BIO tags
        assert "123" in results[0]['tokens']
        assert "B-45.5" in results[0]['tags']  # 45.5 becomes B-45.5 entity tag
    
    def test_extra_columns_in_dataframe(self):
        """Extra columns should be ignored"""
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER", "extra_col": "ignore"},
            {"sentence_id": 0, "word": "works", "tag": "O", "another_extra": 123}
        ])
        results = tag_from_dataframe(df)
        assert len(results) == 1
        assert results[0]['tokens'] == ["John", "works"]
    
    def test_large_dataframe(self):
        """Large DataFrame (10000 rows, 1000 sentences)"""
        data = []
        for sent_id in range(1000):
            for word_idx in range(10):  # 10 words per sentence
                data.append({
                    "sentence_id": sent_id,
                    "word": f"word_{sent_id}_{word_idx}",
                    "tag": "O" if word_idx % 2 == 0 else "ENT"
                })
        
        df = pd.DataFrame(data)
        results = tag_from_dataframe(df)
        assert len(results) == 1000
        assert all(len(r['tokens']) == 10 for r in results)

class TestValidationEdgeCases:
    """Pydantic validation edge cases"""
    
    def test_validation_with_very_long_lists(self):
        """Validation with very long token/tag lists"""
        long_data = {
            "sentence_id": 0,
            "sentence": " ".join([f"token{i}" for i in range(1000)]),
            "tokens": [f"token{i}" for i in range(1000)],
            "tags": ["O"] * 1000
        }
        validated = validate_bio_output(long_data)
        assert len(validated.tokens) == 1000
    
    def test_validation_empty_lists_should_fail(self):
        """Empty token/tag lists should fail validation"""
        with pytest.raises(ValidationError, match="Cannot be empty"):
            validate_bio_output({
                "sentence_id": 0,
                "sentence": "",
                "tokens": [],  # Empty
                "tags": []     # Empty
            })
    
    def test_validation_unicode_in_tokens(self):
        """Unicode characters in validation"""
        unicode_data = {
            "sentence_id": 0,
            "sentence": "Café München 北京",
            "tokens": ["Café", "München", "北京"],
            "tags": ["O", "LOC", "LOC"]
        }
        validated = validate_bio_output(unicode_data)
        assert validated.tokens == ["Café", "München", "北京"]
    
    def test_validation_numeric_sentence_id(self):
        """Very large sentence IDs"""
        data = {
            "sentence_id": 999999999,
            "sentence": "test",
            "tokens": ["test"],
            "tags": ["O"]
        }
        validated = validate_bio_output(data)
        assert validated.sentence_id == 999999999
    
    def test_validation_negative_sentence_id(self):
        """Negative sentence ID should work"""
        data = {
            "sentence_id": -1,
            "sentence": "test", 
            "tokens": ["test"],
            "tags": ["O"]
        }
        validated = validate_bio_output(data)
        assert validated.sentence_id == -1

class TestErrorHandlingEdgeCases:
    """Error handling and recovery edge cases"""
    
    def test_batch_processing_with_mixed_errors(self):
        """Batch processing where some items fail and some succeed"""
        batch = [
            ("John works", ["PER", "O"]),  # Should succeed
            ("", []),  # Should fail - empty string
            ("Mary hired", ["PER", "O"])   # Should succeed
        ]
        
        # Should fail on the second item and give clear error message
        with pytest.raises(ValueError) as exc_info:
            tag_sentences(batch)
        
        assert "[index 1]" in str(exc_info.value)  # Should indicate which item failed
        assert "cannot be empty" in str(exc_info.value)  # Should mention empty string error
    
    def test_memory_stress_large_entity_names(self):
        """Very long entity names"""
        very_long_entity = "A" * 1000  # 1000 character entity name
        result = tag_sentence("test", [very_long_entity])
        assert result['tags'] == [f'B-{very_long_entity}']
    
    def test_repair_illegal_consistency(self):
        """Ensure repair_illegal parameter works consistently across functions"""
        # Test direct function
        result1 = tag_sentence("test", ["I-PER"], repair_illegal=True)
        assert result1['tags'] == ['B-PER']
        
        # Test through JSON
        json_data = [{"sentence": "test", "tags": ["I-PER"]}]
        result2 = tag_from_json(json_data)  # Should use repair_illegal=True by default
        assert result2[0]['tags'] == ['B-PER']
        
        # Test through DataFrame
        df = pd.DataFrame([{"sentence_id": 0, "word": "test", "tag": "I-PER"}])
        result3 = tag_from_dataframe(df)  # Should use repair_illegal=True by default
        assert result3[0]['tags'] == ['B-PER']

if __name__ == "__main__":
    print("Running comprehensive edge case tests...")
    
    # Run a few critical tests manually
    from biotagging import tag_sentence
    
    # Test illegal I-tag repair
    result = tag_sentence("John", ["I-PER"], repair_illegal=True)
    assert result['tags'] == ['B-PER'], f"Expected ['B-PER'], got {result['tags']}"
    print("✓ Illegal I-tag repair works")
    
    # Test empty tag continuation
    result = tag_sentence("New York", ["LOC", ""], repair_illegal=True)
    assert result['tags'] == ['B-LOC', 'I-LOC'], f"Expected ['B-LOC', 'I-LOC'], got {result['tags']}"
    print("✓ Empty tag continuation works")
    
    # Test malformed tag repair
    result = tag_sentence("John", ["B-"], repair_illegal=True)
    assert result['tags'] == ['O'], f"Expected ['O'], got {result['tags']}"
    print("✓ Malformed tag repair works")
    
    # Test large data
    large_tokens = ["token"] * 1000
    large_tags = ["O"] * 1000
    result = tag_sentence(large_tokens, large_tags)
    assert len(result['tokens']) == 1000
    print("✓ Large data handling works")
    
    print("All critical edge case tests passed! ✓")