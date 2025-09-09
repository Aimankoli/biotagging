import pytest
import pandas as pd
import numpy as np
from biotagging import tag_sentence, tag_sentences, tag_from_json, tag_from_dataframe
from biotagging.schema.jsonresponse import validate_bio_output, validate_bio_input
from pydantic import ValidationError

class TestExtremeInputs:
    """Test with extreme and unusual inputs"""
    
    def test_extremely_long_tokens(self):
        """Tokens with thousands of characters"""
        super_long_token = "a" * 10000  # 10k character token
        result = tag_sentence([super_long_token, "test"], ["ENT", "O"])
        assert len(result['tokens'][0]) == 10000
        assert result['tags'] == ['B-ENT', 'O']
    
    def test_extremely_long_entity_names(self):
        """Entity names with thousands of characters"""
        super_long_entity = "ENTITY_" + "X" * 5000  # 5k+ character entity
        result = tag_sentence(["test"], [super_long_entity])
        assert result['tags'] == [f'B-{super_long_entity}']
    
    def test_maximum_sentence_length(self):
        """Test with 100k tokens"""
        huge_tokens = [f"token{i}" for i in range(100000)]
        huge_tags = ["O"] * 100000
        huge_sentence = " ".join(huge_tokens)
        
        result = tag_sentence(huge_sentence, huge_tags)
        assert len(result['tokens']) == 100000
        assert len(result['tags']) == 100000
    
    def test_special_characters_in_tokens(self):
        """Tokens with special characters, emojis, etc."""
        special_tokens = [
            "hello@world.com",
            "$$$money$$$", 
            "👋🌍",  # Emojis
            "café_münchen_北京",  # Mixed unicode
            "test\ttab\nline",  # Control characters
            "a\\b/c|d&e%f#g",  # Special symbols
            "",  # Empty string as token (when passed as list)
        ]
        special_tags = ["EMAIL", "MONEY", "EMOJI", "PLACE", "TEXT", "SYMBOLS", "EMPTY"]
        
        result = tag_sentence(special_tokens, special_tags)
        assert len(result['tokens']) == len(special_tokens)
        assert all(tag.startswith('B-') for tag in result['tags'])
    
    def test_numeric_edge_cases(self):
        """Various numeric formats"""
        numeric_tokens = [
            "0", "123", "-456", "3.14159", 
            "1e10", "0x1A2B", "0b1010", 
            "1,234,567.89", "∞", "NaN"
        ]
        tags = ["NUM"] * len(numeric_tokens)
        
        result = tag_sentence(numeric_tokens, tags)
        assert len(result['tokens']) == len(numeric_tokens)
        assert all(tag == 'B-NUM' for tag in result['tags'])

class TestMalformedBIOSequences:
    """Test malformed BIO sequences and repairs"""
    
    def test_multiple_consecutive_i_tags_different_classes(self):
        """Multiple I-tags with different classes"""
        result = tag_sentence(
            ["John", "Smith", "works", "at", "Big", "Tech", "Corp"], 
            ["I-PER", "I-ORG", "O", "O", "I-COMPANY", "I-PLACE", "I-BUSINESS"],
            repair_illegal=True
        )
        # All I-tags without B-tags should become B-tags
        expected = ['B-PER', 'B-ORG', 'O', 'O', 'B-COMPANY', 'B-PLACE', 'B-BUSINESS']
        assert result['tags'] == expected
    
    def test_alternating_empty_and_entity_tags(self):
        """Alternating empty strings and entity tags"""
        result = tag_sentence(
            ["A", "B", "C", "D", "E"], 
            ["PER", "", "ORG", "", "LOC"],
            repair_illegal=True
        )
        expected = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC']
        assert result['tags'] == expected
    
    def test_mixed_b_i_o_empty_chaos(self):
        """Chaotic mix of B-, I-, O, empty tags"""
        result = tag_sentence(
            ["w1", "w2", "w3", "w4", "w5", "w6", "w7"], 
            ["B-A", "", "I-B", "O", "", "B-", "I-C"],
            repair_illegal=True
        )
        # Expected: B-A continues, I-B becomes B-B, O stays, empty after O becomes O, B- becomes O, I-C becomes B-C
        expected = ['B-A', 'I-A', 'B-B', 'O', 'O', 'O', 'B-C']
        assert result['tags'] == expected
    
    def test_all_empty_tags(self):
        """All empty tags should become all O when repair_illegal=True"""
        result = tag_sentence(["a", "b", "c"], ["", "", ""], repair_illegal=True)
        assert result['tags'] == ['O', 'O', 'O']
    
    def test_nested_entity_syntax_in_names(self):
        """Entity names containing B- and I- patterns"""
        result = tag_sentence(
            ["token1", "token2"], 
            ["B-WEIRD_B-ENTITY", "I-STRANGE_I-TAG"]
        )
        # Should treat the full string as entity name, I-tag becomes B-tag due to class mismatch
        assert result['tags'] == ['B-WEIRD_B-ENTITY', 'B-STRANGE_I-TAG']

class TestUnicodeAndEncoding:
    """Test unicode, encoding, and international text"""
    
    def test_mixed_scripts(self):
        """Mixed writing systems"""
        mixed_tokens = [
            "English",      # Latin
            "العربية",       # Arabic
            "中文",          # Chinese
            "हिन्दी",        # Devanagari
            "русский",      # Cyrillic
            "日本語",        # Japanese
            "한글",         # Korean
            "ελληνικά",     # Greek
        ]
        tags = ["LANG"] * len(mixed_tokens)
        
        result = tag_sentence(mixed_tokens, tags)
        assert len(result['tokens']) == len(mixed_tokens)
        assert result['tokens'] == mixed_tokens  # Should preserve unicode
    
    def test_rtl_text(self):
        """Right-to-left text"""
        rtl_tokens = ["مرحبا", "بالعالم"]
        tags = ["GREETING", "WORLD"]
        
        result = tag_sentence(rtl_tokens, tags)
        assert result['tokens'] == rtl_tokens
        assert result['tags'] == ['B-GREETING', 'B-WORLD']
    
    def test_combining_characters(self):
        """Unicode combining characters"""
        # é can be written as e + combining acute accent
        combining_tokens = ["café", "cafe\u0301"]  # Second is e + combining acute
        tags = ["PLACE", "PLACE"]
        
        result = tag_sentence(combining_tokens, tags)
        assert len(result['tokens']) == 2
        assert result['tags'] == ['B-PLACE', 'B-PLACE']
    
    def test_zero_width_characters(self):
        """Zero-width characters"""
        tokens_with_zwc = [
            "test\u200B",    # Zero-width space
            "word\u200C",    # Zero-width non-joiner
            "another\u200D", # Zero-width joiner
            "invisible\uFEFF" # Zero-width no-break space (BOM)
        ]
        tags = ["TEST"] * len(tokens_with_zwc)
        
        result = tag_sentence(tokens_with_zwc, tags)
        assert len(result['tokens']) == len(tokens_with_zwc)
        assert result['tokens'] == tokens_with_zwc  # Should preserve as-is

class TestDataTypeEdgeCases:
    """Test various data type edge cases"""
    
    def test_numpy_types_as_input(self):
        """NumPy data types as input"""
        np_tokens = [np.str_("numpy_string"), str(np.int64(123)), str(np.float64(3.14))]
        np_tags = [str(np.str_("NUMPY")), "NUM", "FLOAT"]
        
        result = tag_sentence(np_tokens, np_tags)
        assert len(result['tokens']) == 3
        assert result['tags'] == ['B-NUMPY', 'B-NUM', 'B-FLOAT']
    
    def test_boolean_and_none_conversions(self):
        """Boolean and None values"""
        mixed_tokens = ["true", str(True), str(False), str(None)]
        mixed_tags = ["BOOL", "BOOL", "BOOL", "NULL"]
        
        result = tag_sentence(mixed_tokens, mixed_tags)
        assert result['tokens'] == ["true", "True", "False", "None"]
        assert result['tags'] == ['B-BOOL', 'B-BOOL', 'B-BOOL', 'B-NULL']
    
    def test_complex_nested_structures_in_strings(self):
        """Complex nested structures converted to strings"""
        nested_data = [
            str({"key": "value"}),
            str([1, 2, 3]),
            str((1, 2, 3)),
            str({1, 2, 3})
        ]
        tags = ["DICT", "LIST", "TUPLE", "SET"]
        
        result = tag_sentence(nested_data, tags)
        assert len(result['tokens']) == 4
        assert result['tags'] == ['B-DICT', 'B-LIST', 'B-TUPLE', 'B-SET']

class TestMemoryAndPerformance:
    """Test memory usage and performance edge cases"""
    
    def test_deeply_nested_entity_chains(self):
        """Very long entity chains"""
        long_entity_tokens = ["word"] * 10000
        # Alternate between continuing and starting new entities
        long_entity_tags = []
        for i in range(10000):
            if i % 100 == 0:  # Start new entity every 100 tokens
                long_entity_tags.append(f"ENT{i//100}")
            else:
                long_entity_tags.append("")  # Continue previous
        
        result = tag_sentence(long_entity_tokens, long_entity_tags)
        assert len(result['tokens']) == 10000
        assert len(result['tags']) == 10000
        # Check pattern: should have B- tags every 100 positions
        for i in range(0, 10000, 100):
            assert result['tags'][i].startswith('B-ENT')
    
    def test_many_unique_entity_types(self):
        """Thousands of different entity types"""
        tokens = [f"token{i}" for i in range(5000)]
        tags = [f"ENTITY_{i}" for i in range(5000)]  # All unique entities
        
        result = tag_sentence(tokens, tags)
        assert len(result['tokens']) == 5000
        assert len(set(result['tags'])) == 5000  # All tags should be unique
        assert all(tag.startswith('B-ENTITY_') for tag in result['tags'])
    
    def test_repeated_large_batch_processing(self):
        """Process multiple large batches"""
        large_batches = []
        
        # Create 100 batches of 1000 sentences each
        for batch_num in range(100):
            batch = []
            for sent_num in range(1000):
                sentence = f"Sentence {batch_num} number {sent_num}"
                tags = ["O", "NUM", "O", "NUM"]
                batch.append((sentence, tags))
            large_batches.extend(batch)
        
        # Process all 100k sentences
        results = tag_sentences(large_batches)
        assert len(results) == 100000
        assert all(len(r['tokens']) == 4 for r in results)

class TestConcurrencyAndThreadSafety:
    """Test concurrent access patterns"""
    
    def test_same_input_multiple_times(self):
        """Same input processed multiple times should give same result"""
        sentence = "John works at IBM"
        tags = ["PER", "O", "O", "ORG"]
        
        results = []
        for _ in range(100):
            result = tag_sentence(sentence, tags)
            results.append(result)
        
        # All results should be identical
        first_result = results[0]
        assert all(r == first_result for r in results)
    
    def test_state_isolation(self):
        """Ensure no state leakage between calls"""
        # Process with repair_illegal=False first
        try:
            tag_sentence("test", ["I-PER"], repair_illegal=False)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Then with repair_illegal=True - should work independently
        result = tag_sentence("test", ["I-PER"], repair_illegal=True)
        assert result['tags'] == ['B-PER']

class TestJSONDataFrameExtremeEdgeCases:
    """Extreme edge cases for JSON and DataFrame processing"""
    
    def test_json_with_deeply_nested_sentence_ids(self):
        """Very large sentence IDs"""
        json_data = [
            {"sentence_id": 2**63 - 1, "sentence": "test", "tags": ["O"]},  # Max int64
            {"sentence_id": -2**63, "sentence": "test2", "tags": ["O"]},    # Min int64
            {"sentence_id": 0, "sentence": "test3", "tags": ["O"]}
        ]
        
        results = tag_from_json(json_data)
        assert len(results) == 3
        assert results[0]['sentence_id'] == 2**63 - 1
        assert results[1]['sentence_id'] == -2**63
    
    def test_dataframe_with_extreme_sentence_ids(self):
        """DataFrame with extreme sentence ID values"""
        df = pd.DataFrame([
            {"sentence_id": 999999999999, "word": "test1", "tag": "O"},
            {"sentence_id": -999999999999, "word": "test2", "tag": "O"},
            {"sentence_id": 0, "word": "test3", "tag": "O"}
        ])
        
        results = tag_from_dataframe(df)
        assert len(results) == 3
    
    def test_json_with_mixed_data_types_in_tags(self):
        """JSON with various data types as tags"""
        json_data = [
            {"sentence": ["word1", "word2", "word3"], "tags": [123, 45.6, True]},
            {"sentence": "test word", "tags": [None, "O"]}
        ]
        
        results = tag_from_json(json_data)
        assert len(results) == 2
        # Should convert all to strings and process
        assert "B-123" in results[0]['tags']
        assert "B-45.6" in results[0]['tags']
        assert "B-True" in results[0]['tags']
    
    def test_dataframe_with_extreme_data_pollution(self):
        """DataFrame with various data pollution scenarios"""
        df = pd.DataFrame([
            {"sentence_id": 0, "word": "normal", "tag": "O", "junk_col": "ignore"},
            {"sentence_id": 0, "word": None, "tag": "", "extra": [1, 2, 3]},
            {"sentence_id": 0, "word": 123.456, "tag": np.inf, "more_junk": {"a": 1}},
            {"sentence_id": 0, "word": True, "tag": False, "stuff": None}
        ])
        
        results = tag_from_dataframe(df)
        assert len(results) == 1
        assert len(results[0]['tokens']) == 4
        # Should handle all data type conversions gracefully

class TestErrorRecoveryAndRobustness:
    """Test error recovery and robustness"""
    
    def test_partial_failure_in_large_batch(self):
        """Large batch where middle items fail"""
        batch = []
        
        # Add 1000 good items
        for i in range(1000):
            batch.append((f"sentence {i}", ["O", "NUM"]))
        
        # Add bad item in middle
        batch.append(("", []))  # This will fail
        
        # Add 1000 more good items
        for i in range(1000, 2000):
            batch.append((f"sentence {i}", ["O", "NUM"]))
        
        # Should fail on item 1000 and give clear error
        with pytest.raises(ValueError) as exc_info:
            tag_sentences(batch)
        
        assert "[index 1000]" in str(exc_info.value)
    
    def test_recovery_after_errors(self):
        """Ensure system recovers after errors"""
        # Cause an error
        try:
            tag_sentence("", [])
        except ValueError:
            pass
        
        # Should work fine after error
        result = tag_sentence("test", ["O"])
        assert result['tokens'] == ["test"]
        assert result['tags'] == ["O"]
    
    def test_malformed_input_recovery(self):
        """Recovery from various malformed inputs"""
        malformed_inputs = [
            (None, ["O"]),          # None sentence - should fail
            ("test", None),         # None tags - should fail
            ([], []),              # Empty lists - should work (empty result)
            ("test", []),          # Mismatched lengths - should fail
            (123, ["O"]),          # Wrong type for sentence - should fail
            ("test", [123]),       # Wrong type for tags - should work (converts to string)
        ]
        
        successful_results = 0
        failed_results = 0
        
        for sentence, tags in malformed_inputs:
            try:
                result = tag_sentence(sentence, tags, repair_illegal=True)
                successful_results += 1
            except (ValueError, TypeError, AttributeError):
                failed_results += 1
        
        # Should have some successes (empty lists work, numeric tags work) and some failures
        assert successful_results == 2  # ([], []) and ("test", [123]) should work
        assert failed_results == 4      # The rest should fail
        
        # But system should still work after all these errors
        result = tag_sentence("recovery test", ["O", "O"])
        assert result['tokens'] == ["recovery", "test"]

if __name__ == "__main__":
    print("Running extreme edge case tests...")
    
    # Test a few critical extreme cases manually
    from biotagging import tag_sentence, tag_from_json
    import time
    
    # Test very long input
    start = time.time()
    long_tokens = ["token"] * 10000
    long_tags = ["O"] * 10000
    result = tag_sentence(long_tokens, long_tags)
    duration = time.time() - start
    assert len(result['tokens']) == 10000
    print(f"✓ Large input (10k tokens) processed in {duration:.2f}s")
    
    # Test unicode handling
    result = tag_sentence(["café", "北京", "👋"], ["PLACE", "CITY", "EMOJI"])
    assert result['tokens'] == ["café", "北京", "👋"]
    print("✓ Unicode handling works")
    
    # Test extreme BIO repair
    result = tag_sentence(["a", "b", "c"], ["I-X", "I-Y", "I-Z"], repair_illegal=True)
    assert result['tags'] == ['B-X', 'B-Y', 'B-Z']
    print("✓ Complex BIO repair works")
    
    # Test memory with large entity names
    huge_entity = "ENT_" + "X" * 1000
    result = tag_sentence(["test"], [huge_entity])
    assert result['tags'][0] == f'B-{huge_entity}'
    print("✓ Large entity names work")
    
    print("All critical extreme edge case tests passed! ✓")