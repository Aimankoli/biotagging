import pytest
import json
import tempfile
import os
from pathlib import Path
import pandas as pd

from biotagging.cli import (
    load_json_file, save_json_file, load_csv_file, save_csv_file, 
    save_conll_file, main
)
import subprocess
import sys

class TestCLIUtilities:
    """Test CLI utility functions"""
    
    def test_load_save_json_file(self):
        """Test JSON file loading and saving"""
        test_data = [
            {"sentence": "test", "tags": ["O"], "sentence_id": 1},
            {"sentence": "test2", "tags": ["O"], "sentence_id": 2}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Test loading
            loaded_data = load_json_file(temp_path)
            assert loaded_data == test_data
            
            # Test saving
            output_path = temp_path.replace('.json', '_output.json')
            save_json_file(test_data, output_path)
            
            # Verify saved data
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == test_data
            
        finally:
            os.unlink(temp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_load_save_csv_file(self):
        """Test CSV file loading and saving"""
        test_data = [
            {"sentence_id": 0, "sentence": "test", "tokens": ["test"], "tags": ["O"]},
            {"sentence_id": 1, "sentence": "test2", "tokens": ["test2"], "tags": ["O"]}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test loading
            loaded_df = load_csv_file(temp_path)
            assert len(loaded_df) == 2
            assert list(loaded_df.columns) == ["sentence_id", "sentence", "tokens", "tags"]
            
            # Test saving
            output_path = temp_path.replace('.csv', '_output.csv')
            save_csv_file(test_data, output_path)
            
            # Verify saved data
            saved_df = pd.read_csv(output_path)
            assert len(saved_df) == 2
            
        finally:
            os.unlink(temp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_save_conll_file(self):
        """Test CoNLL file saving"""
        test_data = [
            {
                "sentence_id": 0,
                "sentence": "John works",
                "tokens": ["John", "works"],
                "tags": ["B-PER", "O"]
            },
            {
                "sentence_id": 1,
                "sentence": "IBM hired",
                "tokens": ["IBM", "hired"],
                "tags": ["B-ORG", "O"]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conll', delete=False) as f:
            temp_path = f.name
        
        try:
            save_conll_file(test_data, temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Should contain sentence headers and token-tag pairs
            assert "# Sentence 0" in content
            assert "# Sentence 1" in content
            assert "John\tB-PER" in content
            assert "works\tO" in content
            assert "IBM\tB-ORG" in content
            assert "hired\tO" in content
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestCLICommands:
    """Test CLI commands using subprocess"""
    
    def run_cli(self, args, should_succeed=True):
        """Helper to run CLI commands"""
        cmd = [sys.executable, '-m', 'biotagging.cli'] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if should_succeed:
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                assert result.returncode == 0, f"CLI command failed: {' '.join(args)}"
        
        return result
    
    def test_version_command(self):
        """Test version command"""
        result = self.run_cli(['version'])
        assert "biotagging version 0.1.0" in result.stdout
    
    def test_help_command(self):
        """Test help command"""
        result = self.run_cli(['--help'])
        assert "Biotagging CLI" in result.stdout
        assert "Process text with BIO entity tags" in result.stdout
    
    def test_sentence_command(self):
        """Test single sentence processing"""
        result = self.run_cli(['sentence', 'John works at IBM', 'PER O O ORG'])
        output = json.loads(result.stdout)
        
        assert output['tokens'] == ['John', 'works', 'at', 'IBM']
        assert output['tags'] == ['B-PER', 'O', 'O', 'B-ORG']
        assert output['sentence_id'] is None
    
    def test_sentence_command_with_id(self):
        """Test single sentence with sentence ID"""
        result = self.run_cli(['sentence', 'test sentence', 'O O', '--sentence-id', '42'])
        output = json.loads(result.stdout)
        
        assert output['sentence_id'] == 42
        assert output['tokens'] == ['test', 'sentence']
        assert output['tags'] == ['O', 'O']
    
    def test_sentence_command_with_output_file(self):
        """Test single sentence with output file"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            result = self.run_cli(['sentence', 'test', 'O', '--output', output_path])
            assert f"Result saved to {output_path}" in result.stdout
            
            # Verify file was created and contains correct data
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # CLI saves single sentence as a list with one item
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]['tokens'] == ['test']
            assert data[0]['tags'] == ['O']
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_json_command(self):
        """Test JSON file processing"""
        # Create test input file
        test_data = [
            {"sentence": "John works", "tags": ["PER", "O"]},
            {"sentence": "IBM hired", "tags": ["ORG", "O"]}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            input_path = f.name
        
        output_path = input_path.replace('.json', '_tagged.json')
        
        try:
            result = self.run_cli(['json', input_path])
            assert f"Results saved to {output_path}" in result.stdout
            assert "Processed 2 sentences successfully" in result.stdout
            
            # Verify output file
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            assert len(results) == 2
            assert results[0]['tokens'] == ['John', 'works']
            assert results[0]['tags'] == ['B-PER', 'O']
            assert results[1]['tokens'] == ['IBM', 'hired']
            assert results[1]['tags'] == ['B-ORG', 'O']
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_csv_command(self):
        """Test CSV file processing"""
        # Create test CSV file
        test_data = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "works", "tag": "O"},
            {"sentence_id": 1, "word": "IBM", "tag": "ORG"},
            {"sentence_id": 1, "word": "hired", "tag": "O"}
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            input_path = f.name
        
        output_path = input_path.replace('.csv', '_tagged.json')
        
        try:
            result = self.run_cli(['csv', input_path])
            assert f"Results saved to {output_path}" in result.stdout
            assert "Processed 2 sentences successfully" in result.stdout
            
            # Verify output file
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            assert len(results) == 2
            assert results[0]['sentence_id'] == 0
            assert results[1]['sentence_id'] == 1
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_validate_command_success(self):
        """Test validation command with valid data"""
        test_data = [
            {
                "sentence_id": 0,
                "sentence": "John works",
                "tokens": ["John", "works"],
                "tags": ["B-PER", "O"]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            input_path = f.name
        
        try:
            result = self.run_cli(['validate', input_path])
            assert "✓ All 1 items passed validation" in result.stdout
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_validate_command_failure(self):
        """Test validation command with invalid data"""
        test_data = [
            {
                "sentence_id": 0,
                "sentence": "John works",
                "tokens": ["John"],  # Mismatched length
                "tags": ["B-PER", "O"]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            input_path = f.name
        
        try:
            result = self.run_cli(['validate', input_path], should_succeed=False)
            assert "1/1 items failed validation" in result.stdout
            assert result.returncode == 1
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_error_handling_missing_file(self):
        """Test error handling for missing input file"""
        result = self.run_cli(['json', 'nonexistent.json'], should_succeed=False)
        assert result.returncode == 1
        assert "File not found" in result.stderr
    
    def test_error_handling_malformed_tags(self):
        """Test error handling for malformed tags"""
        result = self.run_cli(['sentence', 'John works', 'PER'], should_succeed=False)
        assert result.returncode == 1
        assert "Token/tag length mismatch" in result.stderr

class TestCLIOutputFormats:
    """Test different CLI output formats"""
    
    def run_cli(self, args):
        """Helper to run CLI commands"""
        cmd = [sys.executable, '-m', 'biotagging.cli'] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        return result
    
    def test_json_output_format(self):
        """Test JSON output format"""
        test_data = [{"sentence": "John works", "tags": ["PER", "O"]}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            input_path = f.name
        
        output_path = input_path.replace('.json', '_output.json')
        
        try:
            self.run_cli(['json', input_path, '--output', output_path, '--format', 'json'])
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert len(data) == 1
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_csv_output_format(self):
        """Test CSV output format"""
        test_data = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "works", "tag": "O"}
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            input_path = f.name
        
        output_path = input_path.replace('.csv', '_output.csv')
        
        try:
            self.run_cli(['csv', input_path, '--output', output_path, '--format', 'csv'])
            
            assert os.path.exists(output_path)
            df = pd.read_csv(output_path)
            assert len(df) == 1  # Should be 1 sentence
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_conll_output_format(self):
        """Test CoNLL output format"""
        test_data = pd.DataFrame([
            {"sentence_id": 0, "word": "John", "tag": "PER"},
            {"sentence_id": 0, "word": "works", "tag": "O"}
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            input_path = f.name
        
        output_path = input_path.replace('.csv', '_output.conll')
        
        try:
            self.run_cli(['csv', input_path, '--output', output_path, '--format', 'conll'])
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "# Sentence 0" in content
            assert "John\tB-PER" in content
            assert "works\tO" in content
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

if __name__ == "__main__":
    print("Running CLI tests...")
    
    # Quick smoke test for CLI utilities
    test_data = [{"test": "data"}]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        loaded = load_json_file(temp_path)
        assert loaded == test_data
        print("✓ JSON utilities work")
        
        save_json_file(test_data, temp_path.replace('.json', '_test.json'))
        print("✓ JSON save works")
        
    finally:
        os.unlink(temp_path)
        test_output = temp_path.replace('.json', '_test.json')
        if os.path.exists(test_output):
            os.unlink(test_output)
    
    print("All CLI utility tests passed! ✓")