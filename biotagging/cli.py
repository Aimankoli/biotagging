#!/usr/bin/env python3
"""
Biotagging CLI - Command line interface for biotagging package
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from . import tag_sentence, tag_sentences, tag_from_json, tag_from_dataframe
from .schema.jsonresponse import validate_bio_output


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv_file(file_path: str) -> pd.DataFrame:
    """Load CSV/DataFrame data from file"""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV {file_path}: {e}")


def save_csv_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def save_conll_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to CoNLL format"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            sentence_id = item.get('sentence_id', 0)
            tokens = item['tokens']
            tags = item['tags']
            
            f.write(f"# Sentence {sentence_id}\n")
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


def process_single_sentence(args) -> None:
    """Process a single sentence from command line"""
    sentence = args.sentence
    tags = args.tags.split() if isinstance(args.tags, str) else args.tags
    
    result = tag_sentence(
        sentence=sentence,
        tags=tags,
        sentence_id=args.sentence_id,
        repair_illegal=not args.strict
    )
    
    if args.output:
        save_json_file([result], args.output)
        print(f"Result saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def process_json_file(args) -> None:
    """Process JSON file"""
    print(f"Loading JSON file: {args.input}")
    data = load_json_file(args.input)
    
    print(f"Processing {len(data)} sentences...")
    results = tag_from_json(data)
    
    # Validate if requested
    if args.validate:
        print("Validating results...")
        for i, result in enumerate(results):
            try:
                validate_bio_output(result)
            except Exception as e:
                print(f"Validation error in sentence {i}: {e}", file=sys.stderr)
    
    # Save results
    output_path = args.output or args.input.replace('.json', '_tagged.json')
    
    if args.format == 'json':
        save_json_file(results, output_path)
    elif args.format == 'csv':
        save_csv_file(results, output_path.replace('.json', '.csv'))
    elif args.format == 'conll':
        save_conll_file(results, output_path.replace('.json', '.conll'))
    
    print(f"Results saved to {output_path}")
    print(f"Processed {len(results)} sentences successfully")


def process_csv_file(args) -> None:
    """Process CSV/DataFrame file"""
    print(f"Loading CSV file: {args.input}")
    df = load_csv_file(args.input)
    
    # Validate required columns
    required_cols = ['sentence_id', 'word', 'tag']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Processing {len(df)} rows grouped by sentence_id...")
    results = tag_from_dataframe(df)
    
    # Validate if requested
    if args.validate:
        print("Validating results...")
        for i, result in enumerate(results):
            try:
                validate_bio_output(result)
            except Exception as e:
                print(f"Validation error in sentence {i}: {e}", file=sys.stderr)
    
    # Save results
    output_path = args.output or args.input.replace('.csv', '_tagged.json')
    
    if args.format == 'json':
        save_json_file(results, output_path)
    elif args.format == 'csv':
        save_csv_file(results, output_path.replace('.json', '.csv'))
    elif args.format == 'conll':
        save_conll_file(results, output_path.replace('.json', '.conll'))
    
    print(f"Results saved to {output_path}")
    print(f"Processed {len(results)} sentences successfully")


def validate_file(args) -> None:
    """Validate a file of tagged results"""
    print(f"Validating file: {args.file}")
    
    if args.file.endswith('.json'):
        data = load_json_file(args.file)
    elif args.file.endswith('.csv'):
        df = load_csv_file(args.file)
        data = df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")
    
    errors = 0
    for i, item in enumerate(data):
        try:
            validate_bio_output(item)
        except Exception as e:
            errors += 1
            print(f"Error in item {i}: {e}")
    
    if errors == 0:
        print(f"✓ All {len(data)} items passed validation")
    else:
        print(f"✗ {errors}/{len(data)} items failed validation")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Biotagging CLI - Process text with BIO entity tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single sentence
  biotagging sentence "John works at IBM" "PER O O ORG"
  
  # Process JSON file
  biotagging json input.json --output results.json
  
  # Process CSV file with validation
  biotagging csv data.csv --validate --format conll
  
  # Validate existing results
  biotagging validate results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single sentence command
    sentence_parser = subparsers.add_parser('sentence', help='Process single sentence')
    sentence_parser.add_argument('sentence', help='Input sentence (string or space-separated tokens)')
    sentence_parser.add_argument('tags', help='Space-separated tags')
    sentence_parser.add_argument('--sentence-id', type=int, help='Sentence ID')
    sentence_parser.add_argument('--output', '-o', help='Output JSON file')
    sentence_parser.add_argument('--strict', action='store_true', 
                               help='Strict mode - fail on illegal BIO sequences')
    
    # JSON file command
    json_parser = subparsers.add_parser('json', help='Process JSON file')
    json_parser.add_argument('input', help='Input JSON file')
    json_parser.add_argument('--output', '-o', help='Output file path')
    json_parser.add_argument('--format', choices=['json', 'csv', 'conll'], 
                           default='json', help='Output format')
    json_parser.add_argument('--validate', action='store_true', help='Validate results')
    
    # CSV file command
    csv_parser = subparsers.add_parser('csv', help='Process CSV file')
    csv_parser.add_argument('input', help='Input CSV file (must have sentence_id, word, tag columns)')
    csv_parser.add_argument('--output', '-o', help='Output file path')
    csv_parser.add_argument('--format', choices=['json', 'csv', 'conll'], 
                          default='json', help='Output format')
    csv_parser.add_argument('--validate', action='store_true', help='Validate results')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate tagged data file')
    validate_parser.add_argument('file', help='File to validate (.json or .csv)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'sentence':
            process_single_sentence(args)
        elif args.command == 'json':
            process_json_file(args)
        elif args.command == 'csv':
            process_csv_file(args)
        elif args.command == 'validate':
            validate_file(args)
        elif args.command == 'version':
            from . import __version__
            print(f"biotagging version {__version__}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()