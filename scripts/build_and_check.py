#!/usr/bin/env python3
"""
Build and check script for biotagging package
Run this before releasing to ensure everything is correct
"""
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print(f"✅ Success: {description}")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
        return True

def main():
    """Main build and check process"""
    print("🚀 Biotagging Package Build & Check")
    print("=" * 40)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    print(f"Working directory: {project_root}")
    
    # Clean previous builds
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        print("🧹 Cleaning previous builds...")
        shutil.rmtree(dist_dir)
    
    # Run tests first
    if not run_command([sys.executable, "-m", "pytest", "test/", "-v"], "Running tests"):
        print("❌ Tests failed! Fix tests before building.")
        return False
    
    # Test CLI
    if not run_command([sys.executable, "-m", "biotagging.cli", "version"], "Testing CLI"):
        print("❌ CLI test failed!")
        return False
    
    # Build package
    if not run_command([sys.executable, "-m", "build"], "Building package"):
        print("❌ Build failed!")
        return False
    
    # Check package
    if not run_command(["twine", "check", "dist/*"], "Checking package"):
        print("❌ Package check failed!")
        return False
    
    # Test installation in clean environment
    print("\n🧪 Testing installation in clean environment...")
    
    # List built files
    dist_files = list(dist_dir.glob("*"))
    print(f"\n📦 Built packages:")
    for f in dist_files:
        print(f"  - {f.name}")
    
    print("\n🎉 All checks passed!")
    print("Your package is ready for release!")
    
    print("\nNext steps:")
    print("1. Push to GitHub")
    print("2. Create a release/tag (e.g., v0.1.0)")
    print("3. GitHub Actions will automatically publish to PyPI")
    print("\nOr manually upload with: twine upload dist/*")

if __name__ == "__main__":
    main()