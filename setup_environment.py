"""
Automated Setup Script for Trustworthy RAG Project
This script helps automate the initial setup process.
"""

import subprocess
import sys
import os
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        # Use list format for better path handling on Windows
        if sys.platform == 'win32':
            # For Windows, use list format to avoid path quoting issues
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
        else:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"Error: {error_msg}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False

def check_virtual_env():
    """Check if virtual environment is activated"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✓ Virtual environment is active")
        return True
    else:
        print("⚠ Virtual environment not detected")
        print("  Please activate your venv first:")
        print("  Windows: .\\venv\\Scripts\\Activate.ps1")
        print("  Linux/Mac: source venv/bin/activate")
        return False

def install_requirements():
    """Install Python packages from requirements.txt"""
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found")
        return False
    
    # Quote the executable path properly for Windows
    python_exe = f'"{sys.executable}"' if ' ' in sys.executable else sys.executable
    
    return run_command(
        f"{python_exe} -m pip install --upgrade pip",
        "Upgrading pip"
    ) and run_command(
        f"{python_exe} -m pip install -r requirements.txt",
        "Installing requirements"
    )

def download_nltk_data():
    """Download required NLTK data"""
    print("\n" + "="*60)
    print("Downloading NLTK Data")
    print("="*60)
    
    try:
        import nltk
        nltk_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for data in nltk_data:
            print(f"Downloading {data}...")
            try:
                nltk.download(data, quiet=True)
                print(f"✓ {data}")
            except Exception as e:
                print(f"✗ {data}: {str(e)}")
        
        return True
    except ImportError:
        print("✗ NLTK not installed. Install requirements first.")
        return False

def download_spacy_model():
    """Download spaCy English model"""
    print("\n" + "="*60)
    print("Downloading spaCy Model")
    print("="*60)
    
    # Quote the executable path properly for Windows
    python_exe = f'"{sys.executable}"' if ' ' in sys.executable else sys.executable
    
    return run_command(
        f"{python_exe} -m spacy download en_core_web_sm",
        "Downloading spaCy en_core_web_sm model"
    )

def create_directories():
    """Create required directories"""
    print("\n" + "="*60)
    print("Creating Directory Structure")
    print("="*60)
    
    directories = [
        "data/raw",
        "data/processed",
        "data/poisoned",
        "logs",
        "experiments",
        "tests",
        "notebooks"
    ]
    
    all_created = True
    for dir_path in directories:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {dir_path}/")
        except Exception as e:
            print(f"✗ {dir_path}/: {str(e)}")
            all_created = False
    
    return all_created

def create_env_file():
    """Create .env file from config.yaml if it doesn't exist"""
    print("\n" + "="*60)
    print("Setting up Configuration")
    print("="*60)
    
    config_path = Path("configs/config.yaml")
    env_path = Path(".env")
    
    if not config_path.exists():
        print("✗ configs/config.yaml not found")
        return False
    
    if env_path.exists():
        print("⚠ .env file already exists (skipping)")
        return True
    
    try:
        # Copy config.yaml to .env
        import shutil
        shutil.copy(config_path, env_path)
        print("✓ Created .env file from configs/config.yaml")
        print("  Please edit .env with your specific settings")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("Trustworthy RAG - Automated Setup")
    print("="*60)
    
    steps = [
        ("Virtual Environment Check", check_virtual_env, True),
        ("Create Directories", create_directories, True),
        ("Install Requirements", install_requirements, True),
        ("Download NLTK Data", download_nltk_data, False),
        ("Download spaCy Model", download_spacy_model, False),
        ("Create .env File", create_env_file, False),
    ]
    
    results = {}
    
    for step_name, step_func, critical in steps:
        try:
            result = step_func()
            results[step_name] = (result, critical)
        except Exception as e:
            print(f"\n✗ Error in {step_name}: {str(e)}")
            results[step_name] = (False, critical)
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    all_critical_passed = True
    for step_name, (result, critical) in results.items():
        status = "✓" if result else "✗"
        critical_mark = " [CRITICAL]" if critical else ""
        print(f"{status} {step_name}{critical_mark}")
        
        if critical and not result:
            all_critical_passed = False
    
    print("\n" + "="*60)
    
    if all_critical_passed:
        print("✓ Core setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your configuration")
        print("2. Request model access from supervisor (see SUPERVISOR_REQUEST_TEMPLATE.md)")
        print("3. Run: python test_setup.py")
        print("4. Download datasets (FEVER, TruthfulQA)")
    else:
        print("✗ Some critical steps failed. Please fix errors above.")
    
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
