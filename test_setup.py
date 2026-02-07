"""
Setup Verification Script for Trustworthy RAG Project
Run this script after completing the preparation steps to verify everything is working.
"""

import sys
import importlib
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA Not Available (CPU mode only)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_models():
    """Check if models can be loaded"""
    results = {}
    
    # Check NLI model
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("\nTesting NLI Model (facebook/bart-large-mnli)...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        print("✓ NLI Model loaded successfully")
        results['nli'] = True
    except Exception as e:
        print(f"✗ NLI Model failed: {str(e)}")
        results['nli'] = False
    
    # Check Embedding model
    try:
        from sentence_transformers import SentenceTransformer
        print("\nTesting Embedding Model (sentence-transformers/all-MiniLM-L6-v2)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode("Test sentence")
        print(f"✓ Embedding Model working (output shape: {test_embedding.shape})")
        results['embedding'] = True
    except Exception as e:
        print(f"✗ Embedding Model failed: {str(e)}")
        results['embedding'] = False
    
    return results

def check_ollama():
    """Check if Ollama is accessible"""
    try:
        import requests
        import os
        from pathlib import Path
        
        # Try to read config
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            print("⚠ Config file not found, skipping Ollama check")
            return None
        
        # Try to connect to Ollama
        base_url = "http://localhost:11434"  # Default
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✓ Ollama accessible at {base_url}")
                print(f"  Available models: {len(models)}")
                if models:
                    print(f"  Models: {', '.join([m['name'] for m in models[:3]])}")
                return True
        except:
            print(f"⚠ Ollama not accessible at {base_url} (may need FARMI endpoint)")
            return None
    except ImportError:
        print("⚠ requests not installed, skipping Ollama check")
        return None

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/poisoned",
        "src",
        "configs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (MISSING)")
            all_exist = False
    
    return all_exist

def check_config():
    """Check if config file exists"""
    config_path = Path("configs/config.yaml")
    env_path = Path(".env")
    
    if config_path.exists():
        print("✓ configs/config.yaml exists")
    else:
        print("✗ configs/config.yaml MISSING")
    
    if env_path.exists():
        print("✓ .env file exists")
        return True
    else:
        print("⚠ .env file not found (copy from configs/config.yaml)")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("Trustworthy RAG - Setup Verification")
    print("=" * 60)
    
    checks = {
        'python': False,
        'packages': False,
        'cuda': False,
        'models': False,
        'directories': False,
        'config': False
    }
    
    # Python version
    print("\n[1] Python Version Check")
    print("-" * 60)
    checks['python'] = check_python_version()
    
    # Required packages
    print("\n[2] Required Packages Check")
    print("-" * 60)
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('langchain', 'langchain'),
        ('sentence-transformers', 'sentence_transformers'),
        ('faiss', 'faiss'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('nltk', 'nltk'),
        ('spacy', 'spacy'),
        ('mlflow', 'mlflow'),
    ]
    
    package_results = []
    for pkg_name, import_name in required_packages:
        package_results.append(check_package(pkg_name, import_name))
    
    checks['packages'] = all(package_results)
    
    # CUDA check
    print("\n[3] CUDA/GPU Check")
    print("-" * 60)
    checks['cuda'] = check_cuda()
    
    # Directory structure
    print("\n[4] Directory Structure Check")
    print("-" * 60)
    checks['directories'] = check_directories()
    
    # Config files
    print("\n[5] Configuration Files Check")
    print("-" * 60)
    checks['config'] = check_config()
    
    # Model access
    print("\n[6] Model Access Check")
    print("-" * 60)
    model_results = check_models()
    checks['models'] = all(model_results.values()) if model_results else False
    
    # Ollama check
    print("\n[7] Ollama Service Check")
    print("-" * 60)
    ollama_status = check_ollama()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    critical_checks = ['python', 'packages', 'directories']
    for check_name in critical_checks:
        status = "✓ PASS" if checks[check_name] else "✗ FAIL"
        print(f"{check_name.upper():15} {status}")
    
    optional_checks = ['cuda', 'models', 'config']
    for check_name in optional_checks:
        status = "✓ PASS" if checks[check_name] else "⚠ WARNING"
        print(f"{check_name.upper():15} {status}")
    
    if ollama_status is None:
        print(f"{'OLLAMA':15} ⚠ SKIPPED")
    elif ollama_status:
        print(f"{'OLLAMA':15} ✓ PASS")
    else:
        print(f"{'OLLAMA':15} ⚠ WARNING")
    
    print("\n" + "=" * 60)
    
    if all(checks[c] for c in critical_checks):
        print("✓ Core setup is complete!")
        print("\nNext steps:")
        print("1. Fix any warnings above")
        print("2. Configure .env file with your settings")
        print("3. Request model access from supervisor if needed")
        print("4. Download datasets (FEVER, TruthfulQA)")
        print("5. Start implementation!")
    else:
        print("✗ Please fix critical issues before proceeding")
        print("\nRun: pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
