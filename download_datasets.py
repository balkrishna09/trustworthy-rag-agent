"""
Dataset Download Script for Trustworthy RAG Project
Downloads FEVER and TruthfulQA datasets from Hugging Face.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import json
from tqdm import tqdm

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def download_fever():
    """Download FEVER dataset"""
    print("\n" + "="*60)
    print("Downloading FEVER Dataset")
    print("="*60)
    
    try:
        print("Loading FEVER dataset from Hugging Face...")
        # FEVER dataset alternatives on Hugging Face
        # Try different dataset names/versions
        dataset = None
        alternatives = [
            ("fever/fever", None),  # Correct FEVER dataset path
            ("fever", None),
            ("tals/fever", None),
            ("facebook/fever", None),
        ]
        
        for dataset_name, config in alternatives:
            try:
                if config:
                    dataset = load_dataset(dataset_name, config, split="train")
                else:
                    dataset = load_dataset(dataset_name, split="train")
                print(f"[OK] FEVER dataset loaded from '{dataset_name}': {len(dataset)} samples")
                break
            except Exception as e:
                continue
        
        if dataset is None:
            # FEVER uses old script format - try alternative fact-checking datasets
            print("FEVER dataset uses outdated format on Hugging Face.")
            print("Trying alternative fact-checking datasets...")
            
            alternative_datasets = [
                ("tals/fever", None),
                ("google/fact_check_claim_verification", None),
                ("potsawee/wiki_bio_dpr", None),
            ]
            
            for alt_name, alt_config in alternative_datasets:
                try:
                    if alt_config:
                        dataset = load_dataset(alt_name, alt_config, split="train")
                    else:
                        # Try train split first, then any available split
                        try:
                            dataset = load_dataset(alt_name, split="train")
                        except:
                            info = load_dataset(alt_name, split="train", streaming=True)
                            dataset = load_dataset(alt_name, split=list(info.keys())[0] if hasattr(info, 'keys') else "train")
                    print(f"[OK] Using alternative dataset '{alt_name}': {len(dataset)} samples")
                    break
                except Exception as e:
                    continue
            
            if dataset is None:
                print("\n[INFO] FEVER dataset needs manual download.")
                print("Reason: FEVER uses outdated format on Hugging Face.")
                print("\nOptions:")
                print("1. Download manually from: https://fever.ai/dataset/fever.html")
                print("2. Use TruthfulQA (already downloaded) for initial testing")
                print("3. Create custom fact-checking dataset for your experiments")
                print("\nFor now, continuing with TruthfulQA only...")
                return False
        
        # Save to data/raw/fever/
        output_dir = Path("data/raw/fever")
        ensure_dir(output_dir)
        
        # Convert to JSON Lines format
        output_file = output_dir / "fever_train.jsonl"
        print(f"Saving to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Saving FEVER"):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"[OK] FEVER dataset saved: {output_file}")
        print(f"  Samples: {len(dataset)}")
        
        # Show sample
        if len(dataset) > 0:
            print("\nSample entry:")
            sample = dataset[0]
            for key in list(sample.keys())[:3]:  # Show first 3 keys
                print(f"  {key}: {str(sample[key])[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"X Error downloading FEVER: {str(e)}")
        print("\nAlternative: Download manually from https://fever.ai/dataset/fever.html")
        print("Or use a fact-checking dataset from Hugging Face")
        return False

def download_truthfulqa():
    """Download TruthfulQA dataset"""
    print("\n" + "="*60)
    print("Downloading TruthfulQA Dataset")
    print("="*60)
    
    try:
        print("Loading TruthfulQA dataset from Hugging Face...")
        
        # TruthfulQA on Hugging Face
        alternatives_truthfulqa = [
            ("truthful_qa", "generation", "validation"),
            ("truthful_qa", None, "validation"),
            ("truthful_qa", None, "train"),
        ]
        
        dataset = None
        for name, config, split in alternatives_truthfulqa:
            try:
                if config:
                    dataset = load_dataset(name, config, split=split)
                else:
                    dataset = load_dataset(name, split=split)
                print(f"[OK] TruthfulQA dataset loaded: {len(dataset)} samples")
                break
            except Exception as e:
                continue
        
        if dataset is None:
            print("[WARNING] TruthfulQA not found. You may need to download manually.")
            print("Visit: https://github.com/sylinrl/TruthfulQA")
            return False
        
        # Save to data/raw/truthfulqa/
        output_dir = Path("data/raw/truthfulqa")
        ensure_dir(output_dir)
        
        output_file = output_dir / "truthfulqa.jsonl"
        print(f"Saving to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Saving TruthfulQA"):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"[OK] TruthfulQA dataset saved: {output_file}")
        print(f"  Samples: {len(dataset)}")
        
        # Show sample
        if len(dataset) > 0:
            print("\nSample entry:")
            sample = dataset[0]
            for key in list(sample.keys())[:3]:  # Show first 3 keys
                print(f"  {key}: {str(sample[key])[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"X Error downloading TruthfulQA: {str(e)}")
        print("\nAlternative: Download manually from https://github.com/sylinrl/TruthfulQA")
        return False

def show_dataset_info():
    """Show information about downloaded datasets"""
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    
    fever_path = Path("data/raw/fever")
    truthfulqa_path = Path("data/raw/truthfulqa")
    
    if fever_path.exists():
        fever_files = list(fever_path.glob("*.jsonl"))
        if fever_files:
            print(f"[OK] FEVER: {len(fever_files)} file(s)")
            for f in fever_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.2f} MB")
    
    if truthfulqa_path.exists():
        truthfulqa_files = list(truthfulqa_path.glob("*.jsonl"))
        if truthfulqa_files:
            print(f"[OK] TruthfulQA: {len(truthfulqa_files)} file(s)")
            for f in truthfulqa_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.2f} MB")

def main():
    """Main download function"""
    print("="*60)
    print("Trustworthy RAG - Dataset Download")
    print("="*60)
    
    # Create directories
    ensure_dir("data/raw")
    ensure_dir("data/processed")
    ensure_dir("data/poisoned")
    
    results = {}
    
    # Download datasets
    results['fever'] = download_fever()
    results['truthfulqa'] = download_truthfulqa()
    
    # Show summary
    show_dataset_info()
    
    # Final summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for dataset_name, success in results.items():
        status = "[OK] SUCCESS" if success else "[X] FAILED"
        print(f"{dataset_name.upper():15} {status}")
    
    print("\n" + "="*60)
    
    if all(results.values()):
        print("[OK] All datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Explore datasets in notebooks/")
        print("2. Create poisoned dataset after pipeline is ready")
        print("3. Use datasets for evaluation")
    else:
        print("⚠ Some datasets failed to download")
        print("Check error messages above for alternatives")
    
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
