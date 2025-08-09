#!/usr/bin/env python3
"""
Installation script for the Multilingual Multi-Agent Support System.
Handles dependency installation with Windows compatibility.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"{'=' * 60}")
    print(f"🔧 {description}")
    print(f"Running: {command}")
    print(f"{'=' * 60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version < (3, 9):
        print(f"❌ Python 3.9+ is required. You have {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_minimal_requirements():
    """Install minimal requirements first."""
    print("\n🚀 Installing minimal requirements...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install minimal requirements
    if run_command(f"{sys.executable} -m pip install -r requirements-minimal.txt", "Installing minimal requirements"):
        print("✅ Minimal requirements installed successfully!")
        return True
    else:
        print("❌ Failed to install minimal requirements")
        return False

def install_optional_packages():
    """Install optional packages one by one."""
    print("\n🔧 Installing optional packages...")
    
    optional_packages = [
        ("googletrans==4.0.0rc1", "Google Translate support"),
        ("stable-baselines3>=2.0.0", "Reinforcement Learning"),
        ("gymnasium>=0.28.1", "RL Environment"),
        ("tensorboard>=2.13.0", "Training visualization"),
        ("gradio>=3.35.0", "Additional UI option"),
        ("redis>=4.5.0", "Caching support"),
        ("pytest>=7.4.0", "Testing framework"),
        ("black>=23.0.0", "Code formatting"),
    ]
    
    success_count = 0
    for package, description in optional_packages:
        if run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {description}"):
            success_count += 1
        else:
            print(f"⚠️ Skipping {package} - not critical for core functionality")
    
    print(f"\n📊 Installed {success_count}/{len(optional_packages)} optional packages")
    return success_count

def download_nltk_data():
    """Download required NLTK data."""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_downloads = [
            ('punkt', 'Sentence tokenizer'),
            ('stopwords', 'Stop words'),
            ('wordnet', 'WordNet lemmatizer'),
        ]
        
        for package, description in nltk_downloads:
            try:
                print(f"📥 Downloading {description}...")
                nltk.download(package, quiet=True)
                print(f"✅ Downloaded {package}")
            except Exception as e:
                print(f"⚠️ Failed to download {package}: {e}")
        
        return True
    except ImportError:
        print("❌ NLTK not installed, skipping data download")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "logs",
        "models", 
        "checkpoints",
        "results",
        "cache",
        "exports"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"⚠️ Could not create {directory}: {e}")

def test_imports():
    """Test if critical packages can be imported."""
    print("\n🧪 Testing critical imports...")
    
    critical_imports = [
        ("torch", "PyTorch"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("streamlit", "Streamlit"),
        ("yaml", "PyYAML"),
        ("langdetect", "Language Detection"),
    ]
    
    failed_imports = []
    
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Some imports failed: {failed_imports}")
        print("The system may have limited functionality.")
        return False
    else:
        print("\n🎉 All critical imports successful!")
        return True

def main():
    """Main installation process."""
    print("🤖 NexaCorp AI Support System - Installation Script")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    
    # # Check Python version
    # if not check_python_version():
    #     sys.exit(1)
    
    # # Install minimal requirements
    # if not install_minimal_requirements():
    #     print("\n❌ Installation failed at minimal requirements stage")
    #     sys.exit(1)
    
    # # Install optional packages
    # install_optional_packages()
    
    # Download NLTK data
    download_nltk_data()
    
    # Setup directories
    setup_directories()
    
    # Test imports
    test_success = test_imports()
    
    print("\n" + "=" * 60)
    if test_success:
        print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("\n🚀 Next steps:")
        print("1. Build knowledge base: python train_system.py --build-kb")
        print("2. Launch UI: streamlit run ui/streamlit_app.py")
        print("3. Start training: python train_system.py --episodes 100")
    else:
        print("⚠️ INSTALLATION COMPLETED WITH WARNINGS")
        print("Some optional features may not work properly.")
        print("You can still use the core functionality.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()