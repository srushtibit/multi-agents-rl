#!/usr/bin/env python3
"""
Setup script for the Multilingual Multi-Agent Support System.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.9+
if sys.version_info < (3, 9):
    sys.exit("Error: Python 3.9 or later is required.")

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Multilingual Multi-Agent Support System with Reinforcement Learning"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Read version
def get_version():
    version_file = os.path.join("utils", "version.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            exec(f.read())
            return locals()["__version__"]
    return "1.0.0"

setup(
    name="nexacorp-support-system",
    version=get_version(),
    author="NexaCorp AI Team",
    author_email="ai-team@nexacorp.com",
    description="Multilingual Multi-Agent Support System with Reinforcement Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/nexacorp/multilingual-support-system",
    project_urls={
        "Bug Reports": "https://github.com/nexacorp/multilingual-support-system/issues",
        "Source": "https://github.com/nexacorp/multilingual-support-system",
        "Documentation": "https://github.com/nexacorp/multilingual-support-system/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Customer Relationship Management",
        "Topic :: Communications :: Email",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "artificial-intelligence",
        "multi-agent-systems", 
        "reinforcement-learning",
        "natural-language-processing",
        "multilingual",
        "customer-support",
        "knowledge-base",
        "semantic-search",
        "pytorch",
        "streamlit"
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "faiss-gpu>=1.7.4",
        ],
        "production": [
            "gunicorn>=21.0.0",
            "uvicorn[standard]>=0.22.0",
            "redis>=4.5.0",
            "psycopg2-binary>=2.9.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "mlflow>=2.4.0",
            "prometheus-client>=0.17.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "gunicorn>=21.0.0",
            "uvicorn[standard]>=0.22.0",
            "redis>=4.5.0",
            "psycopg2-binary>=2.9.0",
            "wandb>=0.15.0",
            "mlflow>=2.4.0",
            "prometheus-client>=0.17.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "nexacorp-support=train_system:main",
            "support-system=train_system:main",
            "nexacorp-ui=ui.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.yml", "email_templates/*.html"],
        "docs": ["*.md"],
        "": ["README.md", "LICENSE", "requirements.txt"],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    test_suite="tests",
)