# setup.py
from setuptools import find_packages, setup

# 读取 README.md 作为 long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="polydiff",
    version="0.1.0",
    author="wang-work",
    author_email="",  # 您可以之后添加您的邮箱
    description="A machine learning framework for polymer search and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tom-wang813/PolyDiff",  # 您可以更新为实际的代码仓库地址
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "pytorch-lightning>=1.6.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "black>=21.7b0",
        "flake8>=3.9.2",
        "isort>=5.9.3",
    ],
    python_requires=">=3.10",
)
