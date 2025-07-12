from setuptools import setup, find_packages

setup(
    name="embedding-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "sentence-transformers>=2.2.2",
        "tiktoken>=0.5.1",
        "uvicorn[standard]>=0.23.2",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.3.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "ruff>=0.0.290",
        ],
    },
    python_requires=">=3.8",
)
