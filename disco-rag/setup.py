from setuptools import setup, find_packages

setup(
    name="disco-rag",
    version="1.0.0",
    description="Disco-RAG: 基于修辞结构理论的检索增强生成框架",
    author="Disco-RAG Implementation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
    ],
    extras_require={
        "semantic": ["sentence-transformers>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "disco-rag=disco_rag.cli:main",
        ],
    },
)
