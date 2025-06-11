from setuptools import setup, find_packages

setup(
    name="legalqa",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "faiss-cpu",
        "pandas",
    ],
) 