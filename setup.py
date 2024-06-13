from setuptools import setup, find_packages

setup(
    name="coir",
    version="0.1.0",
    author="xiangyang Li",
    author_email="xiangyangli@pku.edu.cn",
    description="A package for COIR evaluations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coir",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "datasets>=2.19.0",
        "numpy>=1.0.0",
        "requests>=2.26.0",
        "scikit_learn>=1.0.2",
        "scipy>=0.0.0",
        "sentence_transformers>=3.0.0",
        "torch>1.0.0",
        "tqdm>1.0.0",
        "rich>=0.0.0",
        "pytrec-eval-terrier>=0.5.6",
        "beir"
    ],
)