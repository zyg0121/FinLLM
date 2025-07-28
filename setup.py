from setuptools import setup, find_packages

setup(
    name="finllm",
    version="0.1.0",
    description="FinLLM: A Specialized Large Language Model Architecture for Stock Price Prediction",
    author="Zhou Yiguo",
    author_email="zyg0121@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "tqdm>=4.62.0",
        "transformers>=4.18.0",
        "flask>=2.0.0",
        "ta-lib>=0.4.0"
    ],
    python_requires=">=3.8",
)