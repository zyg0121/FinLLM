# FinLLM: A Specialized Large Language Model Architecture for Stock Price Prediction

FinLLM is a domain-adapted large language model that natively ingests multi-modal market information (prices, fundamentals, news, ESG signals) to predict stock price movements. It implements a hierarchical self-attention pipeline that couples temporal convolutions with cross-modal fusion to capture both micro-structure and macro narratives.

## Key Features

- **Multi-modal data fusion**: Combines structured financial data with unstructured text
- **Hierarchical self-attention**: Advanced transformer architecture for financial data
- **Risk-aware prediction**: Includes expected shortfall objectives for better risk management
- **Deployment-ready**: Low-latency inference engine for production environments

## Architecture

The FinLLM architecture consists of the following key components:

1. **Data Layer**: Collects, preprocesses, and stores data from various sources including historical stock data, real-time market data, news articles, and social media sentiment.

2. **Feature Engineering Layer**: Transforms raw data into meaningful features using techniques like sliding window methods, text preprocessing, and feature fusion.

3. **Core Model Layer**:
   - Self-Attention Mechanism Module: Leverages transformer technology to identify key financial information
   - Time-Series Feature Extraction Module: Uses BiLSTM networks to model historical stock price movements
   - Risk-Aware Head: Incorporates expected shortfall objectives for better risk management

4. **Deployment and Application Layer**: Enables real-time deployment with Docker containers and cloud infrastructure.

## Installation

```bash
git clone https://github.com/zyg0121/finllm.git
cd finllm
pip install -e .