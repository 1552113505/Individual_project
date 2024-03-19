# Rerank Evaluation for TREC-DL

## Overview

This project provides an evaluation framework for reranking tasks on the TREC-DL 2019/2020 datasets. It supports evaluations using several models including LLaMA2, Zephyr, and Vicuna with initial retrieval (first-stage recall) performed by BM25 and SPLADE.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
cd <repository-name>
pip install -r requirements.txt   
```

## Running Evaluations

To run evaluations across all combinations of models, years, and first-stage results, use the following command:
```bash
python evaluate.py --config config.yaml
```


### Configuration

A YAML configuration file (`config.yaml`) is used to specify the parameters for the evaluation. This includes model names, dataset years, and first-stage results methods. 
 
  
