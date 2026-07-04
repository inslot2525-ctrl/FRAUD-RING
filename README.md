# Fraud Ring Detection using Graph Neural Networks, Anomaly Detection and Graph Analytics

## Overview

Fraud Ring Detector is a graph-based financial fraud detection system designed to identify coordinated money laundering and fraudulent account networks rather than individual suspicious transactions.

Unlike traditional machine learning models that classify transactions independently, this system models the complete transaction history as a directed graph where:

- Nodes represent bank accounts
- Edges represent financial transactions
- Node embeddings are learned using Graph Neural Networks
- Anomalous accounts are detected using Isolation Forest
- Fraud rings are discovered using DBSCAN clustering

The project demonstrates an end-to-end Graph Machine Learning pipeline from raw transaction processing to fraud visualization.

---

# Problem Statement

Traditional fraud detection systems suffer from several limitations:

- Detect individual fraudulent transactions but miss organized fraud rings
- Ignore graph relationships between accounts
- Fail to identify shell accounts and money laundering chains
- Depend heavily on labeled fraud data

This project addresses these issues by representing financial transactions as a graph and learning structural account representations for anomaly detection.

---

# Architecture

```
                  PaySim Dataset
                  (Transactions)
                         │
                         ▼
              Graph Construction
          (Sender → Receiver Network)
                         │
                         ▼
             Node Feature Engineering
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
     ANN Encoder                    LSTM Encoder
         │                               │
         └───────────────┬───────────────┘
                         ▼
                  GraphSAGE Encoder
                         │
                  Node Embeddings
                         │
                         ▼
               Isolation Forest
                         │
                  Suspicious Accounts
                         │
                         ▼
                     DBSCAN
                         │
                         ▼
              Fraud Ring Identification
                         │
                         ▼
                Interactive Dashboard
```

---

# Dataset

Dataset:

PaySim Mobile Money Simulation Dataset

Source:

https://www.kaggle.com/datasets/ealaxi/paysim1

Current Prototype Dataset

| Property | Value |
|----------|---------|
| Transactions Used | 100,000 |
| Unique Accounts | 151,551 |
| Graph Type | Directed |
| Nodes | 151,551 |
| Edges | 100,000 |

Future Training

The project is designed to scale to the complete PaySim dataset containing over **6.3 million transactions** using GraphSAGE edge prediction and neighbor sampling.

---

# Features Engineered

For every account:

- In-degree
- Out-degree
- Total amount sent
- Total amount received
- Average outgoing amount
- Average incoming amount
- Fraud transaction count
- Fraud ratio

---

# Machine Learning Pipeline

## ANN

Purpose:

Compress engineered account statistics into dense numerical representations.

Input:

```
8 engineered features
```

Output:

```
16-dimensional embedding
```

---

## LSTM

Purpose:

Capture sequential transaction behaviour of each account.

Input sequence:

```
[
Amount,
Transaction Type,
Time Step
]
```

Sequence Length:

```
5 transactions
```

---

## Graph Neural Network

Architecture:

GraphSAGE

Input:

- Node Features
- Graph Connectivity

Output:

```
16-dimensional graph embedding
```

Current implementation performs supervised embedding learning.

Future version will replace this with:

- Edge Prediction
- Negative Sampling
- Neighbor Sampling

for production-scale graph learning.

---

## Isolation Forest

Purpose:

Detect structurally anomalous accounts.

Configuration:

```
n_estimators = 100
contamination = 1%
```

Output:

Anomaly Score

Higher score indicates higher fraud suspicion.

---

## DBSCAN

Purpose:

Group anomalous accounts into coordinated fraud rings.

Configuration:

```
eps = 0.8

min_samples = 3
```

---

# Results

Current Dataset

| Metric | Value |
|----------|-----------|
| Accounts | 151,551 |
| Transactions | 100,000 |
| Suspicious Accounts | 1,504 |
| Fraud Ring Candidates | 3 |
| Largest Ring Size | 3 Accounts |

Cluster Distribution

```
Noise (-1): 1495

Cluster 0 : 3

Cluster 1 : 3

Cluster 2 : 3
```

---

# Training Summary

## ANN

```
Epochs : 10

Training Loss

82.13

↓

24.08
```

---

## GraphSAGE

```
Epochs : 20

Training Loss

1799

↓

0.0002
```

Current GraphSAGE implementation serves as a proof-of-concept encoder.

Future versions will use edge prediction on the complete 6.3M transaction dataset.

---

# Dashboard

The Streamlit dashboard provides

- Fraud overview statistics
- Suspicious account table
- Fraud ring summary
- Cluster information

---

# Technology Stack

Python

PyTorch

PyTorch Geometric

NetworkX

Pandas

NumPy

Scikit-learn

Isolation Forest

DBSCAN

Streamlit

Plotly

---

# Project Structure

```
fraud-ring-detector/

│

├── data/

│ ├── raw/

│ └── processed/

│

├── src/

│ ├── preprocessing/

│ ├── models/

│ ├── training/

│ └── inference/

│

├── dashboard/

│ └── app.py

│

└── README.md
```

---

# Future Improvements

- GraphSAGE Edge Prediction
- Neighbor Sampling
- Full 6.3 Million Transaction Training
- Graph Contrastive Learning (GraphCL)
- SHAP Explainability
- XGBoost Risk Scoring
- Temporal Graph Neural Networks
- Neo4j Integration
- REST API Deployment
- Docker Support

---

# Skills Demonstrated

- Graph Machine Learning
- Fraud Analytics
- Graph Construction
- Network Analysis
- Deep Learning
- Graph Neural Networks
- Unsupervised Learning
- Anomaly Detection
- Feature Engineering
- Financial Risk Modelling
- Dashboard Development

---

# License

MIT License