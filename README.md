---

# Multimodal Anomaly Detection for Network Operations Centers (NOC)

This project implements a **Multimodal Anomaly Detection System** designed for Network Operations Centers (NOC). It leverages a **Transformer-based architecture** to fuse and analyze three distinct types of data simultaneously:

1. **Text Logs** (Natural Language from routers/servers)
2. **Metrics** (Numerical time-series data like Latency, Bandwidth, Packet Loss)
3. **Alerts** (Categorical severity levels)

The system detects network anomalies by learning correlations across these modalities and provides interpretable insights into possible root causes.

---

## 🚀 Key Features

* **Multimodal Data Fusion:** Combines unstructured text logs, continuous numerical metrics, and categorical alerts into a unified embedding space.
* **Transformer Architecture:** Uses a Transformer Encoder with self-attention to capture complex relationships between logs and metrics (e.g., correlating "high latency" in text with numeric spikes).
* **Anomaly Classification:** Classifies network states as **Normal** or **Anomaly**.
* **Root Cause Hints:** Provides rule-based heuristic explanations for detected anomalies to aid NOC engineers.
* **Interactive Dashboard:** Includes a **Streamlit** web application for real-time visualization of network health, anomalies, and logs.
* **Synthetic Data Generation:** Includes a simulation module to generate realistic NOC data for training and testing without requiring sensitive real-world logs.

---

## 🛠️ System Architecture

1. **Data Ingestion & Simulation:** Generates multimodal data (Logs, Metrics, Alerts) with injected anomalies.
2. **Preprocessing:**
* **Logs:** Tokenization and Embedding (Vector size 16).
* **Metrics:** Linear Projection to match embedding dimension.
* **Alerts:** Embedding lookup.


3. **Fusion Layer:** Concatenates embeddings from all three sources into a single sequence.
4. **Model Core:**
* **Multimodal Transformer Encoder:** Processes the fused sequence using self-attention mechanisms.
* **Pooling & Classifier:** Global average pooling followed by a Multi-Layer Perceptron (MLP) for binary classification.


5. **Visualization:** Plots metrics over time and network topology graphs.

---

## 📂 Project Structure

```bash
├── multimodal_anomaly_detection.ipynb  # Main notebook containing model training & logic
├── app.py                              # Streamlit dashboard application
├── requirements.txt                    # List of dependencies
├── README.md                           # Project documentation
└── logs/                               # Directory for saving training logs (optional)

```

---

## 🔧 Installation & Setup

### Prerequisites

* Python 3.8+
* Google Colab (recommended for GPU access) or a local machine

### 1. Install Dependencies

Run the following command to install the required libraries:

```bash
pip install torch torchvision pandas numpy matplotlib networkx streamlit pyngrok

```

### 2. Train the Model

Open and run the `multimodal_anomaly_detection.ipynb` notebook. This script will:

* Generate synthetic training data.
* Train the Transformer model.
* Evaluate performance and print detected anomalies.
* Save visualization plots (`metrics_with_anomalies.png`, `topology.png`).

### 3. Run the Dashboard

To launch the interactive Streamlit dashboard:

```bash
streamlit run app.py

```

*Note: If running on Google Colab, the notebook includes code to expose the dashboard via a Cloudflare Tunnel.*

---

## 📊 Usage Guide

### Model Output

After training, the model outputs:

* **Anomaly Detection Accuracy:** True positives found vs. actual anomalies injected.
* **Contextual Logs:** Detailed printouts of anomalous logs, their associated metrics, and alert levels.
* **Root Cause Analysis:** Automatic hints such as "High latency spike" or "Error keywords in log."

### Dashboard Interface

The Streamlit app provides:

* A **DataFrame view** of network events (Logs, Metrics, Alerts).
* **Visual Alerts:** Highlighted red boxes for detected anomalies with detailed breakdowns.
* **Root Cause Explanations:** Interpretations of why an event was flagged.

---

## 📝 Example Scenario

**Input Data:**

* **Log:** `"Critical: packet loss spike on server3"`
* **Metric:** `Latency: 95ms`, `Packet Loss: 9.1%`
* **Alert Level:** `5`

**Model Prediction:** `Anomaly`
**Root Cause Hint:** `Error keywords in logs, High latency spike, Packet loss detected`

---

## 👥 Contributors
This project was developed as part of the Introduction Computer Networks (ICN) course at Amrita Vishwa Vidyapeetham.

Vepuri Satya Krishna (DL.AI.U4AID24140)

Gowripriya R (DL.AI.U4AID24113)

Yaalini R (DL.AI.U4AID24043)

---

> *This project is for educational and research purposes only and should not be used as a primary diagnostic tool without clinical validation.*
