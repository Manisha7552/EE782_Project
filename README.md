# EE782_Project
Anushka Sharma(22b1267), Manisha Sahu(22b1273)

# GNN Retraining with Error Flow Graph (EFG) Guided Node Weights

This repository contains a **single Jupyter notebook** implementing retraining of Graph Neural Networks (GNNs) for node classification using **Error Flow Graph (EFG)** guided node weights to handle class imbalance.

---
## Datasets

The imbalanced datasets of cora, citeseer, pubmed  used in this project are available from **IGL Bench** > go to node level>class imbalance folder and download each dataset individually:

- [IGL Bench: Imbalanced GNN Dataset](https://drive.google.com/drive/folders/1GFfu6oXEaaB8-DkgBEsIXMid_i3br7HI?usp=drive_link)

> Download and extract the dataset into the `data/` folder before running the notebook.


## Notebook

- `GNN_EFG_Node_Weight_Retraining.ipynb` includes:

  1. Loading Planetoid datasets: **Cora, CiteSeer, PubMed**  
  2. Training baseline GNNs: **GCN, GraphSAGE, GIN, GAT**  
  3. Logging per-node errors with `GEFLogger`  
  4. Building **Error Flow Graph (EFG)**  
  5. Computing per-node weights  
  6. Retraining GNNs using weighted loss  
  7. Saving metrics and histories  

---

## Requirements

- Python 3.8+  
- PyTorch 2.x  
- PyTorch Geometric  
- NumPy, Pandas, Matplotlib, NetworkX, Scikit-learn  
- XlsxWriter  

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas matplotlib networkx scikit-learn xlsxwriter
