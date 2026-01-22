# ** Embedding Experiments CLI**

This repository contains several **command-line tools** for experimenting with **text embeddings**, **cosine similarity**, **semantic search**, **text classification**, **sentence similarity**, and **embedding visualization**.
All scripts use **Sentence Transformers** and are designed to be lightweight, beginner-friendly, and useful for testing embedding behavior across several models.

---

## **File Descriptions**

### **1. `similarity_search.py`**

A CLI tool that:

* Loads two different embedding models (MiniLM & MPNet)
* Embeds user input text
* Compares it with sample sentences
* Returns the **top-k most similar texts** for each model
  Great for studying how different embedding models represent meaning differently.

---

### **2. `text_classifier.py`**

A simple embedding-based text classifier:

* Compares user input to example labeled sentences
* Predicts the category based on highest cosine similarity
  Useful to understand how embeddings can act as zero-shot classifiers.

---

### **3. `embedding_visualizer.py`**

A visualization script that:

* Embeds sample texts
* Reduces to 2D using PCA
* Plots the embeddings using matplotlib
  Shows how sentences cluster semantically in vector space.

---

### **4. `sentence_similarity.py`**

A small CLI utility that:

* Accepts two sentences
* Computes embedding similarity
  Great for testing STS (Semantic Textual Similarity) tasks.

---

## **Requirements**

Install dependencies:

```bash
pip install sentence-transformers scikit-learn numpy matplotlib
```

---

## **Usage**

### Run any script:

```bash
python similarity_search.py
python text_classifier.py
python embedding_visualizer.py
python sentence_similarity.py
```
---
## **Embedding Visualized in 2 dimensions**
<img width="1920" height="967" alt="embeddings_visualized" src="https://github.com/user-attachments/assets/9c7a0aab-8d98-4dfc-bb0f-ff320516c64a" />

---

## **Future Ideas (Contributions Welcome!)**

* Add FAISS vector indexing for fast nearest-neighbor search
* Add support for local embeddings (e5, bge, jina)
* Add an API version exposed over Flask / FastAPI
* Add GPU / CUDA auto-detection

---

