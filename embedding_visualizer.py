from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load model
model_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample texts
sample_texts = [
    "I am a self-taught data scientist with hands-on experience in building real-world ML systems.",
    "I developed a fraud detection pipeline on the IEEE-CIS dataset using XGBoost and SHAP.",
    "I created a retail price forecasting model using LightGBM and Prophet with 1.7M+ rows.",
    "I built a customer churn prediction web app using Flask, Random Forest, and SQLite.",
    "I specialize in EDA, feature engineering, statistical analysis, and model tuning.",
    "I have experience with anomaly detection using Isolation Forest, LOF, DBSCAN, and clustering.",
    "I am learning about LLMs, embeddings, vector search, and RAG pipelines.",
    "I use Python, SQL, and Scikit-learn extensively in my ML workflows.",
    "I integrate SHAP for explainability in my ML models.",
    "I enjoy learning deep learning, semantic embeddings, and vector similarity search."
]


def get_embeddings(model, texts):
    return model.encode(texts)

def visualize_embeddings(embeddings, texts):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    for i, txt in enumerate(texts):
        plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.title('PCA of Text Embeddings')
    plt.show()

def main():
    print("Embedding Visualization using PCA")

    embeddings = get_embeddings(model_mini, sample_texts)
    visualize_embeddings(embeddings, sample_texts)

if __name__ == "__main__":
    main()

