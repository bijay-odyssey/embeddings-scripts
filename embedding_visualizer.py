from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load model
model_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample texts
sample_texts = [
    "The sky is blue today.",
    "I love playing the guitar.",
    "Machine learning helps computers learn from data.",
    "Pizza is my favorite food.",
    "Deep learning requires large neural networks"
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
