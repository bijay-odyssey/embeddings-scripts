from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models
model_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_large = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Sample texts
sample_texts = [
    "The sky is blue today.",
    "I love playing the guitar.",
    "Machine learning helps computers learn from data.",
    "Pizza is my favorite food.",
    "Deep learning requires large neural networks"
]

def get_top_k(input_vec, db_vectors, k=3):
    sims = cosine_similarity([input_vec], db_vectors)[0]
    top_k_idx = np.argsort(sims)[::-1][:k]
    return [(sample_texts[i], sims[i]) for i in top_k_idx]

def main():
    print("Embedding and Similarity Search CLI")

    while True:
        user_text = input("\nEnter text (or 'exit' to quit): ")
        if user_text.lower() == 'exit':
            break
        emb_mini = model_mini.encode(user_text)
        emb_large = model_large.encode(user_text)

        print("Embedding Sizes:")
        print("all-MiniLM-L6-v2 =>", emb_mini.shape)
        print("all-mpnet-base-v2 =>", emb_large.shape)

        # Compute DB embeddings for both models
        db_emb_mini = model_mini.encode(sample_texts)
        db_emb_bge = model_large.encode(sample_texts)

        print("\nTop 3 similar texts using all-MiniLM-L6-v2:")
        for text, score in get_top_k(emb_mini, db_emb_mini):
            print(f"Text: {text} | Similarity: {score:.4f}")

        print("\nTop 3 similar texts using all-mpnet-base-v2:")
        for text, score in get_top_k(emb_large, db_emb_bge):
            print(f"Text: {text} | Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
    