from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sample_texts = [
    "The sky is blue today.",
    "I love playing the guitar.",
    "Machine learning helps computers learn from data.",
    "Pizza is my favorite food.",
    "Deep learning requires large neural networks"
]

categories = ["Weather", "Music", "Technology", "Food", "Technology"]

def classify_input(user_input, model):
    emb_input = model.encode(user_input)
    db_emb = model.encode(sample_texts)

    sims = cosine_similarity([emb_input], db_emb)[0]
    top_idx = np.argmax(sims)
    return categories[top_idx]

def main():
    print("Text Classification Based on embeddings CLI")

    while True:
        user_text = input("\nEnter text to classify (or 'exit' to quit): ")
        if user_text.lower() == 'exit':
            break

        category = classify_input(user_text, model_mini)
        print(f"Predicted Category: {category}")

if __name__ == "__main__":
    main()