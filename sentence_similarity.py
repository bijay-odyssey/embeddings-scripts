from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_similarity(sentence1, sentence2, model):
    emb1 = model.encode(sentence1)
    emb2 = model.encode(sentence2)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity

def main():
    print("Semantic Textual Similarity CLI")

    while True:
        sentence1 = input("\nEnter first sentence (or 'exit' to quit): ")
        if sentence1.lower() == 'exit':
            break
        sentence2 = input("Enter second sentence: ")

        similarity = calculate_similarity(sentence1, sentence2, model_mini)
        print(f"Cosine Similarity between the sentences: {similarity:.4f}")

if __name__ == "__main__":
    main()
