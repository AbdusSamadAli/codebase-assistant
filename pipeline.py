from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Python is a programming language.",
    "The cat is on the roof.", 
    "I love machine learning.",
    "The sky is blue."
]

embeddings = model.encode(sentences)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = "What color is the sky?"
query_embedding = model.encode([query])
print(embeddings.shape)
distances, indices = index.search(np.array(query_embedding),k=2)

print("most relevant results:")
for i in indices[0]:
    print(f"->{sentences[i]}")