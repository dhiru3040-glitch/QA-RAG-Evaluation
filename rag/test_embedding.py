
from sentence_transformers import SentenceTransformer

# Load Nomic embedding model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# Sample QA document
text = """
search_document: TC-LGN-001 verifies that navigating
to the VWO application redirects the user to the login page.
"""

# Generate embedding
embedding = model.encode(text)

print("Model loaded successfully!")
print("Embedding dimensions:", embedding.shape)
print("First 5 vector values:", embedding[:5])
print("Maximum sequence length:", model.max_seq_length)
