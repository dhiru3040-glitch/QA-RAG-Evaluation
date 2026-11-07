
import re
from rag.ingestion import load_prd

documents = load_prd()

for index, document in enumerate(documents, start=1):
    # Normalize extra spaces and line breaks
    text = re.sub(r"\s+", " ", document["content"])

    print(f"\n========== PRD PAGE {index} ==========")
    print(text[:1500])
