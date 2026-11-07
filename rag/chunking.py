
import re
from sentence_transformers import SentenceTransformer
from rag.ingestion import load_prd, load_test_cases


# ==========================================
# 1. MODEL AND CHUNKING CONFIGURATION
# ==========================================

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
tokenizer = model.tokenizer

CHUNK_SIZE = 900
OVERLAP = 100


# ==========================================
# 2. ACTUAL VWO PRD HEADINGS
# ==========================================

PRD_HEADINGS = [
    ("1", "Product Overview"),
    ("2", "Business Objectives"),
    ("3", "Target Users"),
    ("4", "Core Features & Capabilities"),
    ("4.1", "Experimentation & Testing"),
    ("4.2", "Behavioral Insights"),
    ("4.3", "Personalization"),
    ("4.4", "Program & Workflow Management"),
    ("4.5", "Integrations"),
    ("5", "User Flows"),
    ("6", "Functional Requirements"),
    ("7", "Non-Functional Requirements"),
    ("8", "Success Metrics (KPIs)"),
    ("9", "Pricing & Licensing"),
    ("10", "Risks & Mitigations"),
    ("11", "Future Enhancements"),
    ("12", "Appendices"),
]


# ==========================================
# 3. TOKEN COUNTING
# ==========================================

def count_tokens(text):
    """Count tokens using the Nomic tokenizer."""

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# ==========================================
# 4. SPLIT LONG SECTIONS
# ==========================================

def split_long_section(text):
    """Split long sections into overlapping token chunks."""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []
    start = 0

    while start < len(token_ids):

        end = min(start + CHUNK_SIZE, len(token_ids))

        chunk_text = tokenizer.decode(
            token_ids[start:end]
        )

        chunks.append(chunk_text)

        if end == len(token_ids):
            break

        start = end - OVERLAP

    return chunks


# ==========================================
# 5. IDENTIFY ACTUAL PRD HEADINGS
# ==========================================

def identify_headings(text):
    """Find known PRD headings in normalized PDF text."""

    headings = []

    for number, title in PRD_HEADINGS:

        pattern = (
            rf"(?<![\d.]){re.escape(number)}\s*\.\s*"
            rf"{re.escape(title)}"
            rf"(?=\s|$)"
        )

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            headings.append({
                "number": number,
                "title": title,
                "start": match.start(),
                "end": match.end()
            })

    # Sort headings by their actual position in the document
    headings.sort(key=lambda item: item["start"])

    return headings


# ==========================================
# 6. PRD CHUNKING
# ==========================================

def chunk_prd():
    """Create section-aware PRD chunks with metadata."""

    documents = load_prd()

    page_texts = []

    # Normalize extracted PDF text page by page
    for document in documents:

        text = document["content"]

        text = re.sub(r"\s+", " ", text).strip()

        page_texts.append({
            "page": document["metadata"].get("page"),
            "content": text
        })

    # Combine normalized pages
    full_text = " ".join(
        page["content"] for page in page_texts
    )

    # Identify actual section headings
    headings = identify_headings(full_text)

    if not headings:
        raise ValueError(
            "No PRD headings were identified. "
            "Check PDF extraction and heading definitions."
        )

    sections = []

    # Create content for each section
    for index, heading in enumerate(headings):

        start = heading["start"]

        if index + 1 < len(headings):
            end = headings[index + 1]["start"]
        else:
            end = len(full_text)

        section_text = full_text[start:end].strip()

        # Find page range for this section
        page_start = None
        page_end = None

        current_position = 0

        for page in page_texts:

            page_end_position = (
                current_position + len(page["content"])
            )

            if (
                page_start is None
                and start < page_end_position
            ):
                page_start = page["page"]

            if (
                start < page_end_position
                and end > current_position
            ):
                page_end = page["page"]

            current_position = page_end_position + 1

        sections.append({
            "section_number": heading["number"],
            "section_title": heading["title"],
            "page_start": page_start,
            "page_end": page_end,
            "content": section_text
        })

    all_chunks = []

    # Split sections only when they exceed the token limit
    for section in sections:

        section_text = section["content"]

        if count_tokens(section_text) <= CHUNK_SIZE:

            section_chunks = [section_text]

        else:

            section_chunks = split_long_section(
                section_text
            )

        for index, chunk in enumerate(
            section_chunks,
            start=1
        ):

            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "source_type": "PRD",
                    "section_number": section["section_number"],
                    "section_title": section["section_title"],
                    "section_chunk": index,
                    "page_start": section["page_start"],
                    "page_end": section["page_end"]
                }
            })

    return all_chunks


# ==========================================
# 7. TEST CASE CHUNKING
# ==========================================

def chunk_test_cases():
    """Keep each complete test case as one chunk."""

    documents = load_test_cases()

    all_chunks = []

    for document in documents:

        all_chunks.append({
            "content": document["content"],
            "metadata": document["metadata"]
        })

    return all_chunks


# ==========================================
# 8. CHUNK VALIDATION
# ==========================================

def validate_chunks(prd_chunks, tc_chunks):

    print("\n========== CHUNK VALIDATION ==========")

    oversized_chunks = []

    for index, chunk in enumerate(
        prd_chunks,
        start=1
    ):

        token_count = count_tokens(chunk["content"])

        if token_count > CHUNK_SIZE:
            oversized_chunks.append(index)

    print("PRD chunks:", len(prd_chunks))
    print("Test case chunks:", len(tc_chunks))

    print(
        "Total chunks:",
        len(prd_chunks) + len(tc_chunks)
    )

    print(
        "Oversized PRD chunks:",
        oversized_chunks
    )

    print(
        "PRD metadata present:",
        all(
            all(
                key in chunk["metadata"]
                for key in [
                    "source_type",
                    "section_number",
                    "section_title",
                    "section_chunk",
                    "page_start",
                    "page_end"
                ]
            )
            for chunk in prd_chunks
        )
    )

    print(
        "Test case metadata preserved:",
        all(
            "test_case_id" in chunk["metadata"]
            for chunk in tc_chunks
        )
    )

    print("\n========== ALL PRD CHUNKS ==========")

    for index, chunk in enumerate(
        prd_chunks,
        start=1
    ):

        print(f"\nCHUNK {index}")

        print("Metadata:", chunk["metadata"])

        print(
            "Tokens:",
            count_tokens(chunk["content"])
        )

        print(
            "Content:",
            chunk["content"][:300]
        )

    print("\n========== TEST CASE VALIDATION ==========")

    for chunk in tc_chunks:
        print(chunk["metadata"])


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    prd_chunks = chunk_prd()

    tc_chunks = chunk_test_cases()

    validate_chunks(
        prd_chunks,
        tc_chunks
    )
