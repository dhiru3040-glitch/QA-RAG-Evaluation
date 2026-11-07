
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PRD_DIR = BASE_DIR / "knowledge_base" / "prds"
TEST_CASE_DIR = BASE_DIR / "knowledge_base" / "test_cases"


def load_prd():
    """Extract text from all PDF files in the PRD folder."""

    documents = []

    for pdf_file in PRD_DIR.glob("*.pdf"):
        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": pdf_file.name,
                        "source_type": "PRD",
                        "page": page_number
                    }
                })

    return documents


def load_test_cases():
    """Convert each CSV row into a separate text document."""

    documents = []

    for csv_file in TEST_CASE_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            content = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
            )

            documents.append({
                "content": content,
                "metadata": {
                    "source": csv_file.name,
                    "source_type": "TEST_CASE",
                    "test_case_id": str(row["Scenario TID"])
                }
            })

    return documents


if __name__ == "__main__":

    prd_documents = load_prd()
    test_documents = load_test_cases()

    all_documents = prd_documents + test_documents

    print("PRD pages extracted:", len(prd_documents))
    print("Test cases extracted:", len(test_documents))
    print("Total documents:", len(all_documents))

    for document in all_documents[:2]:
        print("\n--------------------------")
        print(document["content"][:500])
        print(document["metadata"])

