import sys

# print(sys.argv[1])

# Document storage - simple in-memory list of dictionaries
documents = []

def add_document(title, content):
    doc_id = len(documents)
    documents.append({"id": doc_id, "title": title, "content": content})
    print(f"Added document '{title}' with ID {doc_id}")

# Simple document ingest from command line
if len(sys.argv) >= 3:
    title = sys.argv[1]
    content = sys.argv[2]
    add_document(title, content)
    print(f"Total documents: {len(documents)}")
else:
    print("Usage: python main.py <title> <content>")
