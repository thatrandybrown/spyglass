import sys

# print(sys.argv[1])

# Document storage - simple in-memory list of dictionaries
documents = []

def add_document(title, content):
    doc_id = len(documents)
    documents.append({"id": doc_id, "title": title, "content": content})
    print(f"Added document '{title}' with ID {doc_id}")

def query_documents(query_text):
    results = []
    query_words = query_text.lower().split()
    for doc in documents:
        content_lower = doc["content"].lower()
        count = sum(content_lower.count(word) for word in query_words)
        if count > 0:
            results.append((doc, count))
    # Sort by match count (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    command = ""
    # switch on first command line argument
    while command!= "exit":
        command = input("\n> ").strip()
        print(command)
        if command.startswith("add "):
            title, content = command.split(" ", 1)[1].split(" ", 1)
            if not title or not content:
                print("Usage: add <title> <content>")
                continue
            add_document(title, content)
            print(f"Total documents: {len(documents)}")
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py <add> <title> <content>")
