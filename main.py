import sys

# print(sys.argv[1])

# Document storage - simple in-memory list of dictionaries
documents = []

# read document from file path
def read_document(file_path):
    with open(file_path, "r") as file:
        return file.read()

def add_document(title, content):
    doc_id = len(documents)
    documents.append({"id": doc_id, "title": title, "content": content})
    print(f"Added document '{title}' with ID {doc_id}")

def query_documents(query_text):
    results = []
    query_words = query_text.lower().split()
    for doc in documents:
        content_lower = doc["content"].lower()
        content_words = content_lower.split()
        total_words = len(content_words)

        if total_words == 0:  # Avoid division by zero
            continue

        count = sum(content_lower.count(word) for word in query_words)
        if count > 0:
            # Calculate match percentage
            match_percentage = (count / total_words) * 100
            results.append((doc, count, match_percentage))

    # Sort by match percentage (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    return results

if __name__ == "__main__":
    command = ""
    # switch on first command line argument
    while command!= "exit":
        command = input("\n> ").strip()
        print(command)
        if command.startswith("add "):
            filepath = command.split(" ", 1)[1]
            add_document(filepath, read_document(filepath))
            print(f"Total documents: {len(documents)}")
        elif command.startswith("query "):
            query_text = command.split(" ", 1)[1]
            results = query_documents(query_text)
            if results:
                for doc, count, percentage in results:
                    print(f"ID {doc['id']}: '{doc['title']}' (matches: {count}, {percentage:.2f}% of document)")
            else:
                print("No documents found matching your query.")
        elif command == "exit":
            break
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py <add> <title> <content>")
