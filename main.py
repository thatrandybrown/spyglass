import sys

# print(sys.argv[1])

# Document storage - simple in-memory list of dictionaries
documents = []

def add_document(title, content):
    doc_id = len(documents)
    documents.append({"id": doc_id, "title": title, "content": content})
    print(f"Added document '{title}' with ID {doc_id}")

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
