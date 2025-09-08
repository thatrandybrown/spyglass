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

    # handle 'add' command
    if command == "add":
        if len(sys.argv) >= 4:
            title = sys.argv[2]
            content = " ".join(sys.argv[3:])
            add_document(title, content)
            print(f"Total documents: {len(documents)}")
        else:
            print("Usage: python main.py add <title> <content>")
    # handle unknown command
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py <add> <title> <content>")
