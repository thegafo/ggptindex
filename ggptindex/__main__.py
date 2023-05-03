# index.py
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timezone
from prettytable import PrettyTable

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

def format_datetime(datetime_str):
    dt = datetime.fromisoformat(datetime_str)
    dt_local = dt.replace(tzinfo=timezone.utc).astimezone(tz=None)  # Convert to local time
    return dt_local.strftime("%Y-%m-%d %H:%M")

def query_index(tag, index_storage):
    index_dir = index_storage / tag

    if not index_dir.exists():
        print(f"No index found with tag '{tag}'.")
        return

    embeddings = OpenAIEmbeddings()
    db = Chroma(collection_name=tag, persist_directory=str(index_dir), embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name='text-davinci-002', max_tokens=1000),
                                     chain_type="stuff", retriever=retriever)

    print(f"Querying index '{tag}'...")
    while True:
        try:
            query = input(">>> ")
            if query.strip().lower() == 'exit':
                break
            print(qa.run(query))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    print("\nExiting query...")


def chat_with_index(tag, index_storage):
    index_dir = index_storage / tag

    if not index_dir.exists():
        print(f"No index found with tag '{tag}'.")
        return

    embeddings = OpenAIEmbeddings()
    db = Chroma(collection_name=tag, persist_directory=str(index_dir), embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, verbose=True), retriever=retriever)

    chat_history = []

    print(f"Chatting with index '{tag}'...")
    while True:
        try:
            query = input(">>> ")
            if query.strip().lower() == 'exit':
                break
            result = qa({"question": query, "chat_history": chat_history})
            print(result['answer'])
            chat_history.append((query, result["answer"]))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    print("\nExiting chat...")


def create_index(tag, filename, index_storage, description, chunk_size, chunk_overlap):
    embeddings = OpenAIEmbeddings()
    loader = UnstructuredFileLoader(filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    index_dir = index_storage / tag
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=str(index_dir), collection_name=tag)
    vectordb.persist()

    # Extract the page_content from docs
    original_document = documents[0].page_content
    split_docs = [doc.page_content for doc in docs]

    # Store the description, creation datetime, original document, and split docs in a metadata file
    metadata = {
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "original_document": original_document,
        "split_docs": split_docs,
    }
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"Index created with tag '{tag}'.")


def list_indexes(index_storage):
    for index_dir in index_storage.glob('*'):
        if index_dir.is_dir():
            # Load the description from the metadata file
            description = "<no description>"
            metadata_path = index_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    description = metadata.get("description", "")
            
            print(f"{index_dir.name}: {description}")

def table_indexes(index_storage):
    table = PrettyTable()
    table.field_names = ["Created At", "Tag", "Description"]

    for index_dir in index_storage.glob('*'):
        if index_dir.is_dir():
            # Load the description and creation datetime from the metadata file
            description = ""
            created_at = ""
            metadata_path = index_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    description = metadata.get("description", "")
                    created_at = metadata.get("created_at", "")
                    if created_at:
                        created_at = format_datetime(created_at)

            table.add_row([created_at, index_dir.name, description])

    print(table)


def remove_index(tag, index_storage):
    index_dir = index_storage / tag
    if index_dir.exists():
        shutil.rmtree(index_dir)
        print(f"Index with tag '{tag}' has been removed.")
    else:
        print(f"No index found with tag '{tag}'.")


def main():
    parser = argparse.ArgumentParser(description='Utility tool for creating and managing document indexes.')
    subparsers = parser.add_subparsers(dest='command')

    # list command
    list_parser = subparsers.add_parser('list', help='List available indexes by tag.')

    # remove command
    remove_parser = subparsers.add_parser('remove', help='Delete an index by tag.')
    remove_parser.add_argument('tag', help='Tag of the index to remove.')

    # create command
    create_parser = subparsers.add_parser('create', help='Create a new index from a file with a given tag name.')
    create_parser.add_argument('tag', help='Tag name for the new index.')
    create_parser.add_argument('filepath', help='Path to the file to create the index from.')
    create_parser.add_argument('--description', help='Description for the new index.', default="")

    # Add chunk_size and chunk_overlap arguments to create_parser
    create_parser.add_argument('--chunk_size', type=int, help='Size of text chunks for the index.', default=1000)
    create_parser.add_argument('--chunk_overlap', type=int, help='Overlap between text chunks for the index.', default=0)

    # chat command
    chat_parser = subparsers.add_parser('chat', help='Chat with an index by a given tag name.')
    chat_parser.add_argument('tag', help='Tag name of the index to chat with.')

    # query command
    query_parser = subparsers.add_parser('query', help='Query an index by a given tag name.')
    query_parser.add_argument('tag', help='Tag name of the index to query.')

    args = parser.parse_args()

    index_storage = Path.home() / '.gpt-indexes'

    if args.command == 'list':
        table_indexes(index_storage)
    elif args.command == 'remove':
        remove_index(args.tag, index_storage)
    elif args.command == 'create':
        create_index(args.tag, args.filepath, index_storage, args.description, args.chunk_size, args.chunk_overlap)
    elif args.command == 'chat':
        chat_with_index(args.tag, index_storage)
    elif args.command == 'query':
        query_index(args.tag, index_storage)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
