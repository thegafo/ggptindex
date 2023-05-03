# ggptindex

> `ggptindex` is a command-line utility to create and interact with indexes generated from documents using OpenAI's GPT model. It allows users to create indexes from various document types, and chat with those indexes.

# Features

- Create indexes from documents (PDF, text files, etc.) with a given tag name
- Chat with an index using its tag name
- Query an index without maintaining chat history
- List available indexes with their descriptions and creation dates
- Remove indexes by their tags

# Installation

Install the ggptindex utility using pip:

`pip install ggptindex`

# Usage

Here's a list of commands and their descriptions:

1. **list**: List the available indexes by tag.

`ggptindex list`

2. **remove**: Remove an index by its tag.

`ggptindex remove <tag>`

3. **create**: Create a new index with a given tag name from a specific file.

`ggptindex create <tag> <filename> [--description "your description"] [--chunk_size 1000] [--chunk_overlap 0]`

4. **chat**: Chat with an index using its tag name.

`ggptindex chat <tag>`

5. **query**: Query an index without maintaining chat history.

`ggptindex query <tag>`

# Example

1. Create an index from a document:

`ggptindex create cooking cooking.pdf --description "Cooking index" --chunk_size 1000 --chunk_overlap 0`

2. List available indexes:

`ggptindex list`

3. Chat with the "cooking" index:

`ggptindex chat cooking`

4. Query the "cooking" index:

`ggptindex query cooking`

5. Remove the "cooking" index:

`ggptindex remove cooking`

# License

[MIT](LICENSE)
