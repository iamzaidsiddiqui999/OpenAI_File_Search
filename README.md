# OpenAI File Search

A beginner-friendly Python utility that downloads supported files from an Azure Blob Storage container, uploads them into an OpenAI vector store, and enables question answering using the Responses API with the `file_search` tool.

## Features

- Connects to Azure Blob Storage
- Downloads supported file types from a container
- Uploads new or changed files into an OpenAI vector store
- Queries the vector store via Microsoft Foundry / OpenAI Responses API
- Reuses the same vector store across runs

## Supported file types

- `.pdf`
- `.txt`
- `.md`
- `.docx`
- `.csv`
- `.json`

## Requirements

- Python 3.10+
- Azure Storage account and container
- Microsoft Foundry endpoint and deployment
- GitHub repository: `iamzaidsiddiqui999/OpenAI_File_Search`

## Setup

1. Clone the repository:

```bash
git clone https://github.com/iamzaidsiddiqui999/OpenAI_File_Search.git
cd OpenAI_File_Search
```

2. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and set the required values:

```text
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_ACCOUNT_KEY=your_storage_account_key
AZURE_STORAGE_CONTAINER_NAME=your_container_name
AZURE_STORAGE_BLOB_PREFIX=
FOUNDRY_ENDPOINT=https://your-foundry-endpoint
FOUNDRY_DEPLOYMENT_NAME=gpt-4.1
VECTOR_STORE_NAME=azure-files-vector-store
```

- `AZURE_STORAGE_BLOB_PREFIX` is optional. Leave blank to download all supported blobs.
- `FOUNDRY_DEPLOYMENT_NAME` defaults to `gpt-4.1` if not set.
- `VECTOR_STORE_NAME` defaults to `azure-files-vector-store`.

## Usage

Run the application:

```bash
python main.py
```

The script will:

1. Validate required configuration.
2. Connect to Azure Storage and download supported blobs.
3. Create or reuse a vector store in OpenAI.
4. Upload only new or changed files.
5. Prompt you to ask questions interactively.

Type `exit` or `quit` to stop the app.

## State files

- `vector_store_state.json` stores the vector store ID and content hashes of uploaded files to avoid duplicate uploads.
- `downloads/` contains downloaded files from Azure Blob Storage.

## Notes

- The project uses `DefaultAzureCredential` for Foundry authentication, so ensure your Azure identity is configured for access.
- If Azure or Foundry authentication fails, the script prints a clear error and exits.

## License

This repository does not include a license file. Add one if needed.
