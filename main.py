"""
OpenAI_File_Search
-------------------
Beginner-friendly app that:

1. Connects to an Azure Storage Account (Blob Storage).
2. Downloads supported files from a container.
3. Uploads those files into an OpenAI vector store (creating one vector
   store the first time, then reusing it on later runs).
4. Lets you ask questions answered with the Responses API's file_search
   tool, running on a GPT-4.1 deployment in Microsoft Foundry.

Run it with:
    python main.py
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError
from azure.storage.blob import BlobServiceClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Load configuration from .env
# ---------------------------------------------------------------------------
load_dotenv()

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
AZURE_STORAGE_BLOB_PREFIX = os.getenv("AZURE_STORAGE_BLOB_PREFIX", "")

FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT")
FOUNDRY_DEPLOYMENT_NAME = os.getenv("FOUNDRY_DEPLOYMENT_NAME", "gpt-4.1")

VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "azure-files-vector-store")

DOWNLOAD_DIR = Path("downloads")
STATE_FILE = Path("vector_store_state.json")  # remembers vector store id + which files were uploaded

# Files OpenAI's file_search / vector store pipeline can parse today.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".json"}

REQUIRED_SETTINGS = {
    "AZURE_STORAGE_ACCOUNT_NAME": AZURE_STORAGE_ACCOUNT_NAME,
    "AZURE_STORAGE_ACCOUNT_KEY": AZURE_STORAGE_ACCOUNT_KEY,
    "AZURE_STORAGE_CONTAINER_NAME": AZURE_STORAGE_CONTAINER_NAME,
    "FOUNDRY_ENDPOINT": FOUNDRY_ENDPOINT,
}


def check_configuration() -> None:
    """Fail fast with a clear message if required settings are missing."""
    missing = [name for name, value in REQUIRED_SETTINGS.items() if not value]
    if missing:
        print("Missing required settings in your .env file:")
        for name in missing:
            print(f"  - {name}")
        print("\nCopy .env.example to .env and fill in the real values, then re-run.")
        sys.exit(1)

    if AZURE_STORAGE_CONTAINER_NAME == "YOUR_CONTAINER_NAME":
        print(
            "AZURE_STORAGE_CONTAINER_NAME is still set to the placeholder "
            "'YOUR_CONTAINER_NAME'. Open .env and replace it with your real "
            "container name (find it in the Azure Portal under your Storage "
            "Account -> Containers)."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Local state (so we don't re-create a vector store or re-upload files
#    every time the script runs)
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"vector_store_id": None, "uploaded_files": {}}  # filename -> content hash


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def file_hash(path: Path) -> str:
    """Content hash used to detect if a local file changed since last upload."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 3. Azure Blob Storage: list + download files
# ---------------------------------------------------------------------------
def download_blobs() -> list[Path]:
    print("Connecting to Azure Storage...")
    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=AZURE_STORAGE_ACCOUNT_KEY,
        )
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        # Cheap call to confirm the container exists and the key works before listing.
        container_client.get_container_properties()
    except ClientAuthenticationError:
        print(
            "Authentication failed against Azure Storage. Double-check "
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY in .env."
        )
        sys.exit(1)
    except ResourceNotFoundError:
        print(
            f"Container '{AZURE_STORAGE_CONTAINER_NAME}' was not found on account "
            f"'{AZURE_STORAGE_ACCOUNT_NAME}'. Check the name in the Azure Portal."
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - surface any other Azure error clearly
        print(f"Could not connect to Azure Storage: {exc}")
        sys.exit(1)

    print("Connected successfully.\n")

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    downloaded_paths: list[Path] = []

    blobs = container_client.list_blobs(name_starts_with=AZURE_STORAGE_BLOB_PREFIX or None)
    blob_names = [b.name for b in blobs]
    print(f"Found {len(blob_names)} blob(s) under prefix '{AZURE_STORAGE_BLOB_PREFIX or '(root)'}'.\n")

    print("Downloading files...")
    for blob_name in blob_names:
        extension = Path(blob_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            print(f"  Skipping (unsupported type): {blob_name}")
            continue

        # Flatten virtual "folders" (blob names can contain "/") into a safe local filename.
        local_name = blob_name.replace("/", "__")
        local_path = DOWNLOAD_DIR / local_name

        try:
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"  Downloaded: {blob_name}")
            downloaded_paths.append(local_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed to download {blob_name}: {exc}")

    print()
    return downloaded_paths


# ---------------------------------------------------------------------------
# 4. Microsoft Foundry / OpenAI client
# ---------------------------------------------------------------------------
def build_openai_client() -> OpenAI:
    """
    Authenticates against Microsoft Foundry using your logged-in Azure identity
    (DefaultAzureCredential), then wraps it in the standard OpenAI Python SDK
    so we can use the Responses API.
    """
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://ai.azure.com/.default",
    )
    # Passing the callable (not calling it) lets the SDK refresh the token
    # automatically if a long-running session needs a new one.
    return OpenAI(base_url=FOUNDRY_ENDPOINT, api_key=token_provider)


# ---------------------------------------------------------------------------
# 5. Vector store: create/reuse, upload only new or changed files
# ---------------------------------------------------------------------------
def ensure_vector_store(client: OpenAI, state: dict) -> str:
    vector_store_id = state.get("vector_store_id")

    if vector_store_id:
        try:
            client.vector_stores.retrieve(vector_store_id)
            print(f"Reusing existing vector store: {vector_store_id}\n")
            return vector_store_id
        except Exception:
            print("Saved vector store id is no longer valid on the server. Creating a new one.\n")

    print("Creating/reusing vector store...")
    vector_store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    state["vector_store_id"] = vector_store.id
    state["uploaded_files"] = {}  # fresh store -> nothing uploaded yet
    save_state(state)
    print(f"Vector store ID: {vector_store.id}\n")
    return vector_store.id


def sync_files_to_vector_store(client: OpenAI, vector_store_id: str, local_files: list[Path], state: dict) -> None:
    uploaded = state.setdefault("uploaded_files", {})

    files_to_upload = []
    for path in local_files:
        current_hash = file_hash(path)
        if uploaded.get(path.name) == current_hash:
            continue  # already uploaded and unchanged - skip (keeps the app idempotent)
        files_to_upload.append((path, current_hash))

    if not files_to_upload:
        print("All files are already up to date in the vector store. Nothing to upload.\n")
        return

    print(f"Uploading {len(files_to_upload)} new/changed file(s) to OpenAI...")
    file_streams = [open(path, "rb") for path, _ in files_to_upload]
    try:
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=file_streams,
        )
    finally:
        for stream in file_streams:
            stream.close()

    for path, current_hash in files_to_upload:
        uploaded[path.name] = current_hash
    save_state(state)

    print("Files uploaded successfully.")
    print("Waiting for files to be indexed... All files are ready.\n")


# ---------------------------------------------------------------------------
# 6. Ask a question using the Responses API + file_search tool
# ---------------------------------------------------------------------------
def ask_question(client: OpenAI, vector_store_id: str, question: str) -> None:
    print("\nSearching files...\n")
    response = client.responses.create(
        model=FOUNDRY_DEPLOYMENT_NAME,
        input=question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
    )

    print("Answer:")
    print(response.output_text)

    # Pull out citation info (which file each answer chunk came from), if present.
    sources = set()
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            for annotation in getattr(content, "annotations", []) or []:
                filename = getattr(annotation, "filename", None)
                if filename:
                    sources.add(filename)

    if sources:
        print("\nSources:")
        for name in sorted(sources):
            print(f"  - {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    check_configuration()

    state = load_state()

    local_files = download_blobs()

    client = build_openai_client()
    vector_store_id = ensure_vector_store(client, state)
    sync_files_to_vector_store(client, vector_store_id, local_files, state)

    print("Ready! Ask questions about your files (type 'exit' to quit).\n")
    while True:
        question = input("Ask a question:\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            ask_question(client, vector_store_id, question)
        except Exception as exc:  # noqa: BLE001
            print(f"\nSomething went wrong calling Microsoft Foundry: {exc}\n")
        print()


if __name__ == "__main__":
    main()
