import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
DATASET_FILE = os.getenv("DATASET_FILE")

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting Chroma ingestion process...")

# ----------------------------
# Initialize embeddings
# ----------------------------
try:
    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    logging.info("Ollama embeddings initialized ✅")
except Exception as e:
    logging.error(f"Error initializing embeddings: {e}")
    raise

# ----------------------------
# Remove existing Chroma DB if exists 
# ----------------------------


#REMOVE COMMENT AFTER CHANGE DB 

# if os.path.exists(DATABASE_LOCATION):
#     logging.info(f"Removing existing ChromaDB at {DATABASE_LOCATION} ...")
#     shutil.rmtree(DATABASE_LOCATION)

# ----------------------------
# Initialize Chroma vector store
# ----------------------------
try:
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DATABASE_LOCATION,
    )
    logging.info(f"Chroma vector store initialized at {DATABASE_LOCATION} with collection '{COLLECTION_NAME}'")
except Exception as e:
    logging.error(f"Error initializing Chroma: {e}")
    raise

# ----------------------------
# Initialize text splitter
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
logging.info("Text splitter initialized ✅")

# ----------------------------
# Read dataset file (plain text)
# ----------------------------
dataset_path = os.path.join(DATASET_FILE)
if not os.path.exists(dataset_path):
    logging.error(f"Dataset file not found at {dataset_path}. Run 1_fetch_from_sql.py first!")
    raise FileNotFoundError(f"{dataset_path} not found!")

with open(dataset_path, "r", encoding="utf-8") as f:
    file_content = [line.strip() for line in f if line.strip()]

logging.info(f"Loaded {len(file_content)} rows from dataset ✅")

# ----------------------------
# Chunking, embedding, and ingestion
# ----------------------------
logging.info("Starting chunking and embedding process...")

for idx, text in enumerate(file_content, start=1):
    try:
        # Split text into chunks
        documents = text_splitter.create_documents(
            [text],
            metadatas=[{"source": "bank_data", "row_index": idx}]
        )

        # Generate UUIDs for each chunk
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # Add to Chroma
        vector_store.add_documents(documents=documents, ids=uuids)

        if idx % 1000 == 0:
            logging.info(f"{idx} rows processed...")

    except Exception as e:
        logging.error(f"Error processing row {idx}: {e}")

# # ----------------------------
# # Persist Chroma DB
# # ----------------------------
# try:
#     vector_store.persist()
#     logging.info(f"ChromaDB persisted successfully at {DATABASE_LOCATION} ✅")
# except Exception as e:
#     logging.error(f"Error persisting ChromaDB: {e}")

logging.info("Ingestion process completed Successfully! ✅")