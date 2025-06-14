from pymilvus import utility, connections
from config import MILVUS_HOST, MILVUS_PORT, JOB_COLLECTION_NAME

def cleanup():
    """Connects to Milvus and forcefully drops the job postings collection."""
    print(f"Attempting to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Connection successful.")
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return

    print(f"Checking for collection: '{JOB_COLLECTION_NAME}'...")
    if utility.has_collection(JOB_COLLECTION_NAME):
        print(f"Collection found. Dropping collection '{JOB_COLLECTION_NAME}'...")
        try:
            utility.drop_collection(JOB_COLLECTION_NAME)
            print("Collection dropped successfully!")
        except Exception as e:
            print(f"Error dropping collection: {e}")
    else:
        print("Collection not found. No action needed.")

    connections.disconnect("default")
    print("Cleanup complete.")


if __name__ == "__main__":
    cleanup()