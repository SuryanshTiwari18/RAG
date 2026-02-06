import weaviate
import os
from dotenv import load_dotenv
from weaviate.classes.init import Auth

# Load env to get WEAVIATE credentials
load_dotenv()

def reset_schema():
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")

    if not url or not api_key:
        print("Error: WEAVIATE_URL or WEAVIATE_API_KEY not found in .env")
        return

    print(f"Connecting to Weaviate at {url}...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )

    collections = ["FAQ", "story_parts", "story_parts_overlap"]

    for col in collections:
        if client.collections.exists(col):
            print(f"Deleting collection: {col}")
            client.collections.delete(col)
        else:
            print(f"Collection {col} does not exist (skipping).")

    client.close()
    print("Schema reset complete. You can now run ingestion scripts.")

if __name__ == "__main__":
    reset_schema()
