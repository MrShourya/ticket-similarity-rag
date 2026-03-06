from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "tickets_similarity"


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


def recreate_collection(client: QdrantClient, vector_size: int):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )