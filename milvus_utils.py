import time
from typing import List, Dict, Any
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from config import logger, MILVUS_HOST, MILVUS_PORT, JOB_COLLECTION_NAME
import embedding_utils

def get_milvus_connection():
    if not connections.has_connection("default"):
        logger.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to Milvus: {e}")

def ensure_job_collection_exists():
    get_milvus_connection()
    if not utility.has_collection(JOB_COLLECTION_NAME):
        logger.info(f"Creating Milvus collection: {JOB_COLLECTION_NAME}")
        model = embedding_utils.get_embedding_model()
        embedding_dim = model.get_sentence_embedding_dimension()
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="job_id_str", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        schema = CollectionSchema(fields, "Job postings collection")
        collection = Collection(JOB_COLLECTION_NAME, schema)
        index_params = { "metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024} }
        collection.create_index("embedding", index_params)
        logger.info("Collection and index created successfully.")
    
    logger.info(f"Loading collection '{JOB_COLLECTION_NAME}' into memory.")
    try:
        Collection(JOB_COLLECTION_NAME).load()
    except Exception as e:
        logger.error(f"Failed to load Milvus collection into memory: {e}", exc_info=True)
        pass

def upsert_jobs_to_milvus(job_listings: List[Dict[str, Any]]):
    get_milvus_connection()
    collection = Collection(JOB_COLLECTION_NAME)
    model = embedding_utils.get_embedding_model()
    existing_job_ids = set()
    try:
        # We need to make sure we query a field that exists.
        # Let's assume job_id_str is the unique identifier from the source.
        for result in collection.query(expr="id > 0", output_fields=["job_id_str"], limit=16384):
            if result.get('job_id_str'):
                existing_job_ids.add(result['job_id_str'])
    except Exception as e:
        logger.warning(f"Could not query existing jobs, may insert duplicates: {e}")

    new_jobs = [job for job in job_listings if job.get("job_id") and job.get("job_id") not in existing_job_ids]
    if not new_jobs:
        logger.info("No new jobs to insert into Milvus.")
        return 0

    logger.info(f"Preparing to insert {len(new_jobs)} new jobs into Milvus.")
    
    # --- THIS IS THE FIX ---
    # We must ensure that every piece of data is a string and handle None values BEFORE creating embeddings or inserting.
    
    # 1. Sanitize the data first
    sanitized_jobs = []
    for job in new_jobs:
        sanitized_jobs.append({
            "job_id": str(job.get("job_id", "")),
            "title": str(job.get("title", "Untitled Job")),
            "company": str(job.get("company", "Unknown Company")),
            "location": str(job.get("location", "Unspecified Location")),
            "description": str(job.get("description", ""))
        })

    # 2. Create embeddings from the sanitized data
    full_texts = [f"Title: {j['title']}. Company: {j['company']}. Description: {j['description']}" for j in sanitized_jobs]
    embeddings = model.encode(full_texts, normalize_embeddings=True)
    
    # 3. Create the data payload for Milvus, ensuring truncation
    data_to_insert = [
        [job["job_id"][:512] for job in sanitized_jobs],
        [job["title"][:512] for job in sanitized_jobs],
        [job["company"][:512] for job in sanitized_jobs],
        [job["location"][:256] for job in sanitized_jobs],
        [job["description"][:8192] for job in sanitized_jobs], 
        embeddings,
    ]
    
    collection.insert(data_to_insert)
    collection.flush()
    logger.info(f"Successfully inserted {len(new_jobs)} jobs.")
    return len(new_jobs)

def search_jobs_in_milvus(resume_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Searches for jobs in Milvus that are similar to the provided resume text.
    This version uses a robust two-step process: search for IDs, then query for data.
    """
    get_milvus_connection()
    collection = Collection(JOB_COLLECTION_NAME)
    model = embedding_utils.get_embedding_model()
    
    # 1. Create the query vector
    query_vector = model.encode(resume_text, normalize_embeddings=True)
    search_params = { "metric_type": "L2", "params": {"nprobe": 16} }
    
    # 2. Perform the initial vector search to get the IDs of the top_k matches
    logger.info("Step 1: Performing vector search to find matching IDs.")
    search_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        # We only need the ID and distance from the search
        output_fields=["id"] 
    )

    if not search_results or not search_results[0]:
        logger.warning("Milvus search returned no results.")
        return []

    # 3. Extract the IDs and their corresponding scores
    hit_ids = [hit.id for hit in search_results[0]]
    id_to_score_map = {hit.id: max(0, (2 - hit.distance) / 2) * 100 for hit in search_results[0]}
    logger.info(f"Step 2: Found {len(hit_ids)} matching IDs. Now fetching full data.")

    # 4. Use a Milvus 'query' to retrieve the full data for the matched IDs
    # The 'in' operator is powerful for this.
    query_expression = f"id in {hit_ids}"
    
    try:
        query_results = collection.query(
            expr=query_expression,
            output_fields=["job_id_str", "title", "company", "location", "description", "id"]
        )
    except Exception as e:
        logger.error(f"Milvus query failed after search: {e}", exc_info=True)
        return []

    # 5. Combine the query results with their scores
    recommendations = []
    for item in query_results:
        item_id = item.pop('id') # Remove the internal ID from the final dict
        item['match_score'] = round(id_to_score_map.get(item_id, 0), 2)
        recommendations.append(item)

    # 6. Sort by match score in descending order
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    
    return recommendations