import milvus_utils

# This is where you would get your job data from your company's database or API
my_company_jobs = [
    {
        "job_id": "PY-DEV-001",
        "title": "Senior Python Developer",
        "company": "AI Resume Pro Corp",
        "location": "Remote",
        "description": "Seeking a Senior Python Developer with 5+ years of experience..."
    },
    {
        "job_id": "ML-ENG-002",
        "title": "Machine Learning Engineer",
        "company": "AI Resume Pro Corp",
        "location": "New York, NY",
        "description": "Join our AI team as a Machine Learning Engineer..."
    }
]

print("Populating Milvus with internal jobs...")
milvus_utils.upsert_jobs_to_milvus(my_company_jobs)
print("Population complete.")