import uvicorn
import os
import tempfile
import asyncio
import json
import contextlib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from config import logger, MAX_FILE_SIZE, DEFAULT_USER_ID
import db_utils
import milvus_utils
from app_tools import extract_text_from_file
from app_tools import (
    review_resume as review_resume_tool,
    generate_interview_qa as generate_qa_tool,
    generate_career_roadmap as generate_career_roadmap_tool,
    simulate_interview_step as simulate_interview_step_tool,
    get_upskilling_recommendations as get_upskilling_recommendations_tool,
    recommend_jobs as recommend_jobs_tool,
)
# UPDATED IMPORT: Import the agent_executor directly
from agents import agent_executor

# --- App Initialization ---
app = FastAPI(title="AI Resume Pro API", version="7.1.0") # Version bump for fix
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Startup and Health Checks ---
@app.on_event("startup")
async def startup_event():
    """Initializes database and vector store when the application starts."""
    logger.info("API starting up...")
    try:
        db_utils.ensure_tables_exist()
        milvus_utils.ensure_job_collection_exists()
    except Exception as e:
        logger.critical(f"Fatal startup error: {e}", exc_info=True)
        raise RuntimeError(f"Startup failed: {e}")

@app.get("/health", summary="Perform a health check of the API")
async def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok"}

# --- Reusable File Validation & Handling ---
async def valid_resume_file(file: UploadFile = File(...)):
    """A FastAPI dependency to validate an uploaded file's type and size."""
    if file.filename and not file.filename.lower().endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be .txt or .pdf")
    return file

@contextlib.asynccontextmanager
async def save_temp_file(file: UploadFile):
    """Context manager to save an uploaded file to a temporary path and ensure cleanup."""
    temp_path = None
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# --- API Endpoints (Direct Tool Calls for Reliability) ---

@app.post("/review_resume", summary="Review a single resume")
async def review_resume(file: UploadFile = Depends(valid_resume_file), job_description: str = Form(...)):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    async with save_temp_file(file) as temp_path:
        try:
            resume_text = await extract_text_from_file(temp_path, file.filename)
            review_data = review_resume_tool.invoke({
                "resume_text": resume_text, "job_description": job_description
            })
            resume_id = db_utils.save_resume(DEFAULT_USER_ID, file.filename, resume_text)
            job_id = db_utils.save_job_description(job_description)
            db_utils.save_review(resume_id, job_id, review_data)
            return review_data
        except Exception as e:
            logger.error(f"Error in /review_resume: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank_resumes", summary="Rank multiple resumes")
async def rank_resumes(files: List[UploadFile] = File(...), job_description: str = Form(...)):
    async def process_and_rank(file: UploadFile):
        await valid_resume_file(file)
        async with save_temp_file(file) as temp_path:
            try:
                resume_text = await extract_text_from_file(temp_path, file.filename)
                review_data = review_resume_tool.invoke({
                    "resume_text": resume_text, "job_description": job_description
                })
                return {**review_data, "filename": file.filename}
            except Exception as e:
                return {"filename": file.filename, "score": 0, "feedback": f"Error: {e}", "suggestions": []}
    tasks = [process_and_rank(file) for file in files]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results

@app.post("/generate_qa", summary="Generate interview Q&A")
async def generate_qa(file: UploadFile = Depends(valid_resume_file), job_description: str = Form(...)):
    async with save_temp_file(file) as temp_path:
        try:
            resume_text = await extract_text_from_file(temp_path, file.filename)
            qa_data = generate_qa_tool.invoke({
                "resume_text": resume_text, "job_description": job_description
            })
            return qa_data
        except Exception as e:
            logger.error(f"Error in /generate_qa: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/interview_simulator", summary="Simulate an interview step")
async def interview_simulator(user_answer: str = Form(...), conversation_history_json: str = Form(...)):
    try:
        history = json.loads(conversation_history_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid history format: {e}")
    try:
        response_text = simulate_interview_step_tool.invoke({
            "user_answer": user_answer, "conversation_history": history
        })
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error in /interview_simulator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_career_plan", summary="Generate a career roadmap")
async def generate_career_plan(role: str = Form(...), description: str = Form(...)):
    if not role.strip() or not description.strip():
        raise HTTPException(status_code=400, detail="Role and description cannot be empty.")
    try:
        plan_data = generate_career_roadmap_tool.invoke({
            "role": role, "description": description
        })
        db_utils.save_career_plan(DEFAULT_USER_ID, role, plan_data)
        return plan_data
    except Exception as e:
        logger.error(f"Error in /generate_career_plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_user_dashboard", summary="Get all review history for a user's dashboard")
async def get_user_dashboard():
    try:
        dashboard_data = db_utils.get_reviews_for_user(DEFAULT_USER_ID)
        return dashboard_data
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve dashboard data.")

@app.post("/get_upskilling_recs", summary="Get upskilling and certification recommendations")
async def get_upskilling_recs(job_title: str = Form(...), current_skills: str = Form(...)):
    if not job_title.strip():
        raise HTTPException(status_code=400, detail="Job title cannot be empty.")
    try:
        recommendations = get_upskilling_recommendations_tool.invoke({
            "job_title": job_title, "current_skills": current_skills
        })
        return recommendations
    except Exception as e:
        logger.error(f"Error in /get_upskilling_recs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend_jobs", summary="Recommend jobs based on a resume")
async def recommend_jobs_endpoint(
    file: UploadFile = Depends(valid_resume_file),
    job_title_filter: str = Form(default="")
):
    async with save_temp_file(file) as temp_path:
        try:
            resume_text = await extract_text_from_file(temp_path, file.filename)
            job_list = recommend_jobs_tool.invoke({
                "resume_text": resume_text,
                "job_title_filter": job_title_filter
            })
            return job_list
        except Exception as e:
            logger.error(f"Error in /recommend_jobs: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent_query", summary="Process a natural language query via an agent")
async def agent_query(
    query: str = Form(...),
    job_description: str = Form(default=""),
    current_skills: str = Form(default=""),
    job_title_filter: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None)
):
    """
    This endpoint uses a LangChain agent to interpret a natural language query
    and decide which tool to use. It dynamically constructs the inputs for the agent.
    """
    resume_text = ""
    if file and file.filename:
        async with save_temp_file(file) as temp_path:
            resume_text = await extract_text_from_file(temp_path, file.filename)
    
    # Consolidate all possible inputs for the agent
    tool_inputs = {
        "resume_text": resume_text,
        "job_description": job_description,
        "job_title_filter": job_title_filter or query,
        "role": job_title_filter or query,
        "description": current_skills or "No specific context provided.",
        "job_title": job_title_filter or query,
        "current_skills": current_skills or "No current skills provided.",
    }

    try:
        # CORRECTED ASYNC CALL: Use ainvoke on the imported agent_executor
        result = await agent_executor.ainvoke({"input": query, **tool_inputs})
        return result
    except Exception as e:
        logger.error(f"Error in /agent_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent failed to process request: {e}")

if __name__ == "__main__":
    # Use reload=True for development to automatically restart the server on code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)