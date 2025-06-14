import json
import requests
import fitz
import aiofiles
from pydantic import BaseModel, Field
from typing import Any, List, Dict
from functools import lru_cache
from langchain_core.tools import tool
from config import logger, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
import embedding_utils
import milvus_utils

class ResumeReviewArgs(BaseModel):
    resume_text: str = Field(description="The full text content of the resume.")
    job_description: str = Field(description="The full text of the job description.")

class InterviewQAArgs(BaseModel):
    resume_text: str = Field(description="The full text content of the resume.")
    job_description: str = Field(description="The full text of the job description.")

class InterviewSimulatorArgs(BaseModel):
    user_answer: str = Field(description="The user's most recent answer in the interview.")
    conversation_history: List[Dict[str, str]] = Field(description="A list of previous turns in the conversation.")

class CareerRoadmapArgs(BaseModel):
    role: str = Field(description="The user's desired job role.")
    description: str = Field(description="A description of the user's current situation or context.")

class UpskillingRecsArgs(BaseModel):
    job_title: str = Field(description="The target job title the user wants to upskill for.")
    current_skills: str = Field(description="A brief description of the user's current skills or experience.")

class JobRecsArgs(BaseModel):
    resume_text: str = Field(description="The full text of the user's resume.")
    job_title_filter: str = Field(description="An optional filter to search for specific job titles.")


@lru_cache(maxsize=128)
def _ollama_request(prompt: str, expect_json: bool) -> Any:
    """Sends a cached request to the Ollama API."""
    if not prompt.strip(): raise ValueError("Prompt cannot be empty.")
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {"prompt": prompt, "model": OLLAMA_MODEL, "stream": False}
    if expect_json: payload["format"] = "json"
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        content = response.json()
        response_text = content.get("response", "")
        if not response_text: raise ValueError("Ollama returned an empty response.")
        return json.loads(response_text) if expect_json else response_text.strip()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Could not connect to Ollama service: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from Ollama. Response: {content.get('response', '')}")

async def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extracts text from a local PDF or TXT file asynchronously."""
    try:
        if filename.lower().endswith('.pdf'):
            async with aiofiles.open(file_path, "rb") as f:
                pdf_bytes = await f.read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    return "".join(page.get_text() for page in doc)
        elif filename.lower().endswith('.txt'):
            async with aiofiles.open(file_path, "r", encoding='utf-8') as f:
                return await f.read()
        raise ValueError("Unsupported file type.")
    except Exception as e:
        raise IOError(f"Could not read or process file {filename}: {e}")

@tool("review_resume", args_schema=ResumeReviewArgs)
def review_resume(resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Analyzes a resume against a job description, calculates a similarity score,
    and uses an LLM to provide qualitative feedback and suggestions.
    """
    if not resume_text or not job_description:
        raise ValueError("Resume and job description text cannot be empty.")
    logger.info("Calculating similarity score using embedding model...")
    score = embedding_utils.calculate_similarity(resume_text, job_description)
    logger.info(f"Similarity score: {score:.2f}. Now generating qualitative feedback...")
    prompt = f"""
    You are an expert HR analyst. I have determined that the resume matches the job description
    with a semantic score of {score:.2f} out of 100.
    Based on the texts, provide concise, constructive feedback.
    Return ONLY a valid JSON object with keys "feedback" (a string) and "suggestions" (a list of up to 10 strings).

    Resume: {resume_text[:2500]}
    Job Description: {job_description[:2000]}
    """
    llm_feedback = _ollama_request(prompt, expect_json=True)
    return {"score": round(score, 2), **llm_feedback}

@tool("generate_interview_qa", args_schema=InterviewQAArgs)
def generate_interview_qa(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Generates interview questions, notes, and tips based on a resume and job description."""
    prompt = f"""
    You are an expert interview coach. Based on the resume and job description, generate tailored interview material.
    Return ONLY a valid JSON object with three keys:
    1. "questions": A list of objects, each with a "question" (string) and "category" (string).
    2. "notes": A single string of preparatory notes for the candidate.
    3. "suggestions": A list of 3-5 strings. Each string in this list MUST be a complete sentence providing a pro-tip.

    Resume: {resume_text[:2500]}
    Job Description: {job_description[:2000]}
    """
    return _ollama_request(prompt, expect_json=True)

@tool("simulate_interview_step", args_schema=InterviewSimulatorArgs)
def simulate_interview_step(user_answer: str, conversation_history: List[Dict[str, str]]) -> str:
    """
    Generates the next interview question based on the user's answer and conversation history.
    It's used to create a step-by-step interview simulation.
    """
    history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history[-4:]])
    prompt = f"""
    You are a friendly but professional interviewer. Acknowledge the candidate's answer briefly, then ask a single, concise, relevant follow-up question.
    Return ONLY the follow-up question as a string.

    History:
    {history_str}

    Candidate's Answer:
    {user_answer}

    Your Next Question:
    """
    return _ollama_request(prompt, expect_json=False)

@tool("generate_career_roadmap", args_schema=CareerRoadmapArgs)
def generate_career_roadmap(role: str, description: str) -> Dict[str, Any]:
    """Generates a comprehensive career roadmap, including a Mermaid flowchart diagram."""
    prompt = f"""
    You are an expert career strategist. Create a comprehensive career plan.
    Return ONLY a valid JSON object with keys: "career_goal_summary", "current_assessment", "gap_analysis", "actionable_steps" (list of objects with "step" and "timeline"), "resources_suggestions" (a list of strings, like "Coursera: Google Data Analytics Certificate" or "Book: 'Designing Data-Intensive Applications'"), and "disclaimer".

    Desired Role: {role}
    Role Context: {description}
    """
    result = _ollama_request(prompt, expect_json=True)
    steps = result.get("actionable_steps", [])
    flowchart = f"graph TD\n    A[Start: Current Position] --> B(Skill Assessment)\n"
    node_names = [chr(67 + i) for i in range(len(steps))]
    last_node = "B"
    for i, step_data in enumerate(steps):
        step_text = step_data.get('step', f'Step {i+1}').replace('"', '')
        timeline = step_data.get('timeline', 'N/A').replace('"', '')
        current_node = node_names[i]
        flowchart += f'    {last_node} -->|"{timeline}"| {current_node}["{step_text}"]\n'
        last_node = current_node
    goal_role_sanitized = role.replace('"', '')
    flowchart += f"    {last_node} --> Z[Goal: {goal_role_sanitized}]\n"
    flowchart += "    style Z fill:#d4edda,stroke:#155724,stroke-width:2px"
    result["mermaid_flowchart"] = flowchart
    return result

@tool("get_upskilling_recommendations", args_schema=UpskillingRecsArgs)
def get_upskilling_recommendations(job_title: str, current_skills: str) -> Dict[str, Any]:
    """
    Provides a list of recommended online courses and professional certifications
    to help a user upskill for a specific job title based on their current skills.
    """
    logger.info(f"Generating upskilling recommendations for job title: {job_title}")
    
    prompt = f"""
    You are an expert career advisor. A user wants to upskill for the job title '{job_title}' and has these skills: '{current_skills}'.

    Your task is to generate a list of online courses and professional certifications.

    **CRITICAL INSTRUCTIONS:**
    Return ONLY a valid JSON object with two keys: "recommended_courses" and "recommended_certifications".
    Each key must contain a list of JSON objects.
    EACH object in the lists MUST have the following three keys:
    1. "name": The full name of the course or certification.
    2. "platform" (for courses) or "issuer" (for certifications): The company or platform offering it.
    3. "justification": A concise, one-sentence explanation of WHY this is a valuable resource for the target job. This key is mandatory.

    **EXAMPLE OF THE EXACT OUTPUT FORMAT REQUIRED:**
    {{
        "recommended_courses": [
            {{
                "name": "Google Data Analytics Professional Certificate",
                "platform": "Coursera",
                "justification": "Provides a strong foundational understanding of data analysis tools and techniques essential for this role."
            }}
        ],
        "recommended_certifications": [
            {{
                "name": "AWS Certified Cloud Practitioner",
                "issuer": "Amazon Web Services",
                "justification": "Validates fundamental knowledge of the AWS cloud, which is crucial for many modern tech roles."
            }}
        ]
    }}
    """
    
    recommendations = _ollama_request(prompt, expect_json=True)
    return recommendations

@tool("recommend_jobs", args_schema=JobRecsArgs)
def recommend_jobs(resume_text: str, job_title_filter: str = "") -> List[Dict[str, Any]]:
    """
    Recommends jobs from the internal database (Milvus) that match a given resume.
    This tool DOES NOT fetch new jobs; it only searches existing ones.
    """
    if not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    logger.info("Searching for best matches in the internal Milvus job database...")
    
    if job_title_filter:
        logger.info(f"Note: job_title_filter '{job_title_filter}' is not applied in pure vector search.")

    recommendations = milvus_utils.search_jobs_in_milvus(resume_text, top_k=10)
    
    return recommendations