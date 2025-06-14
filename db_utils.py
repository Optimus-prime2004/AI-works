import sqlite3
from contextlib import contextmanager
from typing import Dict, Any, Generator, List
import json
import bcrypt
from config import DB_PATH, logger

# --- Simplified Connection Management ---
@contextmanager
def db_cursor() -> Generator[sqlite3.Cursor, None, None]:
    """
    Provides a database cursor within a managed context.
    Handles connection opening, closing, commits, and rollbacks automatically.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database transaction failed: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

# --- Schema Management ---
def get_password_hash(password: str) -> str:
    """Generates a secure, hashed password using bcrypt."""
    try:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    except Exception as e:
        logger.error(f"Password hashing failed: {e}", exc_info=True)
        raise ValueError(f"Password hashing failed: {e}")

def ensure_tables_exist() -> None:
    """Verifies that all required SQLite tables and indexes exist, creating them if necessary."""
    logger.info("Verifying database schema...")
    create_statements = [
        # Users Table
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, email TEXT NOT NULL UNIQUE, hashed_password TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);",
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
        # Resumes Table
        "CREATE TABLE IF NOT EXISTS resumes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT NOT NULL, raw_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, UNIQUE(user_id, filename));",
        "CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes(user_id);",
        # Job Descriptions Table
        "CREATE TABLE IF NOT EXISTS job_descriptions (id INTEGER PRIMARY KEY, job_text TEXT NOT NULL UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);",
        # Reviews Table
        "CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, resume_id INTEGER NOT NULL, job_id INTEGER NOT NULL, score REAL, feedback TEXT, suggestions TEXT, reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE, FOREIGN KEY (job_id) REFERENCES job_descriptions(id) ON DELETE CASCADE);",
        "CREATE INDEX IF NOT EXISTS idx_reviews_resume_job ON reviews(resume_id, job_id);",
        # Career Plans Table
        "CREATE TABLE IF NOT EXISTS career_plans (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, career_goal TEXT, plan_data TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE);",
        "CREATE INDEX IF NOT EXISTS idx_career_plans_user ON career_plans(user_id);"
    ]
    try:
        with db_cursor() as cursor:
            for statement in create_statements:
                cursor.execute(statement)

            cursor.execute("SELECT COUNT(id) as user_count FROM users WHERE username = 'default_user'")
            if cursor.fetchone()['user_count'] == 0:
                logger.info("Creating default user...")
                default_hashed_password = get_password_hash("default_password_for_app")
                cursor.execute("INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)", ("default_user", "default@example.com", default_hashed_password))
        logger.info("Database schema verified successfully.")
    except Exception as e:
        logger.error(f"Schema creation failed: {e}", exc_info=True)
        raise

# --- Data Access Functions ---

def save_resume(user_id: int, filename: str, raw_text: str) -> int:
    """Saves or updates a resume text and returns its ID. Uses ON CONFLICT for efficiency."""
    sql = "INSERT INTO resumes (user_id, filename, raw_text, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(user_id, filename) DO UPDATE SET raw_text=excluded.raw_text, updated_at=CURRENT_TIMESTAMP RETURNING id;"
    with db_cursor() as cursor:
        cursor.execute(sql, (user_id, filename, raw_text))
        result = cursor.fetchone()
        if not result: raise RuntimeError("Failed to save resume")
        return result['id']

def save_job_description(job_text: str) -> int:
    """Saves a job description if it doesn't exist and returns its ID."""
    if not job_text.strip(): raise ValueError("Job description cannot be empty")
    with db_cursor() as cursor:
        cursor.execute("SELECT id FROM job_descriptions WHERE job_text = ?", (job_text,))
        existing = cursor.fetchone()
        if existing: return existing['id']
        cursor.execute("INSERT INTO job_descriptions (job_text) VALUES (?) RETURNING id", (job_text,))
        result = cursor.fetchone()
        if not result: raise RuntimeError("Failed to save job description")
        return result['id']

def save_review(resume_id: int, job_id: int, review_data: Dict[str, Any]) -> int:
    """Saves a resume review analysis to the database."""
    sql = "INSERT INTO reviews (resume_id, job_id, score, feedback, suggestions) VALUES (?, ?, ?, ?, ?) RETURNING id"
    with db_cursor() as cursor:
        cursor.execute(sql, (resume_id, job_id, review_data.get('score'), review_data.get('feedback'), json.dumps(review_data.get('suggestions', []))))
        result = cursor.fetchone()
        if not result: raise RuntimeError("Failed to save review")
        return result['id']

def save_career_plan(user_id: int, career_goal: str, plan_data: Dict[str, Any]) -> int:
    """Saves a generated career plan to the database."""
    sql = "INSERT INTO career_plans (user_id, career_goal, plan_data) VALUES (?, ?, ?) RETURNING id"
    with db_cursor() as cursor:
        cursor.execute(sql, (user_id, career_goal, json.dumps(plan_data)))
        result = cursor.fetchone()
        if not result: raise RuntimeError("Failed to save career plan")
        return result['id']

def get_reviews_for_user(user_id: int) -> List[Dict[str, Any]]:
    """
    Fetches all review history for a given user to populate the dashboard.
    Joins across tables to get all relevant data.
    """
    sql = """
        SELECT
            r.filename,
            rev.score,
            rev.feedback,
            rev.suggestions,
            rev.reviewed_at,
            j.job_text
        FROM reviews rev
        JOIN resumes r ON rev.resume_id = r.id
        JOIN job_descriptions j ON rev.job_id = j.id
        WHERE r.user_id = ?
        ORDER BY rev.reviewed_at DESC
    """
    with db_cursor() as cursor:
        cursor.execute(sql, (user_id,))
        # Convert the list of Row objects to a list of standard dictionaries
        return [dict(row) for row in cursor.fetchall()]