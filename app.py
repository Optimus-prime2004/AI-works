import streamlit as st
import requests
import json
import base64
import os
import tempfile
import pandas as pd
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr
from streamlit_mermaid import st_mermaid
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import BACKEND_URL, logger, MAX_FILE_SIZE, REQUEST_TIMEOUT

# Imports for the Live Interview feature
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import av
import queue
import threading
import time
from socket import timeout as SocketTimeoutError

# --- State Initialization ---
def init_session_state():
    """Initializes all necessary session state variables if they don't exist."""
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
        st.session_state.interview_chat_history = []
        st.session_state.current_ai_question = "Hello! I'll be your interviewer today. To start, please tell me about yourself."
        st.session_state.transcribed_text_queue = queue.Queue()
        st.session_state.audio_lock = threading.Lock()
        st.session_state.is_recording = False
        st.session_state.audio_buffer = bytearray()
        st.session_state.processing_audio = False

# --- Audio & API Helpers ---
def text_to_speech_autoplay(text: str) -> str:
    """Converts text to speech and returns HTML for autoplaying audio."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            fp.seek(0)
            data = fp.read()
        os.unlink(fp.name)
        b64 = base64.b64encode(data).decode()
        return f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        logger.error(f"Text-to-speech generation failed: {e}", exc_info=True)
        st.warning("Could not generate audio for the AI's response.")
        return ""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def make_request(endpoint: str, method: str = "post", data: dict | None = None, files: list | None = None) -> dict | list:
    """Makes an HTTP request to the backend with retries and detailed error handling."""
    url = f"{BACKEND_URL.rstrip('/')}{endpoint}"
    try:
        if method.lower() == "post":
            response = requests.post(url, data=data, files=files, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request to {url} failed: {e}", exc_info=True)
        st.error(f"An API error occurred. Please check the backend logs.")
        raise

# --- Live Interview Audio Processing Logic ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = bytearray()
        self.target_sample_rate = 16000
        self.resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=self.target_sample_rate)
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        resampled_frames = self.resampler.resample(frame)
        with st.session_state.audio_lock:
            if st.session_state.get("is_recording", False):
                for resampled_frame in resampled_frames:
                    st.session_state.audio_buffer.extend(resampled_frame.planes[0].to_bytes())
        return frame

def transcribe_and_process():
    """Transcribes the audio buffer and puts the result in the text queue."""
    target_sample_rate = 16000
    sample_width = 2
    with st.session_state.audio_lock:
        if not st.session_state.audio_buffer:
            st.session_state.transcribed_text_queue.put("[INFO: No audio was recorded.]")
            return
        audio_data = bytes(st.session_state.audio_buffer)
        st.session_state.audio_buffer = bytearray()
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio_data, timeout=15)
        st.session_state.transcribed_text_queue.put(text)
    except sr.UnknownValueError:
        st.session_state.transcribed_text_queue.put("[ERROR: Could not understand the audio.]")
    except (sr.RequestError, SocketTimeoutError) as e:
        st.session_state.transcribed_text_queue.put(f"[ERROR: Network or speech service error: {e}]")

# --- UI Rendering Functions ---

def render_interview_simulator_tab():
    st.header("💬 Live AI Interview Simulator")
    
    audio_placeholder = st.empty()
    
    for entry in st.session_state.interview_chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])
    
    st.caption(f"Interviewer: {st.session_state.current_ai_question}")

    if not st.session_state.interview_started:
        if st.button("▶️ Start Live Interview", use_container_width=True):
            st.session_state.interview_started = True
            st.session_state.interview_chat_history.append({"role": "assistant", "content": st.session_state.current_ai_question})
            audio_html = text_to_speech_autoplay(st.session_state.current_ai_question)
            if audio_html:
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                time.sleep(max(5, len(st.session_state.current_ai_question.split()) / 2.5))
                audio_placeholder.empty()
            st.rerun()
        return

    webrtc_ctx = webrtc_streamer(
        key="interview-audio",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    status_indicator = st.empty()
    processing_placeholder = st.empty()

    if webrtc_ctx.state.playing:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Start Answering", use_container_width=True, key="start_answer"):
                with st.session_state.audio_lock:
                    st.session_state.is_recording = True
                    st.session_state.audio_buffer = bytearray()
                status_indicator.info("🔴 Recording... Speak now and press 'Stop' when finished.")
        with col2:
            if st.button("⏹️ Stop and Process", use_container_width=True, key="stop_process"):
                with st.session_state.audio_lock:
                    st.session_state.is_recording = False
                
                with processing_placeholder.container():
                    with st.spinner("🧠 Processing your answer... Please wait."):
                        transcribe_and_process()
                        try:
                            user_answer = st.session_state.transcribed_text_queue.get(timeout=30)
                        except queue.Empty:
                            user_answer = "[ERROR: Transcription timed out.]"
                
                if "[ERROR:" in user_answer or "[INFO:" in user_answer:
                    st.error(user_answer.replace("[ERROR: ", "").replace("[INFO: ", "").replace("]", ""))
                else:
                    st.success(f"You said: \"{user_answer}\"")
                    st.session_state.interview_chat_history.append({"role": "user", "content": user_answer})
                    
                    with st.spinner("AI is formulating the next question..."):
                        try:
                            history_json = json.dumps(st.session_state.interview_chat_history[-4:])
                            data = {"user_answer": user_answer, "conversation_history_json": history_json}
                            ai_response = make_request("/interview_simulator", data=data)
                            ai_message = ai_response["response"]
                            
                            st.session_state.current_ai_question = ai_message
                            st.session_state.interview_chat_history.append({"role": "assistant", "content": ai_message})
                            
                            audio_html = text_to_speech_autoplay(ai_message)
                            if audio_html:
                                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                                time.sleep(max(5, len(ai_message.split()) / 2.5))
                                audio_placeholder.empty()

                        except Exception:
                            st.error("Failed to get a response from the AI. Please try again.")
                            st.session_state.interview_chat_history.pop()
                st.rerun()
    else:
        status_indicator.warning("Click 'START' above to activate your microphone for the interview.")

def render_resume_review_tab():
    st.header("📄 Resume Review")
    st.markdown("Upload your resume and the job description to get a score and detailed feedback.")
    resume_file = st.file_uploader("Upload Your Resume (.txt or .pdf)", type=["txt", "pdf"])
    job_description = st.text_area("Paste the Job Description Here", height=200)
    if st.button("🔬 Analyze Resume", use_container_width=True):
        if resume_file and job_description.strip():
            files = {"file": (resume_file.name, resume_file.getvalue(), resume_file.type)}
            data = {"job_description": job_description}
            with st.spinner("Your resume is being analyzed by our AI expert..."):
                try:
                    result = make_request("/review_resume", data=data, files=files)
                    st.metric("Match Score", f"{result.get('score', 0):.2f} / 100")
                    with st.expander("**Constructive Feedback**", expanded=True):
                        st.markdown(result.get('feedback', 'No feedback provided.'))
                    st.markdown("#### Actionable Suggestions")
                    for sug in result.get('suggestions', []):
                        st.markdown(f"- {sug}")
                except Exception:
                    pass
        else:
            st.warning("Please upload a resume and provide a job description.")

def render_resume_ranking_tab():
    st.header("📊 Resume Ranking")
    st.markdown("Upload multiple resumes to rank them against a single job description.")
    resume_files = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
    job_description = st.text_area("Paste the Job Description to Rank Against", height=200, key="rank_job")
    if st.button("🏆 Rank Resumes", use_container_width=True):
        if resume_files and job_description.strip():
            files_list = [("files", (f.name, f.getvalue(), f.type)) for f in resume_files]
            data = {"job_description": job_description}
            with st.spinner("Ranking resumes... This may take a moment."):
                try:
                    results = make_request("/rank_resumes", data=data, files=files_list)
                    st.subheader("🏅 Ranking Results")
                    for i, res in enumerate(results, 1):
                        with st.expander(f"**#{i}: {res['filename']} (Score: {res.get('score', 0):.2f})**", expanded=(i == 1)):
                            st.info(f"**Feedback:** {res.get('feedback', 'No feedback')}")
                            suggestions = res.get('suggestions', [])
                            if suggestions:
                                st.markdown("**Top Suggestions:**")
                                for sug in suggestions:
                                    st.markdown(f"- {sug}")
                except Exception:
                    pass
        else:
            st.warning("Please upload at least one resume and provide a job description.")

def render_interview_qa_tab():
    st.header("❓ Interview Q&A Generation")
    st.markdown("Generate tailored interview questions based on your resume and a job description.")
    resume_file = st.file_uploader("Upload Resume for Q&A", type=["txt", "pdf"])
    job_description = st.text_area("Paste Job Description for Q&A", height=200)
    if st.button("🧠 Generate Q&A", use_container_width=True):
        if resume_file and job_description.strip():
            files = {"file": (resume_file.name, resume_file.getvalue(), resume_file.type)}
            data = {"job_description": job_description}
            with st.spinner("Crafting interview questions..."):
                try:
                    result = make_request("/generate_qa", data=data, files=files)
                    st.info(f"**Preparation Notes:** {result.get('notes', 'No notes provided.')}")
                    st.markdown("#### Generated Questions")
                    for q in result.get('questions', []):
                        st.markdown(f"- **{q['question']}** `({q.get('category')})`")
                    st.markdown("#### Pro-Tips for Your Interview")
                    
                    # --- SAFEGUARD ADDED ---
                    suggestions = result.get('suggestions', [])
                    if suggestions and any(len(str(s).strip()) < 3 for s in suggestions):
                        logger.warning("Detected malformed suggestions from LLM. Attempting to fix.")
                        char_list = [str(s) for s in suggestions if isinstance(s, (str, int, float)) and str(s).strip()]
                        st.markdown(f"- {''.join(char_list)}")
                    else:
                        for sug in suggestions:
                            st.markdown(f"- {sug}")
                    # --- END OF SAFEGUARD ---
                            
                except Exception:
                    pass
        else:
            st.warning("Please upload a resume and provide a job description.")

def render_career_roadmap_tab():
    st.header("🗺️ Career Roadmap Generator")
    st.markdown("Define your target role and get a strategic plan to achieve your career goals.")
    role = st.text_input("What is your desired job role?")
    description = st.text_area("Briefly describe your current experience or career context.", height=150)
    if st.button("🌱 Generate My Roadmap", use_container_width=True):
        if role.strip() and description.strip():
            data = {"role": role, "description": description}
            with st.spinner("Building your personalized career plan..."):
                try:
                    result = make_request("/generate_career_plan", data=data)
                    st.subheader(f"🚀 Roadmap to: {result.get('career_goal_summary', 'Your Goal')}")
                    st.success(f"**Strengths Assessment:** {result.get('current_assessment', 'N/A')}")
                    st.warning(f"**Potential Gaps to Address:** {result.get('gap_analysis', 'N/A')}")
                    st.markdown("#### Action Plan Flowchart")
                    st_mermaid(result.get("mermaid_flowchart", "graph TD; A[Chart not generated];"))
                    st.markdown("#### Suggested Resources")
                    for res in result.get('resources_suggestions', []):
                        st.markdown(f"- {res}")
                    st.caption(f"Disclaimer: {result.get('disclaimer', '')}")
                except Exception:
                    pass
        else:
            st.warning("Please provide both a desired role and a description.")

def render_upskilling_tab():
    st.header("🌱 Upskilling & Certifications")
    st.markdown("Get personalized recommendations for courses and certifications to advance your career.")

    # --- THIS IS THE FIX ---
    # We wrap all inputs and the button in a single form.
    with st.form("upskilling_form"):
        job_title = st.text_input("What is your target job title?", placeholder="e.g., Senior Data Scientist")
        current_skills = st.text_area("Briefly describe your current key skills and experience level", placeholder="e.g., 3 years of Python, familiar with Scikit-learn and Pandas, basic SQL knowledge.")
        
        # The button is now a form submit button.
        submitted = st.form_submit_button("🚀 Find Learning Resources", use_container_width=True)

    # The processing logic now happens AFTER the form is submitted.
    if submitted:
        if job_title.strip() and current_skills.strip():
            data = {"job_title": job_title, "current_skills": current_skills}
            with st.spinner(f"Finding the best resources for a '{job_title}'..."):
                try:
                    result = make_request("/get_upskilling_recs", data=data)
                    
                    st.subheader("🎓 Recommended Courses")
                    courses = result.get("recommended_courses", [])
                    if courses:
                        for course in courses:
                            with st.container(border=True):
                                st.markdown(f"**{course.get('name', 'N/A')}** on *{course.get('platform', 'N/A')}*")
                                justification_text = course.get('justification', "No justification provided.")
                                st.info(f"**Why it's recommended:** {justification_text}")
                    else:
                        st.markdown("No specific course recommendations were generated.")

                    st.subheader("📜 Recommended Certifications")
                    certs = result.get("recommended_certifications", [])
                    if certs:
                        for cert in certs:
                            with st.container(border=True):
                                st.markdown(f"**{cert.get('name', 'N/A')}** by *{cert.get('issuer', 'N/A')}*")
                                justification_text = cert.get('justification', "No justification provided.")
                                st.info(f"**Why it's recommended:** {justification_text}")
                    else:
                        st.markdown("No specific certification recommendations were generated.")

                except Exception:
                    st.error(f"Could not generate recommendations. Please try again.")
        else:
            st.warning("Please provide both a target job title and your current skills.")

def render_job_recommendation_tab():
    st.header("🔍 Job Recommendations")
    st.markdown("Upload your resume to find matching job openings from our database.")
    with st.form("job_rec_form"):
        search_query = st.text_input("Enter a Job Title to Search", placeholder="e.g., 'Python Developer'")
        resume_file = st.file_uploader("Upload Your Resume to Find Jobs", type=["txt", "pdf"])
        submitted = st.form_submit_button("Find Matching Jobs", use_container_width=True)
    if submitted:
        if resume_file and search_query.strip():
            files = {"file": (resume_file.name, resume_file.getvalue(), resume_file.type)}
            data = {"job_title_filter": search_query}
            with st.spinner(f"Searching for '{search_query}' jobs and matching your resume..."):
                try:
                    results = make_request("/recommend_jobs", data=data, files=files)
                    st.divider()
                    if not results:
                        st.warning("No matching jobs found. Try broadening your filter or using a different resume.")
                    else:
                        st.success(f"Found {len(results)} matching job recommendations for you!")
                        for job in results:
                            with st.container(border=True):
                                st.subheader(f"{job.get('title')}")
                                st.markdown(f"**Company:** {job.get('company')} | **Location:** {job.get('location')}")
                                st.metric(label="Match Score", value=f"{job.get('match_score', 0):.2f}%")
                                with st.expander("View Job Description"):
                                    st.markdown(job.get('description', 'No description available.'))
                except Exception:
                    st.error("An error occurred while searching for jobs. Please try again.")
        else:
            st.warning("Please enter a job title and upload a resume.")

def render_dashboard_tab():
    st.header("📈 My Dashboard")
    st.markdown("Track your progress and review past analyses.")
    try:
        history_data = make_request("/get_user_dashboard", method="get")
        if not history_data:
            st.info("You don't have any review history yet. Use the 'Review' tab to get started!")
            return
        df = pd.DataFrame(history_data)
        df['reviewed_at'] = pd.to_datetime(df['reviewed_at'])
        df['score'] = pd.to_numeric(df['score'])
        st.subheader("Score Improvement Over Time")
        if len(df) > 1:
            st.line_chart(df.set_index('reviewed_at')['score'])
        else:
            st.bar_chart(df.set_index('reviewed_at')['score'])
        st.subheader("Detailed Review History")
        for index, row in df.iterrows():
            with st.expander(f"{row['reviewed_at'].strftime('%Y-%m-%d')} - **{row['filename']}** (Score: {row['score']:.2f})"):
                st.markdown(f"**Analyzed against job:**")
                st.text_area("Job Description", value=row['job_text'], height=100, disabled=True, key=f"job_{index}")
                st.markdown(f"**Feedback Received:**")
                st.info(row['feedback'])
                suggestions = json.loads(row['suggestions'])
                if suggestions:
                    st.markdown(f"**Suggestions:**")
                    for sug in suggestions:
                        st.markdown(f"- {sug}")
    except Exception as e:
        logger.error(f"Failed to render dashboard: {e}", exc_info=True)
        st.error(f"Could not load your dashboard.")

def render_health_check_tab():
    st.header("🩺 System Health Check")
    st.markdown("Verify the status of the backend services.")
    if st.button("Check System Health", use_container_width=True):
        with st.spinner("Pinging services..."):
            try:
                result = make_request("/health", method="get")
                st.success("Backend API is operational!")
                st.json(result)
            except Exception:
                st.error("The backend service is not responding. Please check the logs.")

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="AI Resume Pro", page_icon="🚀")
    st.title("🚀 AI Resume Pro")
    init_session_state()
    tab_list = ["📄 Review", "📊 Rank", "❓ Q&A", "💬 Simulator", "🗺️ Roadmap", "🌱 Upskilling", "🔍 Job Search", "📈 Dashboard", "🩺 Health"]
    tabs = st.tabs(tab_list)
    render_map = {
        tab_list[0]: render_resume_review_tab,
        tab_list[1]: render_resume_ranking_tab,
        tab_list[2]: render_interview_qa_tab,
        tab_list[3]: render_interview_simulator_tab,
        tab_list[4]: render_career_roadmap_tab,
        tab_list[5]: render_upskilling_tab,
        tab_list[6]: render_job_recommendation_tab,
        tab_list[7]: render_dashboard_tab,
        tab_list[8]: render_health_check_tab,
    }
    for i, tab in enumerate(tabs):
        with tab:
            render_map[tab_list[i]]()

if __name__ == "__main__":
    main()