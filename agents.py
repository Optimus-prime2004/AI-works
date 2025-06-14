# This file is now used to orchestrate tool calls using a LangChain agent.
# It provides a higher-level interface for natural language queries.

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
# Corrected import path for the parser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.tools.render import render_text_description

from app_tools import (
    review_resume,
    generate_interview_qa,
    generate_career_roadmap,
    get_upskilling_recommendations,
    recommend_jobs
)
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, logger

# 1. Define the set of tools the agent can use
tools = [
    review_resume,
    generate_interview_qa,
    generate_career_roadmap,
    get_upskilling_recommendations,
    recommend_jobs
]

# 2. Create the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful career assistant. You have access to several tools to help users with their job search.

The user will ask a question. Your goal is to select the single best tool to answer the user's question and respond with a JSON object containing the tool name and its parameters.

You can use the following tools:
{tools}

You should only respond in JSON format with a single action. The JSON you respond with should conform to the following schema:
```json
{{
    "action": string,
    "action_input": {{...}}
}}
Use code with caution.
Python
The "action" should be one of the following tool names:
{tool_names}
The user's question is:
{input}
"""
)
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
agent = (
{
"input": lambda x: x["input"],
"tools": lambda x: render_text_description(tools),
"tool_names": lambda x: ", ".join([t.name for t in tools]),
}
| prompt_template
| llm
# CORRECTED CLASS NAME: Use the imported JsonOutputParser
| JsonOutputParser()
)
agent_executor = AgentExecutor(
agent=agent,
tools=tools,
verbose=True,
handle_parsing_errors=True,
max_iterations=5,
)