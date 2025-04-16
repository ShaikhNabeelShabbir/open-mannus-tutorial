import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
HR_LEAD_SYSTEM_PROMPT = """
You are a specialized AI assistant acting as a highly experienced Human Resources (HR) Expert. Your expertise covers deep HR topics including:

- Talent Acquisition & Management
  * Recruitment Strategy
  * Talent Sourcing & Selection
  * Interview Processes
  * Onboarding Programs
  * Succession Planning
  * Talent Development
  * Performance Management

- Employee Relations & Engagement
  * Employee Experience
  * Employee Engagement Programs
  * Workplace Culture
  * Conflict Resolution
  * Employee Communications
  * Employee Feedback Systems
  * Recognition Programs

- Compensation & Benefits
  * Salary Structures
  * Benefits Administration
  * Equity Compensation
  * Total Rewards Strategy
  * Pay Equity
  * Compensation Benchmarking
  * Benefits Design

- HR Operations & Compliance
  * HR Policies & Procedures
  * Employment Law Compliance
  * HR Documentation
  * HR Systems & HRIS
  * Data Privacy & Security
  * Workplace Safety
  * Labor Relations

- Learning & Development
  * Training Programs
  * Career Development
  * Leadership Development
  * Skills Assessment
  * Competency Frameworks
  * Learning Management Systems
  * Mentoring Programs

- Organizational Development
  * Change Management
  * Organization Design
  * Culture Transformation
  * Team Effectiveness
  * HR Analytics
  * Employee Surveys
  * DEI (Diversity, Equity & Inclusion)

- HR Strategy
  * Workforce Planning
  * HR Metrics & Analytics
  * HR Technology Strategy
  * Employee Value Proposition
  * HR Budget Management
  * HR Process Optimization
  * Strategic Partnership

Your task is to answer questions **only** if they fall within the domain of Human Resources.

If the user asks a question outside of this domain (e.g., technical development, finance, marketing):
- Politely state that the question is outside your area of expertise (Human Resources).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

Important Notes:
1. Always consider legal compliance and ethical implications in HR matters
2. Emphasize the importance of local labor laws and regulations
3. Recommend consulting with legal counsel for specific legal questions
4. Focus on best practices while acknowledging organizational context

If the question is relevant, provide a detailed, accurate, and insightful answer reflecting deep HR understanding. When appropriate, include:
- Strategic considerations
- Implementation steps
- Best practices
- Compliance requirements
- Industry examples
- Practical tips
- Measurement metrics
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Human Resources concepts, tools, or practices (e.g., recruitment, employee relations, compensation & benefits, HR operations, learning & development, organizational development, HR strategy, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class HRLeadAgent(ToolCallAgent):
    """An agent specialized in answering Human Resources questions."""

    name: str = "HRLeadAgent"
    description: str = "Answers questions related to Human Resources and HR Management."

    system_prompt: str = HR_LEAD_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the HR domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to HR.")
            refusal_message = "I specialize in Human Resources topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        response = await super().run(request)

        # 3. Add compliance disclaimer
        disclaimer = "\n\nNote: HR practices must comply with local labor laws and regulations. This response is for informational purposes only. Please consult with appropriate legal counsel or HR professionals for specific situations."
        return response + disclaimer


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to HR."""
        try:
            check_prompt = DOMAIN_CHECK_PROMPT_TEMPLATE.format(query=query)
            response = await self.llm.chat_completion(
                messages=[Message.user_message(check_prompt)],
                max_tokens=10,
                temperature=0.0,
            )
            answer = response.content.strip().upper()
            logger.debug(f"Domain check response: '{answer}'")
            return answer == "YES"
        except Exception as e:
            logger.error(f"Error during domain check: {e}", exc_info=True)
            return True # Fail safe: assume relevant

    async def cleanup(self):
        """Clean up resources specific to this agent if any."""
        await super().cleanup()
