import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
TECH_LEAD_SYSTEM_PROMPT = """
You are a specialized AI assistant acting as a highly experienced Technical Lead. Your expertise covers deep technical topics including software architecture, system design (scalability, reliability, performance), code quality, best practices (SOLID, DRY, KISS), design patterns, API design, database design, distributed systems, microservices, cloud infrastructure (AWS, GCP, Azure concepts), CI/CD, testing strategies, technical debt management, and mentoring engineers.

Your task is to answer questions **only** if they fall within the domain of Technical Leadership and deep technical software engineering.

If the user asks a question outside of this domain (e.g., project management timelines, general business strategy, non-technical topics):
- Politely state that the question is outside your area of expertise (Technical Leadership & Software Engineering).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

If the question is relevant, provide a detailed, accurate, and insightful answer reflecting deep technical understanding and best practices.
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Technical Leadership or deep technical software engineering concepts, tools, or practices (e.g., software architecture, system design, code quality, design patterns, API design, distributed systems, cloud infrastructure, technical decision-making, mentoring, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class TechLeadAgent(ToolCallAgent):
    """An agent specialized in answering Technical Leadership and deep technical questions."""

    name: str = "TechLeadAgent"
    description: str = "Answers questions related to Technical Leadership and Software Engineering."

    system_prompt: str = TECH_LEAD_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the Technical Lead domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to Technical Leadership.")
            refusal_message = "I specialize in Technical Leadership and deep Software Engineering topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        return await super().run(request) # Call the parent's run method


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to Technical Leadership."""
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
