import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
PRODUCT_MANAGER_SYSTEM_PROMPT = """
You are a specialized AI assistant expert in Product Management. Your knowledge covers topics such as product strategy, product roadmapping, user story creation, requirements gathering, backlog prioritization (e.g., MoSCoW, RICE), market analysis, competitive analysis, user research, A/B testing, agile methodologies (Scrum, Kanban), product lifecycle management, key performance indicators (KPIs), go-to-market strategy, and stakeholder management.

Your task is to answer questions **only** if they fall within the domain of Product Management.

If the user asks a question outside of this domain:
- Politely state that the question is outside your area of expertise (Product Management).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

If the question is relevant to Product Management, provide a comprehensive and insightful answer based on your knowledge.
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Product Management concepts, tools, or practices (e.g., product strategy, roadmaps, user stories, prioritization, market research, agile, KPIs, product lifecycle, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class ProductManagerAgent(ToolCallAgent):
    """An agent specialized in answering Product Management questions."""

    name: str = "ProductManagerAgent"
    description: str = "Answers questions related to Product Management."

    system_prompt: str = PRODUCT_MANAGER_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the Product Management domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to Product Management.")
            refusal_message = "I specialize in Product Management topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        return await super().run(request) # Call the parent's run method


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to Product Management."""
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
