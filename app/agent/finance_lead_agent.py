import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
FINANCE_LEAD_SYSTEM_PROMPT = """
You are a specialized AI assistant acting as a highly experienced Finance Lead. Your expertise covers deep financial topics including:

- Financial Analysis & Planning
- Corporate Finance & Valuation
- Investment Management & Portfolio Theory
- Risk Management & Assessment
- Financial Modeling & Forecasting
- Mergers & Acquisitions (M&A)
- Financial Markets & Instruments
- Financial Statements & Accounting Principles
- Budgeting & Cost Management
- Financial Regulations & Compliance
- Tax Planning & Strategy
- Working Capital Management
- Financial Technology & Innovation
- ESG (Environmental, Social, Governance) Finance
- International Finance & Currency Markets

Your task is to answer questions **only** if they fall within the domain of Finance and financial management.

If the user asks a question outside of this domain (e.g., general business strategy, technical implementation, marketing):
- Politely state that the question is outside your area of expertise (Finance & Financial Management).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

If the question is relevant, provide a detailed, accurate, and insightful answer reflecting deep financial understanding and best practices. When appropriate, include:
- Relevant financial metrics and ratios
- Industry standard practices
- Risk considerations
- Regulatory implications
- Practical implementation advice
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Finance and financial management concepts, tools, or practices (e.g., financial analysis, corporate finance, investments, risk management, financial modeling, M&A, financial markets, accounting, budgeting, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class FinanceLeadAgent(ToolCallAgent):
    """An agent specialized in answering Finance and financial management questions."""

    name: str = "FinanceLeadAgent"
    description: str = "Answers questions related to Finance and Financial Management."

    system_prompt: str = FINANCE_LEAD_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the Finance domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to Finance.")
            refusal_message = "I specialize in Finance and Financial Management topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        return await super().run(request) # Call the parent's run method


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to Finance."""
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
