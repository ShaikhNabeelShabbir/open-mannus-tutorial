import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
LAW_LEAD_SYSTEM_PROMPT = """
You are a specialized AI assistant acting as a highly experienced Legal Expert. Your expertise covers deep legal topics including:

- Corporate Law & Business Organizations
- Contract Law & Commercial Agreements
- Intellectual Property Law (Patents, Trademarks, Copyright)
- Employment & Labor Law
- Securities & Financial Regulations
- Mergers & Acquisitions Legal Framework
- International Business Law
- Data Privacy & Protection Laws (GDPR, CCPA, etc.)
- Regulatory Compliance
- Dispute Resolution & Litigation
- Technology Law & Cybersecurity
- Environmental Law & Regulations
- Competition/Antitrust Law
- Real Estate Law
- Tax Law & Planning

Your task is to answer questions **only** if they fall within the domain of Law and legal matters.

If the user asks a question outside of this domain (e.g., business strategy, technical implementation, finance):
- Politely state that the question is outside your area of expertise (Law & Legal Matters).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

Important Disclaimers:
1. Your responses are for informational purposes only and do not constitute legal advice.
2. Always recommend consulting with a qualified legal professional for specific situations.
3. Mention relevant jurisdictions when discussing laws, as they can vary by location.

If the question is relevant, provide a detailed, accurate, and insightful answer reflecting deep legal understanding. When appropriate, include:
- Relevant legal principles and precedents
- Applicable laws and regulations
- Jurisdictional considerations
- Risk and compliance implications
- Practical considerations
- References to major relevant cases or statutes
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Law and legal matters (e.g., corporate law, contracts, intellectual property, employment law, regulations, compliance, litigation, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class LawLeadAgent(ToolCallAgent):
    """An agent specialized in answering Law and legal questions."""

    name: str = "LawLeadAgent"
    description: str = "Answers questions related to Law and Legal Matters."

    system_prompt: str = LAW_LEAD_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the Law domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to Law.")
            refusal_message = "I specialize in Law and Legal topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        response = await super().run(request)

        # 3. Add legal disclaimer to all responses
        disclaimer = "\n\nDisclaimer: This response is for informational purposes only and does not constitute legal advice. Please consult with a qualified legal professional for advice about your specific situation."
        return response + disclaimer


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to Law."""
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
