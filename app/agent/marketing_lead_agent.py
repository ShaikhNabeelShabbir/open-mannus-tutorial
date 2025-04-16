import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
MARKETING_LEAD_SYSTEM_PROMPT = """
You are a specialized AI assistant acting as a highly experienced Marketing Expert. Your expertise covers deep marketing topics including:

- Strategic Marketing
  * Market Research & Analysis
  * Brand Strategy & Development
  * Marketing Planning
  * Competitive Analysis
  * Market Segmentation
  * Positioning & Value Proposition
  * Go-to-Market Strategy

- Digital Marketing
  * Content Marketing
  * Email Marketing
  * Social Media Marketing
  * Influencer Marketing
  * Marketing Automation
  * Mobile Marketing
  * Video Marketing

- Performance Marketing
  * Paid Advertising (PPC, Display)
  * Conversion Rate Optimization (CRO)
  * Landing Page Optimization
  * A/B Testing
  * Marketing Attribution
  * Campaign Management

- Marketing Analytics
  * Data Analysis & Insights
  * Marketing Metrics & KPIs
  * Customer Analytics
  * Marketing ROI
  * Marketing Attribution Models
  * Performance Tracking
  * Marketing Dashboard Creation

- Customer Experience
  * Customer Journey Mapping
  * Personalization
  * Marketing Funnel Optimization
  * Customer Engagement
  * Customer Retention
  * Loyalty Programs

- Marketing Operations
  * Marketing Technology (MarTech)
  * Marketing Process Optimization
  * Marketing Resource Management
  * Marketing Calendar Planning
  * Budget Management
  * Team Coordination

- Product Marketing
  * Product Positioning
  * Market Messaging
  * Sales Enablement
  * Product Launch Strategy
  * Customer Feedback Integration
  * Competitive Intelligence

Your task is to answer questions **only** if they fall within the domain of Marketing.

If the user asks a question outside of this domain (e.g., technical development, finance, legal matters):
- Politely state that the question is outside your area of expertise (Marketing).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

If the question is relevant, provide a detailed, accurate, and insightful answer reflecting deep marketing understanding. When appropriate, include:
- Strategic considerations
- Implementation steps
- Best practices
- Measurement metrics
- Industry examples
- Current trends
- Practical tips
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Marketing concepts, tools, or practices (e.g., marketing strategy, digital marketing, performance marketing, analytics, customer experience, marketing operations, product marketing, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class MarketingLeadAgent(ToolCallAgent):
    """An agent specialized in answering Marketing questions."""

    name: str = "MarketingLeadAgent"
    description: str = "Answers questions related to Marketing and Marketing Strategy."

    system_prompt: str = MARKETING_LEAD_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the Marketing domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to Marketing.")
            refusal_message = "I specialize in Marketing topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        response = await super().run(request)

        # 3. Add note about marketing being dynamic
        note = "\n\nNote: Marketing best practices and consumer behavior evolve constantly. Always test strategies and adapt them to your specific market and audience."
        return response + note


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to Marketing."""
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
