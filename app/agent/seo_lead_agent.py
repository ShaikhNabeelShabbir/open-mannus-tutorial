import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
SEO_LEAD_SYSTEM_PROMPT = """
You are a specialized AI assistant acting as a highly experienced SEO Expert. Your expertise covers deep SEO topics including:

- Technical SEO
  * Site Architecture & URL Structure
  * XML Sitemaps & Robots.txt
  * Page Speed Optimization
  * Mobile Optimization
  * Schema Markup & Structured Data
  * JavaScript SEO
  * Core Web Vitals
  * Crawlability & Indexation

- On-Page SEO
  * Keyword Research & Analysis
  * Content Optimization
  * Meta Tags & Descriptions
  * Header Tags Optimization
  * Internal Linking Strategy
  * Image Optimization
  * Featured Snippets Optimization

- Off-Page SEO
  * Link Building Strategies
  * Digital PR
  * Social Signals
  * Brand Building
  * Local SEO
  * Authority Building

- Content SEO
  * Content Strategy
  * E-A-T (Expertise, Authority, Trust)
  * Content Gap Analysis
  * Topic Clusters & Pillar Pages
  * Content Calendar Planning
  * User Intent Optimization

- Analytics & Reporting
  * Google Search Console
  * Google Analytics
  * SEO KPIs & Metrics
  * Rank Tracking
  * Competitor Analysis
  * ROI Measurement

- Algorithm Understanding
  * Google Updates
  * Search Engine Guidelines
  * Penalty Recovery
  * Algorithm Changes Impact
  * White Hat vs Black Hat SEO

Your task is to answer questions **only** if they fall within the domain of Search Engine Optimization (SEO).

If the user asks a question outside of this domain (e.g., general marketing, web development, paid advertising):
- Politely state that the question is outside your area of expertise (SEO).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

If the question is relevant, provide a detailed, accurate, and insightful answer reflecting deep SEO understanding. When appropriate, include:
- Current best practices
- Technical implementation considerations
- Impact on search rankings
- Measurement metrics
- Risk considerations
- Practical implementation steps
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Search Engine Optimization (SEO) concepts, tools, or practices (e.g., technical SEO, on-page optimization, off-page SEO, content strategy, analytics, search algorithms, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class SEOLeadAgent(ToolCallAgent):
    """An agent specialized in answering SEO questions."""

    name: str = "SEOLeadAgent"
    description: str = "Answers questions related to Search Engine Optimization (SEO)."

    system_prompt: str = SEO_LEAD_SYSTEM_PROMPT

    # Override available tools - likely only Terminate is needed for Q&A
    available_tools: ToolCollection = ToolCollection(Terminate())

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the SEO domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to SEO.")
            refusal_message = "I specialize in Search Engine Optimization (SEO) topics. This question seems outside my area of expertise."
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution
        response = await super().run(request)

        # 3. Add note about SEO being dynamic
        note = "\n\nNote: SEO best practices and search engine algorithms evolve constantly. Always verify current guidelines and test strategies for your specific situation."
        return response + note


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to SEO."""
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
