import asyncio

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState, Message
from app.tool import Terminate, ToolCollection


# System prompt defining the agent's persona and constraints
DATA_ENG_SYSTEM_PROMPT = """
You are a specialized AI assistant expert in Data Engineering. Your knowledge covers topics such as ETL/ELT processes, data pipelines, data warehousing, data lakes, SQL and NoSQL databases, distributed systems (like Spark, Hadoop, Flink), stream processing (like Kafka, Pulsar), data modeling, data governance, data quality, cloud data platforms (AWS, GCP, Azure), orchestration tools (like Airflow, Dagster), and related programming concepts (Python, Scala, Java in the context of data engineering).

Your task is to answer questions **only** if they fall within the domain of Data Engineering.

If the user asks a question outside of this domain:
- Politely state that the question is outside your area of expertise (Data Engineering).
- Do not attempt to answer it.
- Do not suggest other topics you can answer.

If the question is relevant to Data Engineering, provide a comprehensive and accurate answer based on your knowledge.
"""

# Prompt to ask the LLM for domain classification
DOMAIN_CHECK_PROMPT_TEMPLATE = """
Analyze the following user query. Determine if it is primarily related to Data Engineering concepts, tools, or practices (e.g., ETL, databases, data warehousing, data pipelines, SQL, Spark, Kafka, data modeling, cloud data services, Airflow, etc.).

Answer **only** with 'YES' or 'NO'.

User Query: {query}
"""

class DataEngAgent(ToolCallAgent):
    """An agent specialized in answering Data Engineering questions."""

    name: str = "DataEngAgent"
    description: str = "Answers questions related to Data Engineering."

    system_prompt: str = DATA_ENG_SYSTEM_PROMPT

    # Override available tools - only Terminate might be relevant for Q&A
    # We could even make this an empty ToolCollection if Terminate isn't needed.
    available_tools: ToolCollection = ToolCollection(Terminate())
    # Ensure tool choice reflects that we don't *expect* tools for Q&A
    # tool_choices: TOOL_CHOICE_TYPE = ToolChoice.NONE # Could set this if absolutely no tools

    async def run(self, request: str) -> str:
        """
        Run the agent, first checking if the request is within the Data Engineering domain.
        """
        logger.info(f"Received request: {request}")

        # 1. Domain Check
        is_relevant = await self._is_query_relevant(request)

        if not is_relevant:
            logger.warning("Query deemed irrelevant to Data Engineering.")
            refusal_message = "I specialize in Data Engineering topics. This question seems outside my area of expertise."
            # Add messages to memory for potential context/logging, though we return immediately
            self.memory.add_message(Message.user_message(request))
            self.memory.add_message(Message.assistant_message(refusal_message))
            return refusal_message

        logger.info("Query is relevant. Proceeding with standard execution.")
        # 2. If relevant, proceed with the normal agent execution inherited from ReActAgent/ToolCallAgent
        # The inherited run method handles the conversation loop using the specialized system_prompt.
        return await super().run(request) # Call the parent's run method


    async def _is_query_relevant(self, query: str) -> bool:
        """Uses the LLM to quickly check if the query is relevant to Data Engineering."""
        try:
            check_prompt = DOMAIN_CHECK_PROMPT_TEMPLATE.format(query=query)
            # Use a basic chat completion call, not tool-based
            response = await self.llm.chat_completion(
                messages=[Message.user_message(check_prompt)],
                max_tokens=10, # We only need YES or NO
                temperature=0.0, # Be deterministic
            )
            answer = response.content.strip().upper()
            logger.debug(f"Domain check response: '{answer}'")
            return answer == "YES"
        except Exception as e:
            logger.error(f"Error during domain check: {e}", exc_info=True)
            # Fail safe: assume relevant if check fails to avoid blocking valid requests
            return True

    async def cleanup(self):
        """Clean up resources specific to this agent if any."""
        # Add any specific cleanup logic here if needed
        await super().cleanup() # Call parent cleanup for tools etc.
