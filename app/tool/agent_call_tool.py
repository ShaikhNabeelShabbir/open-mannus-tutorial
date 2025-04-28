from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.agent.mcp import MCPAgent
from app.agent.data_eng_agent import DataEngAgent
from app.agent.product_manager_agent import ProductManagerAgent
from app.agent.tech_lead_agent import TechLeadAgent
from app.agent.finance_lead_agent import FinanceLeadAgent
from app.agent.law_lead_agent import LawLeadAgent
from app.agent.seo_lead_agent import SEOLeadAgent
from app.agent.marketing_lead_agent import MarketingLeadAgent
from app.agent.hr_lead_agent import HRLeadAgent
from app.logger import logger
from app.tool import Tool


class AgentCallParams(BaseModel):
    """Parameters for calling an agent."""
    agent_type: str = Field(
        description="Type of agent to call. Options: mcp, data_eng, product_manager, tech_lead, finance_lead, law_lead, seo_lead, marketing_lead, hr_lead"
    )
    query: str = Field(description="Query to send to the agent")
    context: Optional[str] = Field(
        None, description="Additional context to provide to the agent"
    )


class AgentCallTool(Tool):
    """Tool for calling other agents."""

    name: str = "call_agent"
    description: str = (
        "Call another specialized agent to handle a specific part of the task. "
        "Use this when you need expertise in a particular domain. "
        "Available agents: mcp, data_eng, product_manager, tech_lead, finance_lead, "
        "law_lead, seo_lead, marketing_lead, hr_lead"
    )
    parameters: Any = AgentCallParams

    # Store instances of agents that have been created
    _agent_instances = {}

    async def _run(self, agent_type: str, query: str, context: Optional[str] = None) -> str:
        """Run a query using the specified agent."""
        try:
            # Create or retrieve the agent instance
            agent = self._get_agent(agent_type)

            # Combine context and query if context is provided
            full_query = f"{context}\n\nQuery: {query}" if context else query

            # Run the agent with the query
            logger.info(f"🤖 Calling agent '{agent_type}' with query: {query}")
            result = await agent.run(full_query)

            logger.info(f"✅ Agent '{agent_type}' finished processing")
            return f"Result from {agent_type} agent:\n{result}"

        except Exception as e:
            error_msg = f"Error calling agent '{agent_type}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_agent(self, agent_type: str):
        """Get or create an agent of the specified type."""
        if agent_type not in self._agent_instances:
            if agent_type == "mcp":
                self._agent_instances[agent_type] = MCPAgent()
            elif agent_type == "data_eng":
                self._agent_instances[agent_type] = DataEngAgent()
            elif agent_type == "product_manager":
                self._agent_instances[agent_type] = ProductManagerAgent()
            elif agent_type == "tech_lead":
                self._agent_instances[agent_type] = TechLeadAgent()
            elif agent_type == "finance_lead":
                self._agent_instances[agent_type] = FinanceLeadAgent()
            elif agent_type == "law_lead":
                self._agent_instances[agent_type] = LawLeadAgent()
            elif agent_type == "seo_lead":
                self._agent_instances[agent_type] = SEOLeadAgent()
            elif agent_type == "marketing_lead":
                self._agent_instances[agent_type] = MarketingLeadAgent()
            elif agent_type == "hr_lead":
                self._agent_instances[agent_type] = HRLeadAgent()
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        return self._agent_instances[agent_type]

    async def cleanup(self):
        """Clean up resources used by all agent instances."""
        for agent_type, agent in list(self._agent_instances.items()):
            try:
                if hasattr(agent, "cleanup"):
                    logger.info(f"🧹 Cleaning up agent: {agent_type}")
                    await agent.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up agent '{agent_type}': {str(e)}")

        # Clear agent instances
        self._agent_instances.clear()
