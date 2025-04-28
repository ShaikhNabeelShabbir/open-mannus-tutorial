from typing import Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.agent_call_tool import AgentCallTool
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


class Manus(ToolCallAgent):
    """A versatile general-purpose agent."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), BrowserUseTool(), StrReplaceEditor(), AgentCallTool(), Terminate()
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result

    async def execute_tool(self, command) -> str:
        """Override to add logging for agent delegation."""
        # Check if this is an agent call tool
        if command.function.name == "call_agent":
            logger.warning("🧠 Manus is analyzing your request and considering specialized assistance...")

        # Call the parent method to execute the tool
        result = await super().execute_tool(command)

        # Additional logging after executing the agent call
        if command.function.name == "call_agent":
            logger.warning("🔄 Manus has received specialized input and is integrating it with its analysis...")

        return result

    async def run(self, request: Optional[str] = None) -> str:
        """Override to add logging when Manus starts processing."""
        if request:
            logger.warning(f"🔍 Manus is analyzing your request: \"{request}\"")
            logger.warning("🤖 Manus may delegate to specialized agents if needed for this task...")

        result = await super().run(request)

        # Add completion message
        logger.warning("✨ Manus has completed processing your request")

        return result

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()

        # Call parent cleanup to handle tools including AgentCallTool
        await super().cleanup()
