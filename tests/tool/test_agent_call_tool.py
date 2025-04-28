import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from app.tool.agent_call_tool import AgentCallTool


@pytest.fixture
def agent_call_tool():
    return AgentCallTool()


@patch("app.tool.agent_call_tool.MCPAgent")
async def test_agent_call_basic(mock_mcp_agent, agent_call_tool):
    """Test that the agent call tool correctly calls the specified agent."""
    # Set up the mock
    mock_agent_instance = AsyncMock()
    mock_agent_instance.run.return_value = "Test result from MCP agent"
    mock_mcp_agent.return_value = mock_agent_instance

    # Call the tool
    result = await agent_call_tool._run(
        agent_type="mcp",
        query="Test query"
    )

    # Check that the agent was called correctly
    mock_agent_instance.run.assert_called_once_with("Test query")

    # Verify the result
    assert "Result from mcp agent" in result
    assert "Test result from MCP agent" in result


@patch("app.tool.agent_call_tool.MCPAgent")
async def test_agent_call_with_context(mock_mcp_agent, agent_call_tool):
    """Test that the agent call tool correctly includes context when provided."""
    # Set up the mock
    mock_agent_instance = AsyncMock()
    mock_agent_instance.run.return_value = "Test result with context"
    mock_mcp_agent.return_value = mock_agent_instance

    # Call the tool with context
    result = await agent_call_tool._run(
        agent_type="mcp",
        query="Test query",
        context="This is some context"
    )

    # Check that the agent was called with combined context and query
    mock_agent_instance.run.assert_called_once_with("This is some context\n\nQuery: Test query")

    # Verify the result
    assert "Result from mcp agent" in result
    assert "Test result with context" in result


@patch("app.tool.agent_call_tool.MCPAgent")
async def test_agent_call_error_handling(mock_mcp_agent, agent_call_tool):
    """Test that the agent call tool handles errors correctly."""
    # Set up the mock to raise an exception
    mock_agent_instance = AsyncMock()
    mock_agent_instance.run.side_effect = Exception("Test error")
    mock_mcp_agent.return_value = mock_agent_instance

    # Call the tool
    result = await agent_call_tool._run(
        agent_type="mcp",
        query="Test query"
    )

    # Verify that the error is captured and returned as a string
    assert "Error calling agent 'mcp'" in result
    assert "Test error" in result


@patch("app.tool.agent_call_tool.MCPAgent")
async def test_agent_cleanup(mock_mcp_agent, agent_call_tool):
    """Test that the agent call tool cleans up properly."""
    # Set up the mock
    mock_agent_instance = AsyncMock()
    mock_mcp_agent.return_value = mock_agent_instance

    # Call the tool first to initialize the agent
    await agent_call_tool._run(agent_type="mcp", query="Test query")

    # Call cleanup
    await agent_call_tool.cleanup()

    # Check that the agent's cleanup method was called
    mock_agent_instance.cleanup.assert_called_once()

    # Verify that the _agent_instances dict was cleared
    assert not agent_call_tool._agent_instances


if __name__ == "__main__":
    asyncio.run(test_agent_call_basic(AgentCallTool()))
    asyncio.run(test_agent_call_with_context(AgentCallTool()))
    asyncio.run(test_agent_call_error_handling(AgentCallTool()))
    asyncio.run(test_agent_cleanup(AgentCallTool()))
