from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from app.agent.manus import Manus
from app.agent.mcp import MCPAgent
from app.agent.data_eng_agent import DataEngAgent
from app.agent.product_manager_agent import ProductManagerAgent
from app.agent.tech_lead_agent import TechLeadAgent
from app.logger import logger

app = Flask(__name__)
CORS(app)

# Thread pool for handling async operations
executor = ThreadPoolExecutor(max_workers=10)

# Create agent instances (or pool them)
agents = {}

def get_agent(agent_type):
    """Get or create an agent of the specified type"""
    if agent_type not in agents:
        if agent_type == "manus":
            agents[agent_type] = Manus()
        elif agent_type == "mcp":
            agents[agent_type] = MCPAgent()
        elif agent_type == "data_eng":
            agents[agent_type] = DataEngAgent()
        elif agent_type == "product_manager":
            agents[agent_type] = ProductManagerAgent()
        elif agent_type == "tech_lead":
            agents[agent_type] = TechLeadAgent()
        # Add more agent types as needed
    return agents[agent_type]

def run_agent(agent_type, query):
    """Run the agent in a way that can be called from Flask"""
    agent = get_agent(agent_type)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(agent.run(query))
        return result
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/api/query', methods=['POST'])
def query_agent():
    """Handle a query to an agent"""
    data = request.json
    agent_type = data.get('agent_type', 'manus')
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Run the agent asynchronously
    result = executor.submit(run_agent, agent_type, query).result()

    return jsonify({'result': result})

@app.route('/api/agents', methods=['GET'])
def list_agents():
    """List available agent types"""
    available_agents = ["manus", "mcp", "data_eng", "product_manager", "tech_lead"]
    return jsonify({'agents': available_agents})

@app.route('/api/cleanup', methods=['POST'])
def cleanup_agents():
    """Clean up agent resources"""
    data = request.json
    agent_type = data.get('agent_type')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if agent_type and agent_type in agents:
        # Clean up specific agent
        agent = agents[agent_type]
        loop.run_until_complete(agent.cleanup())
        del agents[agent_type]
        return jsonify({'status': f'Agent {agent_type} cleaned up'})
    else:
        # Clean up all agents
        for agent_type, agent in list(agents.items()):
            loop.run_until_complete(agent.cleanup())
            del agents[agent_type]
        return jsonify({'status': 'All agents cleaned up'})

if __name__ == '__main__':
    logger.info("Starting OpenManus API Server...")
    app.run(debug=True, host='0.0.0.0', port=3000)
