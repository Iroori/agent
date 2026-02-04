# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
AI Agent system built with Python using LangChain/LangGraph (ReAct pattern), FastAPI with WebSocket support, and MCP Protocol for external tool integration.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run server (production)
python app.py

# Run with hot reload (development)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run client static server (for frontend development)
cd client && python -m http.server 3000
```

## Environment Setup

Copy `.env.example` to `.env` and configure:
- `AGENT_CONFIG_LOAD_TYPE`: `local-config` or `seedai-api`
- `DEFAULT_MODEL_TYPE`: `gpt`, `claude`, `ollama`, `luxia`, `gemini`, `grok`, `friendli`
- Model API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Development Philosophy
- **Minimal Code**: Avoid unnecessary/redundant code
- **Requirements-First**: Implement exactly what's requested, nothing more
- **Reusability**: Create reusable components but don't over-engineer
- **Debug Mode**: Use debug logging for development visibility
- **No Testing**: Don't create test files unless explicitly requested

## Code Modification Rules

### CRITICAL: Permission Protocol
- **NEVER modify existing code** without explicit permission
- **ASK FIRST** before changing any file
- **SHOW CODE FIRST** before implementing changes
- **WAIT FOR APPROVAL** before proceeding
- **PRESERVE EXISTING CODE** unless specifically told to change it

### Implementation Guidelines
1. **Requirements Only**: Implement exactly what's requested
2. **Suggest After**: Complete requirements first, then suggest improvements
3. **No Assumptions**: Don't add features "for completeness"
4. **Debug Logging**: Use `logger.debug()` for development visibility
5. **No Documentation**: Don't create README.md or API docs unless requested

## Architecture Overview

### Request Flow
```
WebSocket Connection → WebsocketHandler (api/websocket_v2.py)
                     → AIAgentGateway (ai_agent/ai_agent_gateway.py)
                     → AIAgentBuilder creates AIAgent with tools
                     → AIAgent.generate_response() → ReAct Agent (LangGraph)
                     → Token streaming back to client
```

### Key Components
- `app.py`: FastAPI application entry point, lifespan management
- `ai_agent/ai_agent.py`: Main AIAgent class with ReAct pattern via `create_react_agent()`
- `ai_agent/ai_agent_builder.py`: Builder pattern for agent construction, MCP tool filtering
- `ai_agent/ai_agent_gateway.py`: Agent pool management, singleton access
- `ai_agent/model_factory.py`: Multi-provider LLM factory (GPT, Claude, Ollama, Gemini, Grok, Friendli, Luxia)
- `api/websocket_v2.py`: WebSocket connection handling, JWT auth
- `services/mcp_service.py`: MCP client wrapper using `langchain-mcp-adapters`

### Configuration Loading
Two modes controlled by `AGENT_CONFIG_LOAD_TYPE`:
- `local-config`: `ai_agent/config/file_config_loader.py` loads from `config/local_files/`
- `seedai-api`: `ai_agent/config/seed_ai_api_config_loader.py` fetches from remote API

### Memory Architecture
- `ai_agent/memory/ai_agent_memory.py`: Base memory interface
- `ai_agent/memory/connection_conversation_file_memory.py`: File-based storage
- `ai_agent/memory/connection_conversation_api_memory.py`: API-based storage
- Conversation summaries are generated for long conversations

## Development Patterns

### WebSocket Message Types
- `START_STREAM`: Begin response
- `STREAM_DATA`: Token-level content
- `END_STREAM`: Complete response
- `STREAM_ERROR`: Error message
- `CONVERSATION_REFRESH`: New conversation created notification

### Adding New LLM Provider
1. Add provider settings in `config/settings.py`
2. Add model creation logic in `ai_agent/model_factory.py` `create_model_from_ai_agent_info()`
3. Model must extend `BaseChatModel` from langchain_core

### Adding MCP Tools
1. Configure MCP server in agent config JSON (transport: `sse`, `http`, `streamable_http`, or `stdio`)
2. Add tool names to `bind_tools` with `type: "mcp"` in agent config
3. Tools are filtered by `AIAgentBuilder._get_filtered_mcp_tools()`

### Sub-Agent Pattern
- Main agent can have sub-agents defined by `sub_agent_ids` in config
- Sub-agents are converted to `SubAgentTool` and callable as tools
- Each agent has `AgentType.MAIN` or `AgentType.SUB`

## Code Standards

### Naming Conventions (PEP 8)
- Variables/Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_private_method`
- Booleans: `is_`, `has_`, `can_` prefixes

### File Organization
- Follow existing directory structure
- Ask before adding new files
- Import organization: stdlib → third-party → local
- Type hints required for all functions

### Error Handling
```python
try:
    # operation
    logger.debug(f"Operation completed: {result}")
except SpecificError as e:
    logger.error(f"Specific error in operation: {e}")
    raise
```

## Debugging
- Use `logger.debug()` for development
- Environment-based log levels
- No print statements
- Structured error messages

## What NOT to Do
- ❌ Create test files automatically
- ❌ Add features without request
- ❌ Modify code without permission
- ❌ Create documentation files
- ❌ Over-engineer solutions
- ❌ Add unnecessary abstractions
- ❌ Delete existing code arbitrarily
