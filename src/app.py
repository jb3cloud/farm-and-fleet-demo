"""
Farm & Fleet Marketing Insights Platform - PydanticAI Implementation.

A Chainlit-based AI assistant that analyzes customer feedback from social media and surveys,
correlating it with business data to identify actionable insights and quantify financial impact.

This version uses PydanticAI instead of Semantic Kernel for improved type safety, better streaming,
and modern async patterns while preserving Chainlit Step functionality.
"""

from __future__ import annotations

import logging
import os
import traceback

import chainlit as cl
import dotenv
import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider

# Import dependencies and tool registration functions
from dependencies import AppDependencies
from plugins.search.search_client import AzureSearchClientWrapper
from plugins.sqldb.database_client import DatabaseClient
from tools import (
    register_analytics_tools,
    register_code_execution_tools,
    register_customer_insights_tools,
    register_database_tools,
    register_datetime_tools,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
if dotenv.load_dotenv(override=True):
    logger.info("Environment variables loaded")


def load_system_prompt() -> str:
    """Load and format the system prompt from LLM_SYSTEM_PROMPT.md."""
    try:
        # Path to the system prompt file (relative to the project root)
        system_prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts/LLM_SYSTEM_PROMPT.md"
        )
        logging.info(f"Loading system prompt from {system_prompt_path}")

        with open(system_prompt_path, encoding="utf-8") as f:
            content = f.read()

            # Add dynamic context about available capabilities
            dynamic_context = "\n\n## Current Session Context\n"
            dynamic_context += "- Social Media Data: 1,833 documents with engagement metrics and sentiment analysis\n"
            dynamic_context += "- Survey Data: 61,787 responses with detailed customer feedback and location data\n"
            dynamic_context += "- SQL Database: Direct access to structured business data with 5 tables\n"
            dynamic_context += "- Available Functions: CustomerInsights plugin with 6 core analytical functions\n"
            dynamic_context += "- Available Functions: DatabaseInsights plugin with 5 core SQL query functions\n"
            dynamic_context += "- Available Functions: StatisticalAnalytics plugin with 5 predictive analytics functions\n"
            dynamic_context += "- DateTime plugin available for temporal context\n"

            logger.info("Dynamic context added to system prompt")
            return content + dynamic_context

    except Exception as e:
        logger.warning(f"Failed to load system prompt from file: {str(e)}")
        return "You are a helpful AI assistant for Farm & Fleet customer insights analysis."


def convert_chainlit_to_pydantic_messages(
    chainlit_messages: list[dict[str, str]],
) -> list[ModelMessage]:
    """
    Convert Chainlit's OpenAI format messages to PydanticAI ModelMessage format.

    This function bridges the gap between Chainlit's chat history format and PydanticAI's
    message format, enabling conversation continuity by passing chat history to PydanticAI agents.

    Message Mapping:
    - "system" -> ModelRequest with SystemPromptPart
    - "user" -> ModelRequest with UserPromptPart
    - "assistant" -> ModelResponse with TextPart

    Example:
        chainlit_msgs = cl.chat_context.to_openai()
        pydantic_msgs = convert_chainlit_to_pydantic_messages(chainlit_msgs)
        result = agent.run(message_history=pydantic_msgs, ...)

    Args:
        chainlit_messages: List of dict with 'role' and 'content' keys from cl.chat_context.to_openai()

    Returns:
        List of ModelMessage objects compatible with PydanticAI's message_history parameter
    """
    pydantic_messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content=load_system_prompt())])
    ]

    for msg in chainlit_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            # System messages become ModelRequest with SystemPromptPart
            pydantic_messages.append(
                ModelRequest(parts=[SystemPromptPart(content=content)])
            )
        elif role == "user":
            # User messages become ModelRequest with UserPromptPart
            pydantic_messages.append(
                ModelRequest(parts=[UserPromptPart(content=content)])
            )
        elif role == "assistant":
            # Assistant messages become ModelResponse with TextPart
            pydantic_messages.append(ModelResponse(parts=[TextPart(content=content)]))
        # Note: We ignore other roles or handle them as needed

    return pydantic_messages


async def run_agent(
    agent: Agent[AppDependencies, str],
    message_history: list[ModelMessage],
    deps: AppDependencies,
) -> None:
    """
    Run PydanticAI agent with message history and handle tool Steps sequentially like cl.step decorator.

    Args:
        agent: The PydanticAI agent instance
        message_history: List of ModelMessage objects representing the conversation history
        deps: AppDependencies container with initialized clients
    """
    async with agent.iter(
        message_history=message_history,
        deps=deps,
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # A user prompt node => The user has provided input
                logger.info(f"=== UserPromptNode: {node.user_prompt} ===")
            elif Agent.is_model_request_node(node):
                # A model request node => We can stream tokens from the model's request
                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            logger.info(
                                f"[Request] Starting part {event.index}: {event.part!r}"
                            )
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                logger.debug(
                                    f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                                )
                            elif isinstance(event.delta, ThinkingPartDelta):
                                logger.info(
                                    f"[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}"
                                )
                            elif isinstance(event.delta, ToolCallPartDelta):
                                logger.info(
                                    f"[Request] Part {event.index} args delta: {event.delta.args_delta}"
                                )
                        elif isinstance(event, FinalResultEvent):
                            logger.info(
                                f"[Result] The model started producing a final result (tool_name={event.tool_name})"
                            )
                            final_result_found = True
                            break

                    if final_result_found:
                        # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                        # A similar `AgentStream.stream_output()` method is available to stream structured output.
                        msg = cl.Message(content="")
                        async for output in request_stream.stream_text(delta=True):
                            logger.debug(f"[Output] {output}")
                            await msg.stream_token(output)
                        await msg.send()

            elif Agent.is_call_tools_node(node):
                # A handle-response node => The model returned some data, potentially calls a tool
                logger.info(
                    "=== CallToolsNode: streaming partial response & tool usage ==="
                )
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            logger.info(
                                f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            logger.info(
                                f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
                            )
                        else:
                            logger.warning(f"[Tools] Unhandled Event: {event}")

            elif Agent.is_end_node(node):
                # Once an End node is reached, the agent run is complete
                assert run.result is not None
                assert run.result.output == node.data.output
                logger.info(f"=== Final Agent Output: {run.result.output} ===")


def create_agent_with_tools() -> Agent[AppDependencies, str]:
    """
    Create and configure the PydanticAI agent with all tools.

    Returns:
        Configured PydanticAI agent with type-safe dependencies
    """
    # Load Azure OpenAI configuration from environment (matching original app.py)
    deployment_name = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT", "gpt-5-mini")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Configure OpenAI model for Azure AI Foundry with proper provider
    model = OpenAIModel(
        model_name=deployment_name,
        provider=AzureProvider(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        ),
    )

    # Create agent with dependencies and system prompt
    agent: Agent[AppDependencies, str] = Agent(
        model,
        deps_type=AppDependencies,
        retries=3,
    )

    # Register all tool functions
    register_datetime_tools(agent)
    register_code_execution_tools(agent)
    register_customer_insights_tools(agent)
    register_database_tools(agent)
    register_analytics_tools(agent)

    return agent


def create_dependencies() -> AppDependencies:
    """
    Create and initialize the dependencies container.

    Returns:
        AppDependencies instance with initialized clients
    """
    # Initialize optional clients
    search_client = None
    database_client = None

    # Initialize search client if environment variables are available
    try:
        azure_search_vars = [
            "AZURE_SEARCH_SERVICE_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
            "AZURE_SEARCH_SURVEY_INDEX_NAME",
        ]

        if all(os.getenv(var) for var in azure_search_vars):
            search_client = AzureSearchClientWrapper()
            logger.info("Azure Search client initialized successfully")
        else:
            logger.warning(
                "Azure Search environment variables missing - search features disabled"
            )

    except Exception as e:
        logger.warning(f"Failed to initialize Azure Search client: {str(e)}")

    # Initialize database client if environment variables are available
    try:
        # Check for required SQL environment variables
        required_sql_vars = [
            "SQL_SERVER",
            "SQL_DATABASE",
            "SQL_USERNAME",
            "SQL_PASSWORD",
        ]
        if all(os.getenv(var) for var in required_sql_vars):
            database_client = DatabaseClient()
            logger.info("Database client initialized successfully")
        else:
            missing_vars = [var for var in required_sql_vars if not os.getenv(var)]
            logger.warning(
                f"Required SQL environment variables missing: {missing_vars} - database features disabled"
            )

    except Exception as e:
        logger.warning(f"Failed to initialize database client: {str(e)}")

    # Initialize analytics client (doesn't require external dependencies)
    try:
        # Analytics functionality is now built-in
        analytics_available = True
        logger.info("Analytics tools available (built-in)")
    except Exception as e:
        logger.warning(f"Analytics tools unavailable: {str(e)}")
        analytics_available = False

    # Create HTTP client for external requests
    http_client = httpx.AsyncClient(timeout=30.0)

    return AppDependencies(
        search_client=search_client,
        database_client=database_client,
        http_client=http_client,
        azure_search_available=search_client is not None,
        database_available=database_client is not None,
        analytics_available=analytics_available,
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the PydanticAI agent and dependencies for the chat session."""
    try:
        logger.info("Starting PydanticAI chat session initialization...")

        # Check required environment variables for Azure OpenAI
        required_env_vars = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }

        missing_vars = [var for var, value in required_env_vars.items() if not value]
        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            logger.error(error_msg)
            await cl.Message(
                content=f"❌ **Configuration Error**: {error_msg}\n\n"
                "Please check your `.env` file and ensure it contains:\n"
                "- `AZURE_OPENAI_API_KEY`\n"
                "- `AZURE_OPENAI_ENDPOINT`\n"
                "- `AZURE_OPENAI_MODEL_DEPLOYMENT` (optional, defaults to 'gpt-5-mini')"
            ).send()
            return

        # Create agent and dependencies
        logger.info("Creating PydanticAI agent...")
        agent = create_agent_with_tools()

        logger.info("Initializing dependencies...")
        deps = create_dependencies()

        # Store in session (tool Step integration handled via event streaming)
        cl.user_session.set("agent", agent)
        cl.user_session.set("dependencies", deps)

        logger.info("PydanticAI chat session initialization completed successfully")

    except Exception as e:
        error_msg = f"Failed to initialize session: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        await cl.Message(
            content=f"❌ **Initialization Error**: {error_msg}\n\n"
            "Please check your Azure OpenAI configuration and try again.\n"
            f"**Error details**: `{type(e).__name__}: {str(e)}`"
        ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming messages using PydanticAI agent with tool Step integration."""
    try:
        # Retrieve session objects
        agent: Agent[AppDependencies, str] | None = cl.user_session.get("agent")
        deps: AppDependencies | None = cl.user_session.get("dependencies")

        if not agent or not deps:
            await cl.Message(
                content="❌ **Session Error**: Agent not properly initialized. Please refresh the page."
            ).send()
            return

        try:
            # Get chat history from Chainlit and convert to PydanticAI format
            chainlit_history = cl.chat_context.to_openai()
            # Add the current message to the history
            chainlit_history.append({"role": "user", "content": message.content})
            # Convert to PydanticAI format
            pydantic_messages = convert_chainlit_to_pydantic_messages(chainlit_history)

            await run_agent(agent, pydantic_messages, deps)

        except Exception as e:
            logger.error(f"Agent run failed: {str(e)}")
            await cl.Message(
                content=f"❌ **Processing Error**: {str(e)}\n\n"
                "Please try again or check if all required services are available."
            ).send()

        logger.info("Message processed successfully")

    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        await cl.Message(
            content=f"❌ **Processing Error**: {error_msg}\n\n"
            "Please try again or refresh the page if the issue persists.\n"
            f"**Error details**: `{type(e).__name__}: {str(e)}`"
        ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
