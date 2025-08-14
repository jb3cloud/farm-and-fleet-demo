from __future__ import annotations

import logging
import os
import traceback
from datetime import UTC, datetime
from typing import Annotated

import chainlit as cl
import dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory, StreamingChatMessageContent
from semantic_kernel.functions import kernel_function

# Import plugins
from plugins import DatabasePlugin, FrictionPointSearchPlugin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
if dotenv.load_dotenv(override=True):
    logger.info("Environment variables loaded")

request_settings = AzureChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(
        filters={"excluded_plugins": ["ChatBot"]}
    )
)


def load_system_prompt() -> str:
    """Load and format the system prompt from LLM_SYSTEM_PROMPT.md"""
    try:
        # Path to the system prompt file (relative to the project root)
        system_prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "prompts/LLM_SYSTEM_PROMPT.md"
        )

        with open(system_prompt_path, encoding="utf-8") as f:
            content = f.read()

        # Extract the core system instructions (everything after the first header)
        lines = content.split("\n")
        system_prompt_lines = []
        skip_until_role = True

        for line in lines:
            if skip_until_role and line.strip().startswith("## Your Role"):
                skip_until_role = False
                continue
            elif not skip_until_role:
                # Include everything after "Your Role" section
                system_prompt_lines.append(line)

        system_prompt = "\n".join(system_prompt_lines).strip()

        # Add dynamic context about available capabilities
        dynamic_context = "\n\n## Current Session Context\n"
        dynamic_context += "- Social Media Data: 1,833 documents with engagement metrics and sentiment analysis\n"
        dynamic_context += "- Survey Data: 61,787 responses with detailed customer feedback and location data\n"
        dynamic_context += (
            "- SQL Database: Direct access to structured business data with 5 tables\n"
        )
        dynamic_context += "- Available Functions: CustomerInsights plugin with 5 core analytical functions\n"
        dynamic_context += "- Available Functions: DatabaseInsights plugin with 4 core SQL query functions\n"
        dynamic_context += "- DateTime plugin available for temporal context\n"

        return system_prompt + dynamic_context

    except Exception as e:
        logger.warning(f"Failed to load system prompt from file: {str(e)}")
        return ""


# Example Native Plugin (Tool)
class DateTimePlugin:
    @kernel_function(
        name="get_current_datetime", description="Gets the current date and time"
    )
    def get_current_date(self) -> str:
        """Retrieves the current date and time."""
        return datetime.now(UTC).isoformat()


class CodeExecutionPlugin:
    @kernel_function(name="execute_code", description="Executes simple Python code")
    def execute_code(
        self, code: Annotated[str, "Simple python code to execute using exec()"]
    ) -> str:
        """Executes Python code."""
        try:
            exec(code)
            return "Code executed successfully."
        except Exception as e:
            return f"Error executing code: {str(e)}"


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        logger.info("Starting chat session initialization...")

        # Check required environment variables for Azure OpenAI
        required_env_vars = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }

        # Check optional Azure Search environment variables
        azure_search_vars = {
            "AZURE_SEARCH_SERVICE_ENDPOINT": os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
            "AZURE_SEARCH_API_KEY": os.getenv("AZURE_SEARCH_API_KEY"),
            "AZURE_SEARCH_INDEX_NAME": os.getenv("AZURE_SEARCH_INDEX_NAME"),
            "AZURE_SEARCH_SURVEY_INDEX_NAME": os.getenv(
                "AZURE_SEARCH_SURVEY_INDEX_NAME"
            ),
        }

        missing_search_vars = [
            var for var, value in azure_search_vars.items() if not value
        ]
        if missing_search_vars:
            logger.warning(
                f"Missing Azure Search environment variables: {', '.join(missing_search_vars)}"
            )
            logger.warning("Customer Insights plugin may not work properly")

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

        # Setup Semantic Kernel
        logger.info("Creating Semantic Kernel...")
        kernel = sk.Kernel()

        # Initialize Azure OpenAI service
        logger.info("Initializing Azure OpenAI service...")
        deployment_name = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT", "gpt-5-mini")

        ai_service = AzureChatCompletion(
            deployment_name=deployment_name,
            api_key=required_env_vars["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            endpoint=required_env_vars["AZURE_OPENAI_ENDPOINT"],
        )

        logger.info(f"Using deployment: {deployment_name}")
        kernel.add_service(ai_service)

        # Import the WeatherPlugin
        logger.info("Adding WeatherPlugin...")
        kernel.add_plugin(DateTimePlugin(), plugin_name="DateTime")

        # Import the CodeExecutionPlugin
        logger.info("Adding CodeExecutionPlugin...")
        kernel.add_plugin(CodeExecutionPlugin(), plugin_name="CodeExecution")

        # Add the FrictionPointSearchPlugin for customer feedback analysis
        logger.info("Adding FrictionPointSearchPlugin...")
        try:
            friction_plugin = FrictionPointSearchPlugin()
            kernel.add_plugin(friction_plugin, plugin_name="CustomerInsights")
            logger.info("FrictionPointSearchPlugin added successfully")
        except Exception as e:
            logger.warning(f"Failed to add FrictionPointSearchPlugin: {str(e)}")
            # Continue without the plugin rather than failing completely
            await cl.Message(
                content=f"⚠️ **Plugin Warning**: Customer Insights plugin failed to load: {str(e)}\n\n"
                "The chat will continue to work, but customer feedback search functions won't be available."
            ).send()

        # Add the DatabasePlugin for SQL database analysis
        logger.info("Adding DatabasePlugin...")
        try:
            database_plugin = DatabasePlugin()
            kernel.add_plugin(database_plugin, plugin_name="DatabaseInsights")
            logger.info("DatabasePlugin added successfully")
        except Exception as e:
            logger.warning(f"Failed to add DatabasePlugin: {str(e)}")
            # Continue without the plugin rather than failing completely
            await cl.Message(
                content=f"⚠️ **Plugin Warning**: Database Insights plugin failed to load: {str(e)}\n\n"
                "The chat will continue to work, but database query functions won't be available."
            ).send()

        # Instantiate and add the Chainlit filter to the kernel
        logger.info("Setting up Chainlit filter...")
        sk_filter = cl.SemanticKernelFilter(kernel=kernel)  # pyright: ignore[reportUnusedVariable]  # noqa: F841

        # Initialize chat history with system prompt
        logger.info("Initializing chat history with system prompt...")
        chat_history = ChatHistory()

        # Load and customize system prompt based on available plugins
        system_prompt = load_system_prompt()

        # Add dynamic context about plugin availability
        if missing_search_vars:
            system_prompt += "\n\n**Important Session Note**: Some Azure Search environment variables are missing, so customer insights features may be limited. Inform users when CustomerInsights functions are not available."
        else:
            system_prompt += "\n\n**Session Status**: All CustomerInsights functions are fully available and configured."

        # Add system message to chat history following Semantic Kernel best practices
        chat_history.add_system_message(system_prompt)
        logger.info("System prompt added to chat history successfully")

        # Store session variables
        logger.info("Storing session variables...")
        cl.user_session.set("kernel", kernel)
        cl.user_session.set("ai_service", ai_service)
        cl.user_session.set("chat_history", chat_history)

        # Send success message with available capabilities
        capabilities = ["I can help with general questions and tasks"]
        if not missing_search_vars:
            capabilities.append(
                "I can search and analyze customer feedback from social media and surveys"
            )
            capabilities.append(
                "I can find friction points and provide insights about customer pain points"
            )

        # Database capabilities are always available (no dependency on search vars)
        capabilities.append(
            "I can query and analyze data from SQL databases with schema discovery"
        )
        capabilities.append(
            "I can execute safe SQL queries and provide statistical summaries"
        )

        welcome_msg = (
            "Hello! How can I assist you today?\n\n**Available capabilities:**\n"
        )
        welcome_msg += "\n".join(f"• {cap}" for cap in capabilities)

        if missing_search_vars:
            welcome_msg += "\n\n⚠️ *Note: Some Azure Search environment variables are missing, so customer insights features may be limited.*"

        # Create action buttons for common queries
        actions = []

        await cl.Message(content=welcome_msg, actions=actions).send()

        logger.info("Chat session initialization completed successfully")

    except Exception as e:
        error_msg = f"Failed to initialize session: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        await cl.Message(
            content=f"❌ **Initialization Error**: {error_msg}\n\n"
            "Please check your Azure OpenAI configuration and try again.\n"
            f"**Error details**: `{type(e).__name__}: {str(e)}`"
        ).send()


async def handle_query(query: str) -> None:
    """Handle a query from either a button action or direct message"""
    try:
        # Retrieve session variables
        kernel: sk.Kernel | None = cl.user_session.get("kernel")
        ai_service: AzureChatCompletion | None = cl.user_session.get("ai_service")
        chat_history: ChatHistory | None = cl.user_session.get("chat_history")

        # Check session initialization with detailed error reporting
        if kernel is None:
            logger.error("Kernel not found in session")
            await cl.Message(
                content="❌ **Session Error**: Kernel not initialized. Please refresh the page to restart the session."
            ).send()
            return

        if ai_service is None:
            logger.error("AI service not found in session")
            await cl.Message(
                content="❌ **Session Error**: AI service not initialized. Please check your Azure OpenAI configuration and refresh the page."
            ).send()
            return

        if chat_history is None:
            logger.error("Chat history not found in session")
            await cl.Message(
                content="❌ **Session Error**: Chat history not initialized. Please refresh the page to restart the session."
            ).send()
            return

        logger.info(f"Processing query: {query[:50]}...")

        # Add user message to history
        chat_history.add_user_message(query)

        # Create a Chainlit message for the response stream
        answer = cl.Message(content="")

        # Stream the AI response
        full_response = ""
        async for msg in ai_service.get_streaming_chat_message_content(
            chat_history=chat_history,
            user_input=query,
            settings=request_settings,
            kernel=kernel,
        ):
            msg_typed: StreamingChatMessageContent = msg
            if msg_typed.content:
                await answer.stream_token(msg_typed.content)
                full_response += msg_typed.content

        # Add the full assistant response to history
        chat_history.add_assistant_message(full_response)

        # Send the final message
        await answer.send()

        logger.info("Query processed successfully")

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        await cl.Message(
            content=f"❌ **Processing Error**: {error_msg}\n\n"
            "Please try again or refresh the page if the issue persists.\n"
            f"**Error details**: `{type(e).__name__}: {str(e)}`"
        ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle direct messages from users"""
    await handle_query(message.content)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
