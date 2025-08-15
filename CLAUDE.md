# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Farm & Fleet Marketing Insights Platform - A Chainlit-based AI assistant that analyzes customer feedback from social media and surveys, correlating it with business data to identify actionable insights and quantify financial impact on sales and operations.

## Core Architecture

### Technology Stack
- **Python 3.12+** with uv for dependency management
- **Chainlit** for the conversational AI interface
- **PydanticAI** for LLM orchestration and tool management
- **Azure OpenAI** for chat completion
- **Azure Cognitive Search** for semantic search across customer feedback
- **SQL Server/Azure SQL** for transactional business data

### Tool-Based Architecture
The application follows a modular tool-based architecture with two main data sources:

1. **CustomerInsights Tools** (`src/plugins/search/`) - Semantic search across customer feedback
2. **DatabaseInsights Tools** (`src/plugins/sqldb/`) - SQL database querying and analysis

## Essential Commands

### Development Environment
```bash
# Activate virtual environment and install dependencies
uv sync

# Run the application
uv run chainlit run src/app_pydantic.py

# Type checking with basedpyright
uv run basedpyright

# Code formatting and linting with ruff
uv run ruff check
uv run ruff format
```

### Data Processing Scripts
```bash
# Process and upload social media data
uv run python scripts/social_media_upload.py

# Process and upload survey data
uv run python scripts/survey_upload.py

# Run analytics on social media content
uv run python scripts/social_media_analytics.py

# Run analytics on survey data
uv run python scripts/survey_analytics.py
```

## Core Application Flow

### Main Application (`src/app_pydantic.py`)
- **Initialization**: Sets up Azure OpenAI service and loads tools
- **System Prompt**: Loaded from `src/prompts/LLM_SYSTEM_PROMPT.md` with dynamic context
- **Tool Registration**: Automatically registers CustomerInsights and DatabaseInsights tools
- **Session Management**: Handles chat history and streaming responses

### ReAct Framework Implementation
The system prompt enforces a **ReAct framework** (Reason → Act → Observe → Think) where the LLM must:
1. **ALWAYS** start with schema discovery functions before data retrieval
2. Connect customer feedback to quantifiable business metrics
3. Follow structured analysis patterns

## Tool Architecture Details

### CustomerInsights Tool Functions
- `get_analytics_schema()` - **MUST be called first** for feedback analysis
- `search_customer_feedback()` - General semantic search
- `find_friction_points()` - Targeted pain point discovery
- `get_feedback_summary()` - Aggregate metrics
- `search_priority_feedback()` - Business-priority ranking
- `analyze_cross_sources()` - Cross-source comparison

### DatabaseInsights Tool Functions
- `get_database_schema()` - **MUST be called first** for database analysis
- `query_table_data()` - Simple table lookups
- `get_table_summary()` - Table metadata
- `execute_sql_query()` - Custom T-SQL queries for business impact analysis

## SQL Database Requirements

### T-SQL Syntax (Azure SQL/SQL Server)
- Use `TOP(n)` not `LIMIT`
- Always include `ON` clause in JOINs
- Use `OFFSET...FETCH` after `ORDER BY` for pagination
- JSON functions: `OPENJSON`, `JSON_VALUE`, `JSON_QUERY`
- CTEs require semicolon: `;WITH ...`

### Data Dictionary Integration
The database plugin supports optional data dictionaries in `src/plugins/sqldb/data_dictionaries/` that provide:
- Business context for columns
- Usage patterns for JSON columns
- Query examples for complex data types
- Analysis patterns for specific business scenarios

## Environment Configuration

### Required Variables
```bash
# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-service.openai.azure.com/
AZURE_OPENAI_MODEL_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Search (Required for CustomerInsights)
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=social-media-index
AZURE_SEARCH_SURVEY_INDEX_NAME=survey-index

# SQL Database (Required for DatabaseInsights)
SQL_SERVER=your-sql-server
SQL_PORT=1433
SQL_DATABASE=your-database-name
SQL_SCHEMA=dbo
SQL_USERNAME=your-username
SQL_PASSWORD=your-password
SQL_TABLES=table1,table2,table3  # Optional: restrict to specific tables
SQL_DATA_DICTIONARY_PATH=src/plugins/sqldb/data_dictionaries/
```

## Data Sources

### Customer Feedback Data
- **Social Media**: 1,833 documents with engagement metrics and sentiment analysis
- **Survey Data**: 61,787 responses with detailed feedback and location data
- **Structure**: Stored in Azure Cognitive Search with vector embeddings

### Business Data
- **SQL Database**: 5 tables with transactional data
- **Key Tables**: Customer, Orders, Items, Stores, Feedback
- **Special Handling**: JSON columns for AI-extracted insights

## Key Design Patterns

### Schema-First Analysis
- **NEVER** execute data retrieval without first calling schema discovery
- This prevents invalid queries and ensures proper parameter usage
- Schema functions provide available metrics, tables, and valid combinations

### Business Impact Synthesis
The core value proposition is connecting qualitative feedback to quantitative business metrics:
1. **Identify Friction** - What customers are saying (CustomerInsights)
2. **Quantify Impact** - Business metrics affected (DatabaseInsights)
3. **Build Business Case** - Synthesized insights with financial implications

### Context Control Mechanisms
Both plugins implement "just enough" context controls:
- **Quantity Control**: `max_results`, `max_tables` parameters
- **Content Control**: `detail_level` parameter (minimal/standard/detailed)
- **Relevance Control**: Quality filtering and scoring

## Testing and Quality

### Code Quality Tools
- **basedpyright**: Strict type checking with specific configuration for external libraries
- **ruff**: Code formatting and linting with pycodestyle, pyflakes, isort, bugbear rules
- **Print statements**: Allowed for this project (T201 ignored)

### Type Checking Configuration
- Python 3.11+ target with strict type checking
- External library warnings suppressed for routine issues
- Important checks enabled for missing imports, type arguments, optional operations

## File Structure Patterns

```
src/
├── app_pydantic.py           # Main Chainlit application
├── prompts/
│   └── LLM_SYSTEM_PROMPT.md # System prompt with ReAct framework
└── plugins/
    ├── search/              # CustomerInsights tools
    │   └── search_client.py
    └── sqldb/               # DatabaseInsights tools
        ├── database_client.py
        └── data_dictionaries/
            └── medallia_feedback.json

scripts/                     # Data processing utilities
source_data/                 # Raw data files (surveys, social media)
```

## Important Development Notes

### Tool Integration
- Tools are automatically registered in `app_pydantic.py` with graceful failure handling
- Missing environment variables show warnings but don't break the application
- Each tool operates independently with cross-tool synthesis at the LLM level

### Error Handling Philosophy
- Graceful degradation when tools fail to load
- Detailed error messages for configuration issues
- Continue operation with available capabilities rather than failing completely

### Performance Considerations
- Data dictionaries loaded once at startup and cached in memory
- Search results limited by configurable parameters
- SQL queries validated and protected against excessive resource usage

# Reference Documentation
- Deepwiki repository `Chainlit/chainlit`
- Pydantic AI documentation `pydantic/pydantic-ai`
