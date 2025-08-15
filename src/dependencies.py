"""
Farm & Fleet Marketing Insights Platform - Dependencies.

This module contains the AppDependencies class for PydanticAI dependency injection.
"""

from __future__ import annotations

import httpx

# Import plugin clients
from plugins.search.search_client import AzureSearchClientWrapper
from plugins.sqldb.database_client import DatabaseClient


class AppDependencies:
    """
    Dependencies container for PydanticAI agent.

    Contains all the external services and clients needed by the agent tools.
    This follows PydanticAI's dependency injection pattern for type-safe tool access.
    """

    def __init__(
        self,
        search_client: AzureSearchClientWrapper | None = None,
        database_client: DatabaseClient | None = None,
        http_client: httpx.AsyncClient | None = None,
        azure_search_available: bool = False,
        database_available: bool = False,
        analytics_available: bool = False,
    ):
        # Optional clients (may be None if environment variables are missing)
        self.search_client = search_client
        self.database_client = database_client

        # HTTP client for external API calls
        self.http_client = http_client

        # Environment and configuration info
        self.azure_search_available = azure_search_available
        self.database_available = database_available
        self.analytics_available = analytics_available
