"""
Database client wrapper implementing safe SQL operations.
Self-contained module for DatabasePlugin.
"""

import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import URL, Engine, MetaData, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def serialize_value(value: Any) -> Any:
    """Convert non-JSON serializable values to serializable format."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    elif isinstance(value, Decimal):
        return float(value)
    elif value is None:
        return None
    else:
        return value


class DatabaseClient:
    """Database client with safe SQL operations and schema introspection."""

    def __init__(self):
        """Initialize database client from environment variables."""
        self.server = os.getenv("SQL_SERVER")
        self.sql_port = int(os.getenv("SQL_PORT", "1433"))
        self.database = os.getenv("SQL_DATABASE")
        self.schema = os.getenv("SQL_SCHEMA", "dbo")
        self.username = os.getenv("SQL_USERNAME")
        self.password = os.getenv("SQL_PASSWORD")

        # Parse table filter if provided
        sql_tables = os.getenv("SQL_TABLES")
        self.allowed_tables = None
        if sql_tables:
            self.allowed_tables = [table.strip() for table in sql_tables.split(",") if table.strip()]
            logger.info(f"Table access restricted to: {self.allowed_tables}")

        if not all([self.server, self.database, self.username, self.password]):
            raise ValueError(
                "Missing required SQL environment variables: SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD"
            )

        self._engine: Engine | None = None
        self._metadata: MetaData | None = None
        self._table_warnings: list[str] = []
        self._data_dictionaries: dict[str, dict[str, Any]] = {}
        self._dictionary_path = os.getenv("SQL_DATA_DICTIONARY_PATH",
                                        os.path.join(os.path.dirname(__file__), "data_dictionaries"))

        # Load data dictionaries on initialization
        self._load_data_dictionaries()

    def get_engine(self) -> Engine:
        """Get SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            connection_url = URL.create(
                "mssql+pyodbc",
                username=self.username,
                password=self.password,
                host=self.server,
                port=self.sql_port,
                database=self.database,
                query={
                    "driver": "ODBC Driver 18 for SQL Server",
                    "TrustServerCertificate": "Yes",
                    "Connection Timeout": "30",
                },
            )

            self._engine = create_engine(
                connection_url, pool_pre_ping=True, pool_recycle=900, echo=False
            )

        return self._engine

    def get_metadata(self) -> MetaData:
        """Get database metadata with table reflection and filtering."""
        if self._metadata is None:
            engine = self.get_engine()
            self._metadata = MetaData()

            try:
                # Reflect tables from the specified schema
                self._metadata.reflect(bind=engine, schema=self.schema)

                # Apply table filtering if SQL_TABLES is specified
                if self.allowed_tables:
                    all_tables = list(self._metadata.tables.keys())
                    filtered_metadata = MetaData()
                    found_tables = []

                    for allowed_table in self.allowed_tables:
                        full_table_name = f"{self.schema}.{allowed_table}"
                        if full_table_name in all_tables:
                            # Add to found tables list
                            found_tables.append(allowed_table)
                        else:
                            # Generate warning for missing table
                            warning = f"Table '{allowed_table}' specified in SQL_TABLES not found in schema '{self.schema}'"
                            self._table_warnings.append(warning)
                            logger.warning(warning)

                    # Reflect only the allowed tables in a new metadata object
                    if found_tables:
                        filtered_metadata = MetaData()
                        for table_name in found_tables:
                            filtered_metadata.reflect(bind=engine, schema=self.schema, only=[table_name])
                        self._metadata = filtered_metadata
                    else:
                        # No valid tables found, use empty metadata
                        self._metadata = MetaData()

                    self._metadata = filtered_metadata
                    logger.info(f"Filtered to {len(found_tables)} allowed tables: {found_tables}")
                    if self._table_warnings:
                        logger.warning(f"Table warnings: {len(self._table_warnings)} tables not found")
                else:
                    logger.info(f"Reflected {len(self._metadata.tables)} tables from schema '{self.schema}'")

            except Exception as e:
                logger.error(f"Failed to reflect database schema: {str(e)}")
                raise

        return self._metadata

    def get_table_warnings(self) -> list[str]:
        """Get list of warnings about tables specified in SQL_TABLES but not found."""
        return self._table_warnings.copy()

    def is_safe_query(self, query: str) -> tuple[bool, str]:
        """Check if a SQL query is safe to execute (read-only).

        Args:
            query: The SQL query to check

        Returns:
            Tuple of (is_safe, reason)
        """
        query_lower = query.lower().strip()

        # List of unsafe operations
        unsafe_operations = [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "truncate",
            "merge",
            "exec",
            "execute",
            "sp_",
            "xp_",
            "grant",
            "revoke",
            "deny",
        ]

        # Check for unsafe operations
        for operation in unsafe_operations:
            if any(word == operation for word in query_lower.split()):
                return False, f"Query contains unsafe operation: {operation}"

        # Check for multiple statements
        if ";" in query_lower and not query_lower.endswith(";"):
            return False, "Multiple SQL statements are not allowed"

        # Check for comments
        if "--" in query_lower or "/*" in query_lower:
            return False, "Comments in SQL queries are not allowed for security reasons"

        # Must be a read operation
        read_operations = ["select", "with", "show", "describe", "explain"]
        if not any(query_lower.lstrip().startswith(op) for op in read_operations):
            return False, "Query must start with a read operation (SELECT, WITH, etc.)"

        return True, "Query is safe to execute"

    def execute_query(self, query: str, max_rows: int = 50) -> dict[str, Any]:
        """Execute a safe SQL query with row limits.

        Args:
            query: SQL query to execute
            max_rows: Maximum number of rows to return

        Returns:
            Dictionary with results and metadata
        """
        # Validate query safety
        is_safe, reason = self.is_safe_query(query)
        if not is_safe:
            return {
                "results": [],
                "total_count": 0,
                "error": f"Query rejected: {reason}",
                "query": query,
            }

        try:
            engine = self.get_engine()

            # Add row limit if not already present
            limited_query = query
            query_lower = query.lower()
            if "top " not in query_lower and "limit " not in query_lower:
                if query_lower.strip().startswith("select"):
                    # Add TOP clause for SQL Server
                    limited_query = query.replace("select", f"select top {max_rows}", 1)

            with engine.connect() as conn:
                result = conn.execute(text(limited_query))
                rows = result.fetchall()

                # Convert rows to dictionaries with proper serialization
                if rows:
                    columns = list(result.keys())
                    results = []
                    for row in rows:
                        row_dict = {}
                        for col, value in zip(columns, row, strict=False):
                            row_dict[col] = serialize_value(value)
                        results.append(row_dict)
                else:
                    results = []
                    columns = []

                return {
                    "results": results,
                    "total_count": len(results),
                    "columns": columns,
                    "query": limited_query,
                }

        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {str(e)}")
            return {
                "results": [],
                "total_count": 0,
                "error": f"Query execution failed: {str(e)}",
                "query": query,
            }

    def get_table_info(
        self, table_name: str, max_sample_rows: int = 5
    ) -> dict[str, Any]:
        """Get detailed information about a table.

        Args:
            table_name: Name of the table to analyze
            max_sample_rows: Maximum number of sample rows to return

        Returns:
            Dictionary with table information
        """
        try:
            metadata = self.get_metadata()
            full_table_name = f"{self.schema}.{table_name}"

            if full_table_name not in metadata.tables:
                return {
                    "error": f"Table '{table_name}' not found in schema '{self.schema}'",
                    "available_tables": list(metadata.tables.keys()),
                }

            table = metadata.tables[full_table_name]

            # Get basic table info
            table_info: dict[str, Any] = {
                "table_name": table_name,
                "full_name": full_table_name,
                "columns": [],
                "primary_keys": [],
                "foreign_keys": [],
                "indexes": [],
            }

            # Column information
            for column in table.columns:
                col_info = {
                    "name": column.name,
                    "type": str(column.type),
                    "nullable": column.nullable,
                    "default": str(column.default) if column.default else None,
                }
                table_info["columns"].append(col_info)

                if column.primary_key:
                    table_info["primary_keys"].append(column.name)

            # Foreign key information
            for fk in table.foreign_key_constraints:
                fk_info = {
                    "columns": [col.parent.name for col in fk.elements],
                    "referenced_table": fk.referred_table.name,
                    "referenced_columns": [col.column.name for col in fk.elements],
                }
                table_info["foreign_keys"].append(fk_info)

            # Index information
            for index in table.indexes:
                idx_info = {
                    "name": index.name,
                    "columns": [col.name for col in index.columns],
                    "unique": index.unique,
                }
                table_info["indexes"].append(idx_info)

            # Get row count
            try:
                count_query = (
                    f"SELECT COUNT(*) as row_count FROM [{self.schema}].[{table_name}]"
                )
                count_result = self.execute_query(count_query, max_rows=1)
                if count_result["results"]:
                    table_info["row_count"] = count_result["results"][0]["row_count"]
                else:
                    table_info["row_count"] = 0
            except Exception as e:
                logger.warning(f"Could not get row count for {table_name}: {str(e)}")
                table_info["row_count"] = "unknown"

            # Get sample data if requested
            if max_sample_rows > 0:
                sample_query = f"SELECT TOP {max_sample_rows} * FROM [{self.schema}].[{table_name}]"
                sample_result = self.execute_query(
                    sample_query, max_rows=max_sample_rows
                )
                table_info["sample_data"] = sample_result["results"]

            return table_info

        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            return {
                "error": f"Failed to analyze table '{table_name}': {str(e)}",
                "table_name": table_name,
            }

    def _load_data_dictionaries(self) -> None:
        """Load data dictionary files from the dictionary path."""
        try:
            dictionary_path = Path(self._dictionary_path)
            if not dictionary_path.exists():
                logger.info(f"Data dictionary path does not exist: {dictionary_path}")
                return

            # Load all JSON files in the data dictionaries directory
            json_files = list(dictionary_path.glob("*.json"))
            if not json_files:
                logger.info(f"No data dictionary files found in: {dictionary_path}")
                return

            for json_file in json_files:
                try:
                    with open(json_file, encoding='utf-8') as f:
                        dictionary_data = json.load(f)

                    # Extract table name from the dictionary or filename
                    table_name = dictionary_data.get("table", {}).get("name")
                    if not table_name:
                        # Fallback to filename without extension
                        table_name = json_file.stem.upper()

                    self._data_dictionaries[table_name] = dictionary_data
                    logger.info(f"Loaded data dictionary for table: {table_name}")

                except (OSError, json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load data dictionary from {json_file}: {str(e)}")
                    continue

            logger.info(f"Loaded {len(self._data_dictionaries)} data dictionaries")

        except Exception as e:
            logger.warning(f"Error loading data dictionaries: {str(e)}")
            # Continue without dictionaries - graceful degradation

    def get_table_dictionary(self, table_name: str) -> dict[str, Any] | None:
        """Get data dictionary for a specific table.

        Args:
            table_name: Name of the table (case-insensitive)

        Returns:
            Dictionary data for the table, or None if not found
        """
        # Try exact match first
        if table_name in self._data_dictionaries:
            return self._data_dictionaries[table_name]

        # Try case-insensitive match
        table_upper = table_name.upper()
        if table_upper in self._data_dictionaries:
            return self._data_dictionaries[table_upper]

        # Try all available keys case-insensitively
        for key, value in self._data_dictionaries.items():
            if key.upper() == table_upper:
                return value

        return None

    def get_column_description(self, table_name: str, column_name: str) -> dict[str, Any] | None:
        """Get description for a specific column.

        Args:
            table_name: Name of the table
            column_name: Name of the column

        Returns:
            Column description dictionary, or None if not found
        """
        table_dict = self.get_table_dictionary(table_name)
        if not table_dict:
            return None

        columns = table_dict.get("columns", {})

        # Try exact match first
        if column_name in columns:
            return columns[column_name]

        # Try case-insensitive match
        column_upper = column_name.upper()
        for col_key, col_data in columns.items():
            if col_key.upper() == column_upper:
                return col_data

        return None
