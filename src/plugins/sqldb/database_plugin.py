"""
SQL Database Semantic Kernel Plugin for database querying and analysis.

Implements the four core query patterns following the search plugin design with AI-friendly
function signatures and "just enough" context controls for optimal LLM integration.
"""

import json
import logging
from typing import Annotated, Any

from semantic_kernel.functions import kernel_function

from .database_client import DatabaseClient

logger = logging.getLogger(__name__)


class DatabasePlugin:
    """
    SQL Database plugin for database querying and schema analysis.

    Implements AI-friendly functions with strategic optional parameters that influence
    the three "just enough" context control mechanisms:
    - Quantity control (max_rows/max_tables parameters)
    - Content control (detail_level parameter)
    - Relevance control (query safety and performance limits)
    """

    def __init__(self):
        """Initialize the plugin with database client."""
        try:
            self.database_client = DatabaseClient()
            logger.info("DatabasePlugin initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DatabasePlugin: {str(e)}")
            raise

    @kernel_function(
        description="Get available database schema to discover tables, columns, and relationships"
    )
    def get_database_schema(
        self,
        detail_level: Annotated[
            str,
            "Schema detail: 'minimal' for table names only, 'standard' for columns and types, 'detailed' for full schema info",
        ] = "standard",
        max_tables: Annotated[int, "Maximum number of tables to return (1-50)"] = 20,
    ) -> Annotated[
        str,
        "JSON schema describing available tables, columns, and relationships in the database",
    ]:
        """
        Return database schema for table and column discovery.

        Like the search plugin's get_analytics_schema(), this provides the LLM with
        a complete map of what's available in the database for query planning.

        Implements Pattern 1: Database Schema Discovery
        - Uses metadata reflection for comprehensive schema information
        - Applies quantity filtering for manageable results (1.1.1)
        - Controls content fields returned based on detail level (1.1.2)
        - Ensures relevance through table importance ranking (1.1.3)
        """
        try:
            metadata = self.database_client.get_metadata()

            # Get all tables and apply limits
            all_tables = list(metadata.tables.keys())
            limited_tables = all_tables[: min(max_tables, 50)]

            schema_info: dict[str, Any] = {
                "database_summary": {
                    "database": self.database_client.database,
                    "schema": self.database_client.schema,
                    "total_tables": len(all_tables),
                    "returned_tables": len(limited_tables),
                    "detail_level": detail_level,
                },
                "tables": [],
            }

            # Add table filtering warnings if any
            warnings = self.database_client.get_table_warnings()
            if warnings:
                schema_info["warnings"] = warnings

            for full_table_name in limited_tables:
                table_name = full_table_name.split(".")[-1]
                table = metadata.tables[full_table_name]

                table_info: dict[str, Any] = {
                    "name": table_name,
                    "full_name": full_table_name,
                }

                if detail_level in ["standard", "detailed"]:
                    # Add column information with data dictionary enhancements
                    columns = []
                    table_dictionary = self.database_client.get_table_dictionary(
                        table_name
                    )

                    for column in table.columns:
                        col_info = {
                            "name": column.name,
                            "type": str(column.type),
                            "nullable": column.nullable,
                        }

                        if column.primary_key:
                            col_info["primary_key"] = True

                        # Enhance with business context from data dictionary
                        if table_dictionary:
                            column_desc = self.database_client.get_column_description(
                                table_name, column.name
                            )
                            if column_desc:
                                # Add business context (not technical details)
                                if "business_meaning" in column_desc:
                                    col_info["business_meaning"] = column_desc[
                                        "business_meaning"
                                    ]

                                if "usage_notes" in column_desc:
                                    col_info["usage_notes"] = column_desc["usage_notes"]

                                # Add special handling for JSON columns
                                if (
                                    "json" in str(column.type).lower()
                                    or "azure_sql_functions" in column_desc
                                ):
                                    if "azure_sql_functions" in column_desc:
                                        col_info["azure_sql_functions"] = column_desc[
                                            "azure_sql_functions"
                                        ]
                                    if "query_examples" in column_desc:
                                        col_info["query_examples"] = column_desc[
                                            "query_examples"
                                        ]
                                    if "json_structure" in column_desc:
                                        col_info["json_structure"] = column_desc[
                                            "json_structure"
                                        ]

                        columns.append(col_info)

                    table_info["columns"] = columns
                    table_info["column_count"] = len(columns)

                    # Add table-level dictionary information
                    if table_dictionary:
                        table_info["table_description"] = table_dictionary.get(
                            "table", {}
                        ).get("description", "")
                        table_info["business_purpose"] = table_dictionary.get(
                            "table", {}
                        ).get("business_purpose", "")

                        # Add analysis patterns for detailed view
                        if (
                            detail_level == "detailed"
                            and "analysis_patterns" in table_dictionary
                        ):
                            table_info["analysis_patterns"] = table_dictionary[
                                "analysis_patterns"
                            ]

                    # Add primary keys
                    primary_keys = [
                        col.name for col in table.columns if col.primary_key
                    ]
                    if primary_keys:
                        table_info["primary_keys"] = primary_keys

                if detail_level == "detailed":
                    # Add foreign key relationships
                    foreign_keys = []
                    for fk in table.foreign_key_constraints:
                        fk_info = {
                            "columns": [col.parent.name for col in fk.elements],
                            "referenced_table": fk.referred_table.name,
                            "referenced_columns": [
                                col.column.name for col in fk.elements
                            ],
                        }
                        foreign_keys.append(fk_info)

                    if foreign_keys:
                        table_info["foreign_keys"] = foreign_keys

                    # Add indexes
                    indexes = []
                    for index in table.indexes:
                        idx_info = {
                            "name": index.name,
                            "columns": [col.name for col in index.columns],
                            "unique": index.unique,
                        }
                        indexes.append(idx_info)

                    if indexes:
                        table_info["indexes"] = indexes

                schema_info["tables"].append(table_info)

            return json.dumps(schema_info, indent=2)

        except Exception as e:
            logger.error(f"get_database_schema failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Schema discovery failed: {str(e)}",
                    "database_summary": {
                        "database": getattr(
                            self.database_client, "database", "unknown"
                        ),
                        "schema": getattr(self.database_client, "schema", "unknown"),
                    },
                }
            )

    @kernel_function(
        description="Query and analyze data from a specific database table"
    )
    def query_table_data(
        self,
        table_name: Annotated[str, "Name of the table to query"],
        filter_condition: Annotated[
            str, "SQL WHERE condition to filter data (without 'WHERE' keyword)"
        ] = "",
        max_rows: Annotated[int, "Maximum number of rows to return (1-100)"] = 20,
        detail_level: Annotated[
            str,
            "Response detail: 'minimal' for essential data only, 'standard' for typical view",
        ] = "standard",
    ) -> Annotated[str, "JSON formatted table data matching the specified criteria"]:
        """
        Query specific table data with filtering capabilities.

        Implements Pattern 2: Targeted Data Exploration
        - Filters on user-specified conditions with safety validation
        - Uses read-only query validation for security (1.1.3)
        - Row-specific result limiting (1.1.1)
        - Content selection based on detail level (1.1.2)
        """
        try:
            # Validate inputs
            if not table_name.strip():
                return json.dumps(
                    {
                        "error": "Table name cannot be empty",
                        "available_tables": "Use get_database_schema() to see available tables",
                    }
                )

            # Clamp max_rows
            max_rows = max(1, min(max_rows, 100))

            # Build the query
            base_query = f"SELECT TOP {max_rows} * FROM [{self.database_client.schema}].[{table_name.strip()}]"

            if filter_condition.strip():
                # Basic validation of filter condition
                filter_lower = filter_condition.lower().strip()
                unsafe_keywords = [
                    "drop",
                    "delete",
                    "update",
                    "insert",
                    "create",
                    "alter",
                    "exec",
                ]
                if any(keyword in filter_lower for keyword in unsafe_keywords):
                    return json.dumps(
                        {
                            "error": "Filter condition contains unsafe operations",
                            "filter_condition": filter_condition,
                        }
                    )

                base_query += f" WHERE {filter_condition.strip()}"

            # Execute the query
            result = self.database_client.execute_query(base_query, max_rows=max_rows)

            # Format response for LLM consumption
            response = {
                "query_summary": {
                    "table_name": table_name,
                    "filter_condition": filter_condition
                    if filter_condition
                    else "none",
                    "total_returned": result.get("total_count", 0),
                    "detail_level": detail_level,
                    "controls": {
                        "max_rows": max_rows,
                        "query_executed": result.get("query", base_query),
                    },
                },
                "data": result.get("results", []),
            }

            if "error" in result:
                response["error"] = result["error"]
            elif result.get("columns"):
                response["columns"] = result["columns"]

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"query_table_data failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Table query failed: {str(e)}",
                    "table_name": table_name,
                    "filter_condition": filter_condition,
                }
            )

    @kernel_function(
        description="Get statistical summary and metadata about database tables without returning raw data"
    )
    def get_table_summary(
        self,
        table_name: Annotated[str, "Name of the table to analyze"],
        metric_type: Annotated[
            str,
            "What to summarize: 'row_count' for table size, 'column_info' for schema details, 'sample_data' for data preview",
        ],
        detail_level: Annotated[
            str,
            "Analysis depth: 'minimal' for basic stats, 'standard' for comprehensive analysis",
        ] = "standard",
    ) -> Annotated[str, "JSON formatted statistical summary without raw data"]:
        """
        Generate statistical summaries without returning individual records.

        Implements Pattern 3: Database Statistics via Aggregation
        - Uses table metadata and COUNT queries for zero-data analysis (1.1.1)
        - Returns only statistical information (1.1.2)
        - Pre-validates table existence and accessibility (1.1.3)
        """
        try:
            # Validate inputs
            valid_metrics = ["row_count", "column_info", "sample_data"]
            if metric_type not in valid_metrics:
                return json.dumps(
                    {
                        "error": f"Invalid metric_type. Use one of: {valid_metrics}",
                        "provided": metric_type,
                    }
                )

            if not table_name.strip():
                return json.dumps(
                    {
                        "error": "Table name cannot be empty",
                        "available_tables": "Use get_database_schema() to see available tables",
                    }
                )

            # Get table information
            table_info = self.database_client.get_table_info(
                table_name.strip(),
                max_sample_rows=5 if metric_type == "sample_data" else 0,
            )

            if "error" in table_info:
                return json.dumps(
                    {
                        "error": table_info["error"],
                        "table_name": table_name,
                        "metric_type": metric_type,
                    }
                )

            # Format response based on metric type
            response: dict[str, Any] = {
                "summary_info": {
                    "table_name": table_name,
                    "metric_type": metric_type,
                    "detail_level": detail_level,
                    "full_table_name": table_info.get(
                        "full_name", f"{self.database_client.schema}.{table_name}"
                    ),
                },
                "statistics": {},
            }

            if metric_type == "row_count":
                response["statistics"] = {
                    "total_rows": table_info.get("row_count", 0),
                    "table_size_category": "empty"
                    if table_info.get("row_count", 0) == 0
                    else "small"
                    if table_info.get("row_count", 0) < 1000
                    else "medium"
                    if table_info.get("row_count", 0) < 100000
                    else "large",
                }

            elif metric_type == "column_info":
                columns = table_info.get("columns", [])
                column_types: dict[str, int] = {}
                nullable_columns = 0

                # Analyze column types
                for col in columns:
                    col_type = col.get("type", "unknown")
                    type_category = col_type.split("(")[0].upper()  # Get base type
                    column_types[type_category] = column_types.get(type_category, 0) + 1

                    if col.get("nullable", True):
                        nullable_columns += 1

                # Build statistics dictionary
                stats: dict[str, Any] = {
                    "total_columns": len(columns),
                    "column_types": column_types,
                    "nullable_columns": nullable_columns,
                    "primary_key_columns": len(table_info.get("primary_keys", [])),
                    "foreign_key_relationships": len(
                        table_info.get("foreign_keys", [])
                    ),
                    "indexes": len(table_info.get("indexes", [])),
                }

                if detail_level == "standard":
                    column_details = [
                        {
                            "name": col["name"],
                            "type": col["type"],
                            "nullable": col.get("nullable", True),
                            "primary_key": col["name"]
                            in table_info.get("primary_keys", []),
                        }
                        for col in columns[:10]  # Limit to first 10 columns
                    ]
                    stats["column_details"] = column_details

                response["statistics"] = stats

            elif metric_type == "sample_data":
                sample_data = table_info.get("sample_data", [])
                response["statistics"] = {
                    "sample_rows_count": len(sample_data),
                    "columns_with_data": len(sample_data[0].keys())
                    if sample_data
                    else 0,
                    "sample_preview": sample_data[:3]
                    if detail_level == "standard"
                    else [],  # First 3 rows only
                }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"get_table_summary failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Table summary failed: {str(e)}",
                    "table_name": table_name,
                    "metric_type": metric_type,
                }
            )

    @kernel_function(
        description="Execute a custom SQL query with safety validation and result optimization"
    )
    def execute_sql_query(
        self,
        query: Annotated[str, "SQL SELECT query to execute (must be read-only)"],
        max_rows: Annotated[int, "Maximum number of rows to return (1-100)"] = 50,
        performance_mode: Annotated[
            str,
            "Query optimization: 'fast' for quick results, 'accurate' for precise data, 'comprehensive' for detailed output",
        ] = "accurate",
    ) -> Annotated[str, "JSON formatted query results with execution metadata"]:
        """
        Execute custom SQL queries with business-priority optimization.

        Implements Pattern 4: Business-Priority SQL Execution
        - Uses comprehensive safety validation for read-only operations (1.1.3)
        - Combines result limiting with performance monitoring
        - Quality baseline filtering always applied through query validation
        """
        try:
            # Validate inputs
            if not query.strip():
                return json.dumps(
                    {
                        "error": "Query cannot be empty",
                        "example": "SELECT TOP 10 * FROM table_name",
                    }
                )

            # Clamp max_rows
            max_rows = max(1, min(max_rows, 100))

            # Adjust behavior based on performance mode
            if performance_mode == "fast":
                # For fast mode, ensure very low row limits
                max_rows = min(max_rows, 25)
            elif performance_mode == "comprehensive":
                # For comprehensive mode, allow higher limits but add timeout
                max_rows = min(max_rows, 100)

            # Execute the query
            result = self.database_client.execute_query(
                query.strip(), max_rows=max_rows
            )

            # Format response for LLM consumption
            response = {
                "execution_summary": {
                    "query": query.strip(),
                    "performance_mode": performance_mode,
                    "total_returned": result.get("total_count", 0),
                    "max_rows_limit": max_rows,
                    "controls": {
                        "safety_validated": "error" not in result,
                        "query_executed": result.get("query", query.strip()),
                    },
                },
                "results": result.get("results", []),
            }

            if "error" in result:
                response["error"] = result["error"]
            else:
                if result.get("columns"):
                    response["columns"] = result["columns"]

                # Add performance insights
                response["execution_summary"]["result_quality"] = (
                    "complete"
                    if result.get("total_count", 0) < max_rows
                    else "limited"
                    if result.get("total_count", 0) == max_rows
                    else "empty"
                )

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"execute_sql_query failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Query execution failed: {str(e)}",
                    "query": query,
                    "performance_mode": performance_mode,
                }
            )

    @kernel_function(
        description="Get Azure SQL JSON query guidance and examples for working with JSON data types"
    )
    def get_json_query_guidance(
        self,
        table_name: Annotated[str, "Name of the table containing JSON columns"] = "",
        column_name: Annotated[
            str, "Name of the JSON column for specific guidance"
        ] = "",
    ) -> Annotated[
        str, "JSON formatted guidance with Azure SQL JSON functions and query examples"
    ]:
        """
        Provide comprehensive guidance for querying JSON data in Azure SQL Server.

        Implements specialized JSON query pattern guidance:
        - Azure SQL Server compatible JSON functions
        - Query examples for JSON arrays and objects
        - Performance optimization tips
        - Table-specific patterns from data dictionary
        """
        try:
            guidance: dict[str, Any] = {
                "azure_sql_json_functions": {
                    "OPENJSON": {
                        "description": "Parses JSON text and returns objects and properties as rows and columns",
                        "syntax": "OPENJSON(json_text) [WITH (column_definition)]",
                        "use_case": "Converting JSON arrays to table format",
                    },
                    "JSON_VALUE": {
                        "description": "Extracts a scalar value from a JSON string",
                        "syntax": "JSON_VALUE(json_text, '$.path')",
                        "use_case": "Getting single values from JSON",
                    },
                    "JSON_QUERY": {
                        "description": "Extracts an object or array from a JSON string",
                        "syntax": "JSON_QUERY(json_text, '$.path')",
                        "use_case": "Getting objects or arrays from JSON",
                    },
                    "JSON_MODIFY": {
                        "description": "Updates the value of a property in a JSON string (for updates only)",
                        "syntax": "JSON_MODIFY(json_text, '$.path', new_value)",
                        "use_case": "Modifying JSON data (read-only queries don't use this)",
                    },
                },
                "common_patterns": {
                    "extract_array_elements": {
                        "description": "Convert JSON array to rows",
                        "example": "SELECT JSON_VALUE(item.value, '$.text') as text_value FROM table_name CROSS APPLY OPENJSON(json_column) as item",
                    },
                    "filter_by_json_property": {
                        "description": "Filter rows based on JSON property values",
                        "example": "SELECT * FROM table_name WHERE JSON_VALUE(json_column, '$[0].category') = 'Product'",
                    },
                    "aggregate_json_data": {
                        "description": "Count or aggregate JSON array elements",
                        "example": "SELECT COUNT(*) as entity_count FROM table_name CROSS APPLY OPENJSON(named_entities)",
                    },
                },
                "performance_tips": [
                    "Use JSON_VALUE for scalar extraction - it's generally faster than OPENJSON",
                    "Consider computed columns for frequently queried JSON properties",
                    "Use CROSS APPLY OPENJSON to expand arrays efficiently",
                    "JSON path expressions are case-sensitive",
                    "Avoid complex nested path expressions for better performance",
                ],
                "unsupported_functions": [
                    "array_unnest (PostgreSQL function)",
                    "JSON_TABLE (Oracle/MySQL function)",
                    "JSON_EXTRACT (MySQL function)",
                    "jsonb functions (PostgreSQL specific)",
                ],
            }

            # Add table-specific guidance if requested
            table_dict = None
            if table_name:
                table_dict = self.database_client.get_table_dictionary(table_name)
                if table_dict:
                    guidance["table_specific"] = {
                        "table_name": table_name,
                        "description": table_dict.get("table", {}).get(
                            "description", ""
                        ),
                    }

                    # Add column-specific guidance
                    if column_name:
                        column_desc = self.database_client.get_column_description(
                            table_name, column_name
                        )
                        if column_desc:
                            guidance["column_specific"] = {
                                "column_name": column_name,
                                "business_meaning": column_desc.get(
                                    "business_meaning", ""
                                ),
                                "azure_sql_functions": column_desc.get(
                                    "azure_sql_functions", []
                                ),
                                "query_examples": column_desc.get("query_examples", []),
                                "json_structure": column_desc.get("json_structure", {}),
                            }
                    else:
                        # Provide guidance for all JSON columns in the table
                        json_columns = {}
                        columns = table_dict.get("columns", {})
                        for col_name, col_data in columns.items():
                            if "azure_sql_functions" in col_data:
                                json_columns[col_name] = {
                                    "business_meaning": col_data.get(
                                        "business_meaning", ""
                                    ),
                                    "query_examples": col_data.get(
                                        "query_examples", []
                                    ),
                                }

                        if json_columns:
                            guidance["table_json_columns"] = json_columns

                    # Add specialized patterns from data dictionary if available
                    if "specialized_patterns" in table_dict:
                        guidance["specialized_patterns"] = table_dict[
                            "specialized_patterns"
                        ]

            # If no table-specific patterns, provide generic JSON patterns
            if "specialized_patterns" not in guidance:
                guidance["specialized_patterns"] = {
                    "generic_json_array_analysis": {
                        "description": "Generic patterns for analyzing JSON arrays",
                        "examples": [
                            "-- Extract array elements\nSELECT JSON_VALUE(item.value, '$.property') as property_value\nFROM table_name \nCROSS APPLY OPENJSON(json_column) as item",
                            "-- Count by property\nSELECT JSON_VALUE(item.value, '$.property') as property, COUNT(*) as count\nFROM table_name \nCROSS APPLY OPENJSON(json_column) as item\nGROUP BY JSON_VALUE(item.value, '$.property')",
                        ],
                    }
                }

            return json.dumps(guidance, indent=2)

        except Exception as e:
            logger.error(f"get_json_query_guidance failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"JSON guidance failed: {str(e)}",
                    "table_name": table_name,
                    "column_name": column_name,
                }
            )
