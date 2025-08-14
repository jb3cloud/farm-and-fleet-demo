# Database Data Dictionary Integration

This enhancement adds semantic context to database schema information through structured data dictionaries, providing the LLM with rich business context and query guidance.

## Overview

The database plugin now supports optional data dictionary files that enhance schema information with:
- Business-friendly column descriptions
- Usage examples and constraints
- Special handling for JSON columns with Azure SQL function guidance
- Query patterns and performance tips

## Features

### Enhanced Schema Information
- `get_database_schema()` now includes column descriptions, business meanings, and usage notes
- Table-level descriptions and business purposes
- Analysis patterns for complex data types

### JSON Query Guidance
- New `get_json_query_guidance()` function provides Azure SQL Server-specific JSON query examples
- Supports `OPENJSON`, `JSON_VALUE`, `JSON_QUERY` functions
- Includes performance tips and common patterns
- Specializes in `named_entities` and `mined_opinions` analysis

### Graceful Degradation
- System works normally without data dictionaries
- Dictionary loading failures don't break existing functionality
- Optional enhancement that preserves existing behavior

## Configuration

### Environment Variables
- `SQL_DATA_DICTIONARY_PATH`: Path to directory containing dictionary JSON files (optional)
  - Default: `src/plugins/sqldb/data_dictionaries/`

### File Structure
```
src/plugins/sqldb/data_dictionaries/
├── medallia_feedback.json
└── [additional_table].json
```

## Data Dictionary Format

Each JSON file represents one table and focuses on **business context only** (technical details come from database schema):

```json
{
  "table": {
    "name": "TABLE_NAME",
    "description": "Business description of the table",
    "business_purpose": "Why this table exists",
    "data_source": "Where the data comes from"
  },
  "columns": {
    "COLUMN_NAME": {
      "business_meaning": "Why this column matters to the business",
      "usage_notes": "How to use this column effectively for analysis",
      "azure_sql_functions": ["OPENJSON(...)", "JSON_VALUE(...)"],  // JSON columns only
      "query_examples": ["SELECT ... FROM ..."]  // JSON columns only
    }
  },
  "analysis_patterns": {
    "pattern_name": {
      "description": "What this pattern does",
      "key_columns": ["col1", "col2"],
      "query_approach": "How to implement this pattern"
    }
  }
}
```

## JSON Column Support

Special features for JSON columns (like `named_entities` and `mined_opinions`):

### Azure SQL Functions
- `OPENJSON(column_name)` - Expand JSON arrays to rows
- `JSON_VALUE(column_name, '$.path')` - Extract scalar values
- `JSON_QUERY(column_name, '$.path')` - Extract objects/arrays

### Query Examples
```sql
-- Extract named entity categories
SELECT JSON_VALUE(ne.value, '$.category') as category
FROM table_name 
CROSS APPLY OPENJSON(named_entities) as ne

-- Find friction points
SELECT JSON_VALUE(op.value, '$.friction_category') as friction_category
FROM table_name 
CROSS APPLY OPENJSON(mined_opinions) as op
WHERE JSON_VALUE(op.value, '$.sentiment') = 'negative'
```

## API Usage

### Getting Enhanced Schema
```python
# Get schema with dictionary enhancements
schema = plugin.get_database_schema(detail_level="detailed")
# Now includes descriptions, business meanings, and usage notes
```

### Getting JSON Query Guidance
```python
# General JSON guidance
guidance = plugin.get_json_query_guidance()

# Table-specific guidance
guidance = plugin.get_json_query_guidance(table_name="MEDALLIA_FEEDBACK")

# Column-specific guidance
guidance = plugin.get_json_query_guidance(
    table_name="MEDALLIA_FEEDBACK",
    column_name="named_entities"
)
```

## Implementation Details

### DatabaseClient Enhancements
- `_load_data_dictionaries()`: Loads JSON files on initialization
- `get_table_dictionary(table_name)`: Retrieves table dictionary
- `get_column_description(table_name, column_name)`: Gets column details

### DatabasePlugin Integration
- Schema responses enhanced with dictionary data
- JSON columns get special treatment with query examples
- New kernel function for JSON query guidance

### Performance Considerations
- Dictionaries loaded once at startup
- In-memory caching for fast retrieval
- No database queries required for dictionary data
- Graceful handling of missing dictionaries

## Example: MEDALLIA_FEEDBACK

The included `medallia_feedback.json` provides comprehensive descriptions for:

### Standard Columns
- `TRANSACTION_ID`: Unique transaction identifier with joining guidance
- `TRANSACTION_DATE`: Temporal context for time-series analysis
- `STORE`: Location identifier for geographical analysis
- `TRANSACTION_AMOUNT`: Monetary value for customer segmentation

### JSON Columns
- `named_entities`: AI-extracted entities with category analysis patterns
- `mined_opinions`: Sentiment analysis with friction point identification

### Analysis Patterns
- Friction point analysis for operational improvements
- Entity-based insights for product/service mentions
- Temporal satisfaction trends
- Location performance comparison

## Benefits for LLM Integration

1. **Business Context**: LLM understands the "why" behind data, not just the "what"
2. **No Duplication**: Technical details (types, constraints, nullability) come from database schema
3. **Query Guidance**: Specific examples for complex Azure SQL JSON operations
4. **Analysis Patterns**: Pre-defined business analysis approaches
5. **Focused Value**: Dictionary only contains information that can't be derived from schema

## Design Philosophy

- **Database Schema**: Provides technical details (data types, constraints, relationships)
- **Data Dictionary**: Provides business context (meaning, usage, analysis patterns)
- **Combined Output**: LLM receives both technical accuracy and business understanding

## Maintenance

### Adding New Tables
1. Create new JSON file in data_dictionaries directory
2. Follow the established schema format
3. Include business descriptions and usage examples
4. Restart application to load new dictionary

### Updating Existing Tables
1. Edit the corresponding JSON file
2. Restart application to reload dictionaries
3. Test with `get_database_schema()` to verify changes

The system provides immediate value for the MEDALLIA_FEEDBACK table and can be extended to support additional tables as needed.