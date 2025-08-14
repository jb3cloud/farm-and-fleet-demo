# FrictionPointSearchPlugin

Azure Search Semantic Kernel Plugin for customer feedback analysis and friction point discovery.

## Overview

This plugin implements the four core query patterns from DESIGN.md with AI-friendly function signatures and "just enough" context controls for optimal LLM integration. It enables semantic search across customer feedback from social media and survey data sources.

## Features

- **AI-Friendly Design**: Clear function names, minimal parameters, primitive types
- **"Just Enough" Context Controls**: 
  - Quantity control (max_results parameter)
  - Content control (detail_level parameter)  
  - Relevance control (quality_level parameter)
- **Multi-Source Support**: Search across social media and survey data
- **Advanced Search Patterns**: Semantic search, friction point discovery, aggregation, priority ranking

## Functions

### 1. `search_customer_feedback`
General-purpose semantic search across customer feedback.

**Parameters:**
- `query` (str): Natural language search query
- `source` (str): "social" or "surveys" 
- `max_results` (int): Maximum results (1-50, default: 10)
- `quality_level` (str): "high_only", "high_and_medium", "all_quality" (default: "high_and_medium")
- `detail_level` (str): "minimal", "standard", "detailed" (default: "standard")

### 2. `find_friction_points`
Find specific customer pain points and friction categories.

**Parameters:**
- `category` (str): Friction category (e.g., "Product Availability", "Customer Service")
- `source` (str): "social" or "surveys"
- `max_results` (int): Maximum examples (1-50, default: 15)
- `quality_level` (str): Quality filter (default: "high_only")
- `detail_level` (str): Response detail (default: "standard")

### 3. `get_feedback_summary`
High-level statistics without returning individual documents.

**Parameters:**
- `metric_type` (str): "quality_scores", "friction_categories", "sentiment_distribution", "location_distribution"
- `source` (str): "social" or "surveys"
- `quality_level` (str): Quality filter (default: "high_and_medium")

### 4. `search_priority_feedback`
Find most important feedback based on business priority.

**Parameters:**
- `query` (str): Search query
- `priority_type` (str): "quality", "engagement", "recent"
- `source` (str): "social" or "surveys"
- `max_results` (int): Maximum results (1-50, default: 10)
- `detail_level` (str): Response detail (default: "standard")

### 5. `analyze_cross_sources`
Compare insights across both social media and survey data.

**Parameters:**
- `query` (str): What to analyze across sources
- `analysis_type` (str): "friction", "sentiment", "general"
- `quality_level` (str): Quality filter (default: "high_and_medium")

## Environment Variables

Required Azure Search credentials in `.env`:

```
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-api-key
AZURE_SEARCH_INDEX_NAME=social-media-index
AZURE_SEARCH_SURVEY_INDEX_NAME=survey-index
```

## Usage Examples

### LLM Natural Language Queries

The plugin is designed for natural language interaction through the LLM:

- "Search for customer feedback about tire installation problems in social media"
- "Find friction points related to Product Availability in surveys"
- "Get a summary of quality scores from social media data"
- "Show me high-priority feedback about checkout issues from surveys"
- "Compare sentiment about store hours across social media and surveys"

### Direct Function Calls

```python
# Search customer feedback
result = plugin.search_customer_feedback(
    query="tire installation service",
    source="social",
    max_results=10,
    quality_level="high_and_medium",
    detail_level="standard"
)

# Find friction points
result = plugin.find_friction_points(
    category="Product Availability",
    source="surveys", 
    max_results=15,
    quality_level="high_only"
)
```

## Response Format

All functions return JSON-formatted strings optimized for LLM consumption:

```json
{
  "search_summary": {
    "query": "tire installation problems",
    "source": "social",
    "total_found": 127,
    "returned": 10,
    "controls": {
      "quality_level": "high_and_medium",
      "detail_level": "standard", 
      "max_results": 10
    }
  },
  "feedback": [
    {
      "score": 0.89,
      "message": "Had trouble with tire installation...",
      "sentiment": "negative",
      "location": "Madison",
      "friction_categories": ["Service Quality"]
    }
  ]
}
```

## Implementation Details

### "Just Enough" Context Controls

1. **Quantity Control (1.1.1)**: Limits results via `max_results` parameter and internal top limits
2. **Content Control (1.1.2)**: Selective field inclusion based on `detail_level`, excludes vectors  
3. **Relevance Control (1.1.3)**: Quality filtering, scoring profiles, semantic ranking

### Search Patterns

- **Pattern 1**: General semantic search with hybrid keyword+vector queries
- **Pattern 2**: Targeted friction point discovery with category filtering
- **Pattern 3**: Zero-document aggregation using faceted search
- **Pattern 4**: Business-priority ranking with custom scoring profiles

### Error Handling

Robust error handling with graceful degradation:
- Invalid parameters return helpful error messages
- Missing environment variables are detected at initialization
- Search failures return structured error responses
- Cross-source searches continue even if one source fails

## Testing

The plugin has been tested for:
- ✅ Proper imports and initialization
- ✅ Function signature validation
- ✅ Type annotations
- ✅ Linting compliance  
- ✅ Environment variable validation
- ✅ Graceful error handling

## Integration

The plugin is automatically registered in `src/app.py` as "CustomerInsights" and available for LLM function calling through Semantic Kernel's auto function choice behavior.