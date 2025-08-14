"""
Azure Search Semantic Kernel Plugin for Friction Point Discovery.

Implements the four core query patterns from DESIGN.md with AI-friendly function signatures
and "just enough" context controls for optimal LLM integration.
"""

import json
import logging
from typing import Annotated

from semantic_kernel.functions import kernel_function

from .search_client import AzureSearchClientWrapper

logger = logging.getLogger(__name__)


class FrictionPointSearchPlugin:
    """
    Azure Search plugin for customer feedback analysis and friction point discovery.

    Implements AI-friendly functions with strategic optional parameters that influence
    the three "just enough" context control mechanisms:
    - 1.1.1 Controlling Quantity (max_results parameter)
    - 1.1.2 Controlling Content (detail_level parameter)
    - 1.1.3 Ensuring Relevance (quality_level parameter)
    """

    def __init__(self):
        """Initialize the plugin with Azure Search client."""
        try:
            self.search_client = AzureSearchClientWrapper()
            logger.info("FrictionPointSearchPlugin initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FrictionPointSearchPlugin: {str(e)}")
            raise

    @kernel_function(
        description="Get available analytics schema to discover valid data sources, metrics, and parameter combinations"
    )
    def get_analytics_schema(
        self,
    ) -> Annotated[
        str,
        "JSON schema describing available data sources, metrics, quality levels, and valid parameter combinations",
    ]:
        """
        Return analytics schema for parameter validation and discovery.

        This prevents invalid parameter combinations by providing the LLM with
        a complete map of what's available for each data source.

        Like SQL's INFORMATION_SCHEMA or GraphQL introspection, this allows
        the LLM to validate parameters before making analytics requests.
        """
        try:
            schema = {
                "data_sources": {
                    "social": {
                        "description": "Social media content and posts",
                        "available_metrics": {
                            "quality_scores": {
                                "description": "Distribution of content quality ratings",
                                "facet_field": "qualityScore",
                                "example_values": ["high", "medium", "low", "noise"],
                            },
                            "friction_categories": {
                                "description": "Top customer pain points and friction areas",
                                "facet_field": "frictionCategories",
                                "example_values": [
                                    "Store Operations",
                                    "Product Availability",
                                    "Customer Service",
                                    "Digital Experience",
                                    "Pricing",
                                ],
                            },
                            "sentiment_distribution": {
                                "description": "Sentiment breakdown across content",
                                "facet_field": "sentiment",
                                "example_values": ["positive", "negative", "neutral"],
                            },
                            "source_distribution": {
                                "description": "Content distribution by social network",
                                "facet_field": "network",
                                "example_values": [
                                    "twitter",
                                    "facebook",
                                    "instagram",
                                    "reddit",
                                ],
                            },
                        },
                        "quality_levels": {
                            "high_only": "Only highest quality, most relevant content",
                            "high_and_medium": "High and medium quality content (recommended default)",
                            "all_quality": "All content including low quality and noise",
                        },
                        "detail_levels": ["minimal", "standard", "detailed"],
                    },
                    "surveys": {
                        "description": "Customer survey responses and feedback",
                        "available_metrics": {
                            "location_distribution": {
                                "description": "Survey responses by store location",
                                "facet_field": "store_location",
                                "example_values": ["Store locations from survey data"],
                            },
                            "source_distribution": {
                                "description": "Responses by survey title/type",
                                "facet_field": "survey_title",
                                "example_values": ["Various survey titles"],
                            },
                        },
                        "quality_levels": {
                            "all_quality": "All survey responses (quality filtering not available for surveys)"
                        },
                        "detail_levels": ["minimal", "standard", "detailed"],
                        "notes": "Surveys do not support quality_scores, friction_categories, or sentiment_distribution metrics",
                    },
                },
                "valid_combinations": [
                    {
                        "source": "social",
                        "metric_type": "quality_scores",
                        "quality_levels": [
                            "high_only",
                            "high_and_medium",
                            "all_quality",
                        ],
                        "description": "Quality score distribution for social media content",
                    },
                    {
                        "source": "social",
                        "metric_type": "friction_categories",
                        "quality_levels": [
                            "high_only",
                            "high_and_medium",
                            "all_quality",
                        ],
                        "description": "Top friction points identified in social media",
                    },
                    {
                        "source": "social",
                        "metric_type": "sentiment_distribution",
                        "quality_levels": [
                            "high_only",
                            "high_and_medium",
                            "all_quality",
                        ],
                        "description": "Sentiment analysis breakdown for social content",
                    },
                    {
                        "source": "social",
                        "metric_type": "source_distribution",
                        "quality_levels": [
                            "high_only",
                            "high_and_medium",
                            "all_quality",
                        ],
                        "description": "Content distribution across social networks",
                    },
                    {
                        "source": "surveys",
                        "metric_type": "location_distribution",
                        "quality_levels": ["all_quality"],
                        "description": "Survey responses grouped by store location",
                    },
                    {
                        "source": "surveys",
                        "metric_type": "source_distribution",
                        "quality_levels": ["all_quality"],
                        "description": "Survey responses grouped by survey type/title",
                    },
                ],
                "invalid_combinations": [
                    {
                        "source": "surveys",
                        "metric_type": "friction_categories",
                        "reason": "Surveys do not have friction category classification",
                    },
                    {
                        "source": "surveys",
                        "metric_type": "quality_scores",
                        "reason": "Surveys do not have quality score fields",
                    },
                    {
                        "source": "surveys",
                        "metric_type": "sentiment_distribution",
                        "reason": "Surveys do not have sentiment analysis fields",
                    },
                    {
                        "source": "surveys",
                        "quality_level": "high_only",
                        "reason": "Quality filtering only applies to social media data",
                    },
                    {
                        "source": "surveys",
                        "quality_level": "high_and_medium",
                        "reason": "Quality filtering only applies to social media data",
                    },
                ],
                "usage_examples": [
                    {
                        "description": "Get friction categories from social media",
                        "parameters": {
                            "metric_type": "friction_categories",
                            "source": "social",
                            "quality_level": "high_and_medium",
                        },
                    },
                    {
                        "description": "Get survey responses by location",
                        "parameters": {
                            "metric_type": "location_distribution",
                            "source": "surveys",
                            "quality_level": "all_quality",
                        },
                    },
                ],
                "schema_version": "1.0.0",
                "last_updated": "2024-12-19",
            }

            return json.dumps(schema, indent=2)

        except Exception as e:
            logger.error(f"get_analytics_schema failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Schema discovery failed: {str(e)}",
                    "schema_version": "1.0.0",
                }
            )

    @kernel_function(
        description="Search customer feedback with semantic understanding across social media and survey data"
    )
    def search_customer_feedback(
        self,
        query: Annotated[
            str, "What to search for in customer feedback (natural language)"
        ],
        source: Annotated[
            str,
            "Data source to search: 'social' for social media or 'surveys' for survey data",
        ],
        max_results: Annotated[int, "Maximum number of results to return (1-50)"] = 10,
        quality_level: Annotated[
            str,
            "Quality filter: 'high_only' for strictest relevance, 'high_and_medium' for balanced results, 'all_quality' for maximum recall",
        ] = "high_and_medium",
        detail_level: Annotated[
            str,
            "Response detail: 'minimal' for core info only, 'standard' for typical fields, 'detailed' for comprehensive data",
        ] = "standard",
    ) -> Annotated[
        str, "JSON formatted search results with customer feedback matching the query"
    ]:
        """
        Execute general-purpose semantic search across customer feedback.

        Implements Pattern 1 from DESIGN.md: General-Purpose Semantic Search
        - Combines keyword and vector search with semantic ranking
        - Applies quality filtering for relevance (1.1.3)
        - Controls content fields returned (1.1.2)
        - Limits result quantity (1.1.1)
        """
        try:
            # Validate inputs
            if source not in ["social", "surveys"]:
                return json.dumps(
                    {
                        "error": "Invalid source. Use 'social' or 'surveys'",
                        "valid_sources": ["social", "surveys"],
                    }
                )

            if not query.strip():
                return json.dumps(
                    {
                        "error": "Query cannot be empty",
                        "example": "tire installation problems",
                    }
                )

            # Clamp max_results
            max_results = max(1, min(max_results, 50))

            # Execute semantic search
            results = self.search_client.semantic_search(
                query=query,
                source=source,
                max_results=max_results,
                quality_level=quality_level,
                detail_level=detail_level,
            )

            # Format response for LLM consumption
            response = {
                "search_summary": {
                    "query": query,
                    "source": source,
                    "total_found": results["total_count"],
                    "returned": len(results["results"]),
                    "controls": results.get("controls_applied", {}),
                },
                "feedback": results["results"],
            }

            if "error" in results:
                response["error"] = results["error"]

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"search_customer_feedback failed: {str(e)}")
            return json.dumps(
                {"error": f"Search failed: {str(e)}", "query": query, "source": source}
            )

    @kernel_function(
        description="Find specific customer pain points and friction categories in feedback data"
    )
    def find_friction_points(
        self,
        category: Annotated[
            str,
            "Friction category to find (e.g. 'Product Availability', 'Customer Service', 'Store Operations', 'Checkout Process')",
        ],
        source: Annotated[
            str,
            "Data source to search: 'social' for social media or 'surveys' for survey data",
        ],
        max_results: Annotated[int, "Maximum number of examples to return (1-50)"] = 15,
        quality_level: Annotated[
            str,
            "Quality filter: 'high_only' for most reliable examples, 'high_and_medium' for broader coverage, 'all_quality' for all matches",
        ] = "high_only",
        detail_level: Annotated[
            str,
            "Response detail: 'minimal' for core info only, 'standard' for typical fields, 'detailed' for comprehensive data",
        ] = "standard",
    ) -> Annotated[
        str,
        "JSON formatted examples of the specified friction point with customer feedback",
    ]:
        """
        Find feedback for specific friction categories with aggressive filtering.

        Implements Pattern 2 from DESIGN.md: Targeted Friction Point Discovery
        - Filters on predefined friction categories
        - Uses quality-relevance scoring profile
        - High precision with quality filtering (1.1.3)
        - Friction-specific field selection (1.1.2)
        """
        try:
            # Validate inputs
            if source not in ["social", "surveys"]:
                return json.dumps(
                    {
                        "error": "Invalid source. Use 'social' or 'surveys'",
                        "valid_sources": ["social", "surveys"],
                    }
                )

            if not category.strip():
                return json.dumps(
                    {
                        "error": "Category cannot be empty",
                        "examples": [
                            "Product Availability",
                            "Customer Service",
                            "Store Operations",
                            "Checkout Process",
                        ],
                    }
                )

            # Clamp max_results
            max_results = max(1, min(max_results, 50))

            # Execute friction point search
            results = self.search_client.friction_search(
                category=category,
                source=source,
                max_results=max_results,
                quality_level=quality_level,
                detail_level=detail_level,
            )

            # Format response for LLM consumption
            response = {
                "friction_summary": {
                    "category": category,
                    "source": source,
                    "total_found": results["total_count"],
                    "returned": len(results["results"]),
                    "controls": results.get("controls_applied", {}),
                },
                "examples": results["results"],
            }

            if "error" in results:
                response["error"] = results["error"]

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"find_friction_points failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Friction search failed: {str(e)}",
                    "category": category,
                    "source": source,
                }
            )

    @kernel_function(
        description="Get high-level statistics and metrics about customer feedback without returning individual documents"
    )
    def get_feedback_summary(
        self,
        metric_type: Annotated[
            str,
            "What to summarize: 'quality_scores' for data quality distribution, 'friction_categories' for top pain points, 'sentiment_distribution' for sentiment breakdown, 'location_distribution' for geographic patterns",
        ],
        source: Annotated[
            str,
            "Data source to analyze: 'social' for social media or 'surveys' for survey data",
        ],
        quality_level: Annotated[
            str,
            "Quality filter for aggregation: 'high_only', 'high_and_medium', or 'all_quality'",
        ] = "high_and_medium",
    ) -> Annotated[
        str, "JSON formatted statistical summary with counts and distributions"
    ]:
        """
        Generate statistical summaries without returning individual documents.

        Implements Pattern 3 from DESIGN.md: Aggregate Analysis via Faceting
        - Uses faceting with top=0 for zero documents (1.1.1)
        - Returns only aggregate statistics (1.1.2)
        - Pre-filters by quality before aggregation (1.1.3)
        """
        try:
            # Validate inputs
            if source not in ["social", "surveys"]:
                return json.dumps(
                    {
                        "error": "Invalid source. Use 'social' or 'surveys'",
                        "valid_sources": ["social", "surveys"],
                    }
                )

            valid_metrics = [
                "quality_scores",
                "friction_categories",
                "sentiment_distribution",
                "location_distribution",
            ]
            if metric_type not in valid_metrics:
                return json.dumps(
                    {
                        "error": f"Invalid metric_type. Use one of: {valid_metrics}",
                        "provided": metric_type,
                    }
                )

            # Execute aggregation search
            results = self.search_client.aggregate_search(
                metric_type=metric_type, source=source, quality_level=quality_level
            )

            # Format response for LLM consumption
            response = {
                "summary_info": {
                    "metric_type": metric_type,
                    "source": source,
                    "total_documents": results["total_count"],
                    "controls": results.get("controls_applied", {}),
                },
                "statistics": results["aggregation"],
            }

            if "error" in results:
                response["error"] = results["error"]

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"get_feedback_summary failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Summary failed: {str(e)}",
                    "metric_type": metric_type,
                    "source": source,
                }
            )

    @kernel_function(
        description="Find most important customer feedback based on business priority ranking"
    )
    def search_priority_feedback(
        self,
        query: Annotated[str, "What to search for in priority-ranked feedback"],
        priority_type: Annotated[
            str,
            "Priority ranking method: 'quality' for highest quality feedback, 'engagement' for most viral social posts, 'recent' for newest feedback",
        ],
        source: Annotated[
            str,
            "Data source to search: 'social' for social media or 'surveys' for survey data",
        ],
        max_results: Annotated[int, "Maximum number of results to return (1-50)"] = 10,
        detail_level: Annotated[
            str,
            "Response detail: 'minimal' for core info only, 'standard' for typical fields, 'detailed' for comprehensive data",
        ] = "standard",
    ) -> Annotated[
        str,
        "JSON formatted high-priority feedback results ranked by business importance",
    ]:
        """
        Search with business-priority ranking using scoring profiles.

        Implements Pattern 4 from DESIGN.md: Performance & Relevance-Tuned Search
        - Uses custom scoring profiles for business relevance (1.1.3)
        - Combines scoring with semantic ranking
        - Quality baseline filtering always applied
        """
        try:
            # Validate inputs
            if source not in ["social", "surveys"]:
                return json.dumps(
                    {
                        "error": "Invalid source. Use 'social' or 'surveys'",
                        "valid_sources": ["social", "surveys"],
                    }
                )

            valid_priorities = ["quality", "engagement", "recent"]
            if priority_type not in valid_priorities:
                return json.dumps(
                    {
                        "error": f"Invalid priority_type. Use one of: {valid_priorities}",
                        "provided": priority_type,
                    }
                )

            if not query.strip():
                return json.dumps(
                    {"error": "Query cannot be empty", "example": "checkout problems"}
                )

            # Clamp max_results
            max_results = max(1, min(max_results, 50))

            # Execute priority search
            results = self.search_client.priority_search(
                query=query,
                priority_type=priority_type,
                source=source,
                max_results=max_results,
                detail_level=detail_level,
            )

            # Format response for LLM consumption
            response = {
                "priority_summary": {
                    "query": query,
                    "priority_type": priority_type,
                    "source": source,
                    "total_found": results["total_count"],
                    "returned": len(results["results"]),
                    "controls": results.get("controls_applied", {}),
                },
                "priority_feedback": results["results"],
            }

            if "error" in results:
                response["error"] = results["error"]

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"search_priority_feedback failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Priority search failed: {str(e)}",
                    "query": query,
                    "priority_type": priority_type,
                    "source": source,
                }
            )

    @kernel_function(
        description="Compare and analyze insights across both social media and survey data sources"
    )
    def analyze_cross_sources(
        self,
        query: Annotated[
            str, "What to analyze across both social media and survey data"
        ],
        analysis_type: Annotated[
            str,
            "Analysis focus: 'friction' for friction point comparison, 'sentiment' for sentiment analysis, 'general' for overall comparison",
        ],
        quality_level: Annotated[
            str,
            "Quality filter for both sources: 'high_only', 'high_and_medium', or 'all_quality'",
        ] = "high_and_medium",
    ) -> Annotated[
        str,
        "JSON formatted comparative analysis across social media and survey data sources",
    ]:
        """
        Query both data sources simultaneously for comprehensive analysis.

        Implements Multi-Source Pattern from DESIGN.md: Unified Analytics Layer
        - Parallel querying with consistent filtering
        - Source-specific relevance tuning
        - Unified result formatting
        """
        try:
            valid_analyses = ["friction", "sentiment", "general"]
            if analysis_type not in valid_analyses:
                return json.dumps(
                    {
                        "error": f"Invalid analysis_type. Use one of: {valid_analyses}",
                        "provided": analysis_type,
                    }
                )

            if not query.strip():
                return json.dumps(
                    {
                        "error": "Query cannot be empty",
                        "example": "product availability issues",
                    }
                )

            # Execute searches in parallel on both sources
            social_results = None
            survey_results = None

            try:
                # Search social media
                social_results = self.search_client.semantic_search(
                    query=query,
                    source="social",
                    max_results=5,  # Limit per source
                    quality_level=quality_level,
                    detail_level="standard",
                )
            except Exception as e:
                logger.warning(f"Social media search failed: {str(e)}")
                social_results = {"results": [], "total_count": 0, "error": str(e)}

            try:
                # Search surveys
                survey_results = self.search_client.semantic_search(
                    query=query,
                    source="surveys",
                    max_results=5,  # Limit per source
                    quality_level=quality_level,
                    detail_level="standard",
                )
            except Exception as e:
                logger.warning(f"Survey search failed: {str(e)}")
                survey_results = {"results": [], "total_count": 0, "error": str(e)}

            # Format cross-source response
            response = {
                "cross_source_summary": {
                    "query": query,
                    "analysis_type": analysis_type,
                    "quality_level": quality_level,
                    "social_found": social_results["total_count"]
                    if social_results
                    else 0,
                    "survey_found": survey_results["total_count"]
                    if survey_results
                    else 0,
                },
                "social_media_insights": {
                    "results": social_results["results"] if social_results else [],
                    "error": social_results.get("error") if social_results else None,
                },
                "survey_insights": {
                    "results": survey_results["results"] if survey_results else [],
                    "error": survey_results.get("error") if survey_results else None,
                },
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"analyze_cross_sources failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Cross-source analysis failed: {str(e)}",
                    "query": query,
                    "analysis_type": analysis_type,
                }
            )
