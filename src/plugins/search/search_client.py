"""
Azure Search client wrapper implementing "just enough" context controls.
Internal module for FrictionPointSearchPlugin.
"""

import logging
import os
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

logger = logging.getLogger(__name__)


class AzureSearchClientWrapper:
    """Internal Azure Search client with 'just enough' context controls."""

    def __init__(self):
        """Initialize Azure Search client from environment variables."""
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.social_index = os.getenv("AZURE_SEARCH_INDEX_NAME", "social-media-index")
        self.survey_index = os.getenv("AZURE_SEARCH_SURVEY_INDEX_NAME", "survey-index")

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Missing required Azure Search environment variables: AZURE_SEARCH_ENDPOINT (or AZURE_SEARCH_SERVICE_ENDPOINT), AZURE_SEARCH_API_KEY"
            )

        self.credential = AzureKeyCredential(self.api_key)

    def _get_search_client(self, source: str) -> SearchClient:
        """Get SearchClient for specified data source."""
        if not self.endpoint:
            raise ValueError("Azure Search endpoint is not configured")
        index_name = self.social_index if source == "social" else self.survey_index
        return SearchClient(
            endpoint=self.endpoint, index_name=index_name, credential=self.credential
        )

    def _get_quality_filter(
        self, quality_level: str, source: str = "social"
    ) -> str | None:
        """Get quality score filter based on level and source.

        Implements Relevance Control (1.1.3) - Ensuring "The Best Ones"
        Note: Only social media has qualityScore field, surveys don't
        """
        # Only social media has quality scores
        if source != "social":
            return None

        filters = {
            "high_only": "qualityScore eq 'high'",
            "high_and_medium": "(qualityScore eq 'high' or qualityScore eq 'medium')",
            "all_quality": None,
        }
        return filters.get(quality_level, filters["high_and_medium"])

    def _get_field_selection(
        self, detail_level: str, context: str = "general", source: str = "social"
    ) -> list[str]:
        """Get field selection based on detail level and source.

        Implements Content Control (1.1.2) - Controlling "What's Inside"
        Excludes high-volume, low-value fields like vectors
        """
        base_fields = []  # Don't include @search.score in select, it's automatically available

        if source == "social":
            field_sets = {
                "minimal": base_fields + ["message", "sentiment"],
                "standard": base_fields
                + ["message", "sentiment", "network", "frictionCategories"],
                "detailed": base_fields
                + [
                    "message",
                    "messageId",
                    "sentiment",
                    "network",
                    "frictionCategories",
                    "date",
                    "qualityScore",
                ],
            }
        else:  # surveys
            field_sets = {
                "minimal": base_fields + ["searchable_content", "response"],
                "standard": base_fields
                + ["question", "response", "searchable_content", "store_location"],
                "detailed": base_fields
                + [
                    "survey_title",
                    "question",
                    "response",
                    "searchable_content",
                    "store_location",
                    "customer_id",
                    "responded_at",
                ],
            }

        # Context-specific field adjustments
        if context == "friction" and source == "social":
            for level in field_sets:
                if "frictionCategories" not in field_sets[level]:
                    field_sets[level].append("frictionCategories")

        return field_sets.get(detail_level, field_sets["standard"])

    def _combine_filters(self, *filters: str | None) -> str | None:
        """Combine multiple OData filters with AND logic."""
        active_filters = [f for f in filters if f]
        if not active_filters:
            return None
        if len(active_filters) == 1:
            return active_filters[0]
        return " and ".join(f"({f})" for f in active_filters)

    def _get_semantic_config(self, source: str) -> str:
        """Get semantic configuration name for data source."""
        return "social-media-semantic" if source == "social" else "survey-semantic"

    def _get_scoring_profile(self, priority_type: str, source: str) -> str | None:
        """Get scoring profile based on priority type and source.

        Implements Relevance Control (1.1.3) - Performance & Relevance-Tuned Search
        Returns None if no specific scoring profile is needed (default relevance)
        """
        # Based on testing, the indexes don't have custom scoring profiles
        # Return None to use default Azure Search relevance scoring
        return None

    def _format_search_results(
        self, results: Any, detail_level: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Format search results with content truncation.

        Implements Quantity Control (1.1.1) and Content Control (1.1.2)
        """
        formatted_results = []
        count = 0

        for result in results:
            if count >= max_results:
                break

            # Extract core fields based on source
            if "message" in result:  # Social media
                formatted_result = {
                    "score": result.get("@search.score", 0),
                    "message": result.get("message", ""),
                    "sentiment": result.get("sentiment", "unknown"),
                }

                # Add fields based on detail level for social
                if detail_level in ["standard", "detailed"]:
                    formatted_result.update(
                        {
                            "network": result.get("network", ""),
                            "friction_categories": result.get("frictionCategories", []),
                        }
                    )

                if detail_level == "detailed":
                    formatted_result.update(
                        {
                            "message_id": result.get("messageId", ""),
                            "date": result.get("date", ""),
                            "quality_score": result.get("qualityScore", ""),
                        }
                    )

                # Content truncation for token optimization
                if detail_level == "minimal" and len(formatted_result["message"]) > 100:
                    formatted_result["message"] = (
                        formatted_result["message"][:97] + "..."
                    )
                elif (
                    detail_level == "standard"
                    and len(formatted_result["message"]) > 200
                ):
                    formatted_result["message"] = (
                        formatted_result["message"][:197] + "..."
                    )

            else:  # Survey data
                # Use searchable_content as primary content field
                content = result.get("searchable_content", result.get("response", ""))
                formatted_result = {
                    "score": result.get("@search.score", 0),
                    "content": content,
                    "question": result.get("question", ""),
                    "response": result.get("response", ""),
                }

                # Add fields based on detail level for surveys
                if detail_level in ["standard", "detailed"]:
                    formatted_result.update(
                        {
                            "store_location": result.get("store_location", ""),
                        }
                    )

                if detail_level == "detailed":
                    formatted_result.update(
                        {
                            "survey_title": result.get("survey_title", ""),
                            "customer_id": result.get("customer_id", ""),
                            "responded_at": result.get("responded_at", ""),
                        }
                    )

                # Content truncation for token optimization
                if detail_level == "minimal" and len(content) > 100:
                    formatted_result["content"] = content[:97] + "..."
                elif detail_level == "standard" and len(content) > 200:
                    formatted_result["content"] = content[:197] + "..."

            formatted_results.append(formatted_result)
            count += 1

        return formatted_results

    def semantic_search(
        self,
        query: str,
        source: str,
        max_results: int = 10,
        quality_level: str = "high_and_medium",
        detail_level: str = "standard",
        additional_filters: str | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search with full 'just enough' controls."""
        try:
            client = self._get_search_client(source)

            # Apply "just enough" controls
            quality_filter = self._get_quality_filter(quality_level, source)
            combined_filter = self._combine_filters(quality_filter, additional_filters)
            select_fields = self._get_field_selection(detail_level, "general", source)

            # Execute search with semantic ranking
            results = client.search(
                search_text=query,
                filter=combined_filter,
                select=select_fields,
                top=min(max_results, 50),  # Cap at 50 for performance
                query_type="semantic",
                semantic_configuration_name=self._get_semantic_config(source),
                query_caption="extractive",
                include_total_count=True,
            )

            formatted_results = self._format_search_results(
                results, detail_level, max_results
            )

            return {
                "results": formatted_results,
                "total_count": getattr(
                    results, "get_count", lambda: len(formatted_results)
                )(),
                "search_query": query,
                "source": source,
                "controls_applied": {
                    "quality_level": quality_level,
                    "detail_level": detail_level,
                    "max_results": max_results,
                },
            }

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return {
                "results": [],
                "total_count": 0,
                "error": f"Search failed: {str(e)}",
                "search_query": query,
                "source": source,
            }

    def friction_search(
        self,
        category: str,
        source: str,
        max_results: int = 15,
        quality_level: str = "high_only",
        detail_level: str = "standard",
        additional_filters: str | None = None,
    ) -> dict[str, Any]:
        """Search for specific friction points with aggressive filtering."""
        try:
            client = self._get_search_client(source)

            # Build friction-specific filter
            friction_filter = f"frictionCategories/any(cat: cat eq '{category}')"
            quality_filter = self._get_quality_filter(quality_level, source)
            combined_filter = self._combine_filters(
                friction_filter, quality_filter, additional_filters
            )

            select_fields = self._get_field_selection(detail_level, "friction", source)

            # Execute search
            results = client.search(
                search_text="*",  # Search all documents
                filter=combined_filter,
                select=select_fields,
                top=min(max_results, 50),
                include_total_count=True,
            )

            formatted_results = self._format_search_results(
                results, detail_level, max_results
            )

            return {
                "results": formatted_results,
                "total_count": getattr(
                    results, "get_count", lambda: len(formatted_results)
                )(),
                "friction_category": category,
                "source": source,
                "controls_applied": {
                    "quality_level": quality_level,
                    "detail_level": detail_level,
                    "max_results": max_results,
                },
            }

        except Exception as e:
            logger.error(f"Friction search failed: {str(e)}")
            return {
                "results": [],
                "total_count": 0,
                "error": f"Friction search failed: {str(e)}",
                "friction_category": category,
                "source": source,
            }

    def aggregate_search(
        self, metric_type: str, source: str, quality_level: str = "high_and_medium"
    ) -> dict[str, Any]:
        """Execute faceted search for aggregation without returning documents.

        Implements Quantity Control (1.1.1) - top=0 for zero documents
        """
        try:
            client = self._get_search_client(source)

            quality_filter = self._get_quality_filter(quality_level, source)

            # Map metric types to facet fields based on actual schemas
            if source == "social":
                facet_mapping = {
                    "quality_scores": ["qualityScore"],
                    "friction_categories": ["frictionCategories,count:20"],
                    "sentiment_distribution": ["sentiment"],
                    "source_distribution": ["network,count:15"],
                }
            else:  # surveys
                facet_mapping = {
                    "quality_scores": [],  # No quality scores in surveys
                    "friction_categories": [],  # No friction categories in surveys
                    "sentiment_distribution": [],  # No sentiment in surveys
                    "location_distribution": ["store_location,count:15"],
                    "source_distribution": ["survey_title,count:15"],
                }

            facets = facet_mapping.get(metric_type, ["qualityScore"])

            # Execute faceted search with no document results
            results = client.search(
                search_text="*",
                filter=quality_filter,
                top=0,  # No documents, only facets
                facets=facets,
                include_total_count=True,
            )

            # Format facet results
            facet_results = {}
            if hasattr(results, "get_facets"):
                facets = results.get_facets()
                if facets:
                    for facet_name, facet_values in facets.items():
                        if facet_values:
                            facet_results[facet_name] = [
                                {"value": fv["value"], "count": fv["count"]}
                                for fv in facet_values
                            ]

            return {
                "aggregation": facet_results,
                "total_count": getattr(results, "get_count", lambda: 0)(),
                "metric_type": metric_type,
                "source": source,
                "controls_applied": {"quality_level": quality_level},
            }

        except Exception as e:
            logger.error(f"Aggregate search failed: {str(e)}")
            return {
                "aggregation": {},
                "total_count": 0,
                "error": f"Aggregation failed: {str(e)}",
                "metric_type": metric_type,
                "source": source,
            }

    def priority_search(
        self,
        query: str,
        priority_type: str,
        source: str,
        max_results: int = 10,
        detail_level: str = "standard",
    ) -> dict[str, Any]:
        """Execute priority-ranked search using scoring profiles."""
        try:
            client = self._get_search_client(source)

            # Always apply quality filtering for priority search
            quality_filter = self._get_quality_filter("high_and_medium", source)
            select_fields = self._get_field_selection(detail_level, "priority", source)
            scoring_profile = self._get_scoring_profile(priority_type, source)

            # Execute search with priority ranking
            search_kwargs = {
                "search_text": query,
                "filter": quality_filter,
                "select": select_fields,
                "top": min(max_results, 50),
                "query_type": "semantic",
                "semantic_configuration_name": self._get_semantic_config(source),
                "include_total_count": True,
            }

            # Only add scoring profile if one exists
            if scoring_profile:
                search_kwargs["scoring_profile"] = scoring_profile

            results = client.search(**search_kwargs)  # pyright: ignore[reportArgumentType]

            formatted_results = self._format_search_results(
                results, detail_level, max_results
            )

            return {
                "results": formatted_results,
                "total_count": getattr(
                    results, "get_count", lambda: len(formatted_results)
                )(),
                "search_query": query,
                "priority_type": priority_type,
                "source": source,
                "controls_applied": {
                    "detail_level": detail_level,
                    "max_results": max_results,
                    "scoring_profile": scoring_profile,
                },
            }

        except Exception as e:
            logger.error(f"Priority search failed: {str(e)}")
            return {
                "results": [],
                "total_count": 0,
                "error": f"Priority search failed: {str(e)}",
                "search_query": query,
                "priority_type": priority_type,
                "source": source,
            }
