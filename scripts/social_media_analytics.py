#!/usr/bin/env python3
"""
Unified Social Media Analytics Module for Farm and Fleet
=======================================================

This module provides comprehensive social media analytics with enhanced visual output
using Rich console formatting. It combines the best features from both query modules
while integrating quality-based search capabilities and friction point analysis.

Key Features:
- Rich console with professional formatting and colors
- Quality-enhanced search with friction point analysis
- Network sentiment analysis with executive reporting
- Viral content identification and influencer analysis
- Competitive intelligence and market positioning
- Geographic and demographic insights
- Real-time progress indicators and interactive dashboards
- Executive-ready reporting with visual charts and tables

Author: Farm and Fleet Analytics Team
Date: August 2025
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from typing import Any

import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ==================================================================================================
# ENVIRONMENT CONFIGURATION
# ==================================================================================================

# Load environment variables
if dotenv.load_dotenv(override=True):
    console = Console()
    console.print("✅ Environment variables loaded from .env file", style="green")

# --- Azure AI Search Configuration ---
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
AZURE_SEARCH_INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_SEARCH_API_KEY)

# --- Azure OpenAI Configuration for Embeddings ---
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
EMBEDDING_MODEL_DIMENSIONS = 1536  # text-embedding-3-small

# Initialize Rich console
console = Console()

# ==================================================================================================
# ENHANCED DATA MODELS
# ==================================================================================================


class NetworkSentimentAnalysis(BaseModel):
    """Enhanced sentiment analysis results for a specific network."""

    network: str
    positive_count: int = Field(default=0, description="Count of positive posts")
    neutral_count: int = Field(default=0, description="Count of neutral posts")
    negative_count: int = Field(default=0, description="Count of negative posts")
    total_posts: int = Field(default=0, description="Total posts analyzed")
    positive_percentage: float = Field(
        default=0.0, description="Percentage of positive posts"
    )
    sentiment_score: float = Field(default=0.0, description="Overall sentiment score")
    # Quality enhancement fields
    high_quality_count: int = Field(default=0, description="High quality posts")
    medium_quality_count: int = Field(default=0, description="Medium quality posts")
    low_quality_count: int = Field(default=0, description="Low quality posts")
    noise_count: int = Field(default=0, description="Noise posts")


class QualityProductPerformance(BaseModel):
    """Enhanced product performance with quality insights."""

    product_name: str
    network: str
    mention_count: int
    sentiment_score: float
    engagement_metrics: dict[str, float] = Field(default_factory=dict)
    # Quality enhancement fields
    quality_distribution: dict[str, int] = Field(default_factory=dict)
    friction_categories: list[str] = Field(default_factory=list)
    business_relevance_score: float = Field(default=0.0)


class FrictionPointAnalysis(BaseModel):
    """Friction point analysis with categorization."""

    category: str
    mention_count: int
    severity_score: float
    networks: list[str] = Field(default_factory=list)
    sentiment_breakdown: dict[str, int] = Field(default_factory=dict)
    example_messages: list[str] = Field(default_factory=list)
    trending_score: float = Field(default=0.0)


class QualityMetrics(BaseModel):
    """Overall quality metrics for the dataset."""

    total_messages: int
    high_quality_percentage: float
    medium_quality_percentage: float
    low_quality_percentage: float
    noise_percentage: float
    actionable_insights_count: int
    friction_categories_identified: list[str]


class EnhancedViralContent(BaseModel):
    """Enhanced viral content identification with quality insights."""

    message_id: str
    network: str
    message_snippet: str
    total_engagement: float
    engagement_rate: float
    likes: int = Field(default=0)
    shares: int = Field(default=0)
    comments: int = Field(default=0)
    sentiment: str = Field(default="Neutral")
    # Quality enhancement fields
    quality_score: str = Field(default="medium")
    friction_indicators: list[str] = Field(default_factory=list)
    business_relevance: str = Field(default="business_relevant")


class ExecutiveDashboard(BaseModel):
    """Comprehensive executive dashboard with quality insights."""

    analysis_period: tuple[datetime, datetime]
    total_messages_analyzed: int
    quality_metrics: QualityMetrics
    network_sentiment: list[NetworkSentimentAnalysis] = Field(default_factory=list)
    product_performance: list[QualityProductPerformance] = Field(default_factory=list)
    friction_points: list[FrictionPointAnalysis] = Field(default_factory=list)
    viral_content: list[EnhancedViralContent] = Field(default_factory=list)
    competitive_insights: dict[str, Any] = Field(default_factory=dict)
    executive_summary: str = Field(default="")
    key_recommendations: list[str] = Field(default_factory=list)


# ==================================================================================================
# ENHANCED SOCIAL MEDIA ANALYTICS CLIENT
# ==================================================================================================


class SocialMediaAnalyticsClient:
    """
    Unified Social Media Analytics Client with Rich Console Integration.

    This class provides comprehensive marketing analytics with professional
    visual formatting, quality-enhanced search, and friction point analysis.
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the analytics client with Rich console integration."""
        self.console = Console() if not console else console
        self.verbose = verbose

        # Initialize search and OpenAI clients
        self.search_client = SearchClient(
            AZURE_SEARCH_SERVICE_ENDPOINT, AZURE_SEARCH_INDEX_NAME, SEARCH_CREDENTIAL
        )
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        # Quality scoring profiles for enhanced search
        self.quality_profiles = {
            "quality-boost": "quality-relevance",
            "engagement-boost": "social-engagement",
            "freshness-boost": "social-freshness",
            "location-boost": "social-location",
            "theme-boost": "social-themes",
        }

        if self.verbose:
            self.console.print(
                Panel(
                    "[bold green]Social Media Analytics Client Initialized[/bold green]\n"
                    "✅ Azure AI Search connected\n"
                    "✅ Azure OpenAI embeddings ready\n"
                    "✅ Rich console formatting enabled\n"
                    "✅ Quality-enhanced search profiles loaded",
                    title="🎯 Farm & Fleet Analytics",
                    border_style="green",
                )
            )

    def get_embedding(self, text: str) -> list[float]:
        """Generate embeddings using Azure OpenAI with progress indicator."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=None)

            embedding = (
                self.openai_client.embeddings.create(
                    input=[text], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                )
                .data[0]
                .embedding
            )

            progress.update(task, completed=True)

        return embedding

    # ==================================================================================================
    # QUALITY-ENHANCED SEARCH METHODS
    # ==================================================================================================

    def quality_semantic_search(
        self,
        query_text: str,
        quality_filter: str = "high,medium",
        k: int = 10,
        include_friction: bool = False,
        network: str | None = None,
        scoring_profile: str = "quality-boost",
    ) -> Any:
        """
        Perform quality-enhanced semantic search with Rich progress indicators.

        Args:
            query_text: Text to search for
            quality_filter: Comma-separated quality levels (high,medium,low,noise)
            k: Number of results to return
            include_friction: Whether to focus on friction points
            network: Specific network to search (optional)
            scoring_profile: Scoring profile to use for relevance boosting

        Returns:
            Search results enhanced with quality filtering
        """
        # Build quality filter expression
        quality_levels = [q.strip() for q in quality_filter.split(",")]
        quality_filter_expr = " or ".join(
            [f"qualityScore eq '{q}'" for q in quality_levels]
        )

        # Add network filter if specified
        if network:
            network_filter = f"network eq '{network}'"
            filter_expression = f"({quality_filter_expr}) and ({network_filter})"
        else:
            filter_expression = quality_filter_expr

        # Add friction focus if requested
        if include_friction:
            friction_filter = "frictionCategories/any()"  # Has any friction categories
            filter_expression = f"({filter_expression}) and ({friction_filter})"

        # Enhanced field selection including quality fields
        select_fields = [
            "messageId",
            "network",
            "message",
            "sentiment",
            "extractedProducts",
            "extractedLocations",
            "extractedOrganizations",
            "likes",
            "shares",
            "comments",
            "potentialImpressions",
            # Quality enhancement fields
            "qualityScore",
            "frictionCategories",
            "businessRelevance",
            "analysisReason",
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            # Generate embeddings
            embedding_task = progress.add_task(
                "Generating query embeddings...", total=1
            )
            query_vector = self.get_embedding(query_text)
            progress.update(embedding_task, advance=1)

            # Execute search
            search_task = progress.add_task(
                "Executing quality-enhanced search...", total=1
            )

            results = self.search_client.search(
                search_text=query_text,
                vector_queries=[
                    VectorizedQuery(
                        vector=query_vector,
                        k_nearest_neighbors=k,
                        fields="messageVector",
                    )
                ],
                select=select_fields,
                filter=filter_expression,
                scoring_profile=None,  # Disable scoring profile for now
                query_type="semantic",
                semantic_configuration_name="social-media-semantic",
                query_caption="extractive",
                query_answer="extractive",
                top=k,
                include_total_count=True,
            )

            progress.update(search_task, advance=1)

        return results

    def friction_point_search(
        self,
        friction_category: str | None = None,
        network: str | None = None,
        sentiment: str = "Negative",
        limit: int = 20,
    ) -> Any:
        """
        Search specifically for friction points with enhanced filtering.

        Args:
            friction_category: Specific friction category to search
            network: Network to focus on
            sentiment: Sentiment filter (Negative, Neutral, Positive)
            limit: Number of results to return

        Returns:
            Search results focused on friction points
        """
        # Build filter expression
        filters = [
            "frictionCategories/any()",  # Must have friction categories
            f"sentiment eq '{sentiment}'",
        ]

        if friction_category:
            filters.append(f"frictionCategories/any(cat: cat eq '{friction_category}')")

        if network:
            filters.append(f"network eq '{network}'")

        filter_expression = " and ".join(filters)

        return self.search_client.search(
            search_text="*",
            filter=filter_expression,
            select=[
                "messageId",
                "network",
                "message",
                "sentiment",
                "frictionCategories",
                "qualityScore",
                "businessRelevance",
                "analysisReason",
            ],
            top=limit,
            include_total_count=True,
        )

    # ==================================================================================================
    # ENHANCED ANALYTICS METHODS
    # ==================================================================================================

    def analyze_quality_metrics(self) -> QualityMetrics:
        """Analyze overall quality metrics across the dataset."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing quality metrics...", total=None)

            # Get quality distribution using faceted search
            quality_results = self.search_client.search(
                search_text="*",
                facets=["qualityScore"],
                top=0,
                include_total_count=True,
            )

            facets = quality_results.get_facets()
            total_count = quality_results.get_count()

            # Get friction categories
            friction_results = self.search_client.search(
                search_text="*",
                facets=["frictionCategories,count:20"],
                top=0,
            )

            friction_facets = friction_results.get_facets()
            friction_categories = []
            if friction_facets and "frictionCategories" in friction_facets:
                friction_categories = [
                    f["value"] for f in friction_facets["frictionCategories"]
                ]

            # Calculate percentages
            quality_counts = {}
            if facets and "qualityScore" in facets:
                quality_counts = {
                    f["value"]: f["count"] for f in facets["qualityScore"]
                }

            high_quality_count = quality_counts.get("high", 0)
            medium_quality_count = quality_counts.get("medium", 0)
            low_quality_count = quality_counts.get("low", 0)
            noise_count = quality_counts.get("noise", 0)

            actionable_count = high_quality_count + medium_quality_count

            progress.update(task, completed=True)

        return QualityMetrics(
            total_messages=total_count,
            high_quality_percentage=(high_quality_count / total_count) * 100
            if total_count > 0
            else 0,
            medium_quality_percentage=(medium_quality_count / total_count) * 100
            if total_count > 0
            else 0,
            low_quality_percentage=(low_quality_count / total_count) * 100
            if total_count > 0
            else 0,
            noise_percentage=(noise_count / total_count) * 100
            if total_count > 0
            else 0,
            actionable_insights_count=actionable_count,
            friction_categories_identified=friction_categories[:10],  # Top 10
        )

    def analyze_network_sentiment_with_quality(
        self, networks: list[str] | None = None
    ) -> list[NetworkSentimentAnalysis]:
        """
        Analyze sentiment distribution across networks with quality metrics.

        Args:
            networks: Optional list of networks to analyze. If None, analyzes all networks.

        Returns:
            List of NetworkSentimentAnalysis objects with quality insights
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            # Get networks if not specified
            if not networks:
                network_task = progress.add_task("Discovering networks...", total=1)
                network_results = self.search_client.search(
                    search_text="*",
                    facets=["network"],
                    top=0,
                )
                network_facets = network_results.get_facets()
                networks = (
                    [f["value"] for f in network_facets.get("network", [])]
                    if network_facets
                    else []
                )
                progress.update(network_task, advance=1)

            analysis_task = progress.add_task(
                "Analyzing networks...", total=len(networks)
            )
            network_analysis: list[NetworkSentimentAnalysis] = []

            for network in networks:
                # Get comprehensive network data including quality metrics
                network_results = self.search_client.search(
                    search_text="*",
                    filter=f"network eq '{network}'",
                    facets=["sentiment", "qualityScore"],
                    top=0,
                    include_total_count=True,
                )

                facets = network_results.get_facets()
                total_posts = network_results.get_count()

                # Initialize analysis object
                analysis = NetworkSentimentAnalysis(
                    network=network,
                    total_posts=total_posts,
                )

                # Process sentiment data
                if facets and "sentiment" in facets:
                    for sentiment_facet in facets["sentiment"]:
                        sentiment = sentiment_facet["value"]
                        count = sentiment_facet["count"]

                        if sentiment == "Positive":
                            analysis.positive_count = count
                        elif sentiment == "Negative":
                            analysis.negative_count = count
                        else:
                            analysis.neutral_count = count

                # Process quality data
                if facets and "qualityScore" in facets:
                    for quality_facet in facets["qualityScore"]:
                        quality = quality_facet["value"]
                        count = quality_facet["count"]

                        if quality == "high":
                            analysis.high_quality_count = count
                        elif quality == "medium":
                            analysis.medium_quality_count = count
                        elif quality == "low":
                            analysis.low_quality_count = count
                        else:  # noise
                            analysis.noise_count = count

                # Calculate percentages and scores
                if total_posts > 0:
                    analysis.positive_percentage = (
                        analysis.positive_count / total_posts
                    ) * 100
                    analysis.sentiment_score = (
                        analysis.positive_count - analysis.negative_count
                    ) / total_posts

                network_analysis.append(analysis)
                progress.update(analysis_task, advance=1)

        return sorted(network_analysis, key=lambda x: x.total_posts, reverse=True)

    def analyze_friction_points(self) -> list[FrictionPointAnalysis]:
        """
        Analyze friction points across all categories with Rich progress indicators.

        Returns:
            List of FrictionPointAnalysis objects
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing friction points...", total=None)

            # Get friction categories facets
            friction_results = self.search_client.search(
                search_text="*",
                filter="frictionCategories/any()",
                facets=["frictionCategories,count:20"],
                top=0,
                include_total_count=True,
            )

            facets = friction_results.get_facets()
            friction_analysis: list[FrictionPointAnalysis] = []

            if facets and "frictionCategories" in facets:
                category_task = progress.add_task(
                    "Processing categories...", total=len(facets["frictionCategories"])
                )

                for friction_facet in facets["frictionCategories"]:
                    category = friction_facet["value"]
                    mention_count = friction_facet["count"]

                    # Get sentiment breakdown for this category
                    category_results = self.search_client.search(
                        search_text="*",
                        filter=f"frictionCategories/any(cat: cat eq '{category}')",
                        facets=["sentiment", "network"],
                        select=["message"],
                        top=3,  # Get a few examples
                    )

                    category_facets = category_results.get_facets()
                    sentiment_breakdown = {}
                    networks = []

                    if category_facets:
                        if "sentiment" in category_facets:
                            sentiment_breakdown = {
                                f["value"]: f["count"]
                                for f in category_facets["sentiment"]
                            }
                        if "network" in category_facets:
                            networks = [f["value"] for f in category_facets["network"]]

                    # Calculate severity score (higher negative sentiment = higher severity)
                    negative_count = sentiment_breakdown.get("Negative", 0)
                    total_sentiment_count = sum(sentiment_breakdown.values())
                    severity_score = (
                        (negative_count / total_sentiment_count)
                        if total_sentiment_count > 0
                        else 0
                    )

                    # Get example messages
                    examples = [
                        result.get("message", "")[:100] + "..."
                        for result in category_results
                        if result.get("message")
                    ]

                    friction_analysis.append(
                        FrictionPointAnalysis(
                            category=category,
                            mention_count=mention_count,
                            severity_score=severity_score,
                            networks=networks[:3],  # Top 3 networks
                            sentiment_breakdown=sentiment_breakdown,
                            example_messages=examples,
                            trending_score=mention_count
                            * severity_score,  # Simple trending metric
                        )
                    )

                    progress.update(category_task, advance=1)

            progress.update(task, completed=True)

        return sorted(friction_analysis, key=lambda x: x.trending_score, reverse=True)

    def identify_enhanced_viral_content(
        self, limit: int = 20, min_engagement_threshold: float = 50
    ) -> list[EnhancedViralContent]:
        """
        Identify viral content with quality enhancement and friction analysis.

        Args:
            limit: Number of viral posts to return
            min_engagement_threshold: Minimum total engagement for viral classification

        Returns:
            List of EnhancedViralContent objects
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Identifying viral content...", total=None)

            # Search for high-engagement content with quality fields
            posts = self.search_client.search(
                search_text="*",
                filter="likes gt 5 or shares gt 2 or comments gt 3",
                select=[
                    "messageId",
                    "network",
                    "message",
                    "sentiment",
                    "likes",
                    "shares",
                    "comments",
                    "potentialImpressions",
                    "qualityScore",
                    "frictionCategories",
                    "businessRelevance",
                ],
                top=500,
            )

            viral_content: list[EnhancedViralContent] = []

            for post in posts:
                likes = int(post.get("likes", 0))
                shares = int(post.get("shares", 0))
                comments = int(post.get("comments", 0))
                total_engagement = likes + shares + comments

                if total_engagement >= min_engagement_threshold:
                    impressions = int(post.get("potentialImpressions", 1))
                    engagement_rate = (
                        (total_engagement / impressions) if impressions > 0 else 0.0
                    )

                    message = post.get("message", "")
                    message_snippet = (
                        message[:200] + "..." if len(message) > 200 else message
                    )

                    # Get quality enhancement fields
                    quality_score = post.get("qualityScore", "medium")
                    friction_indicators = post.get("frictionCategories", [])
                    business_relevance = post.get(
                        "businessRelevance", "business_relevant"
                    )

                    viral_content.append(
                        EnhancedViralContent(
                            message_id=post.get("messageId", ""),
                            network=post.get("network", "Unknown"),
                            message_snippet=message_snippet,
                            total_engagement=float(total_engagement),
                            engagement_rate=engagement_rate,
                            likes=likes,
                            shares=shares,
                            comments=comments,
                            sentiment=post.get("sentiment", "Neutral"),
                            quality_score=quality_score,
                            friction_indicators=friction_indicators
                            if isinstance(friction_indicators, list)
                            else [],
                            business_relevance=business_relevance,
                        )
                    )

            progress.update(task, completed=True)

        return sorted(viral_content, key=lambda x: x.total_engagement, reverse=True)[
            :limit
        ]

    # ==================================================================================================
    # RICH FORMATTING AND DISPLAY METHODS
    # ==================================================================================================

    def display_quality_metrics(self, metrics: QualityMetrics) -> None:
        """Display quality metrics with Rich formatting."""
        # Create quality distribution table
        quality_table = Table(
            title="📊 Content Quality Distribution",
            show_header=True,
            header_style="bold magenta",
        )
        quality_table.add_column("Quality Level", style="cyan", width=15)
        quality_table.add_column("Percentage", justify="right", style="green", width=12)
        quality_table.add_column("Description", style="white", width=40)
        quality_table.add_column("Business Value", style="yellow", width=25)

        # Add quality rows with color coding
        quality_table.add_row(
            "🟢 High Quality",
            f"{metrics.high_quality_percentage:.1f}%",
            "Actionable insights with friction indicators",
            "Immediate action required",
        )
        quality_table.add_row(
            "🟡 Medium Quality",
            f"{metrics.medium_quality_percentage:.1f}%",
            "Business relevant with substantial content",
            "Context for insights",
        )
        quality_table.add_row(
            "🟠 Low Quality",
            f"{metrics.low_quality_percentage:.1f}%",
            "Tangential mentions, limited value",
            "Monitor for trends",
        )
        quality_table.add_row(
            "🔴 Noise",
            f"{metrics.noise_percentage:.1f}%",
            "Irrelevant content, should be filtered",
            "Exclude from analysis",
        )

        self.console.print(quality_table)

        # Create summary panel
        actionable_percentage = (
            metrics.high_quality_percentage + metrics.medium_quality_percentage
        )
        summary_text = f"""
[bold green]📈 Quality Summary[/bold green]
• Total Messages: [bold cyan]{metrics.total_messages:,}[/bold cyan]
• Actionable Content: [bold green]{actionable_percentage:.1f}%[/bold green]
• Friction Categories: [bold yellow]{len(metrics.friction_categories_identified)}[/bold yellow]
• High-Value Insights: [bold magenta]{metrics.actionable_insights_count:,}[/bold magenta]

[bold blue]💡 Business Impact[/bold blue]
• Ready for vectorization and AI analysis
• Sufficient quality for friction point identification
• Strong foundation for customer insights
        """

        self.console.print(
            Panel(
                summary_text.strip(), title="🎯 Executive Summary", border_style="blue"
            )
        )

    def display_network_sentiment_analysis(
        self, network_sentiment: list[NetworkSentimentAnalysis]
    ) -> None:
        """Display network sentiment analysis with Rich formatting."""

        # Create main sentiment table
        sentiment_table = Table(
            title="📊 Network Sentiment Distribution with Quality Metrics",
            show_header=True,
            header_style="bold magenta",
        )
        sentiment_table.add_column("Network", style="cyan", width=12)
        sentiment_table.add_column(
            "Total Posts", justify="right", style="white", width=12
        )
        sentiment_table.add_column("Positive", justify="right", style="green", width=10)
        sentiment_table.add_column("Negative", justify="right", style="red", width=10)
        sentiment_table.add_column("Neutral", justify="right", style="yellow", width=10)
        sentiment_table.add_column(
            "Sentiment Score", justify="right", style="bright_blue", width=15
        )
        sentiment_table.add_column(
            "Quality Score", justify="right", style="magenta", width=15
        )

        # Sort by total posts
        sorted_sentiment = sorted(
            network_sentiment, key=lambda x: x.total_posts, reverse=True
        )

        for sentiment in sorted_sentiment:
            # Calculate quality percentage
            quality_posts = (
                sentiment.high_quality_count + sentiment.medium_quality_count
            )
            quality_percentage = (
                (quality_posts / sentiment.total_posts * 100)
                if sentiment.total_posts > 0
                else 0
            )

            # Get sentiment indicator
            sentiment_indicator = (
                "🟢"
                if sentiment.sentiment_score > 0.1
                else ("🔴" if sentiment.sentiment_score < -0.1 else "🟡")
            )

            # Get quality indicator
            quality_indicator = (
                "🟢"
                if quality_percentage > 60
                else ("🟡" if quality_percentage > 30 else "🔴")
            )

            sentiment_table.add_row(
                sentiment.network,
                f"{sentiment.total_posts:,}",
                f"{sentiment.positive_count:,} ({sentiment.positive_percentage:.1f}%)",
                f"{sentiment.negative_count:,}",
                f"{sentiment.neutral_count:,}",
                f"{sentiment_indicator} {sentiment.sentiment_score:+.3f}",
                f"{quality_indicator} {quality_percentage:.1f}%",
            )

        self.console.print(sentiment_table)

        # Create quality breakdown table
        quality_table = Table(
            title="🎯 Quality Distribution by Network",
            show_header=True,
            header_style="bold blue",
        )
        quality_table.add_column("Network", style="cyan", width=12)
        quality_table.add_column(
            "High Quality", justify="right", style="green", width=12
        )
        quality_table.add_column(
            "Medium Quality", justify="right", style="yellow", width=14
        )
        quality_table.add_column(
            "Low Quality", justify="right", style="orange1", width=12
        )
        quality_table.add_column("Noise", justify="right", style="red", width=8)
        quality_table.add_column(
            "Actionable %", justify="right", style="bright_green", width=12
        )

        for sentiment in sorted_sentiment:
            actionable_count = (
                sentiment.high_quality_count + sentiment.medium_quality_count
            )
            actionable_percentage = (
                (actionable_count / sentiment.total_posts * 100)
                if sentiment.total_posts > 0
                else 0
            )

            quality_table.add_row(
                sentiment.network,
                f"{sentiment.high_quality_count:,}",
                f"{sentiment.medium_quality_count:,}",
                f"{sentiment.low_quality_count:,}",
                f"{sentiment.noise_count:,}",
                f"{actionable_percentage:.1f}%",
            )

        self.console.print("\n")
        self.console.print(quality_table)

        # Create summary insights
        total_posts = sum(s.total_posts for s in network_sentiment)
        avg_sentiment = (
            sum(s.sentiment_score for s in network_sentiment) / len(network_sentiment)
            if network_sentiment
            else 0
        )
        best_network = (
            max(network_sentiment, key=lambda x: x.sentiment_score)
            if network_sentiment
            else None
        )
        worst_network = (
            min(network_sentiment, key=lambda x: x.sentiment_score)
            if network_sentiment
            else None
        )

        best_network_name = best_network.network if best_network else "N/A"
        best_network_score = best_network.sentiment_score if best_network else 0.0
        worst_network_name = worst_network.network if worst_network else "N/A"
        worst_network_score = worst_network.sentiment_score if worst_network else 0.0

        summary_text = f"""
[bold green]📈 Network Analysis Summary[/bold green]
• Total Posts Analyzed: [bold cyan]{total_posts:,}[/bold cyan]
• Average Sentiment Score: [bold white]{avg_sentiment:+.3f}[/bold white]
• Best Performing Network: [bold green]{best_network_name}[/bold green] ({best_network_score:+.3f} sentiment)
• Network Requiring Attention: [bold red]{worst_network_name}[/bold red] ({worst_network_score:+.3f} sentiment)

[bold blue]💡 Strategic Insights[/bold blue]
• Focus marketing efforts on high-sentiment networks
• Monitor networks with negative sentiment trends
• Leverage quality content for engagement strategies
        """

        self.console.print(
            Panel(
                summary_text.strip(),
                title="📊 Network Intelligence",
                border_style="blue",
            )
        )

    def display_friction_analysis(
        self, friction_points: list[FrictionPointAnalysis]
    ) -> None:
        """Display friction point analysis with Rich formatting."""

        # Create friction points table
        friction_table = Table(
            title="🔥 Customer Friction Point Analysis",
            show_header=True,
            header_style="bold red",
        )
        friction_table.add_column("Category", style="yellow", width=20)
        friction_table.add_column("Mentions", justify="right", style="white", width=10)
        friction_table.add_column("Severity", justify="right", style="red", width=12)
        friction_table.add_column(
            "Trending Score", justify="right", style="magenta", width=15
        )
        friction_table.add_column("Top Networks", style="cyan", width=25)
        friction_table.add_column(
            "Negative %", justify="right", style="bright_red", width=12
        )

        for friction in friction_points[:10]:  # Top 10
            # Calculate negative percentage
            total_sentiment = sum(friction.sentiment_breakdown.values())
            negative_count = friction.sentiment_breakdown.get("Negative", 0)
            negative_percentage = (
                (negative_count / total_sentiment * 100) if total_sentiment > 0 else 0
            )

            # Get severity indicator
            severity_indicator = (
                "🔴"
                if friction.severity_score > 0.7
                else ("🟠" if friction.severity_score > 0.4 else "🟡")
            )

            # Format networks
            networks_text = ", ".join(friction.networks[:3])
            if len(friction.networks) > 3:
                networks_text += "..."

            friction_table.add_row(
                friction.category,
                f"{friction.mention_count:,}",
                f"{severity_indicator} {friction.severity_score:.2f}",
                f"{friction.trending_score:.1f}",
                networks_text,
                f"{negative_percentage:.1f}%",
            )

        self.console.print(friction_table)

        # Create detailed examples for top friction points
        if friction_points:
            top_friction = friction_points[0]

            examples_text = f"""
[bold red]🚨 Top Friction Category: {top_friction.category}[/bold red]

[bold yellow]📊 Statistics:[/bold yellow]
• Total Mentions: [bold white]{top_friction.mention_count:,}[/bold white]
• Severity Score: [bold red]{top_friction.severity_score:.2f}[/bold red]
• Trending Score: [bold magenta]{top_friction.trending_score:.1f}[/bold magenta]

[bold yellow]📱 Network Distribution:[/bold yellow]
• Primary Networks: [bold cyan]{", ".join(top_friction.networks[:3])}[/bold cyan]

[bold yellow]💬 Example Customer Feedback:[/bold yellow]
            """

            for _i, example in enumerate(top_friction.example_messages[:3], 1):
                examples_text += f'• "{example}"\n'

            self.console.print(
                Panel(
                    examples_text.strip(),
                    title="🔍 Detailed Friction Analysis",
                    border_style="red",
                )
            )

    def display_viral_content_analysis(
        self, viral_content: list[EnhancedViralContent]
    ) -> None:
        """Display enhanced viral content analysis with Rich formatting."""

        # Create viral content table
        viral_table = Table(
            title="🔥 Viral Content Analysis with Quality Insights",
            show_header=True,
            header_style="bold magenta",
        )
        viral_table.add_column("Network", style="cyan", width=10)
        viral_table.add_column("Engagement", justify="right", style="green", width=12)
        viral_table.add_column("Quality", style="yellow", width=8)
        viral_table.add_column("Sentiment", style="blue", width=10)
        viral_table.add_column("Message Preview", style="white", width=40)
        viral_table.add_column("Friction", style="red", width=12)

        for content in viral_content:
            # Get quality indicator
            quality_indicator = {
                "high": "🟢 High",
                "medium": "🟡 Med",
                "low": "🟠 Low",
                "noise": "🔴 Noise",
            }.get(content.quality_score, "❓ Unknown")

            # Get sentiment indicator
            sentiment_indicator = {
                "Positive": "😊 Pos",
                "Negative": "😞 Neg",
                "Neutral": "😐 Neu",
            }.get(content.sentiment, "❓ Unknown")

            # Check for friction indicators
            friction_indicator = "🔥 Yes" if content.friction_indicators else "-"

            viral_table.add_row(
                content.network,
                f"{int(content.total_engagement):,}",
                quality_indicator,
                sentiment_indicator,
                content.message_snippet[:60] + "..."
                if len(content.message_snippet) > 60
                else content.message_snippet,
                friction_indicator,
            )

        self.console.print(viral_table)

        # Create viral content insights
        if viral_content:
            high_quality_viral = [v for v in viral_content if v.quality_score == "high"]
            friction_viral = [v for v in viral_content if v.friction_indicators]

            insights_text = f"""
[bold green]📈 Viral Content Insights[/bold green]
• Total Viral Posts: [bold cyan]{len(viral_content)}[/bold cyan]
• High-Quality Viral: [bold green]{len(high_quality_viral)}[/bold green]
• Friction-Related Viral: [bold red]{len(friction_viral)}[/bold red]
• Average Engagement: [bold white]{sum(v.total_engagement for v in viral_content) / len(viral_content):.0f}[/bold white]

[bold blue]💡 Content Strategy Recommendations[/bold blue]
• Analyze high-quality viral content for replication patterns
• Address friction-related viral posts to improve customer experience
• Leverage positive viral content for marketing campaigns
            """

            self.console.print(
                Panel(
                    insights_text.strip(),
                    title="🚀 Viral Content Intelligence",
                    border_style="magenta",
                )
            )

    def display_search_results(
        self,
        results: Any,
        title: str = "Search Results",
        show_quality: bool = True,
        max_results: int = 10,
    ) -> None:
        """Display search results with Rich formatting and quality indicators."""

        # Create results table
        table = Table(title=f"🔍 {title}", show_header=True, header_style="bold blue")
        table.add_column("Network", style="cyan", width=10)
        table.add_column("Quality", style="green", width=8)
        table.add_column("Sentiment", style="yellow", width=10)
        table.add_column("Message Preview", style="white", width=50)
        table.add_column("Engagement", justify="right", style="magenta", width=12)

        count = 0
        for result in results:
            if count >= max_results:
                break

            # Get quality indicator
            quality = result.get("qualityScore", "medium")
            quality_indicator = {
                "high": "🟢 High",
                "medium": "🟡 Med",
                "low": "🟠 Low",
                "noise": "🔴 Noise",
            }.get(quality, "❓ Unknown")

            # Get sentiment indicator
            sentiment = result.get("sentiment", "Neutral")
            sentiment_indicator = {
                "Positive": "😊 Pos",
                "Negative": "😞 Neg",
                "Neutral": "😐 Neu",
            }.get(sentiment, "❓ Unknown")

            # Calculate total engagement
            likes = int(result.get("likes", 0))
            shares = int(result.get("shares", 0))
            comments = int(result.get("comments", 0))
            total_engagement = likes + shares + comments

            # Truncate message
            message = result.get("message", "")
            if len(message) > 200:
                message = message[:197] + "..."

            table.add_row(
                result.get("network", "Unknown"),
                quality_indicator if show_quality else "",
                sentiment_indicator,
                message,
                f"{total_engagement:,}" if total_engagement > 0 else "-",
            )
            count += 1

        self.console.print(table)

        # Show results summary
        total_results = (
            results.get_count() if hasattr(results, "get_count") else len(list(results))
        )
        self.console.print(
            f"\n📊 [bold cyan]Total Results:[/bold cyan] [bold white]{total_results:,}[/bold white] (showing {count})\n"
        )

    def create_executive_dashboard_panel(self, dashboard: ExecutiveDashboard) -> Panel:
        """Create a comprehensive executive dashboard panel."""

        # Create dashboard content
        dashboard_content = f"""
[bold green]📊 EXECUTIVE DASHBOARD - SOCIAL MEDIA ANALYTICS[/bold green]

[bold blue]📈 Overview Metrics[/bold blue]
• Total Messages Analyzed: [bold cyan]{dashboard.total_messages_analyzed:,}[/bold cyan]
• Analysis Period: [bold white]{dashboard.analysis_period[0].strftime("%Y-%m-%d")} to {dashboard.analysis_period[1].strftime("%Y-%m-%d")}[/bold white]
• Quality Score: [bold green]{dashboard.quality_metrics.high_quality_percentage + dashboard.quality_metrics.medium_quality_percentage:.1f}% Actionable[/bold green]

[bold blue]🎯 Key Performance Indicators[/bold blue]
• High-Quality Insights: [bold green]{dashboard.quality_metrics.high_quality_percentage:.1f}%[/bold green]
• Friction Points Identified: [bold yellow]{len(dashboard.friction_points)}[/bold yellow] categories
• Viral Content: [bold magenta]{len(dashboard.viral_content)}[/bold magenta] high-engagement posts
• Networks Analyzed: [bold cyan]{len(dashboard.network_sentiment)}[/bold cyan] platforms

[bold blue]💡 Strategic Recommendations[/bold blue]
"""

        for _i, rec in enumerate(dashboard.key_recommendations[:5], 1):
            dashboard_content += f"• {rec}\n"

        return Panel(
            dashboard_content.strip(),
            title="🎯 Farm & Fleet Executive Dashboard",
            border_style="green",
            padding=(1, 2),
        )

    # ==================================================================================================
    # CLI INTERFACE METHODS
    # ==================================================================================================

    async def run_interactive_session(self) -> None:
        """Run an interactive analytics session with Rich interface."""

        self.console.print(
            Panel(
                "[bold green]🎯 Welcome to Farm & Fleet Social Media Analytics[/bold green]\n\n"
                "This interactive session provides comprehensive social media insights\n"
                "with quality-enhanced search and professional visual formatting.\n\n"
                "[bold blue]Available Commands:[/bold blue]\n"
                "1. Quality Metrics Analysis\n"
                "2. Friction Point Search\n"
                "3. Network Sentiment Analysis\n"
                "4. Friction Point Analysis\n"
                "5. Viral Content Identification\n"
                "6. Executive Dashboard\n"
                "7. Custom Search Query\n"
                "8. Exit\n\n"
                "[italic]Type the number of your choice or 'help' for more information.[/italic]",
                title="🚀 Interactive Analytics",
                border_style="blue",
            )
        )

        while True:
            try:
                choice = self.console.input(
                    "\n[bold cyan]Enter your choice (1-8): [/bold cyan]"
                ).strip()

                if choice in ["8", "exit", "quit", "q"]:
                    self.console.print(
                        "\n[bold green]Thank you for using Farm & Fleet Analytics! 👋[/bold green]"
                    )
                    break
                elif choice == "1":
                    await self._handle_quality_metrics()
                elif choice == "2":
                    await self._handle_friction_search()
                elif choice == "3":
                    await self._handle_network_sentiment()
                elif choice == "4":
                    await self._handle_friction_analysis()
                elif choice == "5":
                    await self._handle_viral_content()
                elif choice == "6":
                    await self._handle_executive_dashboard()
                elif choice == "7":
                    await self._handle_custom_search()
                elif choice in ["help", "h"]:
                    self._show_help()
                else:
                    self.console.print(
                        "[bold red]Invalid choice. Please select 1-8 or 'help'.[/bold red]"
                    )

            except KeyboardInterrupt:
                self.console.print(
                    "\n\n[bold yellow]Session interrupted by user. Goodbye! 👋[/bold yellow]"
                )
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")

    async def _handle_quality_metrics(self) -> None:
        """Handle quality metrics analysis command."""
        self.console.print(
            "\n[bold blue]🔍 Analyzing Content Quality Metrics...[/bold blue]\n"
        )

        metrics = self.analyze_quality_metrics()
        self.display_quality_metrics(metrics)

    async def _handle_friction_search(self) -> None:
        """Handle friction point search command."""
        self.console.print("\n[bold blue]🔥 Searching Friction Points...[/bold blue]\n")

        friction_category = self.console.input(
            "Enter friction category (or press Enter for all): "
        ).strip()
        network = self.console.input("Enter network (or press Enter for all): ").strip()

        results = self.friction_point_search(
            friction_category=friction_category if friction_category else None,
            network=network if network else None,
        )

        self.display_search_results(
            results,
            title=f"Friction Points{f' - {friction_category}' if friction_category else ''}",
            show_quality=True,
        )

    async def _handle_friction_analysis(self) -> None:
        """Handle comprehensive friction point analysis command."""
        self.console.print(
            "\n[bold blue]🔥 Comprehensive Friction Analysis...[/bold blue]\n"
        )

        friction_points = self.analyze_friction_points()
        self.display_friction_analysis(friction_points)

    async def _handle_viral_content(self) -> None:
        """Handle viral content identification command."""
        self.console.print("\n[bold blue]🚀 Identifying Viral Content...[/bold blue]\n")

        min_engagement = self.console.input(
            "Minimum engagement threshold [default: 50]: "
        ).strip()
        min_engagement = int(min_engagement) if min_engagement.isdigit() else 50

        limit = self.console.input("Number of results [default: 15]: ").strip()
        limit = int(limit) if limit.isdigit() else 15

        viral_content = self.identify_enhanced_viral_content(
            limit=limit, min_engagement_threshold=min_engagement
        )
        self.display_viral_content_analysis(viral_content)

    async def _handle_executive_dashboard(self) -> None:
        """Handle executive dashboard generation command."""
        self.console.print(
            "\n[bold blue]📊 Generating Executive Dashboard...[/bold blue]\n"
        )

        # Generate comprehensive dashboard
        dashboard = await self._generate_executive_dashboard()

        # Display the dashboard
        dashboard_panel = self.create_executive_dashboard_panel(dashboard)
        self.console.print(dashboard_panel)

        # Display individual components
        if dashboard.network_sentiment:
            self.display_network_sentiment_analysis(dashboard.network_sentiment)

        if dashboard.friction_points:
            self.console.print("\n")
            self.display_friction_analysis(dashboard.friction_points)

        if dashboard.viral_content:
            self.console.print("\n")
            self.display_viral_content_analysis(dashboard.viral_content)

    async def _generate_executive_dashboard(self) -> ExecutiveDashboard:
        """Generate comprehensive executive dashboard data."""

        # Set analysis period (for demo, using current time)
        start_time = datetime.now(UTC).replace(microsecond=0)
        end_time = start_time

        # Initialize dashboard
        dashboard = ExecutiveDashboard(
            analysis_period=(start_time, end_time),
            total_messages_analyzed=0,
            quality_metrics=QualityMetrics(
                total_messages=0,
                high_quality_percentage=0,
                medium_quality_percentage=0,
                low_quality_percentage=0,
                noise_percentage=0,
                actionable_insights_count=0,
                friction_categories_identified=[],
            ),
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            # Analyze quality metrics
            quality_task = progress.add_task("Analyzing quality metrics...", total=1)
            dashboard.quality_metrics = self.analyze_quality_metrics()
            dashboard.total_messages_analyzed = dashboard.quality_metrics.total_messages
            progress.update(quality_task, advance=1)

            # Analyze network sentiment
            network_task = progress.add_task("Analyzing network sentiment...", total=1)
            dashboard.network_sentiment = self.analyze_network_sentiment_with_quality()
            progress.update(network_task, advance=1)

            # Analyze friction points
            friction_task = progress.add_task("Analyzing friction points...", total=1)
            dashboard.friction_points = self.analyze_friction_points()
            progress.update(friction_task, advance=1)

            # Identify viral content
            viral_task = progress.add_task("Identifying viral content...", total=1)
            dashboard.viral_content = self.identify_enhanced_viral_content(limit=10)
            progress.update(viral_task, advance=1)

            # Generate executive summary
            summary_task = progress.add_task("Generating executive summary...", total=1)
            dashboard.executive_summary = self._generate_executive_summary(dashboard)
            dashboard.key_recommendations = self._generate_key_recommendations(
                dashboard
            )
            progress.update(summary_task, advance=1)

        return dashboard

    def _generate_executive_summary(self, dashboard: ExecutiveDashboard) -> str:
        """Generate executive summary based on dashboard data."""

        if not dashboard.network_sentiment:
            return "Insufficient data for executive summary generation."

        total_posts = dashboard.total_messages_analyzed
        quality_score = (
            dashboard.quality_metrics.high_quality_percentage
            + dashboard.quality_metrics.medium_quality_percentage
        )
        top_network = max(
            dashboard.network_sentiment, key=lambda x: x.total_posts
        ).network
        avg_sentiment = sum(
            n.sentiment_score for n in dashboard.network_sentiment
        ) / len(dashboard.network_sentiment)

        summary = f"""
📊 EXECUTIVE SUMMARY - SOCIAL MEDIA ANALYTICS

🎯 OVERVIEW:
• Total posts analyzed: {total_posts:,}
• Quality actionable content: {quality_score:.1f}%
• Networks monitored: {len(dashboard.network_sentiment)}
• Top performing network: {top_network}
• Average sentiment score: {avg_sentiment:+.3f}

🔥 FRICTION ANALYSIS:
• Friction categories identified: {len(dashboard.friction_points)}
• High-priority friction points requiring immediate attention
• Customer pain points mapped across all networks

🚀 VIRAL CONTENT:
• High-engagement posts: {len(dashboard.viral_content)}
• Quality viral content opportunities identified
• Content strategy insights available

💡 STRATEGIC IMPACT:
• Data-driven insights ready for action
• Customer experience improvement opportunities identified
• Marketing optimization recommendations available
        """

        return summary.strip()

    def _generate_key_recommendations(self, dashboard: ExecutiveDashboard) -> list[str]:
        """Generate key strategic recommendations."""

        recommendations = []

        # Quality-based recommendations
        if dashboard.quality_metrics.high_quality_percentage < 30:
            recommendations.append(
                "Improve content quality monitoring to increase actionable insights"
            )

        if dashboard.quality_metrics.noise_percentage > 5:
            recommendations.append(
                "Implement noise filtering to focus on relevant customer feedback"
            )

        # Network-based recommendations
        if dashboard.network_sentiment:
            best_network = max(
                dashboard.network_sentiment, key=lambda x: x.sentiment_score
            )
            worst_network = min(
                dashboard.network_sentiment, key=lambda x: x.sentiment_score
            )

            recommendations.append(
                f"Focus marketing efforts on {best_network.network} (highest sentiment)"
            )

            if worst_network.sentiment_score < -0.2:
                recommendations.append(
                    f"Address negative sentiment on {worst_network.network}"
                )

        # Friction-based recommendations
        if dashboard.friction_points:
            top_friction = dashboard.friction_points[0]
            recommendations.append(
                f"Prioritize addressing {top_friction.category} friction points ({top_friction.mention_count:,} mentions)"
            )

        # Viral content recommendations
        if dashboard.viral_content:
            high_quality_viral = [
                v for v in dashboard.viral_content if v.quality_score == "high"
            ]
            if high_quality_viral:
                recommendations.append(
                    "Leverage high-quality viral content patterns for marketing campaigns"
                )

        return recommendations[:5]  # Top 5 recommendations

    async def _handle_network_sentiment(self) -> None:
        """Handle network sentiment analysis command."""
        self.console.print(
            "\n[bold blue]📊 Analyzing Network Sentiment...[/bold blue]\n"
        )

        sentiment_analysis = self.analyze_network_sentiment_with_quality()
        self.display_network_sentiment_analysis(sentiment_analysis)

    async def _handle_custom_search(self) -> None:
        """Handle custom search query command."""
        query = self.console.input(
            "\n[bold cyan]Enter your search query: [/bold cyan]"
        ).strip()

        if not query:
            self.console.print("[bold red]Query cannot be empty.[/bold red]")
            return

        quality_filter = self.console.input(
            "Quality filter (high,medium,low,noise) [default: high,medium]: "
        ).strip()
        if not quality_filter:
            quality_filter = "high,medium"

        include_friction = (
            self.console.input("Include friction points only? (y/N): ").strip().lower()
            == "y"
        )

        results = self.quality_semantic_search(
            query_text=query,
            quality_filter=quality_filter,
            include_friction=include_friction,
            k=15,
        )

        self.display_search_results(
            results,
            title=f"Custom Search: '{query}'",
            show_quality=True,
            max_results=15,
        )

    def _show_help(self) -> None:
        """Display help information."""
        help_content = """
[bold green]🎯 Farm & Fleet Analytics Help[/bold green]

[bold blue]Quality Metrics Analysis (1)[/bold blue]
• Analyzes overall content quality distribution
• Shows actionable vs noise content percentages
• Identifies friction categories across dataset

[bold blue]Friction Point Search (2)[/bold blue]
• Searches for customer pain points and complaints
• Filters by friction category and network
• Focuses on negative sentiment content

[bold blue]Network Sentiment Analysis (3)[/bold blue]
• Analyzes sentiment distribution across social networks
• Compares performance between platforms
• Includes quality metrics per network

[bold blue]Custom Search Query (7)[/bold blue]
• Performs semantic search with quality filtering
• Supports friction point focus
• Returns enhanced results with business context

[bold cyan]Quality Levels:[/bold cyan]
• [bold green]High[/bold green]: Actionable insights with friction indicators
• [bold yellow]Medium[/bold yellow]: Business relevant content
• [bold red]Low[/bold red]: Tangential mentions
• [bold red]Noise[/bold red]: Irrelevant content

[bold cyan]Friction Categories:[/bold cyan]
• Customer Service • Product Availability
• Store Operations • Product Quality
• Digital Experience
        """

        self.console.print(
            Panel(
                help_content.strip(), title="📚 Help & Usage Guide", border_style="cyan"
            )
        )


# ==================================================================================================
# CLI INTERFACE
# ==================================================================================================


def main() -> None:
    """Main CLI interface for Social Media Analytics."""
    parser = argparse.ArgumentParser(
        description="Farm & Fleet Social Media Analytics with Rich Console Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python social_media_analytics.py --interactive
  python social_media_analytics.py --quality-metrics
  python social_media_analytics.py --friction-search --category "Customer Service"
  python social_media_analytics.py --search "store experience" --quality high,medium
        """,
    )

    # Main operation modes
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive analytics session",
    )

    parser.add_argument(
        "--quality-metrics",
        "-q",
        action="store_true",
        help="Analyze content quality metrics",
    )

    parser.add_argument(
        "--friction-search",
        "-f",
        action="store_true",
        help="Search for friction points",
    )

    # Search options
    parser.add_argument("--search", "-s", type=str, help="Custom search query")

    parser.add_argument(
        "--quality",
        type=str,
        default="high,medium",
        help="Quality filter (comma-separated: high,medium,low,noise)",
    )

    parser.add_argument("--network", type=str, help="Filter by specific network")

    parser.add_argument(
        "--category", type=str, help="Friction category for friction search"
    )

    parser.add_argument(
        "--include-friction",
        action="store_true",
        help="Include only posts with friction indicators",
    )

    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results to display"
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    args = parser.parse_args()

    # Disable color if requested
    if args.no_color:
        import os

        os.environ["NO_COLOR"] = "1"

    # Initialize analytics client
    try:
        client = SocialMediaAnalyticsClient(verbose=args.verbose)

        if args.interactive:
            asyncio.run(client.run_interactive_session())
        elif args.quality_metrics:
            metrics = client.analyze_quality_metrics()
            client.display_quality_metrics(metrics)
        elif args.friction_search:
            results = client.friction_point_search(
                friction_category=args.category, network=args.network, limit=args.limit
            )
            client.display_search_results(
                results,
                title=f"Friction Points{f' - {args.category}' if args.category else ''}",
                show_quality=True,
                max_results=args.limit,
            )
        elif args.search:
            results = client.quality_semantic_search(
                query_text=args.search,
                quality_filter=args.quality,
                include_friction=args.include_friction,
                network=args.network,
                k=args.limit,
            )
            client.display_search_results(
                results,
                title=f"Search: '{args.search}'",
                show_quality=True,
                max_results=args.limit,
            )
        else:
            # Default to interactive mode
            asyncio.run(client.run_interactive_session())

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
