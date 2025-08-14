from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import dotenv
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    HnswAlgorithmConfiguration,
    MagnitudeScoringFunction,
    MagnitudeScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    TagScoringFunction,
    TagScoringParameters,
    TextWeights,
    VectorSearch,
    VectorSearchProfile,
)
from openai import AzureOpenAI

# Social Media Content Analysis
from social_media_content_analysis import (
    MessageAnalysis,
    SocialMediaContentAnalyzer,
)

# ==================================================================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================================================================
if dotenv.load_dotenv(override=True):
    print("Environment variables loaded from .env file")

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

# --- Azure AI Language (Text Analytics) Configuration for NER ---
AZURE_LANGUAGE_ENDPOINT = os.environ["AZURE_LANGUAGE_ENDPOINT"]
AZURE_LANGUAGE_KEY = os.environ["AZURE_LANGUAGE_KEY"]
TEXT_ANALYTICS_CREDENTIAL = AzureKeyCredential(AZURE_LANGUAGE_KEY)

# ==================================================================================================
# SCRIPT CONFIGURATION
# ==================================================================================================
DEFAULT_CSV_FILE_PATH = "data/Social Media Insights.csv"
DEFAULT_BATCH_SIZE = 5  # Batch size for processing and uploading


# ==================================================================================================
# TYPE DEFINITIONS
# ==================================================================================================
NEREntity = dict[str, str]
NERResult = dict[str, list[NEREntity] | bool]
KeyPhraseResult = dict[str, list[str] | bool]
DocumentData = dict[str, Any]


@dataclass
class CLIArguments:
    """Structured configuration for CLI arguments."""

    csv_file: str
    batch_size: int
    min_quality: str
    include_noise: bool
    dry_run: bool
    skip_index_creation: bool
    recreate_index: bool
    verbose: bool
    quiet: bool
    sample_size: int | None


# ==================================================================================================
# CLI ARGUMENT PARSING
# ==================================================================================================


def parse_arguments() -> CLIArguments:
    """Parse command line arguments and return structured configuration."""
    parser = argparse.ArgumentParser(
        description="Process and upload social media data to Azure AI Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python social_media_upload.py

  # Dry run with verbose output
  python social_media_upload.py --dry-run --verbose

  # Fast dry run with sample data
  python social_media_upload.py --dry-run --sample 50

  # Process with custom file and quality filtering
  python social_media_upload.py --csv-file "custom.csv" --min-quality high

  # Recreate index and include noise-level content
  python social_media_upload.py --recreate-index --include-noise
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview processing without uploading to Azure",
    )
    parser.add_argument(
        "--csv-file",
        default=DEFAULT_CSV_FILE_PATH,
        help=f"Override default CSV file path (default: {DEFAULT_CSV_FILE_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Control batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-quality",
        choices=["high", "medium", "low"],
        default="low",
        help="Set minimum quality threshold (default: low)",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include noise-level content (overrides default filtering)",
    )
    parser.add_argument(
        "--skip-index-creation",
        action="store_true",
        help="Skip index creation (assume exists)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the search index",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Process only the first N rows for faster testing (useful with --dry-run)",
    )

    # Logging level arguments (mutually exclusive)
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed logging",
    )
    logging_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimize output (errors only)",
    )

    args = parser.parse_args()

    return CLIArguments(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        min_quality=args.min_quality,
        include_noise=args.include_noise,
        dry_run=args.dry_run,
        skip_index_creation=args.skip_index_creation,
        recreate_index=args.recreate_index,
        verbose=args.verbose,
        quiet=args.quiet,
        sample_size=args.sample,
    )


def log_message(message: str, config: CLIArguments, level: str = "info") -> None:
    """Print message based on logging configuration."""
    if config.quiet and level != "error":
        return

    if level == "debug" and not config.verbose:
        return

    print(message)


# ==================================================================================================
# HELPER FUNCTIONS
# ==================================================================================================


def should_filter_document(quality_score: str, config: CLIArguments) -> bool:
    """Determine if a document should be filtered based on quality and CLI configuration."""
    # Always filter noise unless explicitly included
    if quality_score == "noise" and not config.include_noise:
        return True

    # Apply minimum quality threshold
    quality_hierarchy = {"low": 0, "medium": 1, "high": 2}
    min_level = quality_hierarchy.get(config.min_quality, 0)
    doc_level = quality_hierarchy.get(quality_score, -1)

    # Filter if document quality is below minimum (but always allow noise if include_noise is True)
    if quality_score != "noise" and doc_level < min_level:
        return True

    return False


def check_document_exists(search_client: SearchClient, message_id: str) -> bool:
    """Check if a document already exists in Azure AI Search."""
    try:
        results = search_client.search(
            search_text="*",
            filter=f"messageId eq '{message_id}'",
            select=["messageId"],
            top=1,
        )
        return len(list(results)) > 0
    except Exception:
        return False


# ==================================================================================================
# STEP 1: CREATE THE AZURE AI SEARCH INDEX
# ==================================================================================================
def create_search_index(config: CLIArguments) -> None:
    """
    Defines and creates the Azure AI Search index with a schema tailored for the social media data.
    Includes scoring profiles for engagement, freshness, location, themes, and quality-relevance.
    The quality-relevance profile prioritizes high-quality content and friction indicators.
    Includes semantic configuration for improved search relevance.
    """
    if config.recreate_index:
        log_message(f"Deleting existing index '{AZURE_SEARCH_INDEX_NAME}'...", config)
        index_client_temp = SearchIndexClient(
            AZURE_SEARCH_SERVICE_ENDPOINT, SEARCH_CREDENTIAL
        )
        try:
            index_client_temp.delete_index(AZURE_SEARCH_INDEX_NAME)
            log_message(
                f"Index '{AZURE_SEARCH_INDEX_NAME}' deleted successfully.", config
            )
        except Exception as e:
            log_message(f"Index deletion failed (may not exist): {e}", config, "debug")

    log_message(f"Creating or updating index '{AZURE_SEARCH_INDEX_NAME}'...", config)
    index_client = SearchIndexClient(AZURE_SEARCH_SERVICE_ENDPOINT, SEARCH_CREDENTIAL)

    fields = [
        SimpleField(name="messageId", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="message",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name="messageVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_MODEL_DIMENSIONS,
            vector_search_profile_name="my-hnsw-profile",
        ),
        SimpleField(
            name="date",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
            retrievable=True,
        ),
        SimpleField(
            name="network",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SimpleField(
            name="sentiment",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SimpleField(
            name="likes",
            type=SearchFieldDataType.Int64,
            filterable=True,
            sortable=True,
            retrievable=True,
        ),
        SimpleField(
            name="comments",
            type=SearchFieldDataType.Int64,
            filterable=True,
            sortable=True,
            retrievable=True,
        ),
        SimpleField(
            name="shares",
            type=SearchFieldDataType.Int64,
            filterable=True,
            sortable=True,
            retrievable=True,
        ),
        SimpleField(
            name="potentialImpressions",
            type=SearchFieldDataType.Int64,
            filterable=True,
            sortable=True,
            retrievable=True,
        ),
        SearchField(
            name="theme",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="hashtags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="location",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SimpleField(
            name="language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SearchableField(
            name="profileName",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SimpleField(
            name="messageUrl", type=SearchFieldDataType.String, retrievable=True
        ),
        SimpleField(
            name="sourceName",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        # Fields for Named Entity Recognition (NER)
        SearchField(
            name="extractedLocations",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="extractedProducts",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="extractedOrganizations",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="keyPhrases",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        # Quality Analysis Fields
        SimpleField(
            name="qualityScore",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SearchField(
            name="frictionCategories",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="businessRelevance",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SearchableField(
            name="analysisReason",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
    ]

    # Configure vector search
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="my-hnsw-profile",
                algorithm_configuration_name="my-hnsw-config",
                vectorizer_name="social-media-vectorizer",
            )
        ],
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="social-media-vectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=AZURE_OPENAI_ENDPOINT,
                    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    api_key=AZURE_OPENAI_API_KEY,
                    model_name="text-embedding-3-small",
                ),
            )
        ],
    )

    # Configure scoring profiles for relevance boosting
    scoring_profiles = [
        # 1. Social Engagement - boost posts by engagement metrics
        ScoringProfile(
            name="social-engagement",
            functions=[
                MagnitudeScoringFunction(
                    field_name="likes",
                    boost=2.0,
                    parameters=MagnitudeScoringParameters(
                        boosting_range_start=0,
                        boosting_range_end=10000,
                        should_boost_beyond_range_by_constant=True,
                    ),
                ),
                MagnitudeScoringFunction(
                    field_name="comments",
                    boost=3.0,
                    parameters=MagnitudeScoringParameters(
                        boosting_range_start=0,
                        boosting_range_end=1000,
                        should_boost_beyond_range_by_constant=True,
                    ),
                ),
                MagnitudeScoringFunction(
                    field_name="shares",
                    boost=4.0,
                    parameters=MagnitudeScoringParameters(
                        boosting_range_start=0,
                        boosting_range_end=1000,
                        should_boost_beyond_range_by_constant=True,
                    ),
                ),
                MagnitudeScoringFunction(
                    field_name="potentialImpressions",
                    boost=1.5,
                    parameters=MagnitudeScoringParameters(
                        boosting_range_start=0,
                        boosting_range_end=100000,
                        should_boost_beyond_range_by_constant=True,
                    ),
                ),
            ],
        ),
        # 2. Social Freshness - boost newer posts
        ScoringProfile(
            name="social-freshness",
            functions=[
                FreshnessScoringFunction(
                    field_name="date",
                    boost=2.0,  # Positive boost for linear interpolation - newer content gets higher scores
                    interpolation="linear",
                    parameters=FreshnessScoringParameters(
                        boosting_duration=timedelta(days=90)  # 90 days
                    ),
                ),
            ],
        ),
        # 3. Social Location - boost posts by location relevance
        ScoringProfile(
            name="social-location",
            functions=[
                TagScoringFunction(
                    field_name="extractedLocations",
                    boost=3.0,
                    parameters=TagScoringParameters(tags_parameter="targetLoc"),
                ),
            ],
        ),
        # 4. Social Themes - boost posts by theme/hashtag relevance with weighted text fields
        ScoringProfile(
            name="social-themes",
            text_weights=TextWeights(
                weights={
                    "message": 2.0,
                    "theme": 3.0,
                    "hashtags": 4.0,
                }
            ),
            functions=[
                TagScoringFunction(
                    field_name="hashtags",
                    boost=2.5,
                    parameters=TagScoringParameters(tags_parameter="targetTags"),
                ),
            ],
        ),
        # 5. Quality-Relevance - boost posts by quality indicators and friction categories
        # This profile prioritizes high-quality content and messages indicating customer friction
        # Quality levels: "high" gets highest boost, "medium" gets moderate boost, "low" gets minimal boost
        # Friction indicators help surface customer pain points and service issues
        ScoringProfile(
            name="quality-relevance",
            functions=[
                # Boost based on quality score - higher quality content should rank higher
                TagScoringFunction(
                    field_name="qualityScore",
                    boost=3.5,  # Strong boost for quality content
                    parameters=TagScoringParameters(
                        tags_parameter="highQuality"  # Boosts documents tagged as "high" quality
                    ),
                ),
                TagScoringFunction(
                    field_name="qualityScore",
                    boost=2.0,  # Medium boost for medium quality
                    parameters=TagScoringParameters(tags_parameter="medQuality"),
                ),
                TagScoringFunction(
                    field_name="qualityScore",
                    boost=0.5,  # Minimal boost for low quality
                    parameters=TagScoringParameters(tags_parameter="lowQuality"),
                ),
                # Boost friction-related content to surface customer pain points
                TagScoringFunction(
                    field_name="frictionCategories",
                    boost=2.5,  # Boost any content with friction indicators
                    parameters=TagScoringParameters(
                        tags_parameter="frictionType"  # Parameter name for friction categories
                    ),
                ),
            ],
        ),
    ]

    # Configure semantic search configuration
    semantic_config = SemanticConfiguration(
        name="social-media-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="message"),
            content_fields=[
                SemanticField(field_name="message"),
                SemanticField(field_name="profileName"),
            ],
            keywords_fields=[
                SemanticField(field_name="hashtags"),
                SemanticField(field_name="theme"),
                SemanticField(field_name="extractedLocations"),
                SemanticField(field_name="extractedProducts"),
                SemanticField(field_name="extractedOrganizations"),
                SemanticField(field_name="keyPhrases"),
                # Quality-related fields for enhanced semantic search
                SemanticField(field_name="qualityScore"),
                SemanticField(field_name="frictionCategories"),
                SemanticField(field_name="businessRelevance"),
            ],
        ),
    )

    # Create semantic search with default configuration
    semantic_search = SemanticSearch(
        configurations=[semantic_config],
        default_configuration_name="social-media-semantic",
    )

    # Create the index with all configurations
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        scoring_profiles=scoring_profiles,
        semantic_search=semantic_search,
    )
    result = index_client.create_or_update_index(index)
    log_message(f"Index '{result.name}' created or updated successfully.", config)
    log_message(
        "Added scoring profiles: social-engagement, social-freshness, social-location, social-themes, quality-relevance",
        config,
        "debug",
    )
    log_message(
        "Added semantic configuration: social-media-semantic (default)", config, "debug"
    )


# ==================================================================================================
# STEP 2: PROCESS AND UPLOAD DATA
# ==================================================================================================
def _extract_entities_by_category(ner_result: NERResult, category: str) -> list[str]:
    """Extract entities of a specific category from NER results."""
    if ner_result.get("is_error", True):
        return []

    entities = ner_result.get("entities", [])
    if not isinstance(entities, list):
        return []

    return [
        e["text"]
        for e in entities
        if isinstance(e, dict) and e.get("category") == category
    ]


def process_and_upload_data(config: CLIArguments) -> None:
    """
    Reads the CSV, processes data in batches to add embeddings and NER, and uploads to the index.
    In dry-run mode, skips Azure API calls for significant performance improvement.
    """
    start_time = time.time()

    if config.dry_run:
        log_message("\n🏃 DRY RUN: Fast preview mode (no Azure API calls)...", config)
        log_message(
            "⚡ Performance optimized for speed - showing realistic preview...", config
        )
    else:
        log_message("\nProcessing and uploading data...", config)

    # Initialize clients only if not in dry-run mode
    openai_client: AzureOpenAI | None = None
    text_analytics_client: TextAnalyticsClient | None = None
    search_client: SearchClient | None = None

    if not config.dry_run:
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        text_analytics_client = TextAnalyticsClient(
            endpoint=AZURE_LANGUAGE_ENDPOINT, credential=TEXT_ANALYTICS_CREDENTIAL
        )
        search_client = SearchClient(
            AZURE_SEARCH_SERVICE_ENDPOINT, AZURE_SEARCH_INDEX_NAME, SEARCH_CREDENTIAL
        )

    # Initialize content quality analyzer
    from pathlib import Path

    content_analyzer = SocialMediaContentAnalyzer(Path(config.csv_file))

    # Read and prepare data
    df: pd.DataFrame = pd.read_csv(config.csv_file)

    # Apply sampling if specified
    original_count = len(df)
    if config.sample_size is not None:
        df = df.head(config.sample_size)
        if config.dry_run:
            log_message(
                f"🎯 Fast sampling: Processing {len(df)} documents (out of {original_count} total)",
                config,
            )
            log_message(
                f"📊 This represents {len(df) / original_count * 100:.1f}% of your full dataset",
                config,
            )
        else:
            log_message(
                f"📊 Sampling first {len(df)} documents (out of {original_count} total)",
                config,
            )

    # Use Message ID as the key, falling back to index for uniqueness
    df["messageId"] = df["Message ID"].fillna(
        pd.Series(df.index.astype(str), index=df.index)
    )
    df = df.where(pd.notna(df), None)  # Replace NaN with None for JSON compatibility

    if config.dry_run:
        log_message(f"🚀 Processing {len(df)} documents in DRY RUN mode...", config)
        log_message(
            f"📋 Batch size: {config.batch_size} | Quality filter: {config.min_quality}+ | Include noise: {config.include_noise}",
            config,
        )
    else:
        log_message(f"Processing {len(df)} documents...", config)

    # Track overall statistics for dry-run summary
    total_quality_distribution = {"high": 0, "medium": 0, "low": 0, "noise": 0}
    total_documents_processed = 0
    total_documents_would_upload = 0

    # Process in batches
    for i in range(0, len(df), config.batch_size):
        batch_df = df[i : i + config.batch_size]
        documents_to_upload: list[DocumentData] = []

        log_message(f"Processing batch {i // config.batch_size + 1}...", config)

        # Collect messages for batch operations
        messages: list[str] = batch_df["Message"].astype(str).tolist()

        # Initialize result containers
        mock_ner_results: list[NERResult] = []
        api_ner_results: list[NERResult] = []
        mock_kp_results: list[KeyPhraseResult] = []
        api_kp_results: list[KeyPhraseResult] = []

        # --- Generate embeddings ---
        if config.dry_run:
            log_message(
                f"  🚀 MOCK: Using mock embeddings for {len(messages)} messages...",
                config,
                "debug",
            )
            # Use mock vectors with realistic dimensions for dry-run preview
            import random

            random.seed(42)  # Consistent mocking for reproducible results
            embeddings = [
                [random.uniform(-1.0, 1.0) for _ in range(EMBEDDING_MODEL_DIMENSIONS)]
                for _ in messages
            ]
        else:
            log_message(
                f"  Generating embeddings for {len(messages)} messages...",
                config,
                "debug",
            )
            assert openai_client is not None  # Type guard
            embedding_response = openai_client.embeddings.create(
                input=messages, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            embeddings = [emb.embedding for emb in embedding_response.data]

        # --- Process NER ---
        if config.dry_run:
            log_message(
                f"  🚀 MOCK: Using mock NER for {len(messages)} messages...",
                config,
                "debug",
            )
            # Generate more realistic mock NER results for dry-run

            # More comprehensive mock entities reflecting Farm and Fleet context
            mock_entities_pool = {
                "Organization": [
                    "Farm and Fleet",
                    "Tractor Supply Co",
                    "Rural King",
                    "Fleet Farm",
                    "Blain's",
                    "Home Depot",
                    "Menards",
                ],
                "Location": [
                    "Wisconsin",
                    "Iowa",
                    "Illinois",
                    "Minnesota",
                    "Milwaukee",
                    "Madison",
                    "Cedar Rapids",
                    "Des Moines",
                ],
                "Product": [
                    "lawn mower",
                    "work boots",
                    "fertilizer",
                    "deer feed",
                    "hunting gear",
                    "hardware",
                    "pet food",
                    "automotive parts",
                ],
            }

            import random

            random.seed(42)  # Consistent results

            for _i, _message in enumerate(messages):
                # Generate 0-3 entities per message based on message content patterns
                num_entities = random.randint(0, 3)
                entities = []

                for _ in range(num_entities):
                    category = random.choice(list(mock_entities_pool.keys()))
                    entity_text = random.choice(mock_entities_pool[category])
                    entities.append({"text": entity_text, "category": category})

                mock_ner_result: NERResult = {
                    "entities": entities,
                    "is_error": False,
                }
                mock_ner_results.append(mock_ner_result)
        else:
            log_message(
                f"  Processing NER for {len(messages)} messages...", config, "debug"
            )
            assert text_analytics_client is not None  # Type guard
            ner_results = text_analytics_client.recognize_entities(documents=messages)
            for result in ner_results:
                if not result.is_error:
                    api_ner_result: NERResult = {
                        "entities": [
                            {"text": e.text, "category": e.category}
                            for e in result.entities
                        ],
                        "is_error": False,
                    }
                else:
                    api_ner_result = {
                        "entities": [],
                        "is_error": True,
                    }
                api_ner_results.append(api_ner_result)

        # --- Process key phrases ---
        if config.dry_run:
            log_message(
                f"  🚀 MOCK: Using mock key phrases for {len(messages)} messages...",
                config,
                "debug",
            )
            # Generate more realistic mock key phrases for dry-run

            # Comprehensive key phrase pools reflecting Farm and Fleet business context
            key_phrase_pools = {
                "service": [
                    "customer service",
                    "store experience",
                    "staff helpful",
                    "checkout process",
                    "return policy",
                ],
                "product": [
                    "product quality",
                    "product availability",
                    "inventory",
                    "stock levels",
                    "selection",
                ],
                "pricing": [
                    "competitive price",
                    "good value",
                    "reasonable cost",
                    "price match",
                    "affordable",
                ],
                "location": [
                    "store location",
                    "parking",
                    "hours",
                    "convenient",
                    "accessibility",
                ],
                "experience": [
                    "shopping experience",
                    "store layout",
                    "clean store",
                    "organized",
                    "easy to find",
                ],
            }

            import random

            random.seed(42)  # Consistent results

            for _i, _message in enumerate(messages):
                # Generate 2-4 key phrases per message
                num_phrases = random.randint(2, 4)
                selected_phrases = []

                # Mix phrases from different categories
                available_categories = list(key_phrase_pools.keys())
                for _ in range(num_phrases):
                    category = random.choice(available_categories)
                    phrase = random.choice(key_phrase_pools[category])
                    if phrase not in selected_phrases:  # Avoid duplicates
                        selected_phrases.append(phrase)

                mock_kp_result: KeyPhraseResult = {
                    "key_phrases": selected_phrases,
                    "is_error": False,
                }
                mock_kp_results.append(mock_kp_result)
        else:
            log_message(
                f"  Processing key phrases for {len(messages)} messages...",
                config,
                "debug",
            )
            assert text_analytics_client is not None  # Type guard
            key_phrase_results = text_analytics_client.extract_key_phrases(
                documents=messages
            )
            for result in key_phrase_results:
                if not result.is_error:
                    api_kp_result: KeyPhraseResult = {
                        "key_phrases": list(result.key_phrases),
                        "is_error": False,
                    }
                else:
                    api_kp_result = {
                        "key_phrases": [],
                        "is_error": True,
                    }
                api_kp_results.append(api_kp_result)

        # --- Process content quality analysis ---
        # This runs in both dry-run and normal mode as it's local processing
        log_message(
            f"  Analyzing content quality for {len(messages)} messages...",
            config,
            "debug",
        )
        quality_analyses: list[MessageAnalysis] = []
        quality_distribution: dict[str, int] = {
            "high": 0,
            "medium": 0,
            "low": 0,
            "noise": 0,
        }

        for batch_idx, message in enumerate(messages):
            # Get corresponding row data for additional context
            row_data = batch_df.iloc[batch_idx]
            theme = (
                str(row_data.get("Theme", ""))
                if row_data.get("Theme") is not None
                else ""
            )
            network = (
                str(row_data.get("Network", ""))
                if row_data.get("Network") is not None
                else ""
            )
            sentiment = (
                str(row_data.get("Sentiment", ""))
                if row_data.get("Sentiment") is not None
                else ""
            )

            analysis = content_analyzer.analyze_message_quality(
                message=message, theme=theme, network=network, sentiment=sentiment
            )
            quality_analyses.append(analysis)
            quality_distribution[analysis.quality_score] += 1
            total_quality_distribution[analysis.quality_score] += 1

        total_documents_processed += len(messages)

        # Log quality distribution for this batch
        batch_num = i // config.batch_size + 1
        total_batch = len(messages)

        if config.dry_run and config.verbose:
            # Enhanced dry-run batch reporting
            log_message(f"  📋 Batch {batch_num} Analysis Summary:", config)
            log_message(
                f"    ✅ High Quality: {quality_distribution['high']} ({quality_distribution['high'] / total_batch * 100:.1f}%) - Actionable insights",
                config,
            )
            log_message(
                f"    🟡 Medium Quality: {quality_distribution['medium']} ({quality_distribution['medium'] / total_batch * 100:.1f}%) - Business relevant",
                config,
            )
            log_message(
                f"    🟠 Low Quality: {quality_distribution['low']} ({quality_distribution['low'] / total_batch * 100:.1f}%) - Tangential mentions",
                config,
            )
            log_message(
                f"    🔴 Noise: {quality_distribution['noise']} ({quality_distribution['noise'] / total_batch * 100:.1f}%) - Would be filtered",
                config,
            )
        elif not config.dry_run:
            log_message(f"  Batch {batch_num} quality distribution:", config, "debug")
            log_message(
                f"    High: {quality_distribution['high']} ({quality_distribution['high'] / total_batch * 100:.1f}%)",
                config,
                "debug",
            )
            log_message(
                f"    Medium: {quality_distribution['medium']} ({quality_distribution['medium'] / total_batch * 100:.1f}%)",
                config,
                "debug",
            )
            log_message(
                f"    Low: {quality_distribution['low']} ({quality_distribution['low'] / total_batch * 100:.1f}%)",
                config,
                "debug",
            )
            log_message(
                f"    Noise: {quality_distribution['noise']} ({quality_distribution['noise'] / total_batch * 100:.1f}%)",
                config,
                "debug",
            )

        # Build documents for upload (filtering based on quality and CLI config)
        for batch_idx, (_, row) in enumerate(batch_df.iterrows()):
            message_id = str(row["messageId"])

            # Get processed results
            embedding = embeddings[batch_idx] if batch_idx < len(embeddings) else []
            # Use appropriate results based on dry-run mode
            if config.dry_run:
                ner_result = (
                    mock_ner_results[batch_idx]
                    if batch_idx < len(mock_ner_results)
                    else {"entities": [], "is_error": True}
                )
            else:
                ner_result = (
                    api_ner_results[batch_idx]
                    if batch_idx < len(api_ner_results)
                    else {"entities": [], "is_error": True}
                )
            if config.dry_run:
                key_phrase_result = (
                    mock_kp_results[batch_idx]
                    if batch_idx < len(mock_kp_results)
                    else {"key_phrases": [], "is_error": True}
                )
            else:
                key_phrase_result = (
                    api_kp_results[batch_idx]
                    if batch_idx < len(api_kp_results)
                    else {"key_phrases": [], "is_error": True}
                )
            quality_analysis = (
                quality_analyses[batch_idx]
                if batch_idx < len(quality_analyses)
                else MessageAnalysis(
                    message="",
                    quality_score="low",
                    friction_indicators=[],
                    business_context="unknown",
                    reason="Analysis unavailable",
                    theme="",
                    network="",
                    sentiment="",
                )
            )

            # Apply quality and noise filtering based on CLI configuration
            if should_filter_document(quality_analysis.quality_score, config):
                continue

            # Parse date, handling potential errors
            date_obj: str | None = None
            try:
                if row["Date"] is not None:
                    date_obj = (
                        datetime.strptime(
                            str(row["Date"]), "%m/%d/%Y %I:%M %p"
                        ).isoformat()
                        + "Z"
                    )
            except (ValueError, TypeError):
                date_obj = None

            # Helper function to safely convert to int
            def safe_int(value: Any) -> int:
                try:
                    return int(value) if value is not None else 0
                except (ValueError, TypeError):
                    return 0

            # Helper function to safely split strings
            def safe_split(value: Any, delimiter: str = ",") -> list[str]:
                if value is None or str(value).strip() == "":
                    return []
                return [
                    item.strip() for item in str(value).split(delimiter) if item.strip()
                ]

            doc: DocumentData = {
                "messageId": message_id,
                "message": str(row["Message"]) if row["Message"] is not None else "",
                "messageVector": embedding,
                "date": date_obj,
                "network": str(row["Network"]) if row["Network"] is not None else "",
                "sentiment": str(row["Sentiment"])
                if row["Sentiment"] is not None
                else "",
                "likes": safe_int(row["Likes"]),
                "comments": safe_int(row["Comments"]),
                "shares": safe_int(row["Shares"]),
                "potentialImpressions": safe_int(row["Potential Impressions"]),
                "theme": safe_split(row["Theme"]),
                "hashtags": safe_split(row["Hashtags"]),
                "location": str(row["Location"]) if row["Location"] is not None else "",
                "language": str(row["Language"]) if row["Language"] is not None else "",
                "profileName": str(row["Profile"])
                if row["Profile"] is not None
                else "",
                "messageUrl": str(row["Message URL"])
                if row["Message URL"] is not None
                else "",
                "sourceName": str(row["Source Name"])
                if row["Source Name"] is not None
                else "",
                "extractedLocations": _extract_entities_by_category(
                    ner_result, "Location"
                ),
                "extractedProducts": _extract_entities_by_category(
                    ner_result, "Product"
                ),
                "extractedOrganizations": _extract_entities_by_category(
                    ner_result, "Organization"
                ),
                "keyPhrases": (
                    key_phrase_result.get("key_phrases", [])
                    if not key_phrase_result.get("is_error", True)
                    else []
                ),
                # Quality Analysis Fields
                "qualityScore": quality_analysis.quality_score,
                "frictionCategories": quality_analysis.friction_indicators,
                "businessRelevance": quality_analysis.business_context,
                "analysisReason": quality_analysis.reason,
            }
            documents_to_upload.append(doc)
            total_documents_would_upload += 1

        # --- Upload the batch ---
        if documents_to_upload:
            # Debug: Print detailed structure of first document
            log_message("  Debug - First document structure:", config, "debug")
            for key, value in documents_to_upload[0].items():
                if isinstance(value, list):
                    if len(value) > 0:
                        log_message(
                            f"    {key}: {type(value)} with {len(value)} items - first item type: {type(value[0])}",
                            config,
                            "debug",
                        )
                        if isinstance(value[0], dict | list):
                            log_message(
                                f"      First item: {repr(value[0])[:100]}{'...' if len(repr(value[0])) > 100 else ''}",
                                config,
                                "debug",
                            )
                        else:
                            log_message(
                                f"      Sample items: {value[:3]}", config, "debug"
                            )
                    else:
                        log_message(
                            f"    {key}: {type(value)} - empty list", config, "debug"
                        )
                else:
                    log_message(
                        f"    {key}: {type(value)} = {repr(value)[:100]}{'...' if len(repr(value)) > 100 else ''}",
                        config,
                        "debug",
                    )
            if config.verbose:
                log_message("", config)

            try:
                if config.dry_run:
                    log_message(
                        f"  DRY RUN: Would upload batch of {len(documents_to_upload)} documents.",
                        config,
                    )
                else:
                    assert search_client is not None  # Type guard
                    result = search_client.upload_documents(
                        documents=documents_to_upload
                    )
                    log_message(f"  Uploaded batch of {len(result)} documents.", config)

            except Exception as e:
                if not config.dry_run:
                    log_message(f"  Upload failed: {e}", config, "error")
                # Try individual uploads to identify issues
                for doc_idx, doc in enumerate(documents_to_upload):
                    try:
                        if not config.dry_run:
                            assert search_client is not None  # Type guard
                            _ = search_client.upload_documents(documents=[doc])
                            log_message(
                                f"    Document {doc_idx} uploaded successfully",
                                config,
                                "debug",
                            )
                    except Exception as single_error:
                        if not config.dry_run:
                            log_message(
                                f"    Document {doc_idx} failed: {single_error}",
                                config,
                                "error",
                            )
                            log_message("    Problematic document:", config, "error")
                            log_message(
                                f"    Raw: {repr(doc)[:500]}...", config, "error"
                            )
                        return  # Stop after first failure to see the exact issue

    # Calculate performance metrics
    end_time = time.time()
    processing_time = end_time - start_time

    if config.dry_run:
        log_message(
            f"\n✨ DRY RUN: Fast preview completed in {processing_time:.2f} seconds!",
            config,
        )
        log_message("🚀 No uploads performed - Azure costs avoided.", config)

        # Comprehensive dry-run summary
        log_message("\n" + "=" * 60, config)
        log_message("📊 DRY RUN PROCESSING SUMMARY", config)
        log_message("=" * 60, config)

        # Document processing stats
        log_message(f"📁 Documents Processed: {total_documents_processed:,}", config)
        log_message(
            f"📤 Documents Would Upload: {total_documents_would_upload:,} ({total_documents_would_upload / total_documents_processed * 100:.1f}%)",
            config,
        )
        filtered_count = total_documents_processed - total_documents_would_upload
        log_message(
            f"🚫 Documents Filtered Out: {filtered_count:,} ({filtered_count / total_documents_processed * 100:.1f}%)",
            config,
        )

        # Quality distribution
        log_message("\n🏆 OVERALL QUALITY DISTRIBUTION:", config)
        log_message(
            f"  ✅ High Quality (Actionable): {total_quality_distribution['high']:,} ({total_quality_distribution['high'] / total_documents_processed * 100:.1f}%)",
            config,
        )
        log_message(
            f"  🟡 Medium Quality (Business): {total_quality_distribution['medium']:,} ({total_quality_distribution['medium'] / total_documents_processed * 100:.1f}%)",
            config,
        )
        log_message(
            f"  🟠 Low Quality (Tangential): {total_quality_distribution['low']:,} ({total_quality_distribution['low'] / total_documents_processed * 100:.1f}%)",
            config,
        )
        log_message(
            f"  🔴 Noise (Irrelevant): {total_quality_distribution['noise']:,} ({total_quality_distribution['noise'] / total_documents_processed * 100:.1f}%)",
            config,
        )

        # Business relevance metrics
        business_relevant = (
            total_quality_distribution["high"] + total_quality_distribution["medium"]
        )
        log_message("\n🎯 BUSINESS METRICS:", config)
        log_message(
            f"  📊 Business Relevant Content: {business_relevant:,} ({business_relevant / total_documents_processed * 100:.1f}%)",
            config,
        )
        log_message(
            f"  🔍 Friction Analysis Ready: {total_quality_distribution['high']:,} ({total_quality_distribution['high'] / total_documents_processed * 100:.1f}%)",
            config,
        )

        # Performance metrics
        log_message("\n⚡ PERFORMANCE METRICS:", config)
        log_message(f"  🕒 Processing Time: {processing_time:.2f} seconds", config)
        log_message(
            f"  🚀 Processing Speed: {total_documents_processed / processing_time:.1f} docs/second",
            config,
        )

        # Show performance comparison estimate
        if config.sample_size:
            estimated_full_time = processing_time * (
                original_count / config.sample_size
            )
        else:
            estimated_full_time = processing_time  # Already processed all data

        estimated_api_time = (
            estimated_full_time * 8
        )  # Conservative estimate of API overhead
        log_message(
            f"  🔥 Estimated time with Azure APIs: ~{estimated_api_time:.1f} seconds",
            config,
        )
        log_message(
            f"  🏃 Speed improvement: ~{estimated_api_time / processing_time:.1f}x faster in dry-run",
            config,
        )

        # Cost savings estimate
        if total_documents_processed > 0:
            embedding_calls = (total_documents_processed // config.batch_size) + (
                1 if total_documents_processed % config.batch_size else 0
            )
            estimated_cost_savings = embedding_calls * 0.02  # Rough estimate per batch
            log_message(
                f"  💰 Estimated API cost savings: ~${estimated_cost_savings:.2f}",
                config,
            )

        log_message("\n" + "=" * 60, config)
    else:
        log_message(
            f"\nData processing and uploading completed in {processing_time:.2f} seconds.",
            config,
        )


# ==================================================================================================
# MAIN EXECUTION
# ==================================================================================================
def main() -> None:
    """Main function to orchestrate the data processing and upload process."""
    # Parse command line arguments
    config = parse_arguments()

    if config.dry_run:
        log_message("Starting social media data processing (DRY RUN mode)...", config)
    else:
        log_message("Starting social media data upload process...", config)

    # Create or update search index (unless skipped or in dry-run mode)
    if config.dry_run:
        log_message("⚡ DRY RUN: Skipping index creation for performance.", config)
    elif not config.skip_index_creation:
        create_search_index(config)
    else:
        log_message("Skipping index creation as requested.", config)

    # Process and upload data
    process_and_upload_data(config)

    if config.dry_run:
        log_message("\n✅ DRY RUN process completed successfully!", config)
        log_message(
            "🔄 Ready for production? Run without --dry-run to upload to Azure.", config
        )
        if config.sample_size:
            log_message(
                f"🔍 Want full analysis? Remove --sample {config.sample_size} to process all documents.",
                config,
            )
    else:
        log_message("\nData upload process completed successfully!", config)
        log_message("To query the data, use the social_media_query.py module.", config)


if __name__ == "__main__":
    """
    Main execution point for the social media upload script.

    This script processes social media data from a CSV file and uploads it
    to Azure AI Search with embeddings and NER extraction.
    """
    main()
