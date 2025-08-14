from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    HnswAlgorithmConfiguration,
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

# Simple analysis dataclass for upload processing


@dataclass
class SimpleAnalysis:
    """Simple analysis results for upload processing."""

    quality_score: str = "medium"
    friction_indicators: list[str] | None = None
    business_context: str = "survey_response"
    reason: str = "Survey response with vectorizable content"

    def __post_init__(self):
        if self.friction_indicators is None:
            self.friction_indicators = []


# ==================================================================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================================================================
if dotenv.load_dotenv(override=True):
    print("Environment variables loaded from .env file")

# --- Azure AI Search Configuration ---
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
AZURE_SEARCH_SURVEY_INDEX_NAME = os.environ.get(
    "AZURE_SEARCH_SURVEY_INDEX_NAME", "farmandfleetdemo-surveys"
)
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_SEARCH_API_KEY)

# --- Azure OpenAI Configuration for Embeddings ---
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
EMBEDDING_MODEL_DIMENSIONS = 1536  # text-embedding-3-small


# ==================================================================================================
# SCRIPT CONFIGURATION
# ==================================================================================================
DEFAULT_DATA_DIRECTORY = "data/Qualtrics Surveys/"
DEFAULT_BATCH_SIZE = 5  # Batch size for processing and uploading


# ==================================================================================================
# TYPE DEFINITIONS
# ==================================================================================================
DocumentData = dict[str, Any]
SurveyRecord = dict[str, Any]


# ==================================================================================================
# QUESTION VECTOR CACHING
# ==================================================================================================
class QuestionVectorCache:
    """Cache for question vectors to avoid repeated API calls for identical questions."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0

    def get_vector(self, question: str) -> list[float] | None:
        """Get cached vector for a question, or None if not cached."""
        if question in self._cache:
            self._hits += 1
            return self._cache[question]
        self._misses += 1
        return None

    def set_vector(self, question: str, vector: list[float]) -> None:
        """Cache a vector for a question."""
        self._cache[question] = vector

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "cache_size": len(self._cache),
            "hit_rate": (self._hits / (self._hits + self._misses) * 100)
            if (self._hits + self._misses) > 0
            else 0,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


@dataclass
class CLIArguments:
    """Structured configuration for CLI arguments."""

    data_directory: str
    batch_size: int
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
        description="Process and upload Qualtrics survey data to Azure AI Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python process_qualtrics_surveys.py

  # Dry run with verbose output
  python process_qualtrics_surveys.py --dry-run --verbose

  # Fast dry run with sample data
  python process_qualtrics_surveys.py --dry-run --sample 50

  # Process with custom directory
  python process_qualtrics_surveys.py --data-directory "custom_surveys/"

  # Recreate index with full coverage
  python process_qualtrics_surveys.py --recreate-index
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview processing without uploading to Azure",
    )
    parser.add_argument(
        "--data-directory",
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Override default data directory path (default: {DEFAULT_DATA_DIRECTORY})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Control batch size for processing (default: {DEFAULT_BATCH_SIZE})",
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
        help="Process only the first N records for faster testing (useful with --dry-run)",
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
        data_directory=args.data_directory,
        batch_size=args.batch_size,
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


def has_vectorizable_content(record: dict[str, Any], config: CLIArguments) -> bool:
    """Determine if a document contains vectorizable content (multiple word text responses)."""

    # Always require valid customer_id for document identification
    customer_id = str(record.get("customer_id", "")).strip()
    if not customer_id or customer_id in ["", "None", "null", "nan"]:
        if config.verbose:
            log_message(
                "  Filtering response with invalid customer_id", config, "debug"
            )
        return False

    question_text = str(record.get("question_text", "")).strip()
    feedback_text = str(record.get("feedback_text", "")).strip()

    # Define noise patterns that shouldn't be vectorized
    noise_patterns = [
        r"^.{1,2}$",  # Very short responses (1-2 characters only)
        r"^(asdf|qwerty|test|xyz|abc|123)\.?$",  # Keyboard mashing/test entries
        r"^(lol|haha|wtf|omg)\.?$",  # Pure social media expressions
        r"^[^a-zA-Z]*$",  # Only numbers/symbols, no letters
    ]

    def is_vectorizable_text(text: str) -> bool:
        """Check if text contains vectorizable content (multiple meaningful words)."""
        if not text:
            return False

        # Check for noise patterns
        for pattern in noise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        # Count meaningful words (letters/numbers, not just punctuation)
        words = [word for word in text.split() if re.search(r"[a-zA-Z0-9]", word)]
        return len(words) >= 2

    # Check if either question or feedback contains vectorizable content
    has_vectorizable_question = is_vectorizable_text(question_text)
    has_vectorizable_feedback = is_vectorizable_text(feedback_text)

    if has_vectorizable_question or has_vectorizable_feedback:
        if config.verbose:
            vectorizable_fields = []
            if has_vectorizable_question:
                vectorizable_fields.append(
                    f"question ({len(question_text.split())} words)"
                )
            if has_vectorizable_feedback:
                vectorizable_fields.append(
                    f"feedback ({len(feedback_text.split())} words)"
                )
            log_message(
                f"  Including vectorizable content: {', '.join(vectorizable_fields)}",
                config,
                "debug",
            )
        return True
    else:
        if config.verbose:
            log_message(
                f"  Filtering non-vectorizable content: question='{question_text[:30]}...', feedback='{feedback_text[:30]}...'",
                config,
                "debug",
            )
        return False


def should_filter_document(record: dict[str, Any], config: CLIArguments) -> bool:
    """Determine if a document should be filtered (legacy function - now inverts has_vectorizable_content)."""
    return not has_vectorizable_content(record, config)


def check_document_exists(search_client: SearchClient, document_id: str) -> bool:
    """Check if a document already exists in Azure AI Search."""
    try:
        results = search_client.search(
            search_text="*",
            filter=f"id eq '{document_id}'",
            select=["id"],
            top=1,
        )
        return len(list(results)) > 0
    except Exception:
        return False


def extract_survey_title(source_file: str) -> str:
    """Extract survey title from source_file field by removing date and .csv suffix."""
    # Remove .csv extension if present
    if source_file.endswith(".csv"):
        source_file = source_file[:-4]

    # Find the last occurrence of underscore followed by date pattern
    # Pattern: "Survey Name_July 24, 2025_08.29"
    import re

    date_pattern = r"_[A-Za-z]+ \d{1,2}, \d{4}_\d{2}\.\d{2}$"
    match = re.search(date_pattern, source_file)

    if match:
        return source_file[: match.start()]

    # Fallback: just return the source_file without extension
    return source_file


def convert_date_to_iso(date_string: str) -> str:
    """Convert date string from '6/23/2025 11:54' format to ISO format with Z suffix."""
    try:
        # Handle escaped forward slashes
        date_string = date_string.replace("\\/", "/")
        dt = datetime.strptime(date_string, "%m/%d/%Y %H:%M")
        return dt.isoformat() + "Z"
    except (ValueError, TypeError):
        # If parsing fails, return current timestamp
        return datetime.now().isoformat() + "Z"


def generate_document_id(customer_id: str, source_column: str, source_file: str) -> str:
    """Generate unique document ID using customer_id + source_column + file hash.

    Azure AI Search document keys can only contain letters, digits, underscore (_),
    dash (-), or equal sign (=). This function ensures all components are sanitized.
    """
    import re

    # Sanitize components by replacing invalid characters with underscores
    # Valid chars: letters, digits, underscore (_), dash (-), equal sign (=)
    def sanitize_key_component(text: str) -> str:
        # Replace any character that's not a letter, digit, underscore, dash, or equals with underscore
        return re.sub(r"[^a-zA-Z0-9_\-=]", "_", str(text))

    # Sanitize all components
    clean_customer_id = sanitize_key_component(customer_id)
    clean_source_column = sanitize_key_component(source_column)

    # Create a hash of the source file for uniqueness (hash will only contain valid hex chars)
    file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]

    return f"{clean_customer_id}_{clean_source_column}_{file_hash}"


def load_jsonl_files(directory_path: str) -> list[SurveyRecord]:
    """Load all .jsonl files from the specified directory."""
    records = []
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    jsonl_files = list(directory.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in directory: {directory_path}")

    for file_path in jsonl_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        records.append(record)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

    return records


# ==================================================================================================
# STEP 1: CREATE THE AZURE AI SEARCH INDEX
# ==================================================================================================
def create_search_index(config: CLIArguments) -> None:
    """
    Defines and creates the Azure AI Search index with a schema tailored for survey data.
    Includes scoring profiles for recency, location, quality-relevance, and survey-specific factors.
    The quality-relevance profile prioritizes high-quality responses and friction indicators.
    Includes semantic configuration for improved search relevance.
    """
    if config.recreate_index:
        log_message(
            f"Deleting existing index '{AZURE_SEARCH_SURVEY_INDEX_NAME}'...", config
        )
        index_client_temp = SearchIndexClient(
            AZURE_SEARCH_SERVICE_ENDPOINT, SEARCH_CREDENTIAL
        )
        try:
            index_client_temp.delete_index(AZURE_SEARCH_SURVEY_INDEX_NAME)
            log_message(
                f"Index '{AZURE_SEARCH_SURVEY_INDEX_NAME}' deleted successfully.",
                config,
            )
        except Exception as e:
            log_message(f"Index deletion failed (may not exist): {e}", config, "debug")

    log_message(
        f"Creating or updating index '{AZURE_SEARCH_SURVEY_INDEX_NAME}'...", config
    )
    index_client = SearchIndexClient(AZURE_SEARCH_SERVICE_ENDPOINT, SEARCH_CREDENTIAL)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="survey_title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SimpleField(
            name="responded_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
            retrievable=True,
        ),
        SearchableField(
            name="store_location",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SimpleField(
            name="customer_id",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SearchableField(
            name="question",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name="question_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_MODEL_DIMENSIONS,
            vector_search_profile_name="my-hnsw-profile",
        ),
        SearchableField(
            name="response",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchableField(
            name="searchable_content",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_MODEL_DIMENSIONS,
            vector_search_profile_name="my-hnsw-profile",
        ),
        # Simple metadata fields for basic filtering and organization
        SimpleField(
            name="content_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
    ]

    # Configure vector search
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="my-hnsw-profile",
                algorithm_configuration_name="my-hnsw-config",
                vectorizer_name="survey-vectorizer",
            )
        ],
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="survey-vectorizer",
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
        # 1. Survey Recency - boost recent responses
        ScoringProfile(
            name="survey-recency",
            functions=[
                FreshnessScoringFunction(
                    field_name="responded_at",
                    boost=2.0,  # Positive boost for linear interpolation - newer content gets higher scores
                    interpolation="linear",
                    parameters=FreshnessScoringParameters(
                        boosting_duration=timedelta(days=180)  # 6 months
                    ),
                ),
            ],
        ),
        # 2. Survey Location - boost responses by store location relevance
        ScoringProfile(
            name="survey-location",
            functions=[
                TagScoringFunction(
                    field_name="store_location",
                    boost=2.5,
                    parameters=TagScoringParameters(tags_parameter="targetStore"),
                ),
            ],
        ),
        # 3. Survey Themes - boost responses by survey type and content relevance
        ScoringProfile(
            name="survey-themes",
            text_weights=TextWeights(
                weights={
                    "searchable_content": 2.5,
                    "question": 2.0,
                    "response": 3.0,
                    "survey_title": 1.5,
                }
            ),
            functions=[
                TagScoringFunction(
                    field_name="survey_title",
                    boost=2.0,
                    parameters=TagScoringParameters(tags_parameter="targetSurvey"),
                ),
            ],
        ),
        # 4. Content-Type - boost responses by content classification
        ScoringProfile(
            name="content-type",
            functions=[
                TagScoringFunction(
                    field_name="content_type",
                    boost=2.0,  # Boost by content type for relevance
                    parameters=TagScoringParameters(tags_parameter="contentType"),
                ),
            ],
        ),
    ]

    # Configure semantic search configuration
    semantic_config = SemanticConfiguration(
        name="survey-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="question"),
            content_fields=[
                SemanticField(field_name="searchable_content"),
                SemanticField(field_name="response"),
                SemanticField(field_name="question"),
            ],
            keywords_fields=[
                SemanticField(field_name="survey_title"),
                SemanticField(field_name="store_location"),
                SemanticField(field_name="content_type"),
            ],
        ),
    )

    # Create semantic search with default configuration
    semantic_search = SemanticSearch(
        configurations=[semantic_config],
        default_configuration_name="survey-semantic",
    )

    # Create the index with all configurations
    index = SearchIndex(
        name=AZURE_SEARCH_SURVEY_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        scoring_profiles=scoring_profiles,
        semantic_search=semantic_search,
    )
    result = index_client.create_or_update_index(index)
    log_message(f"Index '{result.name}' created or updated successfully.", config)
    log_message(
        "Added scoring profiles: survey-recency, survey-location, survey-themes, content-type",
        config,
        "debug",
    )
    log_message(
        "Added semantic configuration: survey-semantic (default)", config, "debug"
    )


# ==================================================================================================
# STEP 2: PROCESS AND UPLOAD DATA
# ==================================================================================================
# Removed _extract_entities_by_category function as NER is no longer used


def process_and_upload_data(config: CLIArguments) -> None:
    """
    Reads .jsonl files from the data directory, processes data in batches to add embeddings and NER,
    and uploads to the index. In dry-run mode, skips Azure API calls for significant performance improvement.
    """
    start_time = time.time()

    if config.dry_run:
        log_message("\n🏃 DRY RUN: Fast preview mode (no Azure API calls)...", config)
        log_message(
            "⚡ Performance optimized for speed - showing realistic preview...", config
        )
    else:
        log_message("\nProcessing and uploading survey data...", config)

    # Initialize clients only if not in dry-run mode
    openai_client: AzureOpenAI | None = None
    search_client: SearchClient | None = None

    # Initialize question vector cache (used in both dry-run and normal mode)
    question_cache = QuestionVectorCache()

    if not config.dry_run:
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        search_client = SearchClient(
            AZURE_SEARCH_SERVICE_ENDPOINT,
            AZURE_SEARCH_SURVEY_INDEX_NAME,
            SEARCH_CREDENTIAL,
        )

    # Simple analysis - just mark all vectorizable content as valid survey responses

    # Load data from .jsonl files
    log_message(f"Loading survey data from directory: {config.data_directory}", config)
    records = load_jsonl_files(config.data_directory)

    # Analyze survey type distribution
    survey_distribution = {}
    for record in records:
        survey_title = extract_survey_title(record.get("source_file", ""))
        survey_distribution[survey_title] = survey_distribution.get(survey_title, 0) + 1

    # Display survey type distribution
    log_message(
        f"\n📋 Survey Type Distribution ({len(records):,} total records):", config
    )
    log_message("=" * 60, config)

    def get_count(item: tuple[str, int]) -> int:
        return item[1]

    sorted_surveys = sorted(survey_distribution.items(), key=get_count, reverse=True)
    for survey_title, count in sorted_surveys:
        percentage = (count / len(records) * 100) if records else 0
        log_message(f"  {survey_title}: {count:,} records ({percentage:.1f}%)", config)
    log_message("=" * 60, config)
    log_message(f"📊 Total Survey Types: {len(survey_distribution)}", config)

    # Apply sampling if specified
    original_count = len(records)
    if config.sample_size is not None:
        records = records[: config.sample_size]
        if config.dry_run:
            log_message(
                f"🎯 Fast sampling: Processing {len(records)} records (out of {original_count} total)",
                config,
            )
            log_message(
                f"📊 This represents {len(records) / original_count * 100:.1f}% of your full dataset",
                config,
            )
        else:
            log_message(
                f"📊 Sampling first {len(records)} records (out of {original_count} total)",
                config,
            )

    if config.dry_run:
        log_message(f"🚀 Processing {len(records)} records in DRY RUN mode...", config)
        log_message(
            f"📋 Batch size: {config.batch_size} | Filtering: Only non-vectorizable content excluded (single words, noise, invalid IDs)",
            config,
        )
    else:
        log_message(f"Processing {len(records)} records...", config)

    # Track overall statistics for summary
    total_quality_distribution = {"high": 0, "medium": 0, "low": 0, "noise": 0}
    total_documents_processed = 0
    total_documents_would_upload = 0
    total_documents_uploaded = 0  # Track actual successful uploads
    survey_types_uploaded = {}
    upload_errors = []  # Track any upload failures

    # Process in batches
    for i in range(0, len(records), config.batch_size):
        batch_records = records[i : i + config.batch_size]
        documents_to_upload: list[DocumentData] = []

        batch_num = i // config.batch_size + 1
        total_batches = (len(records) + config.batch_size - 1) // config.batch_size

        # Analyze survey types in this batch for better logging
        batch_survey_types = {}
        for record in batch_records:
            survey_title = extract_survey_title(record.get("source_file", ""))
            batch_survey_types[survey_title] = (
                batch_survey_types.get(survey_title, 0) + 1
            )

        # Enhanced batch logging
        if len(batch_survey_types) == 1:
            survey_name = list(batch_survey_types.keys())[0]
            log_message(
                f"Processing batch {batch_num}/{total_batches} - {survey_name} ({len(batch_records)} records)",
                config,
            )
        else:
            survey_list = ", ".join(
                [f"{name}({count})" for name, count in batch_survey_types.items()]
            )
            log_message(
                f"Processing batch {batch_num}/{total_batches} - Mixed surveys: {survey_list}",
                config,
            )

        # Collect searchable content for batch operations
        searchable_contents: list[str] = []
        for record in batch_records:
            question_text = record.get("question_text", "")
            feedback_text = record.get("feedback_text", "")
            searchable_content = f"{question_text} {feedback_text}".strip()
            searchable_contents.append(searchable_content)

        # Simplified processing - no NER or key phrase extraction needed

        # --- Generate embeddings ---
        if config.dry_run:
            log_message(
                f"  🚀 MOCK: Using mock embeddings for {len(searchable_contents)} records...",
                config,
                "debug",
            )
            # Use mock vectors with realistic dimensions for dry-run preview
            import random

            random.seed(42)  # Consistent mocking for reproducible results
            embeddings = [
                [random.uniform(-1.0, 1.0) for _ in range(EMBEDDING_MODEL_DIMENSIONS)]
                for _ in searchable_contents
            ]
        else:
            log_message(
                f"  Generating embeddings for {len(searchable_contents)} records...",
                config,
                "debug",
            )
            assert openai_client is not None  # Type guard
            embedding_response = openai_client.embeddings.create(
                input=searchable_contents, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            embeddings = [emb.embedding for emb in embedding_response.data]

        # --- Generate question vectors with caching ---
        questions = [record["question_text"] for record in batch_records]
        unique_questions = list(set(questions))  # Get unique questions for this batch

        if config.dry_run:
            log_message(
                f"  🚀 MOCK: Using mock question vectors for {len(unique_questions)} unique questions...",
                config,
                "debug",
            )
            # Use mock vectors for questions in dry-run mode
            import random

            random.seed(42)
            for question in unique_questions:
                if question_cache.get_vector(question) is None:
                    mock_vector = [
                        random.uniform(-1.0, 1.0)
                        for _ in range(EMBEDDING_MODEL_DIMENSIONS)
                    ]
                    question_cache.set_vector(question, mock_vector)
        else:
            # Check cache for questions we haven't seen before
            questions_to_vectorize = [
                q for q in unique_questions if question_cache.get_vector(q) is None
            ]

            if questions_to_vectorize:
                log_message(
                    f"  Generating question vectors for {len(questions_to_vectorize)} new questions (cache: {len(unique_questions) - len(questions_to_vectorize)} hits)...",
                    config,
                    "debug",
                )
                assert openai_client is not None  # Type guard
                question_embedding_response = openai_client.embeddings.create(
                    input=questions_to_vectorize,
                    model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                )

                # Cache the new question vectors
                for i, question in enumerate(questions_to_vectorize):
                    question_cache.set_vector(
                        question, question_embedding_response.data[i].embedding
                    )
            else:
                log_message(
                    f"  All {len(unique_questions)} questions found in cache - no API calls needed!",
                    config,
                    "debug",
                )

        # Build question vectors list matching the batch order
        question_vectors = [question_cache.get_vector(q) for q in questions]

        # Skip NER and key phrase processing - not needed for simplified vectorizable content upload

        # --- Process content quality analysis ---
        # This runs in both dry-run and normal mode as it's local processing
        log_message(
            f"  Analyzing content quality for {len(searchable_contents)} records...",
            config,
            "debug",
        )
        quality_analyses: list[SimpleAnalysis] = []
        quality_distribution: dict[str, int] = {
            "high": 0,
            "medium": 0,
            "low": 0,
            "noise": 0,
        }

        for batch_idx, _record in enumerate(batch_records):
            # Get corresponding searchable content
            searchable_content = searchable_contents[batch_idx]

            # Simple analysis - all vectorizable content is considered valid
            analysis = SimpleAnalysis(
                quality_score="medium",
                friction_indicators=[],
                business_context="survey_response",
                reason="Survey response with vectorizable content",
            )
            quality_analyses.append(analysis)
            quality_distribution[analysis.quality_score] += 1
            total_quality_distribution[analysis.quality_score] += 1

        total_documents_processed += len(batch_records)

        # Log quality distribution for this batch
        batch_num = i // config.batch_size + 1
        total_batch = len(batch_records)

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
                f"    🟠 Low Quality: {quality_distribution['low']} ({quality_distribution['low'] / total_batch * 100:.1f}%) - Basic responses",
                config,
            )
            log_message(
                f"    🔴 Noise: {quality_distribution['noise']} ({quality_distribution['noise'] / total_batch * 100:.1f}%) - Would be filtered",
                config,
            )
        elif not config.dry_run:
            log_message(f"  Batch {batch_num} quality distribution:", config, "debug")
            for quality, count in quality_distribution.items():
                pct = (count / total_batch * 100) if total_batch > 0 else 0
                log_message(
                    f"    {quality.title()}: {count} ({pct:.1f}%)", config, "debug"
                )

        # Build documents for upload (filtering based on vectorizable content)
        for batch_idx, record in enumerate(batch_records):
            # Get processed results
            embedding = embeddings[batch_idx] if batch_idx < len(embeddings) else []
            question_vector = (
                question_vectors[batch_idx] if batch_idx < len(question_vectors) else []
            )
            # Simplified processing - no NER or key phrase results needed

            # Apply vectorizable content filtering (only include multi-word text responses)
            if should_filter_document(record, config):
                continue

            # Generate document ID
            document_id = generate_document_id(
                record.get("customer_id", ""),
                record.get("source_column", ""),
                record.get("source_file", ""),
            )

            # Parse and convert date
            responded_at = convert_date_to_iso(record.get("date", ""))

            # Extract survey title
            survey_title = extract_survey_title(record.get("source_file", ""))

            # Build searchable content
            question_text = record.get("question_text", "")
            feedback_text = record.get("feedback_text", "")
            searchable_content = f"{question_text} {feedback_text}".strip()

            doc: DocumentData = {
                "id": document_id,
                "survey_title": survey_title,
                "responded_at": responded_at,
                "store_location": record.get("store_location"),
                "customer_id": record.get("customer_id", ""),
                "question": question_text,
                "question_vector": question_vector,
                "response": feedback_text,
                "searchable_content": searchable_content,
                "content_vector": embedding,
                "content_type": "survey_response",  # Simple classification for all survey responses
            }
            documents_to_upload.append(doc)
            total_documents_would_upload += 1

            # Track survey types being uploaded
            survey_types_uploaded[survey_title] = (
                survey_types_uploaded.get(survey_title, 0) + 1
            )

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
                    # Show which survey types would be uploaded in this batch
                    batch_survey_summary = {}
                    for doc in documents_to_upload:
                        survey_type = doc.get("survey_title", "Unknown")
                        batch_survey_summary[survey_type] = (
                            batch_survey_summary.get(survey_type, 0) + 1
                        )

                    if len(batch_survey_summary) == 1:
                        survey_name = list(batch_survey_summary.keys())[0]
                        log_message(
                            f"  🚀 DRY RUN: Would upload {len(documents_to_upload)} documents from {survey_name}",
                            config,
                        )
                    else:
                        survey_details = ", ".join(
                            [
                                f"{name}({count})"
                                for name, count in batch_survey_summary.items()
                            ]
                        )
                        log_message(
                            f"  🚀 DRY RUN: Would upload {len(documents_to_upload)} documents from: {survey_details}",
                            config,
                        )
                else:
                    assert search_client is not None  # Type guard
                    result = search_client.upload_documents(
                        documents=documents_to_upload
                    )

                    # Track successful uploads
                    uploaded_count = len(result)
                    total_documents_uploaded += uploaded_count

                    # Show which survey types were uploaded in this batch
                    batch_survey_summary = {}
                    for doc in documents_to_upload:
                        survey_type = doc.get("survey_title", "Unknown")
                        batch_survey_summary[survey_type] = (
                            batch_survey_summary.get(survey_type, 0) + 1
                        )

                    if len(batch_survey_summary) == 1:
                        survey_name = list(batch_survey_summary.keys())[0]
                        log_message(
                            f"  ✅ Uploaded {uploaded_count} documents from {survey_name}",
                            config,
                        )
                    else:
                        survey_details = ", ".join(
                            [
                                f"{name}({count})"
                                for name, count in batch_survey_summary.items()
                            ]
                        )
                        log_message(
                            f"  ✅ Uploaded {uploaded_count} documents from: {survey_details}",
                            config,
                        )

            except Exception as e:
                if not config.dry_run:
                    error_msg = f"Batch {batch_num} upload failed: {e}"
                    log_message(f"  ❌ {error_msg}", config, "error")
                    upload_errors.append(error_msg)

                    # Try individual uploads to identify issues and continue where possible
                    log_message(
                        "  🔄 Attempting individual document uploads...", config
                    )
                    individual_success = 0
                    for doc_idx, doc in enumerate(documents_to_upload):
                        try:
                            assert search_client is not None  # Type guard
                            _ = search_client.upload_documents(documents=[doc])
                            individual_success += 1
                            total_documents_uploaded += 1
                        except Exception as single_error:
                            error_detail = f"Document {doc_idx} in batch {batch_num}: {single_error}"
                            upload_errors.append(error_detail)
                            if config.verbose:
                                log_message(
                                    f"    ❌ Document {doc_idx} failed: {single_error}",
                                    config,
                                    "error",
                                )
                                # Show the problematic document key for debugging
                                doc_id = doc.get("id", "unknown")
                                survey_title = doc.get("survey_title", "unknown")
                                log_message(
                                    f"    📄 Failed document: {doc_id} from {survey_title}",
                                    config,
                                    "error",
                                )

                    if individual_success > 0:
                        log_message(
                            f"  ✅ Recovered {individual_success}/{len(documents_to_upload)} documents via individual upload",
                            config,
                        )
                    else:
                        log_message(
                            f"  ❌ All {len(documents_to_upload)} documents in batch {batch_num} failed",
                            config,
                            "error",
                        )

    # Calculate performance metrics
    end_time = time.time()
    processing_time = end_time - start_time

    # Get cache statistics
    cache_stats = question_cache.get_stats()

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
        log_message(f"📁 Records Processed: {total_documents_processed:,}", config)
        log_message(
            f"📤 Documents Would Upload: {total_documents_would_upload:,} ({total_documents_would_upload / total_documents_processed * 100:.1f}%)",
            config,
        )
        filtered_count = total_documents_processed - total_documents_would_upload
        log_message(
            f"🚫 Non-Vectorizable Content Filtered: {filtered_count:,} ({filtered_count / total_documents_processed * 100:.1f}%)",
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
            f"  🟠 Low Quality (Basic): {total_quality_distribution['low']:,} ({total_quality_distribution['low'] / total_documents_processed * 100:.1f}%)",
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
            f"  🚀 Processing Speed: {total_documents_processed / processing_time:.1f} records/second",
            config,
        )

        # Question vector cache statistics
        log_message("\n🧠 QUESTION VECTOR CACHE PERFORMANCE:", config)
        log_message(
            f"  📝 Unique Questions Cached: {cache_stats['cache_size']}", config
        )
        log_message(f"  ✅ Cache Hits: {cache_stats['hits']}", config)
        log_message(f"  ❌ Cache Misses: {cache_stats['misses']}", config)
        log_message(f"  📊 Hit Rate: {cache_stats['hit_rate']:.1f}%", config)
        if cache_stats["cache_size"] > 0:
            log_message(
                f"  💰 API Calls Saved: {cache_stats['hits']} embedding requests",
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

        # Survey coverage summary
        log_message("\n📋 SURVEY COVERAGE SUMMARY:", config)
        log_message(f"  📊 Survey Types in Dataset: {len(survey_distribution)}", config)
        log_message(
            f"  📤 Survey Types Would Upload: {len(survey_types_uploaded)}", config
        )

        if len(survey_types_uploaded) > 0:
            log_message("\n📈 Survey Types with Upload Coverage:", config)

            def get_upload_count(item: tuple[str, int]) -> int:
                return item[1]

            sorted_uploaded_surveys = sorted(
                survey_types_uploaded.items(), key=get_upload_count, reverse=True
            )
            for survey_title, count in sorted_uploaded_surveys:
                original_count = survey_distribution.get(survey_title, 0)
                coverage_pct = (
                    (count / original_count * 100) if original_count > 0 else 0
                )
                log_message(
                    f"  ✅ {survey_title}: {count:,}/{original_count:,} records ({coverage_pct:.1f}%)",
                    config,
                )

        # Show any survey types not included
        missing_surveys = set(survey_distribution.keys()) - set(
            survey_types_uploaded.keys()
        )
        if missing_surveys:
            log_message(
                "\n⚠️  Survey Types Not Included (all non-vectorizable content):", config
            )
            for survey_title in sorted(missing_surveys):
                count = survey_distribution[survey_title]
                log_message(
                    f"  ❌ {survey_title}: {count:,} records (100% non-vectorizable)",
                    config,
                )

        log_message("\n" + "=" * 60, config)
    else:
        log_message(
            f"\nSurvey data processing and uploading completed in {processing_time:.2f} seconds.",
            config,
        )

        # Production mode comprehensive upload summary
        log_message("\n📋 FINAL UPLOAD VERIFICATION REPORT:", config)
        log_message("=" * 60, config)

        # Overall statistics
        log_message(
            f"📊 Total Records Processed: {total_documents_processed:,}", config
        )
        log_message(
            f"📤 Documents Ready for Upload: {total_documents_would_upload:,}", config
        )
        log_message(
            f"✅ Documents Successfully Uploaded: {total_documents_uploaded:,}", config
        )

        # Upload success rate
        if total_documents_would_upload > 0:
            success_rate = (
                total_documents_uploaded / total_documents_would_upload
            ) * 100
            if success_rate == 100.0:
                log_message(
                    f"🎉 Upload Success Rate: {success_rate:.1f}% - PERFECT!", config
                )
            elif success_rate >= 95.0:
                log_message(
                    f"✅ Upload Success Rate: {success_rate:.1f}% - Excellent", config
                )
            elif success_rate >= 90.0:
                log_message(
                    f"⚠️  Upload Success Rate: {success_rate:.1f}% - Good with some issues",
                    config,
                )
            else:
                log_message(
                    f"❌ Upload Success Rate: {success_rate:.1f}% - SIGNIFICANT ISSUES",
                    config,
                    "error",
                )

        log_message(f"📊 Survey Types in Dataset: {len(survey_distribution)}", config)
        log_message(
            f"📤 Survey Types Successfully Uploaded: {len(survey_types_uploaded)}",
            config,
        )

        # Survey type completeness check
        if len(survey_types_uploaded) == len(survey_distribution):
            log_message("🎉 ALL SURVEY TYPES SUCCESSFULLY UPLOADED!", config)
        else:
            missing_count = len(survey_distribution) - len(survey_types_uploaded)
            log_message(
                f"⚠️  WARNING: {missing_count} survey type(s) missing from upload",
                config,
                "error",
            )

        # Detailed survey coverage
        if len(survey_types_uploaded) > 0:
            log_message("\n📈 Detailed Survey Coverage by Type:", config)

            def get_detailed_count(item: tuple[str, int]) -> int:
                return item[1]

            sorted_detailed_surveys = sorted(
                survey_types_uploaded.items(), key=get_detailed_count, reverse=True
            )
            for survey_title, uploaded_count in sorted_detailed_surveys:
                original_count = survey_distribution.get(survey_title, 0)
                coverage_pct = (
                    (uploaded_count / original_count * 100) if original_count > 0 else 0
                )
                if coverage_pct == 100.0:
                    log_message(
                        f"  🎉 {survey_title}: {uploaded_count:,}/{original_count:,} records (100.0% - COMPLETE)",
                        config,
                    )
                elif coverage_pct >= 90.0:
                    log_message(
                        f"  ✅ {survey_title}: {uploaded_count:,}/{original_count:,} records ({coverage_pct:.1f}% - Excellent)",
                        config,
                    )
                elif coverage_pct >= 50.0:
                    log_message(
                        f"  ⚠️  {survey_title}: {uploaded_count:,}/{original_count:,} records ({coverage_pct:.1f}% - Partial)",
                        config,
                    )
                else:
                    log_message(
                        f"  ❌ {survey_title}: {uploaded_count:,}/{original_count:,} records ({coverage_pct:.1f}% - Low Coverage)",
                        config,
                    )

        # Show any survey types completely missing
        missing_surveys = set(survey_distribution.keys()) - set(
            survey_types_uploaded.keys()
        )
        if missing_surveys:
            log_message(
                "\n❌ SURVEY TYPES COMPLETELY MISSING FROM UPLOAD:", config, "error"
            )
            for survey_title in sorted(missing_surveys):
                count = survey_distribution[survey_title]
                log_message(
                    f"  💥 {survey_title}: {count:,} records (0% uploaded - ALL NON-VECTORIZABLE/FAILED)",
                    config,
                    "error",
                )

        # Error summary
        if upload_errors:
            log_message(
                f"\n⚠️  UPLOAD ERRORS ENCOUNTERED: {len(upload_errors)} errors",
                config,
                "error",
            )
            if config.verbose:
                log_message("\nError Details:", config, "error")
                for i, error in enumerate(
                    upload_errors[:10], 1
                ):  # Show first 10 errors
                    log_message(f"  {i}. {error}", config, "error")
                if len(upload_errors) > 10:
                    log_message(
                        f"  ... and {len(upload_errors) - 10} more errors",
                        config,
                        "error",
                    )
        else:
            log_message("\n✅ NO UPLOAD ERRORS - Clean execution!", config)

        log_message("=" * 60, config)

        # Final status determination
        if (
            len(survey_types_uploaded) == len(survey_distribution)
            and total_documents_uploaded == total_documents_would_upload
        ):
            log_message("🎉 UPLOAD STATUS: COMPLETE SUCCESS", config)
            log_message("   All survey types uploaded with full coverage!", config)
        elif len(survey_types_uploaded) == len(survey_distribution):
            log_message("✅ UPLOAD STATUS: SUCCESS WITH MINOR ISSUES", config)
            log_message(
                "   All survey types included but some documents may have failed",
                config,
            )
        elif len(survey_types_uploaded) > 0:
            log_message("⚠️  UPLOAD STATUS: PARTIAL SUCCESS", config, "error")
            log_message(
                "   Some survey types missing - investigate vectorizable content requirements or upload errors",
                config,
                "error",
            )
        else:
            log_message("❌ UPLOAD STATUS: FAILED", config, "error")
            log_message("   No survey types successfully uploaded!", config, "error")


# ==================================================================================================
# MAIN EXECUTION
# ==================================================================================================
def main() -> None:
    """Main function to orchestrate the survey data processing and upload process."""
    # Parse command line arguments
    config = parse_arguments()

    if config.dry_run:
        log_message(
            "Starting Qualtrics survey data processing (DRY RUN mode)...", config
        )
    else:
        log_message("Starting Qualtrics survey data upload process...", config)

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
                f"🔍 Want full analysis? Remove --sample {config.sample_size} to process all records.",
                config,
            )
    else:
        log_message("\nSurvey data upload process completed successfully!", config)
        log_message("Survey data is now searchable in Azure AI Search.", config)


if __name__ == "__main__":
    """
    Main execution point for the Qualtrics survey processing script.

    This script processes survey data from .jsonl files and uploads it
    to Azure AI Search with embeddings and NER extraction.
    """
    main()
