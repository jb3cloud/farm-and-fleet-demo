#!/usr/bin/env python3
"""
Social Media Content Quality Analysis for Farm and Fleet
========================================================

This script analyzes the Social Media Insights.csv file to assess content noise
relative to Farm and Fleet's friction point discovery goals. It evaluates:

1. Percentage of messages with actual business relevance vs generic mentions
2. Content quality patterns for vectorization and insight generation
3. Noise identification (generic product mentions, unrelated conversations)
4. Specific examples of high vs low quality content
5. Quantification of useful content for friction point analysis

Author: Farm and Fleet Analytics Team
Date: August 2025
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ContentQualityMetrics:
    """Structured data class for content quality analysis results."""

    total_messages: int
    high_quality_count: int
    medium_quality_count: int
    low_quality_count: int
    noise_count: int

    @property
    def high_quality_percentage(self) -> float:
        """Calculate percentage of high-quality messages."""
        return (self.high_quality_count / self.total_messages) * 100

    @property
    def business_relevant_percentage(self) -> float:
        """Calculate percentage of business-relevant messages (high + medium quality)."""
        relevant = self.high_quality_count + self.medium_quality_count
        return (relevant / self.total_messages) * 100

    @property
    def noise_percentage(self) -> float:
        """Calculate percentage of noise/irrelevant messages."""
        return (self.noise_count / self.total_messages) * 100


@dataclass
class MessageAnalysis:
    """Analysis results for a single social media message."""

    message: str
    quality_score: str  # "high", "medium", "low", "noise"
    friction_indicators: list[str]
    business_context: str
    reason: str
    theme: str
    network: str
    sentiment: str


class SocialMediaContentAnalyzer:
    """Analyzes social media content for Farm and Fleet business relevance and quality."""

    def __init__(self, csv_file_path: Path) -> None:
        """Initialize analyzer with path to CSV file.

        Args:
            csv_file_path: Path to the Social Media Insights.csv file
        """
        self.csv_file_path = csv_file_path
        self.df: pd.DataFrame | None = None
        self.analysis_results: list[MessageAnalysis] = []

        # Define patterns for quality assessment
        self.friction_indicators = [
            # Customer service issues
            r"(?i)\b(waited|wait|line|queue|slow|staff|employee|rude|unhelpful|service)\b",
            r"(?i)\b(checkout|register|cashier|customer service)\b",
            # Product availability issues
            r"(?i)\b(out of stock|sold out|empty|unavailable|shortage|limited)\b",
            r"(?i)\b(restock|restocked|inventory|supply)\b",
            # Store operations issues
            r"(?i)\b(closed|hours|open|parking|location|store)\b",
            r"(?i)\b(price|pricing|expensive|cheap|cost|money)\b",
            # Product quality issues
            r"(?i)\b(broken|defective|quality|return|exchange|warranty)\b",
            r"(?i)\b(disappointed|terrible|awful|great|excellent|love)\b",
            # Digital experience issues
            r"(?i)\b(website|app|online|digital|order|delivery|shipping)\b",
        ]

        # Farm and Fleet specific terms for relevance assessment
        self.farm_fleet_terms = [
            r"(?i)\b(farm\s*and?\s*fleet|blain|blains?)\b",
            r"(?i)\b(tractor supply|tsc)\b",  # competitor comparison context
            r"(?i)\b(rural\s+king|fleet\s+farm)\b",  # competitor mentions
        ]

        # Generic/noise patterns
        self.noise_patterns = [
            r"(?i)\b(just\s+mention|just\s+saying|by\s+the\s+way)\b",
            r"(?i)\b(random|whatever|stuff|things)\b",
            r"(?i)^.{1,20}$",  # Very short messages
            r"(?i)\b(lol|haha|emoji|😀|😂)\b",
        ]

    def load_data(self) -> None:
        """Load CSV data into pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.csv_file_path, encoding="utf-8")
            print(f"Loaded {len(self.df)} messages from {self.csv_file_path}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def analyze_message_quality(
        self, message: str, theme: str = "", network: str = "", sentiment: str = ""
    ) -> MessageAnalysis:
        """Analyze individual message for quality and business relevance.

        Args:
            message: The social media message text
            theme: Theme classification from data
            network: Social media network (Reddit, X, Instagram, etc.)
            sentiment: Sentiment classification from data

        Returns:
            MessageAnalysis object with quality assessment
        """
        if pd.isna(message) or not message.strip():
            return MessageAnalysis(
                message="",
                quality_score="noise",
                friction_indicators=[],
                business_context="empty",
                reason="Empty or null message",
                theme=theme,
                network=network,
                sentiment=sentiment,
            )

        # Check for Farm and Fleet relevance
        farm_fleet_mentioned = any(
            re.search(pattern, message) for pattern in self.farm_fleet_terms
        )

        # Check for friction point indicators
        friction_found = []
        for pattern in self.friction_indicators:
            if re.search(pattern, message):
                friction_found.append(pattern)

        # Check for noise patterns
        is_noise = any(re.search(pattern, message) for pattern in self.noise_patterns)

        # Determine quality score
        if is_noise and not farm_fleet_mentioned:
            quality_score = "noise"
            business_context = "irrelevant"
            reason = "Generic mention or noise pattern detected"
        elif not farm_fleet_mentioned:
            quality_score = "low"
            business_context = "tangential"
            reason = "No direct Farm and Fleet context"
        elif friction_found and farm_fleet_mentioned:
            quality_score = "high"
            business_context = "actionable_insight"
            reason = "Direct F&F mention with friction indicators"
        elif farm_fleet_mentioned and len(message.split()) > 10:
            quality_score = "medium"
            business_context = "business_relevant"
            reason = "Direct F&F mention with substantial content"
        else:
            quality_score = "low"
            business_context = "casual_mention"
            reason = "Brief or casual Farm and Fleet mention"

        return MessageAnalysis(
            message=message[:200] + "..." if len(message) > 200 else message,
            quality_score=quality_score,
            friction_indicators=friction_found,
            business_context=business_context,
            reason=reason,
            theme=theme,
            network=network,
            sentiment=sentiment,
        )

    def analyze_all_messages(self) -> ContentQualityMetrics:
        """Analyze all messages in the dataset.

        Returns:
            ContentQualityMetrics with overall statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.analysis_results = []

        for _, row in self.df.iterrows():
            analysis = self.analyze_message_quality(
                message=str(row.get("Message", "")),
                theme=str(row.get("Theme", "")),
                network=str(row.get("Network", "")),
                sentiment=str(row.get("Sentiment", "")),
            )
            self.analysis_results.append(analysis)

        # Count quality categories
        quality_counts = Counter(
            result.quality_score for result in self.analysis_results
        )

        return ContentQualityMetrics(
            total_messages=len(self.analysis_results),
            high_quality_count=quality_counts["high"],
            medium_quality_count=quality_counts["medium"],
            low_quality_count=quality_counts["low"],
            noise_count=quality_counts["noise"],
        )

    def generate_examples_report(
        self, num_examples: int = 5
    ) -> dict[str, list[MessageAnalysis]]:
        """Generate examples of different quality categories.

        Args:
            num_examples: Number of examples per category

        Returns:
            Dictionary with examples for each quality category
        """
        examples: dict[str, list[MessageAnalysis]] = defaultdict(list)

        for analysis in self.analysis_results:
            if len(examples[analysis.quality_score]) < num_examples:
                examples[analysis.quality_score].append(analysis)

        return dict(examples)

    def generate_theme_analysis(self) -> dict[str, dict[str, int]]:
        """Analyze quality distribution by theme.

        Returns:
            Dictionary mapping themes to quality score distributions
        """
        theme_quality: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for analysis in self.analysis_results:
            theme = (
                analysis.theme
                if analysis.theme and analysis.theme != "nan"
                else "No Theme"
            )
            theme_quality[theme][analysis.quality_score] += 1

        return dict(theme_quality)

    def generate_network_analysis(self) -> dict[str, dict[str, int]]:
        """Analyze quality distribution by social network.

        Returns:
            Dictionary mapping networks to quality score distributions
        """
        network_quality: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for analysis in self.analysis_results:
            network_quality[analysis.network][analysis.quality_score] += 1

        return dict(network_quality)

    def print_comprehensive_report(self) -> None:
        """Print a comprehensive analysis report."""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_all_messages() first.")
            return

        metrics = self.analyze_all_messages()
        examples = self.generate_examples_report()
        theme_analysis = self.generate_theme_analysis()
        network_analysis = self.generate_network_analysis()

        print("=" * 80)
        print("FARM AND FLEET SOCIAL MEDIA CONTENT QUALITY ANALYSIS")
        print("=" * 80)

        print("\n📊 OVERALL CONTENT QUALITY METRICS")
        print("-" * 50)
        print(f"Total Messages Analyzed: {metrics.total_messages:,}")
        print(
            f"High Quality (Actionable): {metrics.high_quality_count:,} ({metrics.high_quality_percentage:.1f}%)"
        )
        print(
            f"Medium Quality (Business Relevant): {metrics.medium_quality_count:,} ({(metrics.medium_quality_count / metrics.total_messages) * 100:.1f}%)"
        )
        print(
            f"Low Quality (Tangential): {metrics.low_quality_count:,} ({(metrics.low_quality_count / metrics.total_messages) * 100:.1f}%)"
        )
        print(
            f"Noise/Irrelevant: {metrics.noise_count:,} ({metrics.noise_percentage:.1f}%)"
        )
        print(
            f"\n🎯 Business Relevant Content: {metrics.business_relevant_percentage:.1f}%"
        )
        print(
            f"📈 Useful for Friction Analysis: {metrics.high_quality_percentage:.1f}%"
        )

        print("\n🔍 CONTENT QUALITY ASSESSMENT FOR VECTORIZATION")
        print("-" * 50)
        vectorization_score = (
            metrics.high_quality_percentage
            + (metrics.medium_quality_count / metrics.total_messages) * 50
        )
        print(f"Vectorization Viability Score: {vectorization_score:.1f}/100")
        if vectorization_score > 70:
            print("✅ EXCELLENT - High potential for meaningful embeddings")
        elif vectorization_score > 50:
            print("⚠️  MODERATE - Some filtering recommended before vectorization")
        elif vectorization_score > 30:
            print("❌ LOW - Significant filtering required")
        else:
            print("🚫 POOR - Dataset requires major cleanup")

        print("\n📱 QUALITY DISTRIBUTION BY NETWORK")
        print("-" * 50)
        for network, qualities in network_analysis.items():
            total = sum(qualities.values())
            high_pct = (qualities.get("high", 0) / total) * 100 if total > 0 else 0
            print(f"{network}: {total:,} messages ({high_pct:.1f}% high quality)")

        print("\n🏷️  QUALITY DISTRIBUTION BY THEME")
        print("-" * 50)
        for theme, qualities in sorted(
            theme_analysis.items(), key=lambda x: sum(x[1].values()), reverse=True
        )[:10]:
            total = sum(qualities.values())
            high_pct = (qualities.get("high", 0) / total) * 100 if total > 0 else 0
            print(f"{theme}: {total:,} messages ({high_pct:.1f}% high quality)")

        print("\n✅ HIGH QUALITY EXAMPLES (Actionable for Friction Analysis)")
        print("-" * 80)
        for i, example in enumerate(examples.get("high", [])[:5], 1):
            print(f"\n{i}. [{example.network}] {example.sentiment} sentiment")
            print(f"   Theme: {example.theme}")
            print(f"   Message: {example.message}")
            print(
                f"   Friction Indicators: {len(example.friction_indicators)} patterns found"
            )
            print(f"   Analysis: {example.reason}")

        print("\n⚠️  MEDIUM QUALITY EXAMPLES (Business Relevant)")
        print("-" * 80)
        for i, example in enumerate(examples.get("medium", [])[:3], 1):
            print(f"\n{i}. [{example.network}] {example.sentiment} sentiment")
            print(f"   Theme: {example.theme}")
            print(f"   Message: {example.message}")
            print(f"   Analysis: {example.reason}")

        print("\n❌ LOW QUALITY EXAMPLES (Tangential Mentions)")
        print("-" * 80)
        for i, example in enumerate(examples.get("low", [])[:3], 1):
            print(f"\n{i}. [{example.network}]")
            print(f"   Message: {example.message}")
            print(f"   Analysis: {example.reason}")

        print("\n🗑️  NOISE EXAMPLES (Should be Filtered)")
        print("-" * 80)
        for i, example in enumerate(examples.get("noise", [])[:3], 1):
            print(f"\n{i}. [{example.network}]")
            print(f"   Message: {example.message}")
            print(f"   Analysis: {example.reason}")

        print("\n🎯 RECOMMENDATIONS FOR FRICTION POINT ANALYSIS")
        print("-" * 80)
        print("1. VECTORIZATION STRATEGY:")
        if metrics.high_quality_percentage > 15:
            print("   ✅ Focus on HIGH quality messages for initial vector embeddings")
            print("   ✅ Include MEDIUM quality messages for context expansion")
        else:
            print(
                "   ⚠️  Consider manual review to identify additional quality patterns"
            )
            print("   ⚠️  May need expanded friction indicator patterns")

        print("\n2. FILTERING RECOMMENDATIONS:")
        print(
            f"   • Remove {metrics.noise_percentage:.1f}% noise content before processing"
        )
        print("   • Consider manual review of low-quality messages")
        print("   • Focus Reddit and Instagram for highest engagement content")

        print("\n3. FRICTION POINT CATEGORIES IDENTIFIED:")
        friction_categories = set()
        for analysis in self.analysis_results:
            if (
                analysis.quality_score in ["high", "medium"]
                and analysis.friction_indicators
            ):
                friction_categories.update(
                    [
                        "Customer Service",
                        "Product Availability",
                        "Store Operations",
                        "Product Quality",
                        "Digital Experience",
                    ]
                )

        for category in sorted(friction_categories):
            print(f"   • {category}")

        print("\n📈 DATA PROCESSING EFFICIENCY:")
        print(f"   • Useful Content Ratio: {metrics.business_relevant_percentage:.1f}%")
        print(
            f"   • Processing Recommendation: Filter to {metrics.high_quality_count + metrics.medium_quality_count:,} messages"
        )
        print(
            f"   • Expected Token Reduction: ~{metrics.noise_percentage:.0f}% savings"
        )


def main() -> None:
    """Main execution function."""
    # Define file path
    data_path = Path(
        "/Users/jonathan/Code/Projects/FarmAndFleet/data/Social Media Insights.csv"
    )

    if not data_path.exists():
        print(f"❌ Error: CSV file not found at {data_path}")
        return

    try:
        # Initialize analyzer
        print("🔍 Initializing Social Media Content Analysis...")
        analyzer = SocialMediaContentAnalyzer(data_path)

        # Load data
        print("📂 Loading social media data...")
        analyzer.load_data()

        # Run analysis
        print("⚙️  Analyzing content quality and business relevance...")
        analyzer.analyze_all_messages()

        # Generate comprehensive report
        print("📊 Generating comprehensive analysis report...\n")
        analyzer.print_comprehensive_report()

        print(f"\n{'=' * 80}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
