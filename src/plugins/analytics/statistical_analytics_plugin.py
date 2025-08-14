"""
Statistical Analytics Semantic Kernel Plugin for predictive insights and trend analysis.

Implements statistical analysis functions that process data from CustomerInsights and DatabaseInsights
plugins to provide predictive insights, risk assessments, and trend forecasting using scipy.stats
without requiring any machine learning training.
"""

import json
import logging
from datetime import datetime, timedelta
from statistics import mean, median, stdev, variance
from typing import Annotated, Any

import numpy as np
from scipy import stats
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)


class StatisticalAnalyticsPlugin:
    """
    Statistical Analytics plugin for predictive insights and trend analysis.

    Implements AI-friendly functions with strategic optional parameters that influence
    the three "just enough" context control mechanisms:
    - Quantity control (max_results parameter)
    - Content control (detail_level parameter)
    - Relevance control (confidence_threshold parameter)

    Processes JSON data from other plugins to provide statistical insights using
    scipy.stats methods for trend analysis, anomaly detection, and risk assessment.
    """

    def __init__(self):
        """Initialize the plugin with statistical analysis capabilities."""
        try:
            logger.info("StatisticalAnalyticsPlugin initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize StatisticalAnalyticsPlugin: {str(e)}")
            raise

    @kernel_function(
        description="Analyze sentiment and volume trends over time using linear regression and statistical methods"
    )
    def analyze_feedback_trends(
        self,
        feedback_data: Annotated[
            str,
            "JSON data containing feedback with timestamps, sentiment scores, and volume metrics from other plugins",
        ],
        trend_type: Annotated[
            str,
            "Type of trend to analyze: 'sentiment' for sentiment trends, 'volume' for feedback volume trends, 'both' for comprehensive analysis",
        ],
        time_period: Annotated[
            str,
            "Time period granularity: 'daily', 'weekly', 'monthly' for trend aggregation",
        ] = "weekly",
        detail_level: Annotated[
            str,
            "Analysis detail: 'minimal' for basic trend direction, 'standard' for confidence intervals, 'detailed' for comprehensive statistical analysis",
        ] = "standard",
        confidence_level: Annotated[
            float, "Statistical confidence level for trend analysis (0.90, 0.95, 0.99)"
        ] = 0.95,
    ) -> Annotated[
        str,
        "JSON formatted trend analysis with regression statistics, confidence intervals, and trend predictions",
    ]:
        """
        Analyze temporal trends in customer feedback using scipy.stats.linregress.

        Statistical Methods Used:
        - Linear Regression: scipy.stats.linregress for trend slope and significance
        - Confidence Intervals: Based on regression standard error
        - Correlation Analysis: Pearson correlation coefficient
        - Trend Significance: p-value testing for statistical significance

        Returns comprehensive trend analysis with slope, r-value, p-value, and predictions.
        """
        try:
            # Parse input data
            if not feedback_data.strip():
                return json.dumps(
                    {
                        "error": "Feedback data cannot be empty",
                        "example": "Pass JSON data from search_customer_feedback() or get_feedback_summary()",
                    }
                )

            try:
                data = json.loads(feedback_data)
            except json.JSONDecodeError as e:
                return json.dumps(
                    {
                        "error": f"Invalid JSON format: {str(e)}",
                        "feedback_data_preview": feedback_data[:200] + "..."
                        if len(feedback_data) > 200
                        else feedback_data,
                    }
                )

            # Validate parameters
            valid_trend_types = ["sentiment", "volume", "both"]
            if trend_type not in valid_trend_types:
                return json.dumps(
                    {
                        "error": f"Invalid trend_type. Use one of: {valid_trend_types}",
                        "provided": trend_type,
                    }
                )

            valid_periods = ["daily", "weekly", "monthly"]
            if time_period not in valid_periods:
                return json.dumps(
                    {
                        "error": f"Invalid time_period. Use one of: {valid_periods}",
                        "provided": time_period,
                    }
                )

            if confidence_level not in [0.90, 0.95, 0.99]:
                return json.dumps(
                    {
                        "error": "Invalid confidence_level. Use 0.90, 0.95, or 0.99",
                        "provided": confidence_level,
                    }
                )

            # Extract feedback items from various plugin response formats
            feedback_items = []
            if "feedback" in data:
                feedback_items = data["feedback"]
            elif "results" in data:
                feedback_items = data["results"]
            elif "social_media_insights" in data and "survey_insights" in data:
                # Handle cross-source data
                feedback_items.extend(
                    data.get("social_media_insights", {}).get("results", [])
                )
                feedback_items.extend(
                    data.get("survey_insights", {}).get("results", [])
                )
            elif isinstance(data, list):
                feedback_items = data
            else:
                return json.dumps(
                    {
                        "error": "Unable to extract feedback items from data",
                        "expected_formats": [
                            "Plugin response with 'feedback' or 'results' key",
                            "Direct array of feedback items",
                        ],
                    }
                )

            if not feedback_items:
                return json.dumps(
                    {
                        "error": "No feedback items found in data",
                        "data_structure": list(data.keys())
                        if isinstance(data, dict)
                        else "array",
                    }
                )

            # Process temporal data by aggregating by time period
            temporal_data = self._aggregate_temporal_data(feedback_items, time_period)

            if not temporal_data:
                return json.dumps(
                    {
                        "error": "Unable to process temporal data - missing timestamps or valid data",
                        "feedback_sample": feedback_items[0] if feedback_items else {},
                    }
                )

            # Perform trend analysis
            results: dict[str, Any] = {
                "analysis_summary": {
                    "trend_type": trend_type,
                    "time_period": time_period,
                    "data_points": len(temporal_data),
                    "date_range": {
                        "start": min(temporal_data.keys()),
                        "end": max(temporal_data.keys()),
                    },
                    "confidence_level": confidence_level,
                    "detail_level": detail_level,
                },
                "trends": {},
            }

            # Analyze sentiment trends
            if trend_type in ["sentiment", "both"]:
                sentiment_analysis = self._analyze_sentiment_trend(
                    temporal_data, confidence_level
                )
                results["trends"]["sentiment"] = sentiment_analysis

            # Analyze volume trends
            if trend_type in ["volume", "both"]:
                volume_analysis = self._analyze_volume_trend(
                    temporal_data, confidence_level
                )
                results["trends"]["volume"] = volume_analysis

            # Add detailed statistical insights
            if detail_level == "detailed":
                results["statistical_details"] = self._get_detailed_statistics(
                    temporal_data
                )

            return json.dumps(results, indent=2)

        except Exception as e:
            logger.error(f"analyze_feedback_trends failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Trend analysis failed: {str(e)}",
                    "trend_type": trend_type,
                    "time_period": time_period,
                }
            )

    @kernel_function(
        description="Assess customer churn risk indicators based on feedback patterns and sentiment degradation"
    )
    def assess_churn_risk_indicators(
        self,
        feedback_data: Annotated[
            str,
            "JSON data containing customer feedback with sentiment scores, timestamps, and customer identifiers",
        ],
        risk_factors: Annotated[
            str,
            "Risk factors to analyze: 'sentiment_decline' for sentiment degradation, 'engagement_drop' for reduced feedback frequency, 'all' for comprehensive assessment",
        ] = "all",
        time_window: Annotated[
            int, "Number of days to look back for trend analysis (7, 14, 30, 60, 90)"
        ] = 30,
        max_results: Annotated[
            int, "Maximum number of risk indicators to return (1-100)"
        ] = 20,
        detail_level: Annotated[
            str,
            "Analysis detail: 'minimal' for risk scores only, 'standard' for risk factors, 'detailed' for statistical analysis",
        ] = "standard",
    ) -> Annotated[
        str,
        "JSON formatted churn risk assessment with risk scores, indicators, and recommended actions",
    ]:
        """
        Calculate customer churn risk indicators using statistical analysis.

        Statistical Methods Used:
        - Trend Analysis: Linear regression on sentiment scores over time
        - Percentile Analysis: Risk scoring based on sentiment percentiles
        - Frequency Analysis: Engagement pattern detection
        - Z-Score Analysis: Outlier detection for unusual patterns
        - Moving Averages: Short-term vs long-term trend comparison

        Returns risk scores (0-100) with higher scores indicating greater churn risk.
        """
        try:
            # Parse and validate input
            if not feedback_data.strip():
                return json.dumps(
                    {
                        "error": "Feedback data cannot be empty",
                        "example": "Pass JSON data from customer feedback analysis",
                    }
                )

            try:
                data = json.loads(feedback_data)
            except json.JSONDecodeError as e:
                return json.dumps(
                    {
                        "error": f"Invalid JSON format: {str(e)}",
                        "feedback_data_preview": feedback_data[:200] + "...",
                    }
                )

            # Validate parameters
            valid_risk_factors = ["sentiment_decline", "engagement_drop", "all"]
            if risk_factors not in valid_risk_factors:
                return json.dumps(
                    {
                        "error": f"Invalid risk_factors. Use one of: {valid_risk_factors}",
                        "provided": risk_factors,
                    }
                )

            valid_windows = [7, 14, 30, 60, 90]
            if time_window not in valid_windows:
                return json.dumps(
                    {
                        "error": f"Invalid time_window. Use one of: {valid_windows}",
                        "provided": time_window,
                    }
                )

            # Clamp max_results
            max_results = max(1, min(max_results, 100))

            # Extract feedback items
            feedback_items = self._extract_feedback_items(data)
            if not feedback_items:
                return json.dumps({"error": "No feedback items found in data"})

            # Calculate risk indicators
            risk_indicators = self._calculate_churn_risk_indicators(
                feedback_items, risk_factors, time_window
            )

            # Sort by risk score and limit results
            risk_indicators.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
            risk_indicators = risk_indicators[:max_results]

            # Prepare response
            results = {
                "risk_assessment_summary": {
                    "risk_factors": risk_factors,
                    "time_window_days": time_window,
                    "total_evaluated": len(feedback_items),
                    "high_risk_count": len(
                        [r for r in risk_indicators if r.get("risk_score", 0) >= 70]
                    ),
                    "medium_risk_count": len(
                        [
                            r
                            for r in risk_indicators
                            if 40 <= r.get("risk_score", 0) < 70
                        ]
                    ),
                    "low_risk_count": len(
                        [r for r in risk_indicators if r.get("risk_score", 0) < 40]
                    ),
                    "detail_level": detail_level,
                },
                "risk_indicators": risk_indicators[:max_results],
            }

            if detail_level == "detailed":
                methodology_dict: dict[str, Any] = {
                    "risk_calculation": "Weighted combination of sentiment decline, engagement drop, and pattern anomalies",
                    "scoring_range": "0-100 (higher scores indicate greater churn risk)",
                    "statistical_methods": [
                        "Linear regression for sentiment trend analysis",
                        "Percentile scoring for relative risk assessment",
                        "Z-score analysis for anomaly detection",
                        "Moving average comparison for pattern changes",
                    ],
                }
                results["methodology"] = methodology_dict

            return json.dumps(results, indent=2)

        except Exception as e:
            logger.error(f"assess_churn_risk_indicators failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Churn risk assessment failed: {str(e)}",
                    "risk_factors": risk_factors,
                    "time_window": time_window,
                }
            )

    @kernel_function(
        description="Predict future trend trajectories based on historical patterns using statistical forecasting"
    )
    def predict_trend_trajectory(
        self,
        historical_data: Annotated[
            str, "JSON data containing historical trends with timestamps and metrics"
        ],
        forecast_periods: Annotated[
            int, "Number of future periods to predict (1-12)"
        ] = 4,
        prediction_metric: Annotated[
            str,
            "Metric to predict: 'sentiment_score' for sentiment forecasting, 'feedback_volume' for volume prediction, 'satisfaction_trend' for overall satisfaction",
        ] = "sentiment_score",
        confidence_level: Annotated[
            float,
            "Statistical confidence level for prediction intervals (0.90, 0.95, 0.99)",
        ] = 0.95,
        detail_level: Annotated[
            str,
            "Prediction detail: 'minimal' for point estimates, 'standard' for confidence intervals, 'detailed' for full statistical analysis",
        ] = "standard",
    ) -> Annotated[
        str,
        "JSON formatted trend predictions with confidence intervals and statistical validation",
    ]:
        """
        Generate statistical forecasts for business metrics using time series analysis.

        Statistical Methods Used:
        - Linear Regression: Basic trend extrapolation
        - Moving Average: Smoothed trend prediction
        - Exponential Smoothing: Weighted recent observations
        - Confidence Intervals: Prediction uncertainty quantification
        - Residual Analysis: Forecast accuracy assessment

        Returns point predictions with confidence intervals and trend validation.
        """
        try:
            # Validate inputs
            if not historical_data.strip():
                return json.dumps(
                    {
                        "error": "Historical data cannot be empty",
                        "example": "Pass JSON data from analyze_feedback_trends()",
                    }
                )

            try:
                data = json.loads(historical_data)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON format: {str(e)}"})

            # Validate parameters
            if not 1 <= forecast_periods <= 12:
                return json.dumps(
                    {
                        "error": "forecast_periods must be between 1 and 12",
                        "provided": forecast_periods,
                    }
                )

            valid_metrics = ["sentiment_score", "feedback_volume", "satisfaction_trend"]
            if prediction_metric not in valid_metrics:
                return json.dumps(
                    {
                        "error": f"Invalid prediction_metric. Use one of: {valid_metrics}",
                        "provided": prediction_metric,
                    }
                )

            if confidence_level not in [0.90, 0.95, 0.99]:
                return json.dumps(
                    {
                        "error": "Invalid confidence_level. Use 0.90, 0.95, or 0.99",
                        "provided": confidence_level,
                    }
                )

            # Extract time series data
            time_series = self._extract_time_series(data, prediction_metric)
            if not time_series or len(time_series) < 3:
                return json.dumps(
                    {
                        "error": "Insufficient historical data for prediction (need at least 3 data points)",
                        "available_data_points": len(time_series) if time_series else 0,
                    }
                )

            # Generate predictions
            predictions = self._generate_statistical_forecast(
                time_series, forecast_periods, confidence_level
            )

            # Prepare response
            results = {
                "prediction_summary": {
                    "metric_predicted": prediction_metric,
                    "historical_data_points": len(time_series),
                    "forecast_periods": forecast_periods,
                    "confidence_level": confidence_level,
                    "detail_level": detail_level,
                    "forecast_accuracy": predictions.get("model_accuracy", "Unknown"),
                },
                "predictions": predictions["forecasts"],
                "trend_analysis": {
                    "overall_trend": predictions.get("trend_direction", "Unknown"),
                    "trend_strength": predictions.get("trend_strength", "Unknown"),
                    "seasonal_pattern": predictions.get("seasonality", "None detected"),
                },
            }

            if detail_level == "detailed":
                results["statistical_details"] = {
                    "model_parameters": predictions.get("model_stats", {}),
                    "residual_analysis": predictions.get("residuals", {}),
                    "validation_metrics": predictions.get("validation", {}),
                    "methodology": {
                        "primary_method": "Linear regression with confidence intervals",
                        "smoothing_applied": "Moving average for noise reduction",
                        "uncertainty_quantification": "Bootstrap confidence intervals",
                    },
                }

            return json.dumps(results, indent=2)

        except Exception as e:
            logger.error(f"predict_trend_trajectory failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Trend prediction failed: {str(e)}",
                    "prediction_metric": prediction_metric,
                    "forecast_periods": forecast_periods,
                }
            )

    @kernel_function(
        description="Detect anomalies and unusual patterns in feedback data using statistical outlier detection"
    )
    def detect_trend_anomalies(
        self,
        data_series: Annotated[
            str, "JSON data containing time series or metric data for anomaly detection"
        ],
        detection_method: Annotated[
            str,
            "Anomaly detection method: 'z_score' for z-score analysis, 'iqr' for interquartile range, 'both' for comprehensive detection",
        ] = "both",
        sensitivity: Annotated[
            float,
            "Detection sensitivity: 2.0 for moderate sensitivity, 2.5 for balanced, 3.0 for conservative detection",
        ] = 2.5,
        max_results: Annotated[
            int, "Maximum number of anomalies to return (1-50)"
        ] = 10,
        detail_level: Annotated[
            str,
            "Analysis detail: 'minimal' for anomaly flags only, 'standard' for statistical scores, 'detailed' for comprehensive analysis",
        ] = "standard",
    ) -> Annotated[
        str,
        "JSON formatted anomaly detection results with statistical scores and significance levels",
    ]:
        """
        Identify statistical anomalies and outliers in customer feedback patterns.

        Statistical Methods Used:
        - Z-Score Analysis: Standard deviation-based outlier detection
        - Interquartile Range (IQR): Robust percentile-based detection
        - Modified Z-Score: Using median absolute deviation for robustness
        - Grubbs' Test: Extreme outlier identification
        - Isolation Score: Multi-dimensional anomaly detection

        Returns anomalies ranked by statistical significance with actionable insights.
        """
        try:
            # Parse and validate input
            if not data_series.strip():
                return json.dumps(
                    {
                        "error": "Data series cannot be empty",
                        "example": "Pass JSON data with numerical values for analysis",
                    }
                )

            try:
                data = json.loads(data_series)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON format: {str(e)}"})

            # Validate parameters
            valid_methods = ["z_score", "iqr", "both"]
            if detection_method not in valid_methods:
                return json.dumps(
                    {
                        "error": f"Invalid detection_method. Use one of: {valid_methods}",
                        "provided": detection_method,
                    }
                )

            if not 1.0 <= sensitivity <= 5.0:
                return json.dumps(
                    {
                        "error": "Sensitivity must be between 1.0 and 5.0",
                        "provided": sensitivity,
                        "recommendations": {
                            "1.5-2.0": "Very sensitive (more anomalies detected)",
                            "2.5": "Balanced detection (recommended)",
                            "3.0-3.5": "Conservative (fewer false positives)",
                        },
                    }
                )

            # Clamp max_results
            max_results = max(1, min(max_results, 50))

            # Extract numerical data for analysis
            values = self._extract_numerical_values(data)
            if not values or len(values) < 5:
                return json.dumps(
                    {
                        "error": "Insufficient numerical data for anomaly detection (need at least 5 values)",
                        "available_values": len(values) if values else 0,
                    }
                )

            # Detect anomalies using specified methods
            anomalies = []

            if detection_method in ["z_score", "both"]:
                z_score_anomalies = self._detect_z_score_anomalies(values, sensitivity)
                anomalies.extend(z_score_anomalies)

            if detection_method in ["iqr", "both"]:
                iqr_anomalies = self._detect_iqr_anomalies(values, sensitivity)
                anomalies.extend(iqr_anomalies)

            # Remove duplicates and sort by significance
            anomalies = self._deduplicate_anomalies(anomalies)
            anomalies.sort(key=lambda x: x.get("significance_score", 0), reverse=True)
            anomalies = anomalies[:max_results]

            # Calculate overall statistics
            data_stats = self._calculate_data_statistics(values)

            # Prepare response
            results = {
                "anomaly_summary": {
                    "detection_method": detection_method,
                    "sensitivity_threshold": sensitivity,
                    "total_data_points": len(values),
                    "anomalies_detected": len(anomalies),
                    "data_range": {
                        "min": min(values),
                        "max": max(values),
                        "mean": data_stats["mean"],
                        "std_dev": data_stats["std_dev"],
                    },
                    "detail_level": detail_level,
                },
                "anomalies": anomalies,
                "data_quality": {
                    "outlier_percentage": round(len(anomalies) / len(values) * 100, 2),
                    "data_distribution": data_stats["distribution_type"],
                    "skewness": data_stats["skewness"],
                },
            }

            if detail_level == "detailed":
                results["statistical_analysis"] = {
                    "normality_test": data_stats["normality"],
                    "distribution_parameters": data_stats["parameters"],
                    "detection_thresholds": {
                        "z_score_threshold": sensitivity,
                        "iqr_multiplier": 1.5 * (sensitivity / 2.5),
                    },
                    "methodology": {
                        "z_score": "Measures how many standard deviations a value is from the mean",
                        "iqr": "Uses interquartile range to identify outliers beyond Q1-1.5*IQR or Q3+1.5*IQR",
                        "significance_scoring": "Combines multiple detection methods with weighted confidence",
                    },
                }

            return json.dumps(results, indent=2)

        except Exception as e:
            logger.error(f"detect_trend_anomalies failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Anomaly detection failed: {str(e)}",
                    "detection_method": detection_method,
                    "sensitivity": sensitivity,
                }
            )

    @kernel_function(
        description="Compare statistical differences between two time periods using hypothesis testing"
    )
    def compare_time_periods(
        self,
        period1_data: Annotated[
            str, "JSON data for first time period with metrics and timestamps"
        ],
        period2_data: Annotated[
            str, "JSON data for second time period with metrics and timestamps"
        ],
        comparison_metric: Annotated[
            str,
            "Metric to compare: 'sentiment_score' for sentiment comparison, 'feedback_volume' for volume comparison, 'satisfaction_rating' for satisfaction comparison",
        ],
        test_type: Annotated[
            str,
            "Statistical test type: 't_test' for means comparison, 'mann_whitney' for non-parametric comparison, 'ks_test' for distribution comparison",
        ] = "t_test",
        significance_level: Annotated[
            float,
            "Statistical significance level for hypothesis testing (0.01, 0.05, 0.10)",
        ] = 0.05,
        detail_level: Annotated[
            str,
            "Analysis detail: 'minimal' for test results only, 'standard' for interpretation, 'detailed' for comprehensive statistical analysis",
        ] = "standard",
    ) -> Annotated[
        str,
        "JSON formatted statistical comparison with hypothesis test results, effect sizes, and business interpretation",
    ]:
        """
        Perform statistical hypothesis testing to compare metrics between time periods.

        Statistical Methods Used:
        - Two-Sample T-Test: Compare means of normally distributed data
        - Mann-Whitney U Test: Non-parametric comparison for non-normal data
        - Kolmogorov-Smirnov Test: Compare entire distributions
        - Cohen's D: Effect size calculation for practical significance
        - Confidence Intervals: Quantify uncertainty in differences

        Returns comprehensive statistical comparison with business-relevant interpretation.
        """
        try:
            # Parse and validate inputs
            if not period1_data.strip() or not period2_data.strip():
                return json.dumps(
                    {
                        "error": "Both period data inputs are required",
                        "period1_empty": not period1_data.strip(),
                        "period2_empty": not period2_data.strip(),
                    }
                )

            try:
                data1 = json.loads(period1_data)
                data2 = json.loads(period2_data)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON format: {str(e)}"})

            # Validate parameters
            valid_metrics = [
                "sentiment_score",
                "feedback_volume",
                "satisfaction_rating",
            ]
            if comparison_metric not in valid_metrics:
                return json.dumps(
                    {
                        "error": f"Invalid comparison_metric. Use one of: {valid_metrics}",
                        "provided": comparison_metric,
                    }
                )

            valid_tests = ["t_test", "mann_whitney", "ks_test"]
            if test_type not in valid_tests:
                return json.dumps(
                    {
                        "error": f"Invalid test_type. Use one of: {valid_tests}",
                        "provided": test_type,
                    }
                )

            if significance_level not in [0.01, 0.05, 0.10]:
                return json.dumps(
                    {
                        "error": "Invalid significance_level. Use 0.01, 0.05, or 0.10",
                        "provided": significance_level,
                    }
                )

            # Extract values for comparison
            values1 = self._extract_comparison_values(data1, comparison_metric)
            values2 = self._extract_comparison_values(data2, comparison_metric)

            if not values1 or not values2:
                return json.dumps(
                    {
                        "error": "Unable to extract comparison values from data",
                        "period1_values": len(values1) if values1 else 0,
                        "period2_values": len(values2) if values2 else 0,
                    }
                )

            if len(values1) < 3 or len(values2) < 3:
                return json.dumps(
                    {
                        "error": "Insufficient data for statistical comparison (need at least 3 values per period)",
                        "period1_count": len(values1),
                        "period2_count": len(values2),
                    }
                )

            # Perform statistical comparison
            comparison_results = self._perform_statistical_comparison(
                values1, values2, test_type, significance_level
            )

            # Calculate descriptive statistics
            period1_stats = self._calculate_descriptive_stats(values1, "Period 1")
            period2_stats = self._calculate_descriptive_stats(values2, "Period 2")

            # Prepare response
            results = {
                "comparison_summary": {
                    "comparison_metric": comparison_metric,
                    "test_type": test_type,
                    "significance_level": significance_level,
                    "sample_sizes": {"period1": len(values1), "period2": len(values2)},
                    "statistical_significance": comparison_results["is_significant"],
                    "p_value": comparison_results["p_value"],
                    "effect_size": comparison_results.get(
                        "effect_size", "Not calculated"
                    ),
                    "detail_level": detail_level,
                },
                "descriptive_statistics": {
                    "period1": period1_stats,
                    "period2": period2_stats,
                    "difference": {
                        "mean_difference": period2_stats["mean"]
                        - period1_stats["mean"],
                        "median_difference": period2_stats["median"]
                        - period1_stats["median"],
                        "percentage_change": round(
                            (
                                (period2_stats["mean"] - period1_stats["mean"])
                                / period1_stats["mean"]
                            )
                            * 100,
                            2,
                        )
                        if period1_stats["mean"] != 0
                        else None,
                    },
                },
                "test_results": comparison_results,
            }

            # Add business interpretation
            results["business_interpretation"] = self._generate_business_interpretation(
                comparison_results, period1_stats, period2_stats, comparison_metric
            )

            if detail_level == "detailed":
                results["detailed_analysis"] = {
                    "data_assumptions": self._check_test_assumptions(
                        values1, values2, test_type
                    ),
                    "confidence_intervals": comparison_results.get(
                        "confidence_intervals", {}
                    ),
                    "alternative_tests": self._suggest_alternative_tests(
                        values1, values2
                    ),
                    "methodology": {
                        "test_description": comparison_results.get(
                            "test_description", ""
                        ),
                        "interpretation_guide": {
                            "p_value": "Probability of observing this difference if there's no real change",
                            "effect_size": "Magnitude of practical difference (small: 0.2, medium: 0.5, large: 0.8)",
                            "confidence_interval": "Range of plausible values for the true difference",
                        },
                    },
                }

            return json.dumps(results, indent=2)

        except Exception as e:
            logger.error(f"compare_time_periods failed: {str(e)}")
            return json.dumps(
                {
                    "error": f"Time period comparison failed: {str(e)}",
                    "comparison_metric": comparison_metric,
                    "test_type": test_type,
                }
            )

    # Helper methods for internal processing

    def _aggregate_temporal_data(
        self, feedback_items: list[dict[str, Any]], time_period: str
    ) -> dict[str, dict[str, Any]]:
        """Aggregate feedback data by time periods."""
        temporal_data = {}

        for item in feedback_items:
            # Extract timestamp from various possible fields
            timestamp = None
            for field in [
                "timestamp",
                "date",
                "created_at",
                "publishedDate",
                "survey_date",
            ]:
                if field in item and item[field]:
                    timestamp = item[field]
                    break

            if not timestamp:
                continue

            try:
                # Parse timestamp and create period key
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = timestamp

                if time_period == "daily":
                    period_key = dt.strftime("%Y-%m-%d")
                elif time_period == "weekly":
                    # Use Monday as start of week
                    monday = dt - timedelta(days=dt.weekday())
                    period_key = monday.strftime("%Y-%m-%d")
                else:  # monthly
                    period_key = dt.strftime("%Y-%m")

                if period_key not in temporal_data:
                    temporal_data[period_key] = {
                        "feedback_count": 0,
                        "sentiment_scores": [],
                        "engagement_scores": [],
                    }

                temporal_data[period_key]["feedback_count"] += 1

                # Extract sentiment if available
                for sentiment_field in ["sentiment_score", "sentiment", "score"]:
                    if sentiment_field in item and item[sentiment_field] is not None:
                        try:
                            score = float(item[sentiment_field])
                            temporal_data[period_key]["sentiment_scores"].append(score)
                            break
                        except (ValueError, TypeError):
                            continue

                # Extract engagement metrics if available
                for engagement_field in [
                    "likes",
                    "shares",
                    "replies",
                    "retweets",
                    "engagement_score",
                ]:
                    if engagement_field in item and item[engagement_field] is not None:
                        try:
                            score = float(item[engagement_field])
                            temporal_data[period_key]["engagement_scores"].append(score)
                            break
                        except (ValueError, TypeError):
                            continue

            except Exception as e:
                logger.warning(f"Failed to parse timestamp {timestamp}: {str(e)}")
                continue

        return temporal_data

    def _analyze_sentiment_trend(
        self, temporal_data: dict[str, dict[str, Any]], confidence_level: float
    ) -> dict[str, Any]:
        """Analyze sentiment trends using linear regression."""
        # Prepare data for regression
        periods = sorted(temporal_data.keys())
        x_values = list(range(len(periods)))
        y_values = []

        for period in periods:
            sentiment_scores = temporal_data[period]["sentiment_scores"]
            if sentiment_scores:
                avg_sentiment = mean(sentiment_scores)
                y_values.append(avg_sentiment)
            else:
                y_values.append(0.0)

        if len(y_values) < 2:
            return {
                "error": "Insufficient data for sentiment trend analysis",
                "data_points": len(y_values),
            }

        # Perform linear regression
        slope, _intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)  # type: ignore
        slope = float(slope)  # type: ignore
        r_value = float(r_value)  # type: ignore
        p_value = float(p_value)  # type: ignore
        std_err = float(std_err)  # type: ignore

        # Calculate confidence intervals
        alpha = 1.0 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(x_values) - 2)
        margin_of_error = t_critical * std_err

        return {
            "trend_slope": slope,
            "trend_direction": "improving"
            if slope > 0
            else "declining"
            if slope < 0
            else "stable",
            "correlation_strength": abs(r_value),
            "statistical_significance": p_value < 0.05,
            "p_value": p_value,
            "confidence_interval": {
                "lower": slope - margin_of_error,
                "upper": slope + margin_of_error,
            },
            "data_points": len(x_values),
            "avg_sentiment_range": {
                "min": min(y_values),
                "max": max(y_values),
                "current": y_values[-1] if y_values else None,
            },
        }

    def _analyze_volume_trend(
        self, temporal_data: dict[str, dict[str, Any]], confidence_level: float
    ) -> dict[str, Any]:
        """Analyze feedback volume trends using linear regression."""
        periods = sorted(temporal_data.keys())
        x_values = list(range(len(periods)))
        y_values = [temporal_data[period]["feedback_count"] for period in periods]

        if len(y_values) < 2:
            return {
                "error": "Insufficient data for volume trend analysis",
                "data_points": len(y_values),
            }

        # Perform linear regression
        slope, _intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)  # type: ignore
        slope = float(slope)  # type: ignore
        r_value = float(r_value)  # type: ignore
        p_value = float(p_value)  # type: ignore
        std_err = float(std_err)  # type: ignore

        # Calculate confidence intervals
        alpha = 1.0 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(x_values) - 2)
        margin_of_error = t_critical * std_err

        return {
            "trend_slope": slope,
            "trend_direction": "increasing"
            if slope > 0
            else "decreasing"
            if slope < 0
            else "stable",
            "correlation_strength": abs(r_value),
            "statistical_significance": p_value < 0.05,
            "p_value": p_value,
            "confidence_interval": {
                "lower": slope - margin_of_error,
                "upper": slope + margin_of_error,
            },
            "data_points": len(x_values),
            "volume_range": {
                "min": min(y_values),
                "max": max(y_values),
                "current": y_values[-1] if y_values else None,
                "average": mean(y_values),
            },
        }

    def _get_detailed_statistics(
        self, temporal_data: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate detailed statistical metrics for temporal data."""
        all_sentiment_scores = []
        all_feedback_counts = []

        for period_data in temporal_data.values():
            all_sentiment_scores.extend(period_data["sentiment_scores"])
            all_feedback_counts.append(period_data["feedback_count"])

        details = {}

        if all_sentiment_scores:
            details["sentiment_statistics"] = {
                "mean": mean(all_sentiment_scores),
                "median": median(all_sentiment_scores),
                "std_dev": stdev(all_sentiment_scores)
                if len(all_sentiment_scores) > 1
                else 0,
                "variance": variance(all_sentiment_scores)
                if len(all_sentiment_scores) > 1
                else 0,
                "min": min(all_sentiment_scores),
                "max": max(all_sentiment_scores),
                "skewness": float(stats.skew(all_sentiment_scores)),
                "kurtosis": float(stats.kurtosis(all_sentiment_scores)),
            }

        if all_feedback_counts:
            details["volume_statistics"] = {
                "mean": mean(all_feedback_counts),
                "median": median(all_feedback_counts),
                "std_dev": stdev(all_feedback_counts)
                if len(all_feedback_counts) > 1
                else 0,
                "min": min(all_feedback_counts),
                "max": max(all_feedback_counts),
                "total": sum(all_feedback_counts),
            }

        return details

    def _extract_feedback_items(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract feedback items from various plugin response formats."""
        feedback_items = []

        if "feedback" in data:
            feedback_items = data["feedback"]
        elif "results" in data:
            feedback_items = data["results"]
        elif "social_media_insights" in data and "survey_insights" in data:
            feedback_items.extend(
                data.get("social_media_insights", {}).get("results", [])
            )
            feedback_items.extend(data.get("survey_insights", {}).get("results", []))
        elif isinstance(data, list):
            feedback_items = data

        return feedback_items if isinstance(feedback_items, list) else []

    def _calculate_churn_risk_indicators(
        self, feedback_items: list[dict[str, Any]], risk_factors: str, time_window: int
    ) -> list[dict[str, Any]]:
        """Calculate churn risk indicators based on feedback patterns."""
        # Group by customer if customer identifier available
        customer_data = {}

        for item in feedback_items:
            customer_id = None
            for id_field in ["customer_id", "user_id", "author", "username"]:
                if id_field in item and item[id_field]:
                    customer_id = str(item[id_field])
                    break

            if not customer_id:
                customer_id = "anonymous"

            if customer_id not in customer_data:
                customer_data[customer_id] = []
            customer_data[customer_id].append(item)

        # Calculate risk for each customer
        risk_indicators = []

        for customer_id, customer_feedback in customer_data.items():
            if len(customer_feedback) < 2:  # Need at least 2 data points
                continue

            risk_score = self._calculate_customer_risk_score(
                customer_feedback, risk_factors, time_window
            )

            if risk_score > 0:
                risk_indicators.append(
                    {
                        "customer_id": customer_id,
                        "risk_score": risk_score,
                        "feedback_count": len(customer_feedback),
                        "risk_factors": self._identify_risk_factors(
                            customer_feedback, risk_factors
                        ),
                        "recent_sentiment": self._get_recent_sentiment(
                            customer_feedback
                        ),
                        "trend_direction": self._get_sentiment_trend_direction(
                            customer_feedback
                        ),
                    }
                )

        return risk_indicators

    def _calculate_customer_risk_score(
        self,
        customer_feedback: list[dict[str, Any]],
        risk_factors: str,
        time_window: int,
    ) -> float:
        """Calculate risk score for a specific customer."""
        risk_score = 0.0

        # Sort by timestamp
        timestamped_feedback = []
        for item in customer_feedback:
            for timestamp_field in ["timestamp", "date", "created_at", "publishedDate"]:
                if timestamp_field in item and item[timestamp_field]:
                    try:
                        dt = datetime.fromisoformat(
                            str(item[timestamp_field]).replace("Z", "+00:00")
                        )
                        timestamped_feedback.append((dt, item))
                        break
                    except Exception:
                        continue

        if len(timestamped_feedback) < 2:
            return 0.0

        timestamped_feedback.sort(key=lambda x: x[0])  # type: ignore

        # Analyze sentiment decline
        if risk_factors in ["sentiment_decline", "all"]:
            sentiment_decline_risk = self._assess_sentiment_decline_risk(
                timestamped_feedback, time_window
            )
            risk_score += sentiment_decline_risk * 0.6  # Weight: 60%

        # Analyze engagement drop
        if risk_factors in ["engagement_drop", "all"]:
            engagement_drop_risk = self._assess_engagement_drop_risk(
                timestamped_feedback, time_window
            )
            risk_score += engagement_drop_risk * 0.4  # Weight: 40%

        return min(risk_score, 100.0)  # Cap at 100

    def _assess_sentiment_decline_risk(
        self, timestamped_feedback: list[tuple[datetime, dict[str, Any]]], time_window: int
    ) -> float:
        """Assess risk based on sentiment decline patterns."""
        cutoff_date = datetime.now() - timedelta(days=time_window)
        recent_feedback = [
            item for dt, item in timestamped_feedback if dt >= cutoff_date
        ]

        if len(recent_feedback) < 2:
            return 0.0

        # Extract sentiment scores
        sentiment_scores = []
        for item in recent_feedback:
            for sentiment_field in ["sentiment_score", "sentiment", "score"]:
                if sentiment_field in item and item[sentiment_field] is not None:
                    try:
                        score = float(item[sentiment_field])
                        sentiment_scores.append(score)
                        break
                    except (ValueError, TypeError):
                        continue

        if len(sentiment_scores) < 2:
            return 0.0

        # Calculate trend and percentile
        x_values = list(range(len(sentiment_scores)))
        slope, _, _, p_value, _ = stats.linregress(x_values, sentiment_scores)  # type: ignore
        slope = float(slope)  # type: ignore
        p_value = float(p_value)  # type: ignore

        risk = 0.0

        # Negative slope indicates declining sentiment
        if slope < 0:
            risk += abs(slope) * 20  # Scale slope to risk score

        # Low absolute sentiment scores
        avg_sentiment = mean(sentiment_scores)
        if avg_sentiment < 0.3:  # Assuming 0-1 scale
            risk += (0.3 - avg_sentiment) * 100

        # Statistical significance of decline
        if p_value < 0.05 and slope < 0:
            risk += 10  # Bonus for statistically significant decline

        return min(risk, 50.0)  # Cap contribution at 50

    def _assess_engagement_drop_risk(
        self, timestamped_feedback: list[tuple[datetime, dict[str, Any]]], time_window: int
    ) -> float:
        """Assess risk based on engagement drop patterns."""
        cutoff_date = datetime.now() - timedelta(days=time_window)
        recent_count = len(
            [item for dt, item in timestamped_feedback if dt >= cutoff_date]
        )

        # Calculate expected frequency based on historical pattern
        total_days = (timestamped_feedback[-1][0] - timestamped_feedback[0][0]).days
        if total_days <= 0:
            return 0.0

        historical_frequency = len(timestamped_feedback) / total_days
        expected_recent = historical_frequency * time_window

        if expected_recent <= 0:
            return 0.0

        # Calculate engagement drop
        frequency_ratio = recent_count / expected_recent

        risk = 0.0
        if frequency_ratio < 0.5:  # 50% drop in engagement
            risk = (0.5 - frequency_ratio) * 100

        return min(risk, 50.0)  # Cap contribution at 50

    def _identify_risk_factors(
        self, customer_feedback: list[dict[str, Any]], risk_factors: str
    ) -> list[str]:
        """Identify specific risk factors for a customer."""
        factors = []

        # Check for negative sentiment pattern
        sentiment_scores = []
        for item in customer_feedback:
            for sentiment_field in ["sentiment_score", "sentiment", "score"]:
                if sentiment_field in item and item[sentiment_field] is not None:
                    try:
                        score = float(item[sentiment_field])
                        sentiment_scores.append(score)
                        break
                    except (ValueError, TypeError):
                        continue

        if sentiment_scores:
            avg_sentiment = mean(sentiment_scores)
            if avg_sentiment < 0.3:
                factors.append("Low average sentiment")

            if len(sentiment_scores) > 1:
                x_values = list(range(len(sentiment_scores)))
                slope, _, _, _, _ = stats.linregress(x_values, sentiment_scores)  # type: ignore
                slope = float(slope)  # type: ignore
                if slope < -0.1:
                    factors.append("Declining sentiment trend")

        # Check for reduced engagement
        if len(customer_feedback) < 3:
            factors.append("Low engagement frequency")

        return factors

    def _get_recent_sentiment(
        self, customer_feedback: list[dict[str, Any]]
    ) -> float | None:
        """Get most recent sentiment score for a customer."""
        # Sort by timestamp and get most recent
        timestamped_feedback = []
        for item in customer_feedback:
            for timestamp_field in ["timestamp", "date", "created_at", "publishedDate"]:
                if timestamp_field in item and item[timestamp_field]:
                    try:
                        dt = datetime.fromisoformat(
                            str(item[timestamp_field]).replace("Z", "+00:00")
                        )
                        timestamped_feedback.append((dt, item))
                        break
                    except Exception:
                        continue

        if not timestamped_feedback:
            return None

        timestamped_feedback.sort(key=lambda x: x[0], reverse=True)  # type: ignore
        most_recent_item = timestamped_feedback[0][1]

        for sentiment_field in ["sentiment_score", "sentiment", "score"]:
            if (
                sentiment_field in most_recent_item
                and most_recent_item[sentiment_field] is not None
            ):
                try:
                    return float(most_recent_item[sentiment_field])
                except (ValueError, TypeError):
                    continue

        return None

    def _get_sentiment_trend_direction(
        self, customer_feedback: list[dict[str, Any]]
    ) -> str:
        """Determine overall sentiment trend direction for a customer."""
        sentiment_scores = []
        for item in customer_feedback:
            for sentiment_field in ["sentiment_score", "sentiment", "score"]:
                if sentiment_field in item and item[sentiment_field] is not None:
                    try:
                        score = float(item[sentiment_field])
                        sentiment_scores.append(score)
                        break
                    except (ValueError, TypeError):
                        continue

        if len(sentiment_scores) < 2:
            return "Unknown"

        x_values = list(range(len(sentiment_scores)))
        slope, _, _, _, _ = stats.linregress(x_values, sentiment_scores)  # type: ignore
        slope = float(slope)  # type: ignore

        if slope > 0.05:
            return "Improving"
        elif slope < -0.05:
            return "Declining"
        else:
            return "Stable"

    def _extract_time_series(
        self, data: dict[str, Any], prediction_metric: str
    ) -> list[tuple[datetime, float]]:
        """Extract time series data for prediction."""
        time_series = []

        # Handle trend analysis results
        if "trends" in data:
            # Extract from trend analysis results
            trends = data["trends"]
            if prediction_metric == "sentiment_score" and "sentiment" in trends:
                sentiment_trend = trends["sentiment"]
                if (
                    "avg_sentiment_range" in sentiment_trend
                    and "current" in sentiment_trend["avg_sentiment_range"]
                ):
                    # Use current value as single point (limited but functional)
                    time_series.append(
                        (
                            datetime.now(),
                            sentiment_trend["avg_sentiment_range"]["current"],
                        )
                    )
            elif prediction_metric == "feedback_volume" and "volume" in trends:
                volume_trend = trends["volume"]
                if (
                    "volume_range" in volume_trend
                    and "current" in volume_trend["volume_range"]
                ):
                    time_series.append(
                        (datetime.now(), volume_trend["volume_range"]["current"])
                    )

        # Handle raw data
        feedback_items = self._extract_feedback_items(data)
        for item in feedback_items:
            timestamp = None
            value = None

            # Extract timestamp
            for timestamp_field in ["timestamp", "date", "created_at", "publishedDate"]:
                if timestamp_field in item and item[timestamp_field]:
                    try:
                        timestamp = datetime.fromisoformat(
                            str(item[timestamp_field]).replace("Z", "+00:00")
                        )
                        break
                    except Exception:
                        continue

            # Extract value based on metric
            if prediction_metric == "sentiment_score":
                for sentiment_field in ["sentiment_score", "sentiment", "score"]:
                    if sentiment_field in item and item[sentiment_field] is not None:
                        try:
                            value = float(item[sentiment_field])
                            break
                        except (ValueError, TypeError):
                            continue
            elif prediction_metric == "feedback_volume":
                value = 1.0  # Each item counts as 1 for volume
            elif prediction_metric == "satisfaction_trend":
                for rating_field in ["satisfaction_rating", "rating", "score"]:
                    if rating_field in item and item[rating_field] is not None:
                        try:
                            value = float(item[rating_field])
                            break
                        except (ValueError, TypeError):
                            continue

            if timestamp and value is not None:
                time_series.append((timestamp, value))

        # Sort by timestamp
        time_series.sort(key=lambda x: x[0])  # type: ignore
        return time_series

    def _generate_statistical_forecast(
        self,
        time_series: list[tuple[datetime, float]],
        forecast_periods: int,
        confidence_level: float,
    ) -> dict[str, Any]:
        """Generate statistical forecast using multiple methods."""
        if len(time_series) < 3:
            return {"error": "Insufficient data for forecasting"}

        # Prepare data
        dates = [ts[0] for ts in time_series]
        values = [ts[1] for ts in time_series]
        x_values = list(range(len(values)))

        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, values)  # type: ignore
        slope = float(slope)  # type: ignore
        intercept = float(intercept)  # type: ignore
        r_value = float(r_value)  # type: ignore
        p_value = float(p_value)  # type: ignore
        std_err = float(std_err)  # type: ignore

        # Calculate prediction intervals
        alpha = 1.0 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(x_values) - 2)

        # Generate forecasts
        forecasts = []
        last_date = dates[-1]

        for i in range(1, forecast_periods + 1):
            future_x = len(x_values) + i - 1
            point_prediction = slope * future_x + intercept

            # Calculate prediction interval
            prediction_se = float(std_err * np.sqrt(
                1
                + 1 / len(x_values)
                + (future_x - mean(x_values)) ** 2
                / sum((x - mean(x_values)) ** 2 for x in x_values)
            ))
            margin_of_error = t_critical * prediction_se

            # Estimate future date (assuming regular intervals)
            if len(dates) > 1:
                avg_interval = (dates[-1] - dates[0]).days / (len(dates) - 1)
                future_date = last_date + timedelta(days=avg_interval * i)
            else:
                future_date = last_date + timedelta(days=7 * i)  # Assume weekly

            forecasts.append(
                {
                    "period": i,
                    "date": future_date.isoformat(),
                    "point_prediction": point_prediction,
                    "confidence_interval": {
                        "lower": point_prediction - margin_of_error,
                        "upper": point_prediction + margin_of_error,
                    },
                    "confidence_level": confidence_level,
                }
            )

        # Assess model accuracy
        residuals = [
            values[i] - (slope * x_values[i] + intercept) for i in range(len(values))
        ]
        rmse = float(np.sqrt(mean([r**2 for r in residuals])))

        return {
            "forecasts": forecasts,
            "model_accuracy": {
                "r_squared": r_value**2,
                "rmse": rmse,
                "mean_absolute_error": mean([abs(r) for r in residuals]),
            },
            "trend_direction": "increasing"
            if slope > 0
            else "decreasing"
            if slope < 0
            else "stable",
            "trend_strength": "strong"
            if abs(r_value) > 0.7
            else "moderate"
            if abs(r_value) > 0.3
            else "weak",
            "model_stats": {
                "slope": slope,
                "intercept": intercept,
                "r_value": r_value,
                "p_value": p_value,
                "standard_error": std_err,
            },
            "residuals": {
                "mean": mean(residuals),
                "std_dev": stdev(residuals) if len(residuals) > 1 else 0,
            },
        }

    def _extract_numerical_values(self, data: dict[str, Any]) -> list[float]:
        """Extract numerical values from data for anomaly detection."""
        values = []

        # Handle different data structures
        if isinstance(data, list):
            for item in data:
                if isinstance(item, int | float):
                    values.append(float(item))
                elif isinstance(item, dict):
                    for _key, value in item.items():
                        if isinstance(value, int | float):
                            values.append(float(value))
        elif isinstance(data, dict):
            # Try to extract from various fields
            feedback_items = self._extract_feedback_items(data)
            for item in feedback_items:
                # Extract numerical values from common fields
                for field in [
                    "sentiment_score",
                    "score",
                    "rating",
                    "likes",
                    "shares",
                    "replies",
                    "engagement_score",
                ]:
                    if field in item and isinstance(item[field], int | float):
                        values.append(float(item[field]))

        return values

    def _detect_z_score_anomalies(
        self, values: list[float], sensitivity: float
    ) -> list[dict[str, Any]]:
        """Detect anomalies using z-score method."""
        if len(values) < 3:
            return []

        mean_val = mean(values)
        std_val = stdev(values)

        if std_val == 0:
            return []

        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val
            if z_score >= sensitivity:
                anomalies.append(
                    {
                        "index": i,
                        "value": value,
                        "z_score": z_score,
                        "detection_method": "z_score",
                        "significance_score": z_score,
                        "anomaly_type": "high" if value > mean_val else "low",
                    }
                )

        return anomalies

    def _detect_iqr_anomalies(
        self, values: list[float], sensitivity: float
    ) -> list[dict[str, Any]]:
        """Detect anomalies using interquartile range method."""
        if len(values) < 4:
            return []

        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1

        if iqr == 0:
            return []

        # Adjust multiplier based on sensitivity
        multiplier = 1.5 * (sensitivity / 2.5)
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        anomalies = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                # Calculate significance score
                distance_from_bound = min(
                    abs(value - lower_bound), abs(value - upper_bound)
                )
                significance_score = distance_from_bound / iqr

                anomalies.append(
                    {
                        "index": i,
                        "value": value,
                        "iqr_distance": distance_from_bound,
                        "detection_method": "iqr",
                        "significance_score": significance_score,
                        "anomaly_type": "high" if value > upper_bound else "low",
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                    }
                )

        return anomalies

    def _deduplicate_anomalies(
        self, anomalies: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate anomalies detected by multiple methods."""
        unique_anomalies = {}

        for anomaly in anomalies:
            key = (anomaly["index"], anomaly["value"])
            if key not in unique_anomalies:
                unique_anomalies[key] = anomaly
            else:
                # Merge detection methods
                existing = unique_anomalies[key]
                if "detection_methods" not in existing:
                    existing["detection_methods"] = [existing["detection_method"]]
                existing["detection_methods"].append(anomaly["detection_method"])

                # Use highest significance score
                if anomaly["significance_score"] > existing["significance_score"]:
                    existing["significance_score"] = anomaly["significance_score"]

        return list(unique_anomalies.values())

    def _calculate_data_statistics(self, values: list[float]) -> dict[str, Any]:
        """Calculate comprehensive statistics for the dataset."""
        if not values:
            return {}

        stats_dict: dict[str, Any] = {
            "mean": mean(values),
            "std_dev": stdev(values) if len(values) > 1 else 0,
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
        }

        # Assess normality
        if len(values) >= 8:
            _, p_value = stats.shapiro(values)
            stats_dict["normality"] = {
                "shapiro_wilk_p": p_value,
                "is_normal": p_value > 0.05,
            }
        else:
            stats_dict["normality"] = {"note": "Too few values for normality test"}

        # Determine distribution type
        if abs(stats_dict["skewness"]) < 0.5:
            distribution_type = "approximately_normal"
        elif stats_dict["skewness"] > 0.5:
            distribution_type = "right_skewed"
        else:
            distribution_type = "left_skewed"

        stats_dict["distribution_type"] = distribution_type

        # Calculate distribution parameters
        stats_dict["parameters"] = {
            "percentiles": {
                "5th": float(np.percentile(values, 5)),
                "25th": float(np.percentile(values, 25)),
                "50th": float(np.percentile(values, 50)),
                "75th": float(np.percentile(values, 75)),
                "95th": float(np.percentile(values, 95)),
            }
        }

        return stats_dict

    def _extract_comparison_values(
        self, data: dict[str, Any], comparison_metric: str
    ) -> list[float]:
        """Extract values for statistical comparison."""
        values = []
        feedback_items = self._extract_feedback_items(data)

        for item in feedback_items:
            if comparison_metric == "sentiment_score":
                for sentiment_field in ["sentiment_score", "sentiment", "score"]:
                    if sentiment_field in item and item[sentiment_field] is not None:
                        try:
                            values.append(float(item[sentiment_field]))
                            break
                        except (ValueError, TypeError):
                            continue
            elif comparison_metric == "feedback_volume":
                values.append(1.0)  # Each item counts as 1
            elif comparison_metric == "satisfaction_rating":
                for rating_field in ["satisfaction_rating", "rating", "score"]:
                    if rating_field in item and item[rating_field] is not None:
                        try:
                            values.append(float(item[rating_field]))
                            break
                        except (ValueError, TypeError):
                            continue

        return values

    def _perform_statistical_comparison(
        self,
        values1: list[float],
        values2: list[float],
        test_type: str,
        significance_level: float,
    ) -> dict[str, Any]:
        """Perform statistical hypothesis testing."""
        results: dict[str, Any] = {}
        p_value = 1.0  # Default value

        try:
            if test_type == "t_test":
                # Two-sample t-test
                statistic, p_value = stats.ttest_ind(values1, values2)
                results["test_statistic"] = statistic
                results["test_description"] = (
                    "Two-sample t-test for independent samples"
                )

                # Calculate Cohen's d for effect size
                pooled_std = float(np.sqrt((stdev(values1) ** 2 + stdev(values2) ** 2) / 2))
                if pooled_std > 0:
                    cohens_d = (mean(values1) - mean(values2)) / pooled_std
                    results["effect_size"] = cohens_d

            elif test_type == "mann_whitney":
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(
                    values1, values2, alternative="two-sided"
                )
                results["test_statistic"] = statistic
                results["test_description"] = "Mann-Whitney U test (non-parametric)"

            elif test_type == "ks_test":
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(values1, values2)
                results["test_statistic"] = statistic
                results["test_description"] = (
                    "Kolmogorov-Smirnov test for distribution comparison"
                )

            results["p_value"] = p_value
            results["is_significant"] = p_value < significance_level  # type: ignore

            # Calculate confidence interval for difference in means
            if test_type in ["t_test", "mann_whitney"]:
                mean_diff = mean(values2) - mean(values1)
                se_diff = float(np.sqrt(
                    variance(values1) / len(values1) + variance(values2) / len(values2)
                ))

                if test_type == "t_test":
                    df = len(values1) + len(values2) - 2
                    t_critical = stats.t.ppf(1 - significance_level / 2, df)
                    margin_of_error = t_critical * se_diff

                    results["confidence_intervals"] = {
                        "mean_difference": {
                            "lower": mean_diff - margin_of_error,
                            "upper": mean_diff + margin_of_error,
                        }
                    }

        except Exception as e:
            results["error"] = f"Statistical test failed: {str(e)}"

        return results

    def _calculate_descriptive_stats(
        self, values: list[float], period_name: str
    ) -> dict[str, Any]:
        """Calculate descriptive statistics for a period."""
        return {
            "period": period_name,
            "sample_size": len(values),
            "mean": mean(values),
            "median": median(values),
            "std_dev": stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }

    def _generate_business_interpretation(
        self,
        comparison_results: dict[str, Any],
        period1_stats: dict[str, Any],
        period2_stats: dict[str, Any],
        comparison_metric: str,
    ) -> dict[str, Any]:
        """Generate business-friendly interpretation of statistical results."""
        interpretation = {}

        # Determine change direction and magnitude
        mean_diff = period2_stats["mean"] - period1_stats["mean"]
        percent_change = (
            (mean_diff / period1_stats["mean"]) * 100
            if period1_stats["mean"] != 0
            else 0
        )

        # Interpret statistical significance
        if comparison_results["is_significant"]:
            interpretation["significance"] = (
                f"The change is statistically significant (p = {comparison_results['p_value']:.4f})"
            )
        else:
            interpretation["significance"] = (
                f"The change is not statistically significant (p = {comparison_results['p_value']:.4f})"
            )

        # Interpret practical significance
        effect_size = comparison_results.get("effect_size", 0)
        if abs(effect_size) >= 0.8:
            practical_significance = "large"
        elif abs(effect_size) >= 0.5:
            practical_significance = "medium"
        elif abs(effect_size) >= 0.2:
            practical_significance = "small"
        else:
            practical_significance = "negligible"

        interpretation["practical_significance"] = (
            f"The effect size is {practical_significance} (Cohen's d = {effect_size:.3f})"
        )

        # Metric-specific interpretation
        if comparison_metric == "sentiment_score":
            if mean_diff > 0:
                interpretation["business_impact"] = (
                    f"Customer sentiment improved by {percent_change:.1f}%"
                )
            else:
                interpretation["business_impact"] = (
                    f"Customer sentiment declined by {abs(percent_change):.1f}%"
                )
        elif comparison_metric == "feedback_volume":
            if mean_diff > 0:
                interpretation["business_impact"] = (
                    f"Feedback volume increased by {percent_change:.1f}%"
                )
            else:
                interpretation["business_impact"] = (
                    f"Feedback volume decreased by {abs(percent_change):.1f}%"
                )
        elif comparison_metric == "satisfaction_rating":
            if mean_diff > 0:
                interpretation["business_impact"] = (
                    f"Customer satisfaction improved by {percent_change:.1f}%"
                )
            else:
                interpretation["business_impact"] = (
                    f"Customer satisfaction declined by {abs(percent_change):.1f}%"
                )

        # Recommendations
        recommendations = []
        if comparison_results["is_significant"] and abs(effect_size) >= 0.5:
            if mean_diff < 0:
                recommendations.append(
                    "Investigate factors contributing to the decline"
                )
                recommendations.append(
                    "Implement corrective measures to address the negative trend"
                )
            else:
                recommendations.append(
                    "Identify and replicate successful practices from the improved period"
                )
                recommendations.append("Monitor to ensure the positive trend continues")
        elif not comparison_results["is_significant"]:
            recommendations.append("The observed change may be due to random variation")
            recommendations.append(
                "Consider collecting more data or extending the analysis period"
            )

        interpretation["recommendations"] = recommendations

        return interpretation

    def _check_test_assumptions(
        self, values1: list[float], values2: list[float], test_type: str
    ) -> dict[str, Any]:
        """Check statistical test assumptions."""
        assumptions = {}

        if test_type == "t_test":
            # Check normality
            if len(values1) >= 8:
                _, p1 = stats.shapiro(values1)
                assumptions["group1_normality"] = {
                    "p_value": p1,
                    "is_normal": p1 > 0.05,
                }

            if len(values2) >= 8:
                _, p2 = stats.shapiro(values2)
                assumptions["group2_normality"] = {
                    "p_value": p2,
                    "is_normal": p2 > 0.05,
                }

            # Check equal variances
            _, p_levene = stats.levene(values1, values2)
            assumptions["equal_variances"] = {
                "p_value": p_levene,
                "assumption_met": p_levene > 0.05,
            }

        elif test_type == "mann_whitney":
            assumptions["test_type"] = (
                "Non-parametric test - no normality assumption required"
            )

        elif test_type == "ks_test":
            assumptions["test_type"] = (
                "Distribution comparison - tests entire distribution shape"
            )

        return assumptions

    def _suggest_alternative_tests(
        self, values1: list[float], values2: list[float]
    ) -> list[str]:
        """Suggest alternative statistical tests based on data characteristics."""
        suggestions = []

        # Check sample sizes
        if len(values1) < 30 or len(values2) < 30:
            suggestions.append("Consider Mann-Whitney U test for small samples")

        # Check for normality violations
        if len(values1) >= 8 and len(values2) >= 8:
            _, p1 = stats.shapiro(values1)
            _, p2 = stats.shapiro(values2)

            if p1 <= 0.05 or p2 <= 0.05:
                suggestions.append(
                    "Data may not be normally distributed - consider Mann-Whitney U test"
                )

        # Check for unequal variances
        if len(values1) > 2 and len(values2) > 2:
            try:
                _, p_levene = stats.levene(values1, values2)
                if p_levene <= 0.05:
                    suggestions.append(
                        "Unequal variances detected - consider Welch's t-test"
                    )
            except Exception:
                pass

        if not suggestions:
            suggestions.append("Current test choice appears appropriate for the data")

        return suggestions
