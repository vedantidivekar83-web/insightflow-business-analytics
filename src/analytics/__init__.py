"""Analytics, statistical computing, and decision-support engines.

Responsible for business KPIs, trend detection, anomaly discovery, and statistical metrics.
"""

from src.analytics.statistics import compute_numerical_statistics
from src.analytics.quality import (
    compute_data_quality_summary,
    compute_column_quality_table,
)
from src.analytics.kpi import (
    detect_business_columns,
    compute_business_kpis,
)
from src.analytics.correlation import (
    compute_correlation_matrix,
    get_top_correlations,
)
from src.analytics.insights import (
    generate_smart_insights,
    generate_quality_insights,
    generate_numerical_insights,
    generate_categorical_insights,
    generate_correlation_insights,
    generate_time_insights,
    generate_business_insights,
)

__all__ = [
    "compute_numerical_statistics",
    "compute_data_quality_summary",
    "compute_column_quality_table",
    "detect_business_columns",
    "compute_business_kpis",
    "compute_correlation_matrix",
    "get_top_correlations",
    "generate_smart_insights",
    "generate_quality_insights",
    "generate_numerical_insights",
    "generate_categorical_insights",
    "generate_correlation_insights",
    "generate_time_insights",
    "generate_business_insights",
]
