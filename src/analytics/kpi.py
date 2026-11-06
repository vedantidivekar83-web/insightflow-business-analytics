"""Business KPI computation engine for InsightFlow.

Infers domain-specific metrics (revenue, AOV, order status) when recognizable
columns exist, while providing generic dimensional summaries for non-sales datasets.
Clearly labels all assumptions.
"""

from typing import Any
import re
import unicodedata
import pandas as pd
import numpy as np


REVENUE_CANDIDATES: list[str] = [
    "revenue",
    "sales",
    "turnover",
    "total_revenue",
    "total_sales",
    "total_amount",
    "total_price",
    "amount",
    "grand_total",
    "sales_amount",
    "salesamount",
    "order_amount",
    "orderamount",
    "order_value",
    "ordervalue",
    "gross_sales",
    "grosssales",
    "net_sales",
    "netsales",
    "invoiced_amount",
    "billing_amount",
    "subtotal",
    "ventes",
    "chiffre_affaires",
    "chiffredaffaires",
    "ingresos",
    "benefice",
    "umsatz",
]

QUANTITY_CANDIDATES: list[str] = [
    "quantity",
    "qty",
    "units",
    "volume",
    "item_count",
    "order_qty",
    "units_sold",
    "order_quantity",
    "orderquantity",
    "quantity_ordered",
    "quantityordered",
    "qty_ordered",
    "qtyordered",
    "num_units",
    "numunits",
    "items_count",
    "itemscount",
    "items",
    "no_of_items",
    "quantite",
    "cantidad",
    "unites",
    "menge",
]

UNIT_PRICE_CANDIDATES: list[str] = [
    "unit_price",
    "unitprice",
    "price",
    "item_price",
    "rate",
    "price_per_unit",
    "price_each",
    "priceeach",
    "unit_cost",
    "unitcost",
    "cost_per_unit",
    "costperunit",
    "selling_price",
    "sellingprice",
    "prix_unitaire",
    "prixunitaire",
    "prix",
    "cout",
    "cout_unitaire",
    "coutunitaire",
    "precio_unitario",
    "preciounitario",
    "precio",
    "einzelpreis",
]

STATUS_CANDIDATES: list[str] = [
    "status",
    "order_status",
    "state",
    "order_state",
    "delivery_status",
    "payment_status",
    "fulfillment_status",
    "statut",
    "etat",
    "estado",
]


def _clean_header(name: str) -> str:
    """Normalize a column header by stripping brackets, accents, punctuation, and whitespace."""
    # Strip parenthetical/bracketed qualifiers like ($), [USD], (Units)
    no_brackets = re.sub(r"[\(\[\{].*?[\)\]\}]", "", str(name))
    # Normalize unicode (NFKD decomposes accented characters, e.g., 'é' -> 'e' + combining acute accent)
    normalized = unicodedata.normalize("NFKD", no_brackets)
    # Strip combining diacritical marks (Unicode category 'Mn') to extract base ASCII Latin characters
    ascii_clean = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", ascii_clean.lower())


def _match_column(df_columns: list[str], candidates: list[str]) -> str | None:
    """Find the best column matching any candidate name case-insensitively with punctuation stripping."""
    normalized = {_clean_header(c): str(c) for c in df_columns}
    for cand in candidates:
        norm_cand = _clean_header(cand)
        if norm_cand in normalized:
            return normalized[norm_cand]
    return None


def _try_parse_numeric_series(series: pd.Series) -> pd.Series | None:
    """Extract clean float values from numeric or currency/comma-formatted string columns."""
    if pd.api.types.is_numeric_dtype(series):
        clean = pd.to_numeric(series, errors="coerce")
        if clean.notna().sum() > 0:
            return clean
    elif pd.api.types.is_string_dtype(series) or series.dtype == object:
        # Strip currency symbols, commas, and whitespace
        cleaned = series.astype(str).str.replace(r"[^\d.-]", "", regex=True)
        clean = pd.to_numeric(cleaned, errors="coerce")
        # Validate that at least half of the non-null entries were parsed as numbers
        non_null_count = series.dropna().shape[0]
        if non_null_count > 0 and clean.notna().sum() >= max(1, int(0.5 * non_null_count)):
            return clean
    return None


def detect_business_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Inspect DataFrame headers for identifiable business dimensions."""
    if df.empty:
        return {
            "revenue": None,
            "quantity": None,
            "unit_price": None,
            "status": None,
        }

    cols = list(df.columns)
    return {
        "revenue": _match_column(cols, REVENUE_CANDIDATES),
        "quantity": _match_column(cols, QUANTITY_CANDIDATES),
        "unit_price": _match_column(cols, UNIT_PRICE_CANDIDATES),
        "status": _match_column(cols, STATUS_CANDIDATES),
    }


def compute_business_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute business KPIs based on detected schema or generic dataset summaries.

    Returns:
        dict containing total_records, total_columns, inferred_kpis (with assumptions),
        status_distribution, and primary_numerical_metric.
    """
    if df.empty:
        return {
            "total_records": 0,
            "total_columns": 0,
            "inferred_kpis": [],
            "status_distribution": None,
            "primary_metric": None,
            "detected_columns": {},
        }

    total_records = len(df)
    total_columns = len(df.columns)
    detected = detect_business_columns(df)
    inferred_kpis: list[dict[str, Any]] = []

    # 1. Revenue & Transaction Metrics (if applicable)
    rev_col = detected["revenue"]
    qty_col = detected["quantity"]
    price_col = detected["unit_price"]

    clean_rev = _try_parse_numeric_series(df[rev_col]) if rev_col else None
    clean_qty = _try_parse_numeric_series(df[qty_col]) if qty_col else None
    clean_price = _try_parse_numeric_series(df[price_col]) if price_col else None

    if rev_col and clean_rev is not None:
        total_rev = float(clean_rev.dropna().sum())
        avg_rev = float(clean_rev.dropna().mean()) if len(clean_rev.dropna()) > 0 else 0.0

        inferred_kpis.append(
            {
                "name": "Total Revenue",
                "value": total_rev,
                "formatted": f"${total_rev:,.2f}",
                "assumption": f"Directly aggregated from detected numeric column '{rev_col}'.",
                "source_columns": [rev_col],
            }
        )
        inferred_kpis.append(
            {
                "name": "Average Transaction Value",
                "value": avg_rev,
                "formatted": f"${avg_rev:,.2f}",
                "assumption": f"Mean value of detected revenue column '{rev_col}'.",
                "source_columns": [rev_col],
            }
        )
    elif qty_col and price_col and clean_qty is not None and clean_price is not None:
        computed_rev = (clean_qty.fillna(0) * clean_price.fillna(0)).sum()
        avg_order = computed_rev / total_records if total_records > 0 else 0.0

        inferred_kpis.append(
            {
                "name": "Total Inferred Revenue",
                "value": float(computed_rev),
                "formatted": f"${float(computed_rev):,.2f}",
                "assumption": f"Inferred by multiplying '{qty_col}' × '{price_col}'.",
                "source_columns": [qty_col, price_col],
            }
        )
        inferred_kpis.append(
            {
                "name": "Inferred Average Order Value (AOV)",
                "value": float(avg_order),
                "formatted": f"${float(avg_order):,.2f}",
                "assumption": f"Inferred total revenue divided by {total_records:,} records.",
                "source_columns": [qty_col, price_col],
            }
        )

    # 2. Status Distribution (if applicable)
    status_distribution: pd.DataFrame | None = None
    status_col = detected["status"]
    if status_col:
        status_series = df[status_col].dropna().astype(str)
        if not status_series.empty:
            counts = status_series.value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            counts["Percentage (%)"] = (counts["Count"] / total_records * 100).round(1)
            status_distribution = counts

    # 3. Generic Numerical Metric (for non-sales or secondary summary)
    primary_metric: dict[str, Any] | None = None
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID-like or index columns, and exclude revenue if already shown
    metric_candidates = [
        c for c in num_cols
        if not str(c).lower().endswith("id") and (c != rev_col if rev_col else True)
    ]
    selected_metric_col = metric_candidates[0] if metric_candidates else (num_cols[0] if num_cols else None)

    if selected_metric_col:
        series = pd.to_numeric(df[selected_metric_col], errors="coerce").dropna()
        if not series.empty:
            primary_metric = {
                "column_name": selected_metric_col,
                "sum": float(series.sum()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(len(series)),
                "assumption": f"Primary numerical metric identified as '{selected_metric_col}'.",
            }

    return {
        "total_records": total_records,
        "total_columns": total_columns,
        "inferred_kpis": inferred_kpis,
        "status_distribution": status_distribution,
        "primary_metric": primary_metric,
        "detected_columns": detected,
    }
