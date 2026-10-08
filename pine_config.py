"""Central configuration: settings from secrets/env and domain constants.

Secrets are read from Streamlit secrets first (Streamlit Community Cloud), then from
environment variables (local `.env` / Docker). Nothing sensitive is hard-coded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _secret(key: str, default: str | None = None) -> str | None:
    """Return a secret from st.secrets, falling back to the environment."""
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no secrets.toml is a normal local state
        pass
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    """Runtime settings."""

    database_url: str
    admin_username: str | None
    admin_password: str | None
    admin_name: str
    session_hours: int
    max_failed_logins: int
    lockout_minutes: int
    live_ttl_seconds: int
    random_seed: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings once per process."""
    return Settings(
        database_url=_secret("DATABASE_URL", "sqlite:///pine_dashboard.db") or "",
        admin_username=_secret("ADMIN_USERNAME"),
        admin_password=_secret("ADMIN_PASSWORD"),
        admin_name=_secret("ADMIN_NAME", "Administrator") or "Administrator",
        session_hours=int(_secret("SESSION_HOURS", "8") or 8),
        max_failed_logins=int(_secret("MAX_FAILED_LOGINS", "5") or 5),
        lockout_minutes=int(_secret("LOCKOUT_MINUTES", "15") or 15),
        live_ttl_seconds=int(_secret("LIVE_TTL_SECONDS", "3600") or 3600),
        random_seed=int(_secret("RANDOM_SEED", "7") or 7),
    )


# --------------------------------------------------------------------------- market
TICKER = "PINE4.SA"
BENCHMARK = "^BVSP"
PEERS: dict[str, str] = {
    "ABCB4.SA": "ABC Brasil",
    "BPAN4.SA": "Banco Pan",
    "BMGB4.SA": "Banco BMG",
    "BRSR6.SA": "Banrisul",
    "BPAC11.SA": "BTG Pactual",
    "ITUB4.SA": "Itau Unibanco",
}
TICKER_LABELS: dict[str, str] = {TICKER: "Banco Pine", BENCHMARK: "Ibovespa", **PEERS}

# --------------------------------------------------------------- Banco Central (SGS)
# https://dadosabertos.bcb.gov.br - series codes of the SGS time-series system.
SGS_SERIES: dict[str, int] = {
    "selic_target": 432,  # Selic target, % p.a. (daily)
    "ipca_month": 433,  # IPCA, % monthly change
    "usd_brl": 1,  # PTAX USD/BRL sell rate (daily)
    "npl_total": 21082,  # System NPL 90+ (% of portfolio), total
    "npl_pj": 21083,  # System NPL 90+, companies
    "npl_pf": 21084,  # System NPL 90+, households
}
SGS_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
FOCUS_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
    "ExpectativasMercadoAnuais"
)
HTTP_TIMEOUT_S = 20

# ------------------------------------------------------------------- credit book
SEGMENTS: list[str] = [
    "Corporate",
    "Consignado Privado",
    "Consignado INSS",
    "FGTS Antecipacao",
    "Home Equity",
    "Cartao Consignado",
]
RETAIL_SEGMENTS: list[str] = SEGMENTS[1:]
REGIONS: list[str] = ["Southeast", "South", "Center-West", "Northeast", "North"]
RATINGS: list[str] = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "Default"]

# Regulatory minimum total capital incl. conservation buffer (Res. CMN 4.958/4.955).
BASEL_MINIMUM = 0.105
PRIVATE_PAYROLL_RATE_CAP = 0.0199  # monthly cap for FGTS-backed private payroll loans

ROLES: tuple[str, ...] = ("admin", "analyst", "viewer")
