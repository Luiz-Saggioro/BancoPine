# BancoPine

Author: Luiz Saggioro
Date: 10/7/2026
Purpose: Create an interactive dashboard with live data and predictive AI in order to provide value to Banco Pine and its shareholders.

**Banco Pine Intelligence** is a Streamlit app with five analysis pages, 15 cross-filtering visuals and 12 predictive models. It has its own login and a prediction ledger that stores every forecast the models make and scores it once the actual value is known.

## The 15 visuals

| # | Page | Visual | Model | Business question it answers |
|---|------|--------|-------|-------------------------------|
| 01 | Market | PINE4 price and 20-day forecast fan | Quantile gradient boosting (P10/P50/P90), direct multi-horizon | Where is the stock likely to trade, with what uncertainty? Timing for buybacks / follow-ons |
| 02 | Market | Listed-bank peers: realised return vs model outlook | Same model per peer | How does Pine screen against ABC, Pan, BMG, Banrisul, BTG, Itau? |
| 03 | Market | Market regime detection | Gaussian mixture on return/volatility | Is the market in a stressed regime (funding / offering risk)? |
| 04 | Macro | Selic and IPCA: actuals, Focus consensus, model | Seasonal ARIMA (IPCA) + Focus-anchored Selic path | What rate and inflation path drives margins and credit? |
| 05 | Macro | System delinquency outlook under Selic scenarios | Ridge regression on lagged Selic, IPCA, NPL | How much will household / corporate NPL move? Feeds PD multipliers |
| 06 | Credit | Exposure map: segment x region, colored by PD | Gradient-boosted PD model | Where is the risk concentrated? |
| 07 | Credit | Portfolio growth and 4-quarter forecast | Damped-trend exponential smoothing | Is the 40% YoY growth sustainable, by segment? |
| 08 | Credit | PD model calibration and drivers | Gradient boosting + permutation importance | Can the PD be trusted, and what drives default? |
| 09 | Credit | Rating migration and projected mix | Markov chain on the empirical transition matrix | How will rating quality evolve over 1-2 years? |
| 10 | Credit | Expected credit loss by segment and scenario | PD x LGD x EAD with macro multipliers from 05 | Provisions under Base / Adverse / Severe and a Selic shock |
| 11 | Credit | Vintage curves by cohort | Weibull curve fit (shared shape, cohort level) | Are new consignado privado cohorts performing worse? |
| 12 | Clients | Portability risk and retention playbook | Gradient-boosted churn model + next-best-offer propensities | Which clients will be refinanced away, how much margin is at risk, what to offer |
| 13 | Clients | Consignado privado demand and pricing under the 1.99% cap | ETS demand forecast + logistic acceptance model | What rate maximises contribution under the legal cap? |
| 14 | Treasury | Repricing gap and NII simulation | Balance-sheet repricing simulator on Selic paths | How does NII react to rate cuts or hikes? |
| 15 | Treasury | Capital adequacy projection and growth capacity | Monte Carlo (3,000 paths) of reference equity and RWA | How fast can the book grow before Basel nears 10.5%? |

### How the visuals interact
- **Click a bubble (06)**: filters segment and region on the Credit and Clients pages.
- **Click a bar (07, 10)**: filters segment. **Click a rating (09)**: filters rating. **Click a cohort (11)**: shows its forecast band.
- **Click a bank (02)**: switches visuals 01 and 03 to that bank. **Drag on 01**: sets the analysis window for 02 and 03.
- **Lasso clients (12)**: builds a retention list you can export to CSV.
- **Sidebar Selic shock**: flows into 04, 05, the PD multipliers in 10, pricing in 13 and NII in 14. **Credit scenario** drives the ECL headline.
- **Active filters** are listed in the sidebar, with one button to clear them.

## Data
- **Live**: B3 prices from Yahoo Finance (`yfinance`). Banco Central data: SGS series 432 (Selic), 433 (IPCA), 1 (USD/BRL), 21083/21084 (NPL), plus the Focus survey (Olinda OData). Cached for `LIVE_TTL_SECONDS` (default 1 hour).
- **Fallback**: if an upstream source is unreachable, the app switches to a deterministic simulation so it never breaks. Every page shows a tag saying whether it is using **live** or **fallback** data.
- **Loan-level book**: loan-level data is confidential, so the app uses a simulated book calibrated to the published 2Q26 figures: R$21.8 bn expanded portfolio, R$6.5 bn consignado privado, R$7.1 bn wholesale, over-90 2.7%, Basel 14.3%. The sources are listed in `pine_reference.py`. For production, swap `generate_loan_book()` for a warehouse loader that returns the same columns (`LOAN_COLUMNS`). Nothing downstream changes.

## Authentication
- Passwords are hashed with bcrypt (cost 12). There are three roles: `admin` (manages users), `analyst` (can save scenarios), `viewer` (read only).
- An account locks after `MAX_FAILED_LOGINS` failures within `LOCKOUT_MINUTES`. Login error messages are generic, usernames are whitelisted, and sessions expire after `SESSION_HOURS`.
- The first admin is created from `ADMIN_USERNAME` / `ADMIN_PASSWORD` when the users table is empty. Add everyone else on the **Users and access** page.

## Prediction ledger
- Every model writes its forecasts to the `predictions` table once per day per model and scenario. Writes are idempotent, enforced by a unique key.
- Validation metrics go to `model_runs`. Analysts can also save named scenarios, such as a capital plan, on demand.
- The **Prediction ledger** page scores forecasts on live targets (prices, IPCA, NPL) once their date passes. It reports median absolute % error and P10-P90 band coverage, and offers a CSV export.
- Storage is SQLite locally and PostgreSQL in production (set `DATABASE_URL`).

## Upload to GitHub (web, no command line)
1. Unzip `BancoPine.zip`. All the files sit in one folder, plus one small hidden folder called `.streamlit`.
2. On github.com/Luiz-Saggioro/BancoPine, click **Add file > Upload files**. Select every file in the folder, drag them in and commit.
3. Click **Add file > Create new file** and name it `.streamlit/config.toml`. Typing the `/` creates the folder. Paste in the contents of the `config.toml` from the zip and commit. This file sets the light theme. The app still runs without it, but then follows the viewer's system theme.

## Deploy on Streamlit Community Cloud
1. Create a free PostgreSQL database (Supabase or Neon) and copy its connection string. Use the form `postgresql+psycopg2://user:pass@host:5432/db`. The Streamlit Cloud filesystem is wiped on restart, so SQLite would lose the users and the ledger.
2. On share.streamlit.io, choose **New app**, pick the repo, branch `main` and main file `streamlit_app.py`.
3. Under **Advanced settings > Secrets**, paste the contents of `secrets.toml.example` with real values.
4. Deploy, sign in with the admin account and create the other users.

## Run locally (optional)
```bash
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                                  # macOS/Linux: cp; then edit the admin password
streamlit run streamlit_app.py
pytest -q                                               # 34 tests, no network needed
```

## File guide (single folder)
| Files | Role |
|-------|------|
| `streamlit_app.py` | Entry point: login gate and page navigation |
| `pine_config.py` | Settings (secrets/env) and constants |
| `pine_live.py`, `pine_reference.py`, `pine_portfolio.py` | Live B3 / Banco Central data, published Pine figures, calibrated loan book |
| `model_forecasting.py`, `model_credit.py`, `model_clients.py`, `model_treasury.py` | The predictive models |
| `pine_db.py`, `pine_ledger.py` | Database schema and prediction ledger |
| `pine_auth.py`, `pine_state.py`, `pine_datasets.py` | Login, cross-filter state, cached data and models |
| `ui_theme.py`, `ui_charts.py`, `ui_components.py` | Palette, the 15 chart builders, shared UI pieces |
| `page_*.py` | One file per page |
| `test_*.py`, `conftest.py` | Automated tests |
| `requirements.txt`, `secrets.toml.example`, `.env.example`, `.gitignore`, `pyproject.toml` | Setup |

Disclaimer: this is an analytical tool, not investment advice. Simulated loan-level results are illustrative until the warehouse loader is connected.
