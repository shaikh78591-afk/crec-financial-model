# CREC Financial Digital Twin Simulator

A Streamlit application for exploring, simulating, and interrogating the CREC financial model. The app ingests the existing `crec_model.xlsx` workbook and layers interactive controls, custom KPIs, rich data visualizations, and an LLM-powered chat assistant backed by retrieval-augmented generation (RAG).

## Features

- **Scenario Controls** – Adjust MSW/Tires throughput, power price, inflation, and CapEx multiplier from the sidebar and rerun the full mass-balance + financial stack with one click.
- **Dynamic Metrics** – IRR, NPV, cash flows, and other headline statistics are recomputed instantly for every scenario you select.
- **Custom KPIs** – Define your own metrics (e.g., operating margin, cash-on-cash ratios) using formula fields. KPIs appear alongside IRR/NPV for every scenario and feed into the chat assistant.
- **Interactive Visualizations** – Plotly charts cover revenue vs expenses, cumulative cash flows, IRR/NPV comparisons, sensitivity tornado, Sankey mass balance, salary breakdown, CapEx allocation, power production, heat maps, scatter plots, and distribution box plots.
- **Explainability Blocks** – Each chart includes a short interpretation, source reference (Excel tab), and actionable recommendations.
- **Data Inferences & Insights** – Automatically generated narrative insights plus dynamic recommendations based on simulation outcomes.
- **Monte Carlo Engine** – Stress-test power price and throughput assumptions with configurable volatility; view IRR/NPV distributions, scatter plots, and summary stats.
- **Forecast Explorer** – Trend-based projections (linear regression) for revenue/expense/cash flow with configurable horizons and summaries.
- **Scenario Optimizer** – Randomized search that hunts for input combinations meeting a target IRR; saves the best configuration back into the scenario list.
- **AI Scenario Suggestions** – Automatic “what-if” recommendations (e.g., boost MSW, renegotiate CapEx) that flow into both the UI and chat responses.
- **LLM Chat Assistant** – Ask natural-language questions about the model. The assistant uses a RAG index built from every workbook tab, plus scenario metrics and recent chat history, to answer with citations. It also understands commands like “set power price to 0.09” and updates the sliders accordingly.
- **Exportable Report** – One-click PDF export summarizing each scenario with key metrics and insights.

## Requirements

- Python 3.10+
- Streamlit 1.37+
- Plotly, pandas, numpy, scikit-learn, numpy-financial, openai (installed via `pip install -r requirements.txt` if provided)
- `crec_model.xlsx` located at the project root

## Configuration

### OpenAI API Key

The chat assistant requires an OpenAI key. Place it in `.streamlit/secrets.toml` (already scaffolded):

```
OPENAI_API_KEY = "sk-..."
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # optional override
```

Alternatively, export `OPENAI_API_KEY` in your terminal environment before launching Streamlit.

### Custom KPI Examples

- Operating Margin (%)
  ```
  (total_revenue - total_expenses) / total_revenue * 100
  ```
- Cash-on-Cash Multiple
  ```
  cash_on_cash
  ```
- Average Annual Cash Flow ($M)
  ```
  avg_cf / 1_000_000
  ```
- Payback Year
  ```
  payback_year
  ```
- Revenue per CapEx Dollar
  ```
  total_revenue / abs(capex)
  ```

KPI fields available inside formulas: `irr`, `npv`, `capex`, `total_revenue`, `total_expenses`, `total_cf`, `avg_cf`, `peak_cf`, `min_cf`, `margin`, `cash_on_cash`, `payback_year`, `payback_year_label`, plus Python helpers like `abs`, `min`, `max`, `round`, and `np`.

## Running the App

1. Install dependencies (example):
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `crec_model.xlsx` is present in the repository root.
3. Add your OpenAI key to `.streamlit/secrets.toml` (or export `OPENAI_API_KEY`).
4. Launch Streamlit:
   ```bash
   streamlit run crec_report_fixed.py
   ```
5. Open the provided local URL in your browser.

## Usage Tips

- **Scenario Comparison** – Use the multi-select dropdown to compare multiple saved scenarios; all charts and KPIs will overlay accordingly.
- **Chat Commands** – The chat assistant understands direct questions and simple commands such as “set MSW to 650” or “increase capex multiplier to 1.1”. If a command is recognized, the inputs update and the assistant confirms the change.
- **Monte Carlo Runs** – Expand “Monte Carlo (Revenue Drivers)” in the sidebar, choose volatility assumptions, and click “Run Monte Carlo” to see distribution charts and summary tables in the main view. The assistant now cites Monte Carlo results when you ask about risk ranges.
- **Evidence Trail** – Every chat response lists the Excel tabs (“sources”) used to generate the answer. If you don’t see a relevant source, rerun the simulation or add a custom KPI so it becomes part of the context.
- **Report Export** – After running scenarios, click “Export PDF Report” near the bottom to share a compact summary with stakeholders.

## Project Structure

```
├── crec_report_fixed.py    # Streamlit application (kept intact)
├── src/                    # Lightweight, production-friendly module facades
│   ├── scenario.py         # build_scenario, metrics, mass balance (facade)
│   ├── optimizer.py        # optimize_scenario, explanation (facade)
│   ├── monte_carlo.py      # run_monte_carlo (facade)
│   ├── switches.py         # parse Data Sources switches (facade)
│   ├── excel_io.py         # Excel loading helpers (facade)
│   └── chatbot.py          # chat helpers (facade)
├── assets/
│   └── theme.css           # External CSS overrides (loaded automatically)
├── crec_model.xlsx         # Excel workbook consumed by the app
├── .streamlit/
│   └── secrets.toml        # OpenAI credentials / configuration
│   └── config.toml         # Theme + server configuration
├── README.md               # This document
└── ...                     # (Optional) requirements, virtual env, etc.
```

## Switches (Data Sources)

- The app reads the Excel `Data Sources` sheet and exposes three switches in the sidebar:
  - Graphene Switch
  - RNG Switch
  - Tires Only Switch
- Impacts:
  - Tires Only: Gates specific CapEx line items (per CapEx sheet logic). Throughput is not altered in the app.
  - RNG: Sets electricity price to $0 (mirrors `Data Sources!C30 = IF(C26="N", D30, 0)`), dropping power revenue while leaving throughput unchanged. Optionally, if RNG price/yield parameters are provided in `Data Sources`, RNG revenue can be added.
  - Graphene: Toggles a specific CapEx item (e.g., `CapEx!D87`) per Excel. Optionally, if graphene price/yield parameters are provided, incremental revenue can be modeled.

### Validation & Diagnostics

- In the sidebar, open “Validation & Diagnostics” and click “Validate Switch Impacts”.
- The app runs quick checks to confirm the modeled impacts:
  - RNG: revenue decreases via zero electricity price (scale unchanged).
  - Tires Only: no forced MSW change; primarily CapEx gating.
  - Graphene: CapEx toggle; revenue check runs only if supporting params are present.

## Developer Notes (Organization)

- The current app remains fully functional and monolithic for continuity, while `src/` provides importable facades around core features (scenario, optimizer, Monte Carlo, switches, Excel IO, chat).
- This allows incremental migration into modules without breaking any behavior today.
- External theme overrides live in `assets/theme.css` (auto-injected on startup).
- Streamlit config lives in `.streamlit/config.toml` (theme, server settings).

## Extending Further

- Hook into an ERP feed or data lake for automated workbook refreshes.
- Persist scenarios and custom KPIs in a shared database so teams can collaborate.
- Add governance features like access controls, audit trails, and alerting for KPI threshold breaches.
- Expand the LLM assistant with multi-step “what-if” automations (e.g., “optimize MSW to hit 20% IRR”).

---
Feel free to customize the README with company-specific deployment steps or security notes before sharing with your colleague.
