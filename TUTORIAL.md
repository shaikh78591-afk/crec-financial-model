# CREC Financial Digital Twin — Team Tutorial

Welcome! This guide walks you through every part of the app so you can explore scenarios, run simulations, and interrogate the model on your own.

---

## 1. Getting Started

### Launch the App
```bash
pip install -r requirements.txt
streamlit run crec_report_fixed.py
```
Open the URL shown in your terminal (usually `http://localhost:8501`).

### First Look
- **Sidebar** (left): Controls, switches, tuning sliders, and tools.
- **Main area**: KPIs, charts, insights, and chat.

---

## 2. Sidebar Overview

### Navigation Radio
At the top you'll see **Base Model** and **Scenario Explorer** tabs. Pick one to switch context.

### Switches (from Data Sources)
Expand this section to toggle the three Excel-driven switches:

| Switch | What It Does |
|--------|--------------|
| **Graphene Switch** | Toggles a CapEx line item in Excel. If graphene yield/price params exist, adds graphene revenue. |
| **RNG Switch** | Sets electricity price to $0 (diverts power to RNG). Revenue drops; scale stays the same. |
| **Tires Only Switch** | Gates certain CapEx rows in Excel. Throughput unchanged in app. |

Click **Reset from Excel** to reload the workbook values.

### Scenario Controls
Adjust these sliders to build "what-if" scenarios:

| Slider | Range | Effect |
|--------|-------|--------|
| MSW TPD | 400–800 | Municipal solid waste per day |
| Tires TPD | 50–200 | Tire feedstock per day |
| Power Price ($/kWh) | 0.05–0.15 | Electricity sale price |
| Inflation Rate | 1%–5% | Annual inflation assumption |
| CapEx Multiplier | 0.8×–1.2× | Scale initial capital outlay |

- **Save Scenario**: Stores the current inputs under a name you provide.
- **Run Simulation**: Recalculates IRR, NPV, cash flows, and updates all charts.

### What-if Tuning (Base Model tab)
Quick percentage shocks without changing core inputs:

| Slider | Effect |
|--------|--------|
| Revenue adjustment (%) | Scales revenue curve |
| Expense adjustment (%) | Scales operating expenses |
| CapEx adjustment (%) | Scales initial capital |
| Discount rate shift (bps) | Adjusts WACC for NPV |

Click **Reset tuning** to revert.

### Forecast Explorer
- Pick a saved scenario and metric (Revenue, Expenses, Cash Flow).
- Set a horizon (1–10 years).
- Click **Generate Forecast** to see a trend-based projection.

### Monte Carlo (Revenue Drivers)
Stress-test assumptions:
1. Set iterations (e.g., 500).
2. Adjust volatility sliders for MSW, Tires, Power Price.
3. Click **Run Monte Carlo**.
4. View IRR/NPV distributions, scatter plots, and percentile summaries.

### Scenario Optimizer
Find the inputs that hit a target IRR:
1. Enter a target (e.g., 20%).
2. Set sample count and tolerance.
3. Click **Optimize Scenario**.
4. Review the best-found configuration and save it.

### Validation & Diagnostics
Click **Validate Switch Impacts** to run automated checks confirming the three switches behave as expected.

---

## 3. Main Area — KPIs & Charts

### Base Model Highlights
Six headline metrics with deltas vs. the Excel Summary tab:

| Metric | What It Means |
|--------|---------------|
| Project IRR | Internal rate of return for the project |
| Equity IRR | Return to equity holders |
| Net Present Value | Discounted value of future cash flows |
| Total Project Cash Flow | Cumulative operating cash |
| Total Equity Contributed | Capital injected by sponsors |
| Loan-to-Value | Debt as a share of asset value |

Teal badges = positive delta; red = negative.

### Charts (scroll down)
1. **Revenue vs Expenses** — annual trajectories by scenario.
2. **Cumulative Cash Flow** — payback visualization.
3. **IRR Comparison** — bar chart across scenarios.
4. **NPV Comparison** — bar chart across scenarios.
5. **Sensitivity Tornado** — proxy sensitivity on revenue drivers.
6. **Revenue Breakdown (Donut)** — power vs other streams.
7. **Mass-Balance Sankey** — MSW → RDF → Syngas → Power flow.
8. **Salary Breakdown (Pie)** — labor cost distribution.
9. **CapEx by Country (Bar)** — geographic allocation.
10. **Power Production Profile** — output over time.
11. **Heatmap** — year × metric intensity.
12. **IRR vs NPV Scatter** — risk/return positioning.
13. **Cash Flow Box Plot** — distribution by scenario.

Each chart includes:
- **Source**: the Excel tab it draws from.
- **Insight**: a short interpretation.
- **Recommended Actions**: next steps.

---

## 4. Chat with the Model

Located at the bottom of each tab. Type natural-language questions or commands.

### Example Questions
| You Type | What Happens |
|----------|--------------|
| "What is the project IRR?" | Returns the current IRR with sources. |
| "Compare Base vs Optimized" | Shows IRR, NPV, cash-on-cash side by side. |
| "Summarize Monte Carlo results" | Gives percentile ranges for IRR/NPV. |
| "Explain why IRR changed" | Describes delta drivers. |

### Example Commands
| You Type | What Happens |
|----------|--------------|
| "Set MSW TPD to 650" | Updates the slider and confirms. |
| "Set power price to 0.09" | Adjusts pricing assumption. |
| "Optimize to hit 20% IRR" | Runs optimizer, saves best scenario, returns summary. |

### Sources
Every response ends with a **Sources** line listing the Excel tabs or app modules used. This is your audit trail.

---

## 5. Exporting a Report

1. Scroll to the bottom of the main area.
2. Click **Export PDF Report**.
3. A PDF is generated with:
   - Scenario summaries
   - Key metrics and deltas
   - Custom KPIs
   - Optimizer results (if run)
   - Monte Carlo summary (if run)

Share this with stakeholders who don't have app access.

---

## 6. Custom KPIs

Define your own metrics using Python-style formulas.

### How to Add
1. In the app, find the **Custom KPIs** section.
2. Enter a name (e.g., "Operating Margin %").
3. Enter a formula (e.g., `(total_revenue - total_expenses) / total_revenue * 100`).
4. Click **Add KPI**.

### Available Fields
`irr`, `npv`, `capex`, `total_revenue`, `total_expenses`, `total_cf`, `avg_cf`, `peak_cf`, `min_cf`, `margin`, `cash_on_cash`, `payback_year`, `payback_year_label`

Plus helpers: `abs`, `min`, `max`, `round`, `np`.

---

## 7. Quick Reference — Keyboard & Tips

| Tip | Details |
|-----|---------|
| **Collapse sidebar** | Click the `<` arrow to get more chart space. |
| **Compare scenarios** | Use the multi-select dropdown to overlay multiple scenarios on charts. |
| **Reset everything** | Refresh the browser; session state clears. |
| **Check Excel alignment** | Use Validation & Diagnostics to confirm switch behavior. |

---

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| "LLM not configured" | Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or export it in your terminal. |
| Charts not updating | Click **Run Simulation** after changing sliders. |
| Switches not reflecting | Click **Reset from Excel** in the Switches expander. |
| Metrics show N/A | Ensure `crec_model.xlsx` is present and sheets are intact. |

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **IRR** | Internal Rate of Return — the discount rate at which NPV = 0. |
| **NPV** | Net Present Value — sum of discounted future cash flows. |
| **CapEx** | Capital Expenditure — upfront investment. |
| **WACC** | Weighted Average Cost of Capital — discount rate. |
| **RDF** | Refuse-Derived Fuel — processed MSW. |
| **TDF** | Tire-Derived Fuel — processed tires. |
| **Syngas** | Synthesis gas produced from RDF. |
| **LTV** | Loan-to-Value ratio. |

---

## 10. Getting Help

- **In-app**: Use the chat to ask clarifying questions.
- **README.md**: Developer notes and project structure.
- **Contact**: Reach out to the project maintainer for deeper Excel or model questions.

---

Happy exploring! 🚀

