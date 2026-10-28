# QuantAgent-Desk Risk Methodology Summary

## Value-at-Risk (VaR)

VaR at confidence level α is the loss threshold such that the probability of losses exceeding VaR is at most (1-α).

- **Historical VaR**: Sorted historical P&L vector; take the empirical quantile.
- **Parametric VaR**: Assumes normal distribution of P&L. VaR = -(μ + σ * Φ⁻¹(1-α)).
- **Expected Shortfall (ES)**: Average of losses beyond VaR (also called CVaR). Preferred under Basel for tail risk.

Default levels used: VaR 95%, VaR 99%, ES 97.5%.

## Stress Testing

Stress tests apply extreme but plausible market moves to the portfolio.

Pre-defined historical scenarios:
- **2008 GFC**: Equity -40%, rates down 150 bps, credit spreads +150 bps, VIX +40.
- **COVID March 2020**: Equity -34%, rates down 120 bps, oil -60%, VIX +50.
- **2022 Rates Shock**: USD 10Y +250 bps, equity -20%, credit +80 bps.

Generative scenarios combine historical analogues with additional narrative-driven shocks produced by the agent.

## Limit Framework

Soft warning at 80% utilization, hard breach at 100%.
Monitored metrics: VaR 95%, VaR 99%, ES 97.5%, Peak PFE, CVA.

## Portfolio Hierarchy

Firm → Desk → Book → Positions.
Aggregation currently uses simple sum (no diversification benefit) for conservative reporting.

## Numeric Grounding Policy

Every numeric claim in agent responses must be traceable to either:
1. A calculation engine output (MarketRiskEngine / StressEngine), or
2. A retrieved document snippet with explicit citation.
