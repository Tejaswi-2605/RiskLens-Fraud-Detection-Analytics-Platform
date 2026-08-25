# Transaction Risk Scoring and Decision Policy

*Illustrative internal policy written for the RiskLens project. Not a real
policy of any institution.*

## Purpose

This policy defines how the RiskLens model score is converted into an
approve, review or decline decision, and who is accountable for each outcome.

## Scope

Applies to all card-present and card-not-present transactions scored by the
RiskLens production model.

## Risk bands

The model outputs a calibrated probability of fraud. That probability is
mapped to one of five risk bands. Bands, not raw probabilities, drive
operational routing.

| Band | Probability | Decision | Owner |
|---|---|---|---|
| MINIMAL | below 0.15 | Approve automatically | Automated |
| LOW | 0.15 to 0.40 | Approve, log for trend monitoring | Automated |
| MEDIUM | 0.40 to 0.70 | Queue for same-day analyst review | Fraud Operations |
| HIGH | 0.70 to 0.90 | Hold and review within one hour | Fraud Operations |
| CRITICAL | 0.90 and above | Decline and contact cardholder immediately | Fraud Operations |

## Threshold setting

Decision thresholds must be set by minimising expected financial loss, not by
maximising a statistical metric such as F1 score or accuracy.

The cost model is:

- A false negative costs the full transaction amount, because the institution
  refunds the cardholder and absorbs the loss.
- A false positive costs the analyst review time plus estimated customer
  friction. The standing assumption is 15 currency units per false positive.
- A true positive recovers approximately 90 per cent of the transaction value;
  recovery is not complete because some funds have already moved.

Because the false negative cost varies with transaction amount while the false
positive cost is roughly fixed, a single global probability threshold is
suboptimal. Higher-value transactions justify intervention at a lower
probability. Amount-banded thresholds are permitted where they can be shown to
reduce expected loss.

## Alert budget constraint

Thresholds must also respect the operational alert budget. Fraud Operations
can review a fixed number of alerts per day. A threshold that produces more
alerts than the team can review is not viable regardless of its precision or
recall, because unreviewed alerts provide no protection.

Where the cost-optimal threshold exceeds the alert budget, the budget
constraint takes precedence and the shortfall must be recorded in the monthly
risk report.

## Precision floor

Alert precision must not fall below 20 per cent sustained over any rolling
seven-day period. Below this level analysts lose confidence in the queue and
review quality degrades. Breach of the precision floor requires threshold
recalibration within five working days.

## Override authority

- A Fraud Analyst may override a MEDIUM or HIGH decision with documented
  reasoning.
- A CRITICAL decline may only be overridden by a Fraud Team Lead.
- All overrides are logged and reviewed weekly. Override patterns are a
  leading indicator of model degradation.

## Review cadence

Thresholds are reviewed monthly, and immediately following any of:

- a Population Stability Index above 0.25 on any monitored feature
- a sustained precision floor breach
- a material change to the product mix or customer channel
