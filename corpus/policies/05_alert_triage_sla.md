# Alert Triage and Service Levels

*Illustrative internal policy written for the RiskLens project. Not a real
policy of any institution.*

## Queue structure

Alerts are routed to one of three queues by risk band.

| Queue | Bands | Target response |
|---|---|---|
| Immediate | CRITICAL | Within fifteen minutes |
| Priority | HIGH | Within one hour |
| Standard | MEDIUM | Same working day |

LOW and MINIMAL band transactions are not queued. They are logged for trend
analysis only.

## Triage sequence

For each alert the analyst must, in order:

1. Read the model reason codes. These state which features drove the score and
   in which direction. Reason codes are generated from the model itself and
   are faithful to it; they are not an analyst interpretation.
2. Review similar historical cases retrieved by the case search system, and
   note their recorded outcomes.
3. Check the relevant control policy for the indicated fraud pattern.
4. Review account and card history for velocity or pattern breaks.
5. Record a decision with written reasoning.

## Evidence standard

A decision to decline must cite at least two independent indicators. A single
elevated model score is not sufficient grounds for decline, because the model
is a probabilistic instrument and false positives are expected at every
operating point.

The reasoning recorded must be specific enough that a second analyst reading
it could reach the same conclusion without re-reviewing the raw data.

## Escalation

Escalate to a Fraud Team Lead where:

- The transaction value exceeds the analyst authority limit.
- The customer is flagged as vulnerable.
- The case indicates a pattern affecting multiple accounts, which may signal a
  coordinated attack rather than an isolated incident.
- The analyst assesses the model score as materially wrong. These cases are
  the most valuable input to model improvement and must be tagged for review.

## Quality assurance

A random five per cent sample of closed alerts is reviewed weekly. Reviews
assess whether the recorded reasoning supports the decision that was taken,
not whether the decision was correct with hindsight. Judging analysts on
hindsight discourages them from acting on genuine uncertainty.
