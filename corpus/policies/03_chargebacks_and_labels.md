# Chargebacks, Dispute Handling and Label Quality

*Illustrative internal policy written for the RiskLens project. Not a real
policy of any institution.*

## What a chargeback is

A chargeback occurs when a cardholder disputes a transaction and the issuing
bank reverses the payment. Chargebacks are the primary mechanism by which
fraud is discovered and confirmed after the fact.

## Label lag and label maturity

Fraud labels are derived from reported chargebacks. Chargebacks arrive weeks
or months after the transaction they dispute. A transaction is therefore not
reliably labelled until the dispute window has closed.

The standing assumption is a one hundred and twenty day maturity window.
Transactions more recent than this must be treated as having immature labels.

Consequences for modelling:

- Training data must exclude the most recent one hundred and twenty days,
  because negatives in that period include fraud that has not yet been
  reported.
- Performance reported on immature periods will be pessimistic on recall and
  optimistic on precision, and must be marked as provisional.
- Any sudden apparent improvement in model precision on recent data should be
  investigated as a label maturity artefact before being reported as a genuine
  gain.

## Label noise in the negative class

Not all fraud is reported. Unreported fraud is recorded as legitimate. The
negative class therefore contains an unknown proportion of true positives.

This means the problem is more accurately described as positive-unlabelled
than as clean binary classification. Two implications follow:

- Measured precision is a lower bound. Some apparent false positives are
  genuine fraud that was never disputed.
- Achievable precision is capped below one hundred per cent by construction,
  and performance targets must be set with that in mind.

## Label propagation

Where a chargeback is confirmed, fraud status is propagated forward to
subsequent transactions linked to the same card, email address or billing
address.

Modellers must be aware that this creates a dependency between the label and
any feature that counts prior activity on the same entity. Such features carry
a circularity risk, because they partially encode the labelling mechanism
itself rather than independent evidence. Their contribution should be
monitored and reported.

## Dispute handling service levels

- Acknowledge cardholder dispute within one working day.
- Provisional credit issued within five working days where the claim meets the
  presumption criteria.
- Final determination within forty-five days.
