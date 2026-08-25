# Card-Not-Present Fraud Controls

*Illustrative internal policy written for the RiskLens project. Not a real
policy of any institution.*

## Background

Card-not-present transactions carry materially higher fraud rates than
card-present transactions, because the criminal does not need physical
possession of the card.

Analysis of the RiskLens training population found fraud rates approximately
four times higher on transactions where device and identity data were
captured. Device telemetry is only captured on online channels. Analysts must
therefore treat the presence of device data as an indicator of channel, not as
an indicator of trustworthiness. The absence of device data most often means
the transaction was card-present, which is inherently lower risk.

## Elevated risk indicators

The following are recognised elevated risk indicators for card-not-present
transactions. No single indicator is sufficient grounds for decline.

- Billing address and transaction location separated by an unusual distance.
- Payer and recipient email domains that do not match.
- Recently created or rarely seen email domains.
- A card identifier seen only a small number of times in the trailing window.
- A device fingerprint or browser string seen only a small number of times.
- Transactions occurring during overnight hours relative to the usual pattern
  for that cardholder.
- Multiple low-value authorisations on a single card in a short window, which
  is characteristic of card testing.

## Card testing

Card testing is the practice of making small, deliberately unremarkable
purchases to confirm that a stolen card is live before selling or exploiting
it. Because the amounts are chosen to look normal, transaction value alone is
a weak signal and must not be used as a primary control.

Where three or more low-value authorisations occur on one card within sixty
minutes, the card must be placed under enhanced monitoring for twenty-four
hours regardless of the individual transaction scores.

## Step-up authentication

Step-up authentication should be applied in preference to outright decline
where the risk band is MEDIUM or HIGH and the transaction value exceeds the
step-up threshold. Step-up preserves the customer relationship while
transferring risk to the authentication provider.

Step-up is not appropriate for CRITICAL band transactions, where immediate
decline and cardholder contact is required.

## Velocity controls

Velocity controls limit the number and value of transactions permitted on a
single card, device or billing address within a rolling window. Velocity
breaches are evaluated independently of the model score and may trigger a hold
even where the model score is low.
