# Account Takeover Detection and Response

*Illustrative internal policy written for the RiskLens project. Not a real
policy of any institution.*

## Definition

Account takeover occurs when an unauthorised party gains control of a
legitimate customer account and transacts as that customer. It is distinct
from stolen card fraud because the account itself, including its transaction
history and accumulated trust signals, is compromised.

Account takeover is difficult to detect precisely because the account has a
legitimate history. Controls that rely on account age or prior good behaviour
are ineffective against it.

## Primary indicators

- A change to the registered email address, followed by transaction activity
  within a short window.
- A mismatch between the payer email domain and the recipient email domain
  where these previously matched.
- A new device or browser fingerprint transacting on an established account.
- A change in the billing address shortly before a high-value transaction.
- Transaction timing that departs materially from the established daily
  pattern for that account.
- A sudden increase in transaction velocity on a previously stable account.

## Combination rule

Account takeover is rarely indicated by a single signal. The presence of two
or more primary indicators within a twenty-four hour window escalates the case
to HIGH regardless of the model score, and requires analyst review before any
transaction above the review threshold is released.

## Response procedure

1. Suspend outbound transaction authority on the account.
2. Contact the cardholder using a contact method held on file prior to the
   suspected compromise date. Contact details changed within the suspicion
   window must not be used, as they may belong to the attacker.
3. Where compromise is confirmed, reset credentials, revoke all active
   sessions and reissue the card.
4. Review all transactions on the account within the preceding thirty days.
5. Record the confirmed compromise date, because it determines which
   historical transactions must be relabelled as fraud.

## Customer treatment

Customers affected by account takeover are victims. Communication must be
non-accusatory, and provisional credit should be issued in line with the
disputes policy while the investigation proceeds.
