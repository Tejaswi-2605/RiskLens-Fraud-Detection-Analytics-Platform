# RiskLens Policy Corpus

## ⚠️ These documents are synthetic

They were **written for this project**. They are **not** real policies from any
bank, regulator, card network, or other institution, and must not be presented
or used as such. Every file repeats this in its header.

## Why they exist

Stages 8 and 9 of RiskLens — policy retrieval and the investigation copilot —
are retrieval-augmented. Retrieval needs a corpus, and the IEEE-CIS dataset
contains no free text at all.

There were two options:

1. **Index real public material** — regulatory guidance, card-network
   chargeback documentation.
2. **Write a synthetic corpus** modelled on the kind of internal manual a
   fraud-operations team maintains.

This project takes the second option, for one substantive reason: **coherence**.
Real external guidance does not mention RiskLens's own risk bands, its
cost-optimal threshold, or its alert-budget constraints. The copilot would then
retrieve policy that did not describe the system it was advising on, and the
demonstration would be less useful, not more.

The trade-off is stated plainly rather than hidden: retrieving from documents
written for the same project is a weaker claim than indexing genuine
regulatory text. Anyone wanting the stronger version can drop real documents
into this folder — the loader reads any markdown files it finds.

## What they cover

| File | Topic |
|---|---|
| `01_risk_scoring_and_decisions.md` | risk bands, threshold policy, override authority |
| `02_card_not_present_controls.md` | CNP indicators, card testing, step-up authentication |
| `03_chargebacks_and_labels.md` | label maturity, the 120-day window, label noise |
| `04_account_takeover.md` | ATO indicators, combination rules, response procedure |
| `05_alert_triage_sla.md` | queue structure, triage sequence, evidence standards |
| `06_model_governance.md` | validation, prohibited practices, PSI monitoring |

They are modelled on the *kinds* of controls such manuals contain — decision
thresholds, escalation paths, service levels, model-risk monitoring — so that
retrieval and grounding behave realistically.

## Using your own

The loader reads every `.md` file in this directory and chunks it with
overlap. Replace or extend these files with any plain-text policy documents
and Stages 8 and 9 will index them without code changes.
