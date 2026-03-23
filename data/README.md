# Ticket Datasets

This directory contains the labeled datasets used to evaluate the AutoTriage pipeline.

## Dataset v1

`ticket_dataset_v1.json` and `ticket_dataset_v1.csv` are the CP2 baseline datasets.

- 210 tickets total
- Balanced across the six project categories
- Used for prompt-version comparison and CP2 baseline reporting

## Dataset v2

`ticket_dataset_v2.json` is the CP3 evaluation dataset.

- 300 tickets total
- Uses the project taxonomy only: `Auth`, `Billing`, `Outage`, `Performance`, `Security`, `Feature Request`
- Balanced severity distribution: 75 tickets each for `P0`, `P1`, `P2`, and `P3`
- Includes 20 explicit edge cases across:
  - ambiguous severity boundaries
  - multi-category phrasing such as auth-outage or billing-impact incidents
  - minimal descriptions with very short ticket bodies

## Expansion Rationale

Checkpoint 3 focuses on improving severity prediction, routing correctness, and live evaluation. Dataset v2 expands on v1 by:

- rebalancing severity so all priority levels are represented evenly
- keeping the taxonomy aligned with the classifier and router services
- adding edge-case tickets that stress prompt reasoning and low-confidence review behavior

## Sources

- CP2 Dataset v1 tickets
- Additional synthetic CP3 edge-case tickets authored to test severity ambiguity, mixed-signal incidents, and sparse descriptions

## Fields

Each ticket contains:

- `ticket_id`
- `title`
- `description`
- `category`
- `severity`
- `source`
