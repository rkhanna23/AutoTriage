# Ticket Dataset v1 - (CP2-KL-02)

This dataset contains labeled support tickets used to evaluate the AutoTriage ticket classification system.

## Dataset Composition

Ticket Dataset v1 includes:

- **200+ synthetic tickets** created to simulate common support issues
- **30 real-world examples** derived from public GitHub issue trackers

The synthetic examples were generated to ensure balanced category coverage.  
The real examples were manually summarized and labeled.

## Fields

Each ticket contains the following fields:

- `title` – short description of the issue
- `description` – summary of the problem reported
- `category` – ticket type classification
- `severity` – priority level of the issue
- `source` – origin of the ticket

## Categories

Tickets are classified into the following categories:

- Auth
- Billing
- Outage
- Performance
- Security
- Feature Request

## Severity Levels

Severity labels represent issue urgency:

- **P0** – critical system failure
- **P1** – major functionality broken
- **P2** – moderate issue
- **P3** – minor issue or usability problem

## Real-World Sources

Real ticket examples were summarized from public GitHub issue trackers, including:

- Supabase
- Kubernetes
- Redis
- Stripe
- Firebase

Issues were paraphrased and manually labeled to match the project taxonomy.

## Purpose

This dataset is used to benchmark the AutoTriage classification model during Checkpoint 2 by measuring category and severity prediction accuracy.