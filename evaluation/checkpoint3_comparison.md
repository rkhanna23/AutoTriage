# Checkpoint 3 Prompt Comparison

Dataset: `data/ticket_dataset_v1.json`
Recommended default: `v2.1`

| Prompt | Category Accuracy | Severity Accuracy | Needs Review Rate | ECE |
| --- | --- | --- | --- | --- |
| v1.0 | 99.52% | 44.29% | 0.00% | 0.4584 |
| v2.0 | 100.00% | 100.00% | 0.00% | 0.1500 |
| v2.1 | 100.00% | 100.00% | 0.00% | 0.0700 |

## Recommendation

`v2.1` should ship as the default because it produced the strongest severity accuracy while also minimizing calibration error and review volume.
