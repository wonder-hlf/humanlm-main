# Human Macro Statistics

Computed from 10 human JUSThink team trajectories.
Repair-after-mismatch uses a window of 10 subsequent events.

## Requested Metrics

| Metric | Human value |
|---|---:|
| Mean turn count | 58.8000 |
| Mean edit count | 105.3000 |
| Mean submit count | 9.2000 |
| Mean best cost gap | 1.8000 |
| Match-to-mismatch ratio | 2.3577 |
| Mean first submit progress | 0.1064 |
| Mean first match progress | 0.1580 |
| Repair-after-mismatch rate | 0.3861 |
| Found optimal rate | 0.3000 |
| Total match | 290 |
| Total mismatch | 123 |
| Total nonmatch | 606 |

Progress values are normalized event positions from 0 (task start) to 1 (task end).

## Additional Metrics

- Action counts: `{"edit_add": 904, "submit": 338, "edit_remove": 115, "press": 151, "check": 405, "edit_load": 34}`
- Discourse counts: `{"other_utterance": 1868, "confirmation": 1123, "disagreement": 503, "filler_or_hesitation": 1394, "submit_proposal": 124, "route_instruction": 87, "repair_talk": 23, "cost_query": 39, "cost_report": 81}`
- Additional retained signals: event count, utterance count, first edit progress, cost-gap trajectory, found-optimal rate, check/press counts, and discourse categories.

## Top Behavior Transitions

| Transition | Count |
|---|---:|
| `nonmatch -> edit_add` | 538 |
| `other_utterance -> other_utterance` | 512 |
| `filler_or_hesitation -> other_utterance` | 450 |
| `other_utterance -> filler_or_hesitation` | 396 |
| `filler_or_hesitation -> filler_or_hesitation` | 352 |
| `other_utterance -> confirmation` | 327 |
| `confirmation -> other_utterance` | 285 |
| `match -> edit_add` | 277 |
| `edit_add -> other_utterance` | 245 |
| `check -> check` | 228 |
| `filler_or_hesitation -> confirmation` | 205 |
| `confirmation -> filler_or_hesitation` | 188 |
| `confirmation -> confirmation` | 187 |
| `edit_add -> confirmation` | 177 |
| `other_utterance -> nonmatch` | 165 |
| `edit_add -> filler_or_hesitation` | 163 |
| `disagreement -> other_utterance` | 145 |
| `submit -> submit` | 144 |
| `edit_add -> nonmatch` | 142 |
| `other_utterance -> disagreement` | 140 |
| `confirmation -> nonmatch` | 104 |
| `other_utterance -> match` | 98 |
| `disagreement -> filler_or_hesitation` | 96 |
| `mismatch -> edit_add` | 89 |
| `filler_or_hesitation -> nonmatch` | 88 |

## Core Behavior Transition Matrix

| From / To | route_instruction | confirmation | disagreement | repair_talk | feedback_interpretation | match | mismatch | nonmatch | edit_add | edit_remove | edit_load | check | submit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_instruction | 2 | 7 | 7 | 1 | 0 | 14 | 5 | 8 | 0 | 0 | 0 | 3 | 4 |
| confirmation | 24 | 187 | 79 | 4 | 0 | 83 | 25 | 104 | 0 | 0 | 0 | 42 | 46 |
| disagreement | 2 | 84 | 66 | 1 | 0 | 15 | 9 | 38 | 0 | 0 | 0 | 16 | 14 |
| repair_talk | 0 | 4 | 2 | 0 | 0 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| feedback_interpretation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| match | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 277 | 13 | 0 | 0 | 0 |
| mismatch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 34 | 0 | 0 | 0 |
| nonmatch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 538 | 68 | 0 | 0 | 0 |
| edit_add | 21 | 177 | 54 | 2 | 0 | 3 | 11 | 142 | 0 | 0 | 0 | 17 | 22 |
| edit_remove | 3 | 27 | 3 | 2 | 0 | 1 | 2 | 14 | 0 | 0 | 0 | 3 | 0 |
| edit_load | 0 | 5 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| check | 2 | 23 | 5 | 1 | 0 | 2 | 0 | 18 | 0 | 0 | 0 | 228 | 3 |
| submit | 3 | 27 | 43 | 0 | 0 | 0 | 1 | 6 | 0 | 0 | 0 | 2 | 144 |
