# CXL_AT — Experiment Code

This repository contains the experiment code for both open-source models and closed-source models.

## Directory Structure

### `openSourcesModel/`
Experiment code for all open-source models.

- **`AttentionTriger/`** — Comparison experiments of AT against the other baseline methods on open-source models.
- **`ablationStudy/`** — (Discussion) Experiments on the poisoning ratio.
- **`AFBAP_2/`** — (Discussion) Code for the semantic trigger.
- **`AFBAP_3/`** — (Discussion) Code for the special-symbol trigger.
- **`semanticAndAttentionTrigger/`** — (Discussion) Code for combining the semantic trigger with the attention trigger.
- **`Defense/`** — Code for the defense part.

### `closedModel/`
- **`AttentionTriger/`** — The corresponding AT experiments on closed-source models.

## Environment

- transformers == 4.51.3
