---
name: execute
description: Run the executor sub-agent on one ticket. Usage: /execute plans/001-dataset-split-pipeline.md
disable-model-invocation: true
context: fork
agent: executor
---

Implement and verify this ticket as the Executor: $ARGUMENTS

If no ticket path is given above, do nothing and report that a ticket path is required.
