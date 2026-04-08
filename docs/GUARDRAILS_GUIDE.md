# NeuralForge Guardrails Guide

Configuring NeMo Guardrails for input validation, output safety, and retrieval quality control.

---

## Overview

NeuralForge uses NeMo Guardrails to protect three stages of the pipeline:

```
User Input ──> [Input Rails] ──> Processing ──> [Output Rails] ──> Response
                                      ^
                                      |
                              [Retrieval Rails]
```

All rails are configurable, auditable, and can be disabled individually.

---

## Configuration Files

```
forge/guardrails/config/
  config.yml    # Model backend + rail registration
  flows.co      # Colang flow definitions
```

### config.yml

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct

rails:
  input:
    flows:
      - check pii
      - check jailbreak
      - check topic relevance
  output:
    flows:
      - check hallucination
      - check attribution
      - add provenance
  retrieval:
    flows:
      - check retrieval relevance

actions:
  - name: check_hallucination
    module: forge.guardrails.actions
  # ... (6 custom actions registered)
```

### flows.co

Colang flow definitions that implement the rail logic. See the [NeMo Guardrails documentation](https://docs.nvidia.com/nemo/guardrails/) for the Colang language reference.

---

## Input Rails

### PII Scrubbing (`check pii`)

**What it does:** Strips personal information from user queries before they enter the processing pipeline.

**What it catches:**
- Email addresses
- Phone numbers (US + international)
- Social Security Numbers
- Physical addresses
- Credit card numbers

**Behavior:** PII is replaced with `[REDACTED]`. The user is informed that scrubbing occurred.

**Disable:** Remove `check pii` from the `input.flows` list in `config.yml`.

### Jailbreak Detection (`check jailbreak`)

**What it does:** Detects prompt injection and jailbreak attempts.

**Patterns detected:**
- "Ignore your instructions"
- "You are now DAN"
- Role-playing prompts designed to bypass safety
- Encoded/obfuscated injection attempts

**Behavior:** The query is blocked and a refusal message is returned.

**Disable:** Remove `check jailbreak` from `input.flows`.

### Topic Relevance (`check topic relevance`)

**What it does:** Ensures the query is related to the knowledge domain covered by ingested experts.

**Behavior:** Off-topic queries receive a redirect message suggesting the user rephrase.

**Disable:** Remove `check topic relevance` from `input.flows`.

---

## Output Rails

### Hallucination Check (`check hallucination`)

**What it does:** After the LLM generates a response, this rail verifies that any expert citations actually exist in the knowledge graph.

**How it works:**
1. Extracts expert names mentioned in the response
2. Checks each name against graph nodes
3. If a non-existent expert is cited, triggers self-correction

**Self-correction:** When a hallucination is detected, the response is re-generated with an explicit instruction to only cite known experts.

**Disable:** Remove `check hallucination` from `output.flows`.

### Attribution Verification (`check attribution`)

**What it does:** A stricter check than hallucination -- ensures every named reference in the response corresponds to a real graph node (expert, paper, tool, etc.).

**Behavior:** If unverifiable references are found, the user is warned to treat those claims with caution.

**Disable:** Remove `check attribution` from `output.flows`.

### Provenance Tracking (`add provenance`)

**What it does:** Appends source metadata to the response, including:
- Source URLs for cited content
- Timestamps of when content was ingested
- Confidence scores from the retrieval pipeline

**Behavior:** Always runs (does not block). Adds a provenance section to the response.

**Disable:** Remove `add provenance` from `output.flows`.

---

## Retrieval Rails

### Relevance Filtering (`check retrieval relevance`)

**What it does:** After chunks are retrieved from Qdrant but before they reach the LLM, this rail validates that the chunks are actually relevant to the query.

**How it works:** Compares the query embedding against each chunk's embedding. Chunks below a relevance threshold are filtered out, preventing the LLM from being distracted by irrelevant context.

**Disable:** Remove `check retrieval relevance` from `retrieval.flows`.

---

## Custom Actions

NeuralForge implements 6 custom actions in `forge/guardrails/actions.py`:

| Action | Rail Stage | Purpose |
|:---|:---|:---|
| `scrub_pii_input` | Input | Regex-based PII detection and removal |
| `check_hallucination` | Output | Cross-reference citations with graph |
| `check_attribution` | Output | Verify all named entities exist in graph |
| `add_provenance` | Output | Append source metadata |
| `self_correction` | Output | Re-generate when hallucination detected |
| `log_rail_decision` | All | Audit logging for every rail decision |

### Adding Custom Actions

1. Add your action function to `forge/guardrails/actions.py`
2. Register it in `config.yml` under the `actions` section
3. Reference it in a Colang flow in `flows.co`

```python
# forge/guardrails/actions.py
async def my_custom_check(text: str, **kwargs) -> bool:
    """Custom rail action."""
    # Your logic here
    return True  # allowed
```

```yaml
# config.yml
actions:
  - name: my_custom_check
    module: forge.guardrails.actions
```

```colang
# flows.co
define flow check my custom thing
  $ok = execute my_custom_check(text=$user_message)
  if not $ok
    bot refuse my custom thing
```

---

## Enabling/Disabling Guardrails

### Global Toggle

Set `GUARDRAILS_ENABLED=false` in your `.env` to disable all guardrails. The system will still function -- all checks return `allowed=True`.

### Per-Rail Toggle

Remove individual flows from the `input.flows`, `output.flows`, or `retrieval.flows` lists in `config.yml`.

### Graceful Degradation

If the `nemoguardrails` package is not installed, NeuralForge runs without any rails. This is logged as a warning:

```
WARNING: NeMo Guardrails not installed -- running without safety rails
```

This is intentional for development and testing environments.

---

## Audit Logging

Every rail decision is logged via the `log_rail_decision` action:

```json
{
  "timestamp": "2025-04-07T12:00:00",
  "rail_name": "input_pii",
  "decision": false,
  "reason": "PII detected and scrubbed"
}
```

Logs are written to `data/logs/` and can be used for compliance auditing.

---

## The Full Pipeline

```
1. User sends query
2. [Input Rails]
   a. check pii ──> scrub PII, inform user
   b. check jailbreak ──> block if detected
   c. check topic relevance ──> redirect if off-topic
3. [Retrieval]
   a. Embed query via Triton
   b. Search Qdrant for top-K chunks
   c. [Retrieval Rails] ──> filter irrelevant chunks
   d. Rerank via Triton cross-encoder
4. [Context Assembly]
   a. Layer 0: System identity
   b. Layer 1: cuGraph context (PageRank + contradictions)
   c. Layer 2: AAAK-compressed vector search chunks
   d. Layer 3: Deep search uncompressed passages
5. [NIM Generation]
   a. Send layered context + query to NIM
   b. Receive LLM response
6. [Output Rails]
   a. check hallucination ──> verify citations, self-correct if needed
   b. check attribution ──> verify named entities
   c. add provenance ──> append source metadata
7. Return response + sources + provenance
```
