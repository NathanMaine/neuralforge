# Roadmap: OpenShell-Native Knowledge Platform

**Status:** Proposed
**Date:** 2026-04-08
**Depends on:** NVIDIA OpenShell (Apache 2.0, part of NVIDIA Agent Toolkit)

---

## Vision

Position NeuralForge as an **OpenShell-native knowledge platform** — deployed inside an OpenShell sandbox with two layers of control:

1. **NeMo Guardrails** (content-level) — prompt filtering, output validation, PII scrubbing, hallucination checks. Already built.
2. **NVIDIA OpenShell** (infrastructure-level) — process isolation, network policy enforcement, privacy-aware inference routing, full audit trail. New integration layer.

This gives users defense-in-depth: even if an agent or prompt injection compromises application-level guardrails, the infrastructure-level controls remain outside the agent's reach.

---

## Why OpenShell + NeuralForge

### The Problem

NeuralForge already enforces content-level safety via NeMo Guardrails. But as usage grows toward agentic workflows — ingestion agents crawling sources, spawning sub-agents for parsing, installing new skills — the threat model changes:

- Long-running agents with shell access and accumulated context
- Sub-agents that can inherit permissions they shouldn't have
- Third-party skills/tools with unreviewed filesystem access
- Prompt injection risks that could leak credentials or bypass application guardrails

### The Solution

OpenShell's out-of-process policy enforcement wraps NeuralForge's runtime. The agent cannot override infrastructure-level constraints, even if compromised.

| Layer | Tool | Controls |
|:------|:-----|:---------|
| **Content** | NeMo Guardrails | Prompt filtering, topicality enforcement, PII scrubbing, output validation, hallucination checks |
| **Infrastructure** | OpenShell | Process sandboxing, filesystem/network policy, inference routing, credential isolation, audit trail |

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────┐
│  OpenShell Sandbox                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Policy Engine                                │  │
│  │  - Filesystem: read-only except /data/ingest  │  │
│  │  - Network: allow localhost:8000-8090, 6333   │  │
│  │  - Process: no unverified binary execution    │  │
│  │  - Audit: log every allow/deny decision       │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Privacy Router                               │  │
│  │  - Compliance queries → local NIM (on-device) │  │
│  │  - General queries → policy-permitted models  │  │
│  │  - PII-flagged content → NEVER routes to cloud│  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  NeuralForge (existing stack)                 │  │
│  │  ┌─────────┐ ┌─────┐ ┌──────┐ ┌───────┐     │  │
│  │  │ API     │ │ NIM │ │Triton│ │ Qdrant│     │  │
│  │  │ +NeMo   │ │     │ │      │ │       │     │  │
│  │  │ Guards  │ │     │ │      │ │       │     │  │
│  │  └─────────┘ └─────┘ └──────┘ └───────┘     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Deployment Target

Single command deployment inside OpenShell:

```bash
openshell sandbox create --remote spark --from neuralforge
```

This should launch the full NeuralForge stack (4 containers) inside an isolated OpenShell sandbox with pre-configured policies.

---

## Implementation Phases

### Phase 1: Sandbox Deployment

- Create OpenShell-compatible NeuralForge image/config
- Define baseline policy file (filesystem, network, process rules)
- Validate all 4 containers run correctly inside sandbox
- Test that existing NeMo Guardrails function unchanged

### Phase 2: Privacy Router Integration

- Configure OpenShell privacy router with NeuralForge inference targets
- Define routing policies:
  - Compliance-domain queries → local NIM only
  - PII-containing content → block cloud routing
  - General knowledge queries → policy-based routing
- Integrate router decisions into NeuralForge audit logs

### Phase 3: Agentic Workflow Governance

- Wrap ingestion agents (blog, YouTube, arXiv crawlers) in scoped sub-sandboxes
- Define per-agent permission policies (e.g., arXiv agent gets network access to arxiv.org only)
- Skill verification for any new tools agents install at runtime
- Live policy updates as new ingestion sources are added

### Phase 4: Enterprise Audit Trail

- Unified audit log combining NeMo Guardrails decisions + OpenShell policy decisions
- Dashboard for reviewing allow/deny history
- Compliance reporting: prove that sensitive data never left the device

---

## Key Differentiator

No other open-source knowledge platform offers both content-level AND infrastructure-level AI safety as a single deployment. This positions NeuralForge uniquely for:

- **Regulated industries** (defense, healthcare, finance) where proving data containment matters
- **Enterprise teams** deploying agentic knowledge workflows that need governance
- **DGX Spark / RTX users** who want production-grade safety without cloud dependency
