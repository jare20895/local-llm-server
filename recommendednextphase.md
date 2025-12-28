# Recommended Next Phase: Homelab LLM Server

## 1) Current State Snapshot
- **API + orchestration**: FastAPI monolith handles registry, orchestration, configuration, and inference in `main.py`, leaning on a singleton `ModelManager` for model state and a SQLite-backed `ModelRegistry`/`PerformanceLog` schema. Compatibility status is tracked per model with crash-aware transitions (e.g., `testing` ➜ `incompatible`) and load-time metadata capture (parameters, architecture, commits).
- **Caching + storage**: Cache abstraction via `CacheManager` chooses primary/secondary/custom Hugging Face cache roots, with periodic disk checks and optional garbage collection for orphaned blobs. Model deletion/update paths can reclaim space but rely on manual triggers.
- **Performance logging**: Inference path collects GPU/system metrics and generation params when `performance_logging` is on, persisting to `PerformanceLog` rows. TTFT is approximated; true streaming TTFT is not yet implemented. Efficiency toggle disables logging for throughput gains.
- **Frontend surfaces**: Two UIs coexist: a static SPA (`/static`) bundled with FastAPI for core operations and the separate Next.js **Vector-Tester** app (dockerized) for load runs, log ingestion, and Hugging Face metadata/config captures with its own SQLite DB.
- **Docs + tooling**: Rich docs for ROCm/WSL, cache behavior, and model workflows; helper scripts include setup verification and ROCm installers. Automated testing/CI is minimal.

## 2) Key Gaps & Risks
- **Process resilience**: Single-process, in-memory model state means crashes or GPU resets interrupt compatibility transitions and in-flight generation; no request-level timeouts, retry strategy, or circuit breakers around `generate`/`load`.
- **Streaming + TTFT fidelity**: `/api/generate` is synchronous; `/api/generate/stream` scaffolding exists but needs completion, backpressure, and heartbeat/timeout handling to avoid stalled streams.
- **Safety + configuration**: CORS is wide open and `trust_remote_code` can be enabled per model without centralized policy. No server-side quota/rate limits, authN/Z, or guardrails for prompt inputs.
- **Data integrity**: SQLite access lacks migrations/indexes for growing telemetry, and there is no structured retention/archival strategy for `PerformanceLog` or Vector-Tester tables; cache metadata (size, commits) can drift without scheduled reconciliations.
- **Observability**: Logs are mostly print statements; no structured logging, distributed traces, or metrics export (Prometheus/OpenTelemetry). GPU/system checks are best-effort and can fail silently.
- **UX cohesion**: Static SPA and Vector-Tester provide overlapping model/test workflows but live in separate stacks with no shared design system or navigation; onboarding across both surfaces is fragmented.
- **Automation debt**: No CI pipeline, smoke tests, or load/regression suites. GPU-dependent code paths and HF downloads lack fakes/mocks for unit coverage.

## 3) Recommended Next-Phase Plan (prioritized)

### A. Reliability & Safety (short term)
- **Finalize streaming**: Complete `/api/generate/stream` with `TextIteratorStreamer`, server-sent events framing, max duration safeguards, and client disconnect handling; emit TTFT per stream for accurate UX metrics.
- **Harden lifecycle**: Wrap `load`/`generate` with timeouts, error taxonomies, and retry/backoff for transient ROCm/CUDA resets; add background watchdog to demote stuck `testing` models and release VRAM after crashes.
- **Access controls**: Introduce optional API key or token auth, tighten CORS defaults, and enforce a centralized `trust_remote_code` policy plus model-level allow/deny lists.
- **Cache governance**: Add scheduled reconciler to refresh cache stats, recompute model sizes, and auto-trigger blob GC; surface warnings in both UIs when approaching thresholds.

### B. Observability & Performance (short term)
- **Structured telemetry**: Adopt structured logging (JSON) with request IDs and correlation to model/test IDs; export Prometheus metrics for throughput, TTFT, VRAM, cache usage, and compatibility outcomes.
- **Profiling hooks**: Add lightweight profiler toggles for tokenization vs. generation time, plus GPU memory snapshots before/after loads; persist failure fingerprints (error codes, stack hashes) to aid root-cause analysis.
- **Alerting surfaces**: Expose health endpoints (app, DB, cache monitor) and optional webhook/Slack notifications for load failures, cache pressure, and model incompatibility flips.

### C. Data Layer (short–mid term)
- **Migrations + indexing**: Introduce Alembic migrations; add indexes on `model_registry_id`, timestamps, and compatibility fields. Provide data retention policies (e.g., prune `PerformanceLog` beyond N days, summarize aggregates).
- **Schema normalization**: Version `ModelRegistry` metadata (load configs, compatibility notes) and Vector-Tester HF captures to enable comparisons across model versions; ensure commit hashes and cache paths stay in sync after updates.
- **Synthetic fixtures**: Create HF/cache/test DB fixtures for offline tests without large downloads; add faker-based prompt/generation rows to validate analytics queries.

### D. API & UX Cohesion (mid term)
- **Contract cleanup**: Expand OpenAPI with detailed error models and rate-limit headers; add validation on generation params (token limits per architecture) and cache location enums.
- **Unified UI direction**: Decide on a single UI stack (likely Next.js) and port core dashboard features from the static SPA; share components between main UI and Vector-Tester, aligning navigation and theming.
- **Model management flows**: Add bulk operations (multi-delete, batch compatibility retests), staged downloads with progress, and inline cache selection/space indicators during registration.

### E. Operations & Packaging (mid term)
- **Container polish**: Harden Dockerfiles for ROCm/CUDA variants, pin HF/transformers/torch versions, and add image health checks. Provide compose overrides for CPU-only dev and GPU-enabled prod.
- **Config system**: Centralize configuration (Pydantic settings/env files) with environment parity between API, worker tasks, and Vector-Tester; document HF token handling and cache directory permissions.
- **Backup & recovery**: Add automated backups for SQLite databases and cache manifests; document disaster-recovery steps and verification scripts.

### F. Quality & Testing (ongoing)
- **CI/CD**: Add lint/format/type checks, unit tests with torch/transformers mocks, and contract tests for key endpoints. Gate Docker builds and Next.js lint on PRs.
- **Load & resilience testing**: Use Vector-Tester to script load ramps and crash/timeout scenarios; assert compatibility status transitions and cache cleanup behavior.
- **User-facing diagnostics**: Extend `test_setup.py` into a richer diagnostics suite (GPU visibility, cache permissions, env config) and surface results in the UI for self-serve troubleshooting.

## 4) 60–90 Day Outcomes
- Reliable streaming with accurate TTFT metrics; structured telemetry and alerting in place.
- Authenticated API with safer defaults (CORS, trust_remote_code policy) and hardened lifecycle for loads/generation.
- Unified UI direction with shared components and clearer model/cache workflows.
- Repeatable migrations, retention policies, and CI coverage to prevent regressions as the registry and telemetry datasets grow.
