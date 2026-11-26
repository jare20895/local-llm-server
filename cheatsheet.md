# local-llm-server Integration Cheatsheet

- Path: `/home/jare16/Projects/ArcoreAgent/All_Arcore_Repos/local-llm-server`
- Generated at: `2026-05-12 21:24:07Z`
- Branch: `main`
- Remote: `https://github.com/jare20895/local-llm-server.git`
- Last commit: `254bb83f5f2680eecc416a0f5c59003b6a673143`
- Commit date: `2025-12-28 09:46:25 -0500`
- Commit subject: `Add recommended next-phase roadmap`

- Language mix (top): `ts:41, js:30, gz:30, md:20, json:15, py:12, [no_ext]:7, sh:6, tsx:4, yml:3, css:3, example:2`

## Quick Integration Pointers

### Docker services
- llm-server: 8002:8000 (from docker-compose-build.yml)
- llm-server: 18085:8000 (from docker-compose.yml)
- llm-frontend: 4173:3000 (from docker-compose.yml)
- llm-frontend-dev: 4174:3000 (from docker-compose.yml)

### Run commands
- none detected

### Recent commits

- 254bb83 Add recommended next-phase roadmap
- 4d95454 chore(ci): configure auto PR creation workflow
- c191ebe chore(ci): configure auto PR creation workflow
- d226164 chore(ci): add auto PR creation workflow
- 79c956b chore: sync local updates in local-llm-server

### Integration touchpoints

- `Dockerfile`
- `docker-compose.yml`
- `docs/MODEL_DOWNLOAD_PIPELINE.md`
- `qwen3_api_server.py`
- `requirements.txt`
- `vector-tester/.next/package.json`
- `vector-tester/.next/server/app/api/llm/status/route.js`
- `vector-tester/.next/server/app/api/log-events/route.js`
- `vector-tester/.next/server/app/api/models/config-tests/[id]/route.js`
- `vector-tester/.next/server/app/api/models/config-tests/route.js`
- `vector-tester/.next/server/app/api/models/configs/[id]/route.js`
- `vector-tester/.next/server/app/api/models/huggingface/[id]/route.js`
- `vector-tester/.next/server/app/api/models/huggingface/config/route.js`
- `vector-tester/.next/server/app/api/models/huggingface/route.js`
- `vector-tester/.next/server/app/api/models/offline/route.js`
- `vector-tester/.next/server/interception-route-rewrite-manifest.js`
- `vector-tester/.next/types/app/api/llm/status/route.ts`
- `vector-tester/.next/types/app/api/log-events/route.ts`
- `vector-tester/.next/types/app/api/models/config-tests/[id]/route.ts`
- `vector-tester/.next/types/app/api/models/config-tests/route.ts`
- `vector-tester/.next/types/app/api/models/configs/[id]/route.ts`
- `vector-tester/.next/types/app/api/models/huggingface/[id]/route.ts`
- `vector-tester/.next/types/app/api/models/huggingface/config/route.ts`
- `vector-tester/.next/types/app/api/models/huggingface/route.ts`
- `vector-tester/.next/types/app/api/models/offline/route.ts`
- `vector-tester/app/api/docs/route.ts`
- `vector-tester/app/api/huggingface/meta/route.ts`
- `vector-tester/app/api/llm/health/route.ts`
- `vector-tester/app/api/llm/inference/generate/route.ts`
- `vector-tester/app/api/llm/models/route.ts`
- `vector-tester/app/api/llm/orchestrate/load/route.ts`
- `vector-tester/app/api/llm/orchestrate/unload/route.ts`
- `vector-tester/app/api/llm/orchestrate/validate/route.ts`
- `vector-tester/app/api/llm/status/route.ts`
- `vector-tester/app/api/log-events/route.ts`
- `vector-tester/app/api/models/config-tests/[id]/route.ts`
- `vector-tester/app/api/models/config-tests/entry/route.ts`
- `vector-tester/app/api/models/config-tests/route.ts`
- `vector-tester/app/api/models/config-tests/test/route.ts`
- `vector-tester/app/api/models/configs/[id]/route.ts`
- `vector-tester/app/api/models/huggingface/[id]/route.ts`
- `vector-tester/app/api/models/huggingface/config/route.ts`
- `vector-tester/app/api/models/huggingface/route.ts`
- `vector-tester/app/api/models/offline/route.ts`
- `vector-tester/app/api/test-config/server-environments/route.ts`
- `vector-tester/app/api/test-config/swagger/route.ts`
- `vector-tester/app/api/test-config/test-profiles/route.ts`
- `vector-tester/app/api/test-config/test-steps/route.ts`
- `vector-tester/app/api/test-runs/route.ts`
- `vector-tester/lib/llm.ts`
- `vector-tester/lib/llmProxy.ts`
- `vector-tester/scripts/sync_hf_configs.py`
- `vector-tester/scripts/sync_hf_metadata.py`

## Main API Endpoints (best-effort)

High-confidence endpoints:

Additional endpoints (lower confidence):
- GET    /  [medium]   (main.py:1775)  from framework-router
- GET    /api/analytics/logs/{model_name}  [medium]   (main.py:1544)  from framework-router
- GET    /api/cache/config  [medium]   (main.py:1714)  from framework-router
- GET    /api/cache/orphaned  [medium]   (main.py:1576)  from framework-router
- GET    /api/cache/recommend  [medium]   (main.py:1740)  from framework-router
- GET    /api/cache/stats  [medium]   (main.py:1707)  from framework-router
- GET    /api/config/logging  [medium]   (main.py:1106)  from framework-router
- GET    /api/models  [medium]   (main.py:577)  from framework-router
- GET    /api/models/{model_name}/updates  [medium]   (main.py:736)  from framework-router
- GET    /api/orchestrate/validate  [medium]   (main.py:1097)  from framework-router
- GET    /api/status  [medium]   (main.py:915)  from framework-router
- GET    /health  [medium]   (main.py:1762)  from framework-router
- GET    /vector-tester/api/docs  [medium]   (vector-tester/app/api/docs/route.ts:219)  from nextjs
- GET    /vector-tester/api/huggingface/meta  [medium]   (vector-tester/app/api/huggingface/meta/route.ts:15)  from nextjs
- GET    /vector-tester/api/llm/health  [medium]   (vector-tester/app/api/llm/health/route.ts:5)  from nextjs
- GET    /vector-tester/api/llm/models  [medium]   (vector-tester/app/api/llm/models/route.ts:5)  from nextjs
- GET    /vector-tester/api/llm/orchestrate/validate  [medium]   (vector-tester/app/api/llm/orchestrate/validate/route.ts:5)  from nextjs
- GET    /vector-tester/api/llm/status  [medium]   (vector-tester/app/api/llm/status/route.ts:5)  from nextjs
- GET    /vector-tester/api/log-events  [medium]   (vector-tester/app/api/log-events/route.ts:14)  from nextjs
- GET    /vector-tester/api/models/config-tests/[id]  [medium]   (vector-tester/app/api/models/config-tests/[id]/route.ts:11)  from nextjs
- GET    /vector-tester/api/models/configs/[id]  [medium]   (vector-tester/app/api/models/configs/[id]/route.ts:11)  from nextjs
- GET    /vector-tester/api/models/huggingface/[id]  [medium]   (vector-tester/app/api/models/huggingface/[id]/route.ts:10)  from nextjs
- GET    /vector-tester/api/test-config/server-environments  [medium]   (vector-tester/app/api/test-config/server-environments/route.ts:24)  from nextjs
- GET    /vector-tester/api/test-config/swagger  [medium]   (vector-tester/app/api/test-config/swagger/route.ts:19)  from nextjs
- GET    /vector-tester/api/test-config/test-profiles  [medium]   (vector-tester/app/api/test-config/test-profiles/route.ts:24)  from nextjs
- GET    /vector-tester/api/test-config/test-steps  [medium]   (vector-tester/app/api/test-config/test-steps/route.ts:26)  from nextjs
- GET    /vector-tester/api/test-runs  [medium]   (vector-tester/app/api/test-runs/route.ts:23)  from nextjs
- POST   /api/cache/check  [medium]   (main.py:1721)  from framework-router
- POST   /api/cache/cleanup  [medium]   (main.py:1632)  from framework-router
- POST   /api/config/logging  [medium]   (main.py:1112)  from framework-router
- POST   /api/generate/stream  [medium]   (main.py:1314)  from framework-router
- POST   /api/models/{model_name}/update  [medium]   (main.py:801)  from framework-router
- POST   /api/orchestrate/load  [medium]   (main.py:935)  from framework-router
- POST   /api/orchestrate/unload  [medium]   (main.py:1083)  from framework-router
- POST   /generate  [medium]   (qwen3_api_server.py:53)  from framework-router
- POST   /vector-tester/api/llm/inference/generate  [medium]   (vector-tester/app/api/llm/inference/generate/route.ts:5)  from nextjs
- POST   /vector-tester/api/llm/orchestrate/load  [medium]   (vector-tester/app/api/llm/orchestrate/load/route.ts:6)  from nextjs
- POST   /vector-tester/api/llm/orchestrate/unload  [medium]   (vector-tester/app/api/llm/orchestrate/unload/route.ts:5)  from nextjs
- POST   /vector-tester/api/log-events  [medium]   (vector-tester/app/api/log-events/route.ts:19)  from nextjs
- POST   /vector-tester/api/models/config-tests  [medium]   (vector-tester/app/api/models/config-tests/route.ts:15)  from nextjs

## Post-scan checks

- ⚠️ No high-confidence API map for this repo. Verify API file/docs for explicit endpoint specs.
