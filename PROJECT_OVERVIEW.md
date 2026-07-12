# TodoGPT / task_mAIstro — Project Overview

A long-term-memory ToDo chatbot. A LangGraph agent (backed by Google Gemini) remembers
a user's profile, their task list, and their stated preferences across conversations,
stored in a Postgres-backed store. A small FastAPI service handles username/password
auth, and a single static HTML page is the chat UI.

## Architecture at a glance

```
┌─────────────────┐      cookie auth       ┌──────────────────────┐
│  index.html      │ ─────────────────────▶ │  auth_service (8124)  │
│  (static SPA)    │                        │  FastAPI + Postgres   │
│                   │                        └──────────────────────┘
│                   │      chat / runs                     │
│                   │ ─────────────────────▶ ┌──────────────────────┐
└──────────────────┘                        │  langgraph-api (8123)  │
                                             │  task_maistro.py graph │
                                             │  Gemini 2.5 Flash      │
                                             └───────────┬────────────┘
                                                          │
                                        ┌─────────────────┴─────────────────┐
                                        │  Postgres (5432) — checkpoints/store │
                                        │  Redis (6379) — LangGraph task queue │
                                        └───────────────────────────────────┘
```

Two independent backend services share one Postgres container but keep separate
concerns: `auth_service` owns `users`/`sessions` tables; the LangGraph API owns its
own checkpoint/store tables (created automatically by `langgraph`).

## Files and what they do

### Root

| File | Purpose |
|---|---|
| [task_maistro.py](task_maistro.py) | The LangGraph agent itself — graph definition, node functions, prompts, and Pydantic schemas. This is the core of the project. |
| [configuration.py](configuration.py) | `Configuration` dataclass (`user_id`, `todo_category`, `task_maistro_role`) read from `RunnableConfig` or environment variables at run time. |
| [langgraph.json](langgraph.json) | LangGraph CLI/API manifest: registers `task_maistro.py:graph` as the `task_maistro` assistant, points at `.env` for secrets, sets permissive CORS for local dev. |
| [langgraph_client.py](langgraph_client.py) | Standalone CLI script (not used by the web UI) for talking to the LangGraph API directly from Python — create threads, send messages, inspect history/graphs. Useful for debugging the agent outside the browser. |
| [frontend/index.html](frontend/index.html) | The entire frontend: a single-file HTML/CSS/vanilla-JS chat SPA ("TodoGPT") with login/register screens, a thread sidebar, and a chat pane. No build step, no framework. Lives in its own folder so Vercel's root-directory setting doesn't collide with the Python files at repo root. |
| [requirements.txt](requirements.txt) | Python dependencies for the LangGraph API container. |
| [Dockerfile](Dockerfile) | Builds the LangGraph API image; runs `langgraph dev` on port 2024 (mapped to host 8123). |
| [docker-compose.yml](docker-compose.yml) | Orchestrates 4 services: `langgraph-redis`, `langgraph-postgres`, `langgraph-api`, `auth-api`. |
| [.env](.env) | Local secrets (gitignored in spirit, but **no `.gitignore` currently exists — see Recommendations**). Expected keys: `Google_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `huggingfacehub_api_token`. |
| `.langgraph_api/`, `__pycache__/` | Local build/runtime artifacts from `langgraph dev`; not source. |

### `auth_service/` — standalone authentication microservice

| File | Purpose |
|---|---|
| [auth_service/main.py](auth_service/main.py) | FastAPI app exposing `/register`, `/login`, `/logout`, `/me`. Issues an httpOnly `session_token` cookie backed by a `sessions` table; passwords hashed with `bcrypt`. |
| [auth_service/db.py](auth_service/db.py) | `psycopg` connection helper + inline schema (`users`, `sessions` tables), created on startup via `init_db()`. |
| [auth_service/requirements.txt](auth_service/requirements.txt) | `fastapi`, `uvicorn`, `psycopg[binary]`, `bcrypt`. |
| [auth_service/Dockerfile](auth_service/Dockerfile) | Builds the auth image; runs `uvicorn main:app` on port 8000 (mapped to host 8124). |

This service is completely decoupled from LangGraph — it only proves *who* the user is.
The frontend attaches `currentUser` (the username) as `user_id` when calling the
LangGraph API, which is how per-user memory isolation happens (see below).

## How the agent works (`task_maistro.py`)

The agent is a **LangGraph `StateGraph`** with one router node and three worker nodes,
using LangGraph's persistent `Store` (namespaced key-value memory, separate from the
message-history checkpointer) to keep memory across threads/sessions for a given user.

**Memory namespaces** (each keyed by `(kind, todo_category, user_id)`):
- `profile` — a single `Profile` doc (name, location, job, connections, interests)
- `todo` — many `ToDo` docs (task, deadline, priority, status, tags, recurrence, subtasks)
- `instructions` — free-text user preferences for how the ToDo list should be managed

**Graph flow:**
1. `task_mAIstro` (entry node) — loads profile/todos/instructions from the store, builds
   a system prompt (including an overdue/due-soon deadline summary), and asks Gemini
   whether to respond directly or call the `UpdateMemory` tool.
2. `route_message` (conditional edge) — for each `UpdateMemory` tool call in the
   response, fans out a `Send()` to the matching worker node (`update_profile`,
   `update_todos`, or `update_instructions`). Multiple memory types can update in the
   same turn.
3. Worker nodes use **Trustcall** (`trustcall.create_extractor`) to have the LLM emit
   structured patches/inserts against existing memory docs:
   - `update_profile` — patches/creates the `Profile` doc.
   - `update_todos` — patches/creates `ToDo` docs; runs a cheap `difflib`-based
     duplicate check (`is_duplicate_task`) before inserting a new task, so paraphrased
     re-asks of the same task don't create dupes.
   - `update_instructions` — asks the model to rewrite the free-text instructions doc
     based on user feedback in the conversation.
4. All worker nodes loop back to `task_mAIstro`, which re-runs with fresh memory until
   the model responds without further tool calls (`route_message` returns `END`).

**Model:** `gemini-2.5-flash` via `langchain-google-genai`, temperature 0. The code
comments explain this choice was for free-tier quota headroom vs. newer Gemini models.

## Frontend (`frontend/index.html`)

A dependency-free single HTML file (only external asset: Google Fonts + the `marked`
CDN script for Markdown rendering). No React/Vue/build tooling.

- **Auth screens** (`#authScreen`): login/register forms hitting `auth_service` at
  `http://localhost:8124`, `credentials:"include"` so the session cookie is sent.
- **Chat app** (`#appScreen`): sidebar of threads (queried via LangGraph's
  `/threads/search`) + a chat pane that posts to `/threads/{id}/runs/wait` on the
  LangGraph API at `http://localhost:8123`, passing `configurable.user_id = currentUser`
  so memory is scoped per logged-in user.
- Handles quota/429 errors, connection timeouts, and renders agent replies as Markdown.
- Polls `/docs` every 15s as a lightweight backend health check (status dot in header).

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, `Send` fan-out, `BaseStore`/`InMemoryStore`, `MemorySaver`) served via `langgraph-cli` / `langgraph-api` |
| LLM | Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai` |
| Structured extraction | [Trustcall](https://github.com/hinthornw/trustcall) for schema-constrained patch/insert tool calls |
| Agent framework glue | `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langsmith` (tracing) |
| Auth backend | FastAPI + Uvicorn, `psycopg` (Postgres driver), `bcrypt` for password hashing |
| Datastore | PostgreSQL 16 (shared by LangGraph checkpoints/store and auth `users`/`sessions`) |
| Task queue | Redis 6 (used internally by `langgraph-api`) |
| Frontend | Vanilla HTML/CSS/JS single-page app, Google Fonts (Fraunces/Inter/JetBrains Mono), `marked` for Markdown rendering |
| Containerization | Docker + Docker Compose (4 services: redis, postgres, langgraph-api, auth-api) |
| Language runtime | Python 3.11 (slim) in both containers |

## Running it

```
docker compose up
```
- LangGraph API → `http://localhost:8123` (from container port 2024)
- Auth API → `http://localhost:8124` (from container port 8000)
- Open `frontend/index.html` directly in a browser (or serve it — its `FRONTEND_ORIGIN` CORS
  default in `docker-compose.yml` assumes `http://localhost:5500`, e.g. VS Code Live
  Server).

Required secrets in `.env` at the project root: a Google Generative AI API key
(`Google_API_KEY` — see note below) and optionally a LangSmith API key for tracing.

## Notes / recommendations

- **No `.gitignore` exists.** `.env` (containing API key placeholders/secrets),
  `__pycache__/`, and `.langgraph_api/` are all currently untracked-but-unignored. Add a
  `.gitignore` covering `.env`, `__pycache__/`, `.langgraph_api/` before committing, to
  avoid accidentally staging secrets or build artifacts.
- **Env var casing mismatch:** `configuration.py`/Gemini client and `docker-compose.yml`
  both expect `GOOGLE_API_KEY` (uppercase), but `.env` currently defines
  `Google_API_KEY` (mixed case). Since env var names are case-sensitive on Linux
  containers, this likely needs to be corrected for the API key to actually be picked
  up. `huggingfacehub_api_token` and `TAVILY_API_KEY` in `.env` don't appear to be
  referenced anywhere in the current code.
- `langgraph_client.py` is a dev/debug utility, not part of the served app — it's not
  invoked by `frontend/index.html` or any Dockerfile.
