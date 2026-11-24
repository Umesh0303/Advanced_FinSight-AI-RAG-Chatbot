# FinSight AI Role-Based RAG API

FinSight AI is a Python Retrieval-Augmented Generation service for financial
documents. The REST API wraps the existing secure RAG chain; retrieval,
collection routing, source validation, and role restrictions remain in the
existing application modules.

## Architecture

- FastAPI exposes `GET /health` and `POST /api/v1/chat`.
- The API initializes the existing Gemini embedding model and the four
  persistent Chroma collections once during application startup.
- Chat requests call `src.chains.rag_chain.ask_question`; the API does not
  implement a parallel retrieval or generation pipeline.
- A Supabase access token is verified against the project's asymmetric JWKS.
  Its verified `sub` UUID is resolved against `public.profiles.id` using a
  backend-only Supabase service-role credential.
- Only `profiles.id`, `role`, `account_status`, and `customer_id` are selected.
  The role is mapped centrally only when `account_status` is exactly `active`.
  Request roles, role headers, usernames, emails, mutable user metadata, and
  Supabase's Postgres `role` claim are never application authorization sources.
- Existing access control restricts collection retrieval. Admin can retrieve
  from all authorized collections; other roles remain limited to their own
  collection.

## Installation

Use Python 3.13 and PowerShell from the project root:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set the values in `.env` locally. Keep `.env` private; it is ignored by Git.

### Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Server-side Groq credential for the language model |
| `GROQ_MODEL` | Groq model name |
| `GOOGLE_API_KEY` | Server-side Gemini embedding credential |
| `INGESTION_MANIFEST_HMAC_KEY` | Stable, private key used to authenticate ingestion decisions |
| `SUPABASE_URL` | Supabase project URL; used to form the Auth issuer and JWKS URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Backend-only Supabase secret key or legacy service-role JWT used for the trusted profile lookup; never put this in the frontend |
| `SUPABASE_JWT_AUDIENCE` | Expected JWT audience; defaults to `authenticated` |
| `API_CORS_ORIGINS` | Comma-separated, exact browser origins allowed to call the API |

Application roles map exactly as follows:

| `public.profiles.role` | Internal RAG role |
|---|---|
| `Admin` | `admin` |
| `Analyst` | `business_analyst` |
| `Business Analyst` | `business_analyst` |
| `Customer` | `customer` |
| `Support Agent` | `risk_analyst` |
| `Risk Analyst` | `risk_analyst` |

Role matching is exact and case-sensitive. Unknown values and any
`account_status` other than the exact value `active` are denied. In
particular, Support Agent and Risk Analyst map only to `risk_analyst`.

### Supabase identity lookup

The verified JWT subject is already the Supabase Auth UUID and is matched only
against `public.profiles.id`. The operational `public.app_users` dataset and
its `user_id` field are not used for identity or authorization. The lookup
selects only `id`, `role`, `account_status`, and `customer_id`; profile status
must be exactly `active`. It fails closed for missing profiles, inactive or
unknown statuses, and unmapped roles.

The service-role credential bypasses RLS and must remain in the backend's
private environment. It is used only after JWT verification and only for the
verified subject's exact profile lookup. The service-role key is never accepted
from requests or sent to the frontend. No database migration or live Supabase
schema change is required for this identity relationship.

JWT verification uses the Supabase project's asymmetric signing keys. Configure
the Supabase project to use an asymmetric signing key and ensure the API host
can reach its Auth JWKS endpoint. Legacy shared-secret HS256 tokens are not
accepted by this API.

`API_CORS_ORIGINS` must contain exact origins (scheme, host, and optional port),
not paths or a wildcard. Configure the deployed Google AI Studio application's
browser origin here. CORS is not authentication.

## Run locally

```powershell
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

The interactive CLI remains available with:

```powershell
python main.py
```

The CLI's role picker is for local testing only and is not authentication.

## API

### `GET /health`

Unauthenticated process health check:

```json
{"status": "ok"}
```

### `POST /api/v1/chat`

Requires `Authorization: Bearer <Supabase access token>`. The request does not
accept a role or user ID.

Request:

```json
{
  "message": "What revenue does the report describe?",
  "session_id": "optional-client-correlation-id"
}
```

Response:

```json
{
  "success": true,
  "answer": "The retrieved documents report ...",
  "session_id": "optional-client-correlation-id",
  "sources": [
    {
      "source": "annual_report.pdf",
      "page_number": 12,
      "section_title": "Revenue",
      "document_id": "validated-document-id",
      "chunk_id": "validated-chunk-id"
    }
  ]
}
```

When omitted, `session_id` is generated by the API. It is currently only a
correlation identifier: the repository has no conversation-history store, so
requests are stateless and sessions cannot be listed or resumed.

Invalid requests return a sanitized `422`; missing or invalid tokens return
`401`; an inactive user, missing identity row, or unknown role returns `403`.
A missing profile schema, invalid server-side service credential, or unavailable
Supabase lookup returns a safe service error without internal exception details.

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`
when the API is running.

## Google AI Studio integration

Configure the frontend to call the deployed API's `/api/v1/chat` endpoint and
send the current Supabase access token as a bearer token. Do not send a role or
user ID as authorization data, and do not place Groq or Gemini credentials in
the frontend. Set the frontend's exact browser origin in `API_CORS_ORIGINS`.
The separate frontend project is not modified here.

## Unsupported endpoints and current limitations

The repository has no session persistence, document-listing service, or feedback
store. Therefore `/api/v1/sessions`, `/api/v1/documents`, and
`/api/v1/feedback` are not exposed; no new database or storage system has been
added. Authenticated chat remains unavailable until the approved schema is
applied, trusted UUID mappings are backfilled, and the Supabase publishable key
is configured.

## Tests

```powershell
python -m compileall src main.py
python -m pytest -q
```
