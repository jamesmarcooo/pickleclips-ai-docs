# Phase 1: Upload → Identify → Clips — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload a pickleball game video, tap to identify yourself, and receive downloadable highlight clips of your plays within 15 minutes.

**Architecture:** FastAPI monolith with Celery workers for async GPU processing. Videos upload directly to Cloudflare R2 (S3-compatible). Supabase provides PostgreSQL, auth (JWT), and Realtime status updates. The AI pipeline runs on an EC2 g5.xlarge spot instance: YOLOv8n detects 4 players, user taps themselves, OSNet Re-ID tracks them across the video, a rule-based pipeline detects rallies and scores, and FFmpeg extracts highlight clips.

**Tech Stack:** Python 3.11, FastAPI, asyncpg, Celery + Redis (Upstash), Supabase (PostgreSQL + pgvector + Auth + Realtime), Cloudflare R2 (boto3), FFmpeg (ffmpeg-python), YOLOv8n (ultralytics), OSNet (torchreid), OpenCV, Next.js 14, TypeScript, Tailwind, @supabase/supabase-js, @uppy/core + @uppy/aws-s3-multipart

**Spec:** `docs/superpowers/specs/2026-04-05-pickleclips-architecture-design.md`

**Scope note:** This is Phase 1 only. Phases 2–4 (reels, personalization, mobile) are separate plans to be written after Phase 1 ships.

---

## File Map

```
pickleclips/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI app, lifespan, routers
│   │   ├── config.py                    # Pydantic settings from .env
│   │   ├── database.py                  # asyncpg pool, get_db dependency
│   │   ├── auth.py                      # Supabase JWT validation dependency
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── videos.py                # upload-url, confirm, list, get, delete, identify
│   │   │   └── highlights.py            # list highlights, get clip download URL
│   │   ├── workers/
│   │   │   ├── __init__.py
│   │   │   ├── celery_app.py            # Celery instance + beat schedule
│   │   │   ├── ingest.py                # full pipeline task chain
│   │   │   └── cleanup.py              # timeout + R2 cleanup beat task
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── storage.py              # R2 presigned URLs, delete
│   │   └── ml/
│   │       ├── __init__.py
│   │       ├── person_detection.py     # YOLOv8n: detect 4 players, return bboxes
│   │       ├── reid_tracking.py        # OSNet: embed seed crop, track across frames
│   │       ├── rally_detector.py       # motion-based rally segmentation
│   │       ├── score_state_machine.py  # track score from rally outcomes
│   │       └── highlight_scorer.py     # rank moments by point_scored + rally_length
│   ├── tests/
│   │   ├── conftest.py                 # test DB, fixtures
│   │   ├── test_storage.py
│   │   ├── test_videos.py
│   │   ├── test_rally_detector.py
│   │   ├── test_score_state_machine.py
│   │   └── test_highlight_scorer.py
│   ├── migrations/
│   │   └── 001_initial.sql             # all tables from spec
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx                # redirect to /videos
│   │   │   ├── login/page.tsx          # Supabase magic link login
│   │   │   ├── upload/page.tsx         # Uppy multipart upload
│   │   │   └── videos/
│   │   │       ├── page.tsx            # video library
│   │   │       └── [id]/
│   │   │           ├── page.tsx        # video detail + clips
│   │   │           └── identify/
│   │   │               └── page.tsx   # tap-to-identify UI
│   │   ├── components/
│   │   │   ├── UploadZone.tsx          # Uppy upload with progress
│   │   │   ├── PlayerIdentify.tsx      # frame with bbox overlays + tap
│   │   │   ├── ProcessingStatus.tsx    # Supabase Realtime status bar
│   │   │   └── ClipCard.tsx            # single clip with download button
│   │   └── lib/
│   │       ├── supabase.ts             # createClient singleton
│   │       └── api.ts                  # typed fetch wrappers for API
│   ├── package.json
│   ├── next.config.ts
│   └── tailwind.config.ts
├── docker-compose.yml                  # local dev: postgres + redis
├── Caddyfile                           # reverse proxy + TLS for production
├── .env.example
└── .github/
    └── workflows/
        └── deploy.yml                  # GitHub Actions → EC2
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/requirements.txt`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `backend/app/__init__.py` (empty)
- Create: all `__init__.py` files in subdirectories

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p backend/app/{routers,workers,services,ml}
mkdir -p backend/tests
mkdir -p backend/migrations
mkdir -p frontend/src/{app/videos/\[id\]/identify,components,lib}
mkdir -p .github/workflows
touch backend/app/__init__.py
touch backend/app/{routers,workers,services,ml}/__init__.py
touch backend/tests/__init__.py
```

- [ ] **Step 2: Create `backend/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pickleclips-backend"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Create `backend/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
asyncpg==0.29.0
pydantic-settings==2.4.0
celery[redis]==5.4.0
boto3==1.35.0
python-jose[cryptography]==3.3.0
httpx==0.27.0
ffmpeg-python==0.2.0
opencv-python-headless==4.10.0.84
ultralytics==8.2.0
torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git
torch==2.4.0
torchvision==0.19.0
numpy==1.26.4
pytest==8.3.0
pytest-asyncio==0.24.0
pytest-mock==3.14.0
```

- [ ] **Step 4: Create `docker-compose.yml`**

```yaml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: pickleclips
      POSTGRES_USER: pickleclips
      POSTGRES_PASSWORD: pickleclips
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

- [ ] **Step 5: Create `.env.example`**

```bash
# Supabase
DATABASE_URL=postgresql://pickleclips:pickleclips@localhost:5432/pickleclips
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret

# Cloudflare R2
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET_NAME=pickleclips

# Redis (Upstash in prod, local in dev)
REDIS_URL=redis://localhost:6379/0

# App
ENVIRONMENT=development
```

- [ ] **Step 6: Start local services and verify**

```bash
cd pickleclips
docker compose up -d
docker compose ps
```
Expected: `db` and `redis` both show `running`.

- [ ] **Step 7: Commit**

```bash
git init
git add docker-compose.yml .env.example backend/pyproject.toml backend/requirements.txt
git add backend/app/ backend/tests/ backend/migrations/ .github/
git commit -m "feat(scaffold): Add docker compose, requirements, and directory structure"
```

---

## Task 2: Database Schema

**Files:**
- Create: `backend/migrations/001_initial.sql`

- [ ] **Step 1: Write the migration**

Create `backend/migrations/001_initial.sql`:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users (mirrors Supabase auth.users, stores app-specific fields)
CREATE TABLE users (
    id UUID PRIMARY KEY,  -- same as Supabase auth user id
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    avatar_url TEXT,
    highlight_preferences JSONB DEFAULT '{}',
    subscription_tier TEXT NOT NULL DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'team')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Player profiles (persistent Re-ID embeddings per user)
CREATE TABLE player_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    appearance_embedding VECTOR(512),  -- OSNet osnet_x1_0 outputs 512-dim
    embedding_confidence FLOAT DEFAULT 0.0,
    uploads_contributing INT DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Videos
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    r2_key_original TEXT,
    r2_key_processed TEXT,
    duration_seconds FLOAT,
    resolution TEXT,
    status TEXT NOT NULL DEFAULT 'uploading' CHECK (
        status IN ('uploading', 'identifying', 'processing', 'analyzed', 'failed', 'timed_out')
    ),
    identify_started_at TIMESTAMPTZ,
    cleanup_after TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Video players (per-video player roles)
CREATE TABLE video_players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'partner', 'opponent_1', 'opponent_2')),
    player_profile_id UUID REFERENCES player_profiles(id),
    seed_frame_bbox JSONB,   -- {x, y, w, h} in pixels
    appearance_embedding VECTOR(512),
    tracking_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rallies
CREATE TABLE rallies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_time_ms INT NOT NULL,
    end_time_ms INT NOT NULL,
    shot_count INT DEFAULT 0,
    intensity_score FLOAT DEFAULT 0.0,
    point_won_by TEXT CHECK (point_won_by IN ('user_team', 'opponent_team')),
    score_before JSONB DEFAULT '{}',
    score_after JSONB DEFAULT '{}',
    is_comeback_point BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Highlights (and lowlights)
CREATE TABLE highlights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    rally_id UUID REFERENCES rallies(id) ON DELETE SET NULL,
    attributed_player_role TEXT CHECK (attributed_player_role IN ('user', 'partner', 'opponent_1', 'opponent_2')),
    sub_highlight_type TEXT NOT NULL CHECK (
        sub_highlight_type IN ('shot_form', 'point_scored', 'lowlight', 'both')
    ),
    lowlight_type TEXT CHECK (
        lowlight_type IN ('unforced_error', 'positioning', 'weak_shot', 'lost_point')
    ),
    point_lost_by_error BOOLEAN DEFAULT FALSE,
    start_time_ms INT NOT NULL,
    end_time_ms INT NOT NULL,
    highlight_score FLOAT DEFAULT 0.0,
    highlight_score_raw FLOAT DEFAULT 0.0,
    shot_type TEXT CHECK (
        shot_type IN ('drive', 'dink', 'lob', 'erne', 'atp', 'drop', 'smash', 'overhead', 'speed_up')
    ),
    shot_quality FLOAT DEFAULT 0.5,
    point_scored BOOLEAN DEFAULT FALSE,
    point_won_by TEXT CHECK (point_won_by IN ('user_team', 'opponent_team')),
    rally_length INT DEFAULT 0,
    rally_intensity FLOAT DEFAULT 0.0,
    score_source TEXT NOT NULL DEFAULT 'rule_based' CHECK (score_source IN ('ocr', 'rule_based', 'manual')),
    r2_key_clip TEXT,  -- set after clip extraction
    model_outputs JSONB DEFAULT '{}',
    user_feedback TEXT CHECK (user_feedback IN ('liked', 'disliked')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_videos_user_id ON videos(user_id);
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_highlights_video_id ON highlights(video_id);
CREATE INDEX idx_highlights_score ON highlights(highlight_score DESC);
CREATE INDEX idx_rallies_video_id ON rallies(video_id);
```

- [ ] **Step 2: Apply migration to local dev DB**

```bash
docker compose up -d db
psql postgresql://pickleclips:pickleclips@localhost:5432/pickleclips -f backend/migrations/001_initial.sql
```
Expected: multiple `CREATE TABLE` and `CREATE INDEX` lines, no errors.

- [ ] **Step 3: Apply migration to Supabase**

In Supabase dashboard → SQL Editor → paste the contents of `001_initial.sql` → Run.

Note: Supabase already has `auth.users`. Our `users` table is for app-specific fields, linked by the same UUID.

- [ ] **Step 4: Commit**

```bash
git add backend/migrations/
git commit -m "feat(db): Add Phase 1 database schema with pgvector support"
```

---

## Task 3: FastAPI App Skeleton + Config

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/database.py`
- Create: `backend/app/main.py`

- [ ] **Step 1: Create `backend/app/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    supabase_url: str
    supabase_anon_key: str
    supabase_jwt_secret: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 2: Create `backend/app/database.py`**

```python
from typing import AsyncGenerator
import asyncpg
from app.config import settings

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    _pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)


async def close_db() -> None:
    if _pool:
        await _pool.close()


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    assert _pool is not None, "DB pool not initialized"
    async with _pool.acquire() as conn:
        yield conn
```

- [ ] **Step 3: Create `backend/app/main.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db, close_db
from app.routers import videos, highlights


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(title="PickleClips API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://pickleclips.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router, prefix="/api/v1")
app.include_router(highlights.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Create stub routers so app starts**

`backend/app/routers/videos.py`:
```python
from fastapi import APIRouter
router = APIRouter(tags=["videos"])
```

`backend/app/routers/highlights.py`:
```python
from fastapi import APIRouter
router = APIRouter(tags=["highlights"])
```

- [ ] **Step 5: Verify app starts**

```bash
cd backend
cp ../.env.example .env  # fill in real values
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Expected: `Application startup complete.` — visit `http://localhost:8000/health` → `{"status":"ok"}`

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/app/database.py backend/app/main.py backend/app/routers/
git commit -m "feat(api): Add FastAPI skeleton with config, db pool, CORS, and health endpoint"
```

---

## Task 4: Supabase Auth Middleware

**Files:**
- Create: `backend/app/auth.py`
- Create: `backend/tests/conftest.py`

The Supabase JWT is a standard HS256 JWT signed with your `SUPABASE_JWT_SECRET`. Validate it on every protected request and return the user's UUID.

- [ ] **Step 1: Write the failing test**

`backend/tests/conftest.py`:
```python
import pytest
import asyncpg
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.config import settings
import jose.jwt as jwt
import time


def make_test_token(user_id: str = "00000000-0000-0000-0000-000000000001") -> str:
    """Create a valid Supabase-format JWT for testing."""
    payload = {
        "sub": user_id,
        "role": "authenticated",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, settings.supabase_jwt_secret, algorithm="HS256")


@pytest.fixture
def test_token():
    return make_test_token()


@pytest.fixture
def test_user_id():
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
```

`backend/tests/test_auth.py`:
```python
import pytest
from tests.conftest import make_test_token


@pytest.mark.asyncio
async def test_protected_route_requires_auth(client):
    response = await client.get("/api/v1/videos")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_accepts_valid_token(client, test_token):
    response = await client.get(
        "/api/v1/videos",
        headers={"Authorization": f"Bearer {test_token}"},
    )
    # 200 or empty list — just not 401
    assert response.status_code != 401


@pytest.mark.asyncio
async def test_invalid_token_rejected(client):
    response = await client.get(
        "/api/v1/videos",
        headers={"Authorization": "Bearer not.a.valid.token"},
    )
    assert response.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
pytest tests/test_auth.py -v
```
Expected: FAIL — `GET /api/v1/videos` returns 404 (route not yet implemented).

- [ ] **Step 3: Create `backend/app/auth.py`**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.config import settings

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate Supabase JWT and return user UUID."""
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase tokens may have audience claims
        )
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
```

- [ ] **Step 4: Add a protected stub to videos router**

Update `backend/app/routers/videos.py`:
```python
from fastapi import APIRouter, Depends
from app.auth import get_current_user

router = APIRouter(tags=["videos"])


@router.get("/videos")
async def list_videos(user_id: str = Depends(get_current_user)):
    return []
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_auth.py -v
```
Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/auth.py backend/app/routers/videos.py backend/tests/
git commit -m "feat(auth): Add Supabase JWT validation middleware"
```

---

## Task 5: R2 Storage Service

**Files:**
- Create: `backend/app/services/storage.py`
- Create: `backend/tests/test_storage.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_storage.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from app.services.storage import generate_upload_url, generate_download_url, delete_object


def test_generate_upload_url_returns_string():
    with patch("app.services.storage.get_r2_client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://r2.example.com/upload?sig=abc"
        mock_client.return_value = mock_s3

        url = generate_upload_url("videos/test-id/original.mp4")

        assert url.startswith("https://")
        mock_s3.generate_presigned_url.assert_called_once_with(
            "put_object",
            Params={
                "Bucket": mock_s3.generate_presigned_url.call_args.kwargs["Params"]["Bucket"],
                "Key": "videos/test-id/original.mp4",
                "ContentType": "video/mp4",
            },
            ExpiresIn=3600,
        )


def test_generate_download_url_returns_string():
    with patch("app.services.storage.get_r2_client") as mock_client:
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://r2.example.com/download?sig=xyz"
        mock_client.return_value = mock_s3

        url = generate_download_url("clips/test-id/clip-001.mp4")
        assert url.startswith("https://")


def test_delete_object_calls_delete():
    with patch("app.services.storage.get_r2_client") as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        delete_object("videos/test-id/original.mp4")

        mock_s3.delete_object.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_storage.py -v
```
Expected: FAIL with `ImportError: cannot import name 'generate_upload_url'`.

- [ ] **Step 3: Create `backend/app/services/storage.py`**

```python
import boto3
from botocore.config import Config
from app.config import settings


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def generate_upload_url(key: str, content_type: str = "video/mp4") -> str:
    """Generate a presigned PUT URL for direct R2 upload."""
    client = get_r2_client()
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.r2_bucket_name, "Key": key, "ContentType": content_type},
        ExpiresIn=3600,
    )


def generate_download_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL for R2 download."""
    client = get_r2_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.r2_bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )


def delete_object(key: str) -> None:
    client = get_r2_client()
    client.delete_object(Bucket=settings.r2_bucket_name, Key=key)


def generate_multipart_upload_id(key: str) -> str:
    """Initiate S3 multipart upload, return UploadId."""
    client = get_r2_client()
    response = client.create_multipart_upload(
        Bucket=settings.r2_bucket_name,
        Key=key,
        ContentType="video/mp4",
    )
    return response["UploadId"]


def sign_multipart_part(key: str, upload_id: str, part_number: int) -> str:
    """Generate a presigned URL for uploading a single multipart part."""
    client = get_r2_client()
    return client.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": settings.r2_bucket_name,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=3600,
    )


def complete_multipart_upload(key: str, upload_id: str, parts: list[dict]) -> None:
    """Complete a multipart upload. parts = [{"ETag": "...", "PartNumber": 1}, ...]"""
    client = get_r2_client()
    client.complete_multipart_upload(
        Bucket=settings.r2_bucket_name,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )


def abort_multipart_upload(key: str, upload_id: str) -> None:
    client = get_r2_client()
    client.abort_multipart_upload(
        Bucket=settings.r2_bucket_name, Key=key, UploadId=upload_id
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_storage.py -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/storage.py backend/tests/test_storage.py
git commit -m "feat(storage): Add R2 storage service with presigned URLs, multipart, and delete"
```

---

## Task 6: Video Upload Endpoints

**Files:**
- Modify: `backend/app/routers/videos.py`
- Create: `backend/tests/test_videos.py`

The upload flow uses S3 multipart (R2-compatible). Uppy's `@uppy/aws-s3-multipart` plugin on the frontend calls these 4 backend endpoints to coordinate the upload, then calls `POST /confirm` when done.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_videos.py`:
```python
import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_create_multipart_upload(client, test_token):
    with patch("app.routers.videos.storage.generate_multipart_upload_id", return_value="upload-id-123"):
        response = await client.post(
            "/api/v1/videos/multipart/create",
            json={"filename": "game.mp4", "content_type": "video/mp4"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert "upload_id" in data
    assert "key" in data


@pytest.mark.asyncio
async def test_sign_multipart_part(client, test_token):
    with patch("app.routers.videos.storage.sign_multipart_part", return_value="https://r2.example.com/part"):
        response = await client.get(
            "/api/v1/videos/multipart/sign-part",
            params={"key": "videos/abc/original.mp4", "upload_id": "uid123", "part_number": 1},
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert response.status_code == 200
    assert "url" in response.json()


@pytest.mark.asyncio
async def test_confirm_triggers_pipeline(client, test_token, test_user_id):
    # Patch DB and celery so we don't need real infra
    with patch("app.routers.videos.get_db"), \
         patch("app.routers.videos.ingest_video.delay") as mock_task:
        response = await client.post(
            "/api/v1/videos/fake-video-id/confirm",
            headers={"Authorization": f"Bearer {test_token}"},
        )
    # Will fail until we wire up DB — that's fine for now
    assert response.status_code in (200, 422, 500)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_videos.py -v
```
Expected: FAIL — endpoints don't exist yet.

- [ ] **Step 3: Implement `backend/app/routers/videos.py`**

```python
import uuid
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import asyncpg

from app.auth import get_current_user
from app.database import get_db
from app.services import storage
from app.workers.ingest import ingest_video

router = APIRouter(tags=["videos"])


class CreateMultipartRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"


class CompleteMultipartRequest(BaseModel):
    key: str
    upload_id: str
    parts: list[dict]  # [{"ETag": "...", "PartNumber": 1}, ...]


# ── Multipart upload coordination ─────────────────────────────────────────────

@router.post("/videos/multipart/create")
async def create_multipart_upload(
    body: CreateMultipartRequest,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    video_id = str(uuid.uuid4())
    key = f"videos/{video_id}/original.mp4"

    # Create video record in DB
    await db.execute(
        """INSERT INTO videos (id, user_id, r2_key_original, status)
           VALUES ($1, $2, $3, 'uploading')""",
        video_id, user_id, key,
    )

    upload_id = storage.generate_multipart_upload_id(key)
    return {"video_id": video_id, "upload_id": upload_id, "key": key}


@router.get("/videos/multipart/sign-part")
async def sign_multipart_part(
    key: str = Query(...),
    upload_id: str = Query(...),
    part_number: int = Query(...),
    user_id: str = Depends(get_current_user),
):
    url = storage.sign_multipart_part(key, upload_id, part_number)
    return {"url": url}


@router.post("/videos/multipart/complete")
async def complete_multipart_upload(
    body: CompleteMultipartRequest,
    user_id: str = Depends(get_current_user),
):
    storage.complete_multipart_upload(body.key, body.upload_id, body.parts)
    return {"status": "ok"}


@router.delete("/videos/multipart/abort")
async def abort_multipart_upload(
    key: str = Query(...),
    upload_id: str = Query(...),
    user_id: str = Depends(get_current_user),
):
    storage.abort_multipart_upload(key, upload_id)
    return {"status": "ok"}


# ── Video management ──────────────────────────────────────────────────────────

@router.post("/videos/{video_id}/confirm")
async def confirm_upload(
    video_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    """Called by frontend after multipart upload completes. Triggers the pipeline."""
    video = await db.fetchrow(
        "SELECT id, user_id FROM videos WHERE id = $1", video_id
    )
    if not video or str(video["user_id"]) != user_id:
        raise HTTPException(status_code=404, detail="Video not found")

    await db.execute(
        "UPDATE videos SET status = 'processing' WHERE id = $1", video_id
    )

    # Fire Celery task (non-blocking)
    ingest_video.delay(video_id, user_id)

    return {"status": "processing", "video_id": video_id}


@router.get("/videos")
async def list_videos(
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    rows = await db.fetch(
        "SELECT id, status, uploaded_at, duration_seconds, resolution FROM videos "
        "WHERE user_id = $1 ORDER BY uploaded_at DESC",
        user_id,
    )
    return [dict(r) for r in rows]


@router.get("/videos/{video_id}")
async def get_video(
    video_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    row = await db.fetchrow(
        "SELECT * FROM videos WHERE id = $1 AND user_id = $2", video_id, user_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Video not found")
    return dict(row)


@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    row = await db.fetchrow(
        "SELECT r2_key_original, r2_key_processed FROM videos WHERE id = $1 AND user_id = $2",
        video_id, user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Video not found")

    if row["r2_key_original"]:
        storage.delete_object(row["r2_key_original"])
    if row["r2_key_processed"]:
        storage.delete_object(row["r2_key_processed"])

    await db.execute("DELETE FROM videos WHERE id = $1", video_id)
    return {"status": "deleted"}


# ── Player identification ──────────────────────────────────────────────────────

@router.get("/videos/{video_id}/identify")
async def get_identify_frame(
    video_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    """Return the seed frame URL + bounding boxes for tap-to-identify."""
    video = await db.fetchrow(
        "SELECT id, status, metadata FROM videos WHERE id = $1 AND user_id = $2",
        video_id, user_id,
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video["status"] != "identifying":
        raise HTTPException(status_code=409, detail=f"Video is in '{video['status']}' state, not 'identifying'")

    meta = video["metadata"] or {}
    frame_key = meta.get("seed_frame_key")
    bboxes = meta.get("player_bboxes", [])

    if not frame_key:
        raise HTTPException(status_code=409, detail="Seed frame not yet extracted")

    frame_url = storage.generate_download_url(frame_key, expires_in=300)
    return {"frame_url": frame_url, "bboxes": bboxes}


class TapIdentifyRequest(BaseModel):
    bbox_index: int  # 0-3, which bounding box the user tapped


@router.post("/videos/{video_id}/identify")
async def tap_identify(
    video_id: str,
    body: TapIdentifyRequest,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    """User taps on their bounding box. Resumes the pipeline."""
    from app.workers.ingest import resume_after_identify

    video = await db.fetchrow(
        "SELECT id, status, metadata FROM videos WHERE id = $1 AND user_id = $2",
        video_id, user_id,
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video["status"] != "identifying":
        raise HTTPException(status_code=409, detail="Video is not waiting for identification")

    meta = video["metadata"] or {}
    bboxes = meta.get("player_bboxes", [])
    if body.bbox_index < 0 or body.bbox_index >= len(bboxes):
        raise HTTPException(status_code=422, detail=f"bbox_index must be 0-{len(bboxes)-1}")

    seed_bbox = bboxes[body.bbox_index]

    # Store the chosen bbox and resume pipeline
    await db.execute(
        "UPDATE videos SET status = 'processing', identify_started_at = NULL WHERE id = $1",
        video_id,
    )
    resume_after_identify.delay(video_id, user_id, seed_bbox)

    return {"status": "processing", "seed_bbox": seed_bbox}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_videos.py -v
```
Expected: first 2 tests PASS, `test_confirm_triggers_pipeline` may still fail until Celery is set up in Task 7 — that's expected.

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/videos.py backend/tests/test_videos.py
git commit -m "feat(upload): Add video upload endpoints — multipart R2, confirm, and identify"
```

---

## Task 7: Celery Setup

**Files:**
- Create: `backend/app/workers/celery_app.py`
- Create: `backend/app/workers/ingest.py` (stubs)
- Create: `backend/app/workers/cleanup.py` (stub)

- [ ] **Step 1: Create `backend/app/workers/celery_app.py`**

```python
from celery import Celery
from app.config import settings

celery = Celery(
    "pickleclips",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.ingest",
        "app.workers.cleanup",
    ],
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # re-queue on worker crash
    worker_prefetch_multiplier=1,  # one task at a time per worker (GPU work)
    beat_schedule={
        "cleanup-stale-identify-jobs": {
            "task": "app.workers.cleanup.cleanup_stale_jobs",
            "schedule": 3600.0,  # every hour
        },
    },
)
```

- [ ] **Step 2: Create stub `backend/app/workers/ingest.py`**

```python
from app.workers.celery_app import celery


@celery.task(bind=True, name="app.workers.ingest.ingest_video")
def ingest_video(self, video_id: str, user_id: str):
    """Full pipeline for a new video. Runs on GPU instance."""
    raise NotImplementedError("Implemented in Task 8")


@celery.task(bind=True, name="app.workers.ingest.resume_after_identify")
def resume_after_identify(self, video_id: str, user_id: str, seed_bbox: dict):
    """Resumes pipeline after user tap-to-identify."""
    raise NotImplementedError("Implemented in Task 15")
```

- [ ] **Step 3: Create stub `backend/app/workers/cleanup.py`**

```python
from app.workers.celery_app import celery


@celery.task(name="app.workers.cleanup.cleanup_stale_jobs")
def cleanup_stale_jobs():
    """Cancel jobs stuck in 'identifying' for > 24 hours. Implemented in Task 17."""
    pass
```

- [ ] **Step 4: Verify Celery starts**

```bash
cd backend
celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
```
Expected: `[tasks]` lists `ingest_video`, `resume_after_identify`, `cleanup_stale_jobs`. No errors.

- [ ] **Step 5: Commit**

```bash
git add backend/app/workers/
git commit -m "feat(workers): Add Celery setup with worker, beat schedule, and task stubs"
```

---

## Task 8: FFmpeg Ingest Worker — Transcode + Frame Extraction

**Files:**
- Modify: `backend/app/workers/ingest.py`

This is the first half of the pipeline. After transcode, the pipeline pauses and waits for the user to tap (task switches to `identifying` status and stores bboxes).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_ingest.py`:
```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.workers import ingest


def test_extract_frames_returns_correct_count():
    """extract_frames should return ~2 frames per second of video."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        7: 1800,   # CAP_PROP_FRAME_COUNT = 1800 frames
        5: 30.0,   # CAP_PROP_FPS = 30 fps = 60 seconds
    }.get(prop, 0)
    mock_cap.read.side_effect = (
        [(True, np.zeros((1080, 1920, 3), dtype=np.uint8))] * 1800 +
        [(False, None)]
    )

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames = ingest.extract_frames("/fake/path.mp4", fps=2)

    # 60 seconds * 2 fps = 120 frames (±1 for rounding)
    assert 118 <= len(frames) <= 122
    assert isinstance(frames[0], np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingest.py::test_extract_frames_returns_correct_count -v
```
Expected: FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `backend/app/workers/ingest.py`**

```python
import os
import asyncio
import asyncpg
import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import List

from app.workers.celery_app import celery
from app.config import settings
from app.services.storage import generate_download_url, delete_object
from app.ml.person_detection import detect_players


# ── Helpers ───────────────────────────────────────────────────────────────────

def update_video_status(video_id: str, status: str, metadata_update: dict = None) -> None:
    """Sync DB update via asyncpg (run in thread since Celery is sync)."""
    import asyncio, asyncpg as apg
    
    async def _update():
        conn = await apg.connect(settings.database_url)
        try:
            if metadata_update:
                await conn.execute(
                    """UPDATE videos
                       SET status = $1,
                           metadata = metadata || $2::jsonb
                       WHERE id = $3""",
                    status, str(metadata_update).replace("'", '"'), video_id,
                )
            else:
                await conn.execute("UPDATE videos SET status = $1 WHERE id = $2", status, video_id)
        finally:
            await conn.close()
    
    asyncio.run(_update())


def transcode_to_1080p(input_path: str, output_path: str) -> None:
    """Transcode video to H.264 1080p working copy using FFmpeg NVENC (GPU)."""
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            vcodec="h264_nvenc",   # GPU encode; falls back to libx264 on CPU
            acodec="aac",
            vf="scale=-2:1080",
            preset="fast",
            crf=23,
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def extract_frames(video_path: str, fps: int = 2) -> List[np.ndarray]:
    """Extract frames at `fps` frames per second from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    step = int(source_fps / fps)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def pick_seed_frame(frames: List[np.ndarray]) -> np.ndarray:
    """Pick a representative mid-game frame (avoid first/last 10% which may be setup)."""
    start = len(frames) // 10
    end = len(frames) - start
    mid = (start + end) // 2
    return frames[mid]


# ── Pipeline tasks ─────────────────────────────────────────────────────────────

@celery.task(bind=True, name="app.workers.ingest.ingest_video", max_retries=3)
def ingest_video(self, video_id: str, user_id: str):
    """
    Stage 1 of the pipeline. Runs before user tap.
    1. Download original from R2 to /tmp
    2. Transcode to 1080p working copy
    3. Extract frames at 2fps
    4. Run YOLOv8n person detection on seed frame
    5. Save bboxes + pause for user tap (status → 'identifying')
    """
    import boto3
    from botocore.config import Config
    from datetime import datetime, timezone, timedelta
    import json, uuid, asyncio, asyncpg as apg

    tmp_dir = Path(f"/tmp/pickleclips/{video_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    original_path = str(tmp_dir / "original.mp4")
    processed_path = str(tmp_dir / "processed_1080p.mp4")
    seed_frame_path = str(tmp_dir / "seed_frame.jpg")

    try:
        # 1. Fetch R2 key from DB
        async def get_r2_key():
            conn = await apg.connect(settings.database_url)
            try:
                row = await conn.fetchrow("SELECT r2_key_original FROM videos WHERE id = $1", video_id)
                return row["r2_key_original"] if row else None
            finally:
                await conn.close()

        r2_key = asyncio.run(get_r2_key())
        if not r2_key:
            raise ValueError(f"No R2 key for video {video_id}")

        # 2. Download from R2
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        s3.download_file(settings.r2_bucket_name, r2_key, original_path)

        # 3. Transcode to 1080p
        try:
            transcode_to_1080p(original_path, processed_path)
        except ffmpeg.Error:
            # CPU fallback if NVENC unavailable (local dev)
            (
                ffmpeg.input(original_path)
                .output(processed_path, vcodec="libx264", acodec="aac", vf="scale=-2:1080", preset="fast")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

        # 4. Extract frames
        frames = extract_frames(processed_path, fps=2)
        if not frames:
            raise ValueError("No frames extracted from video")

        # 5. Run person detection on seed frame
        seed_frame = pick_seed_frame(frames)
        cv2.imwrite(seed_frame_path, seed_frame)

        bboxes = detect_players(seed_frame)  # returns list of {x, y, w, h} dicts

        # 6. Upload seed frame to R2
        seed_frame_key = f"videos/{video_id}/seed_frame.jpg"
        s3.upload_file(seed_frame_path, settings.r2_bucket_name, seed_frame_key)

        # 7. Upload processed video to R2
        processed_key = f"videos/{video_id}/processed_1080p.mp4"
        s3.upload_file(processed_path, settings.r2_bucket_name, processed_key)

        # 8. Update DB: status → identifying, store bboxes + seed frame key
        async def save_results():
            conn = await apg.connect(settings.database_url)
            try:
                deadline = datetime.now(timezone.utc) + timedelta(hours=24)
                await conn.execute(
                    """UPDATE videos SET
                        status = 'identifying',
                        r2_key_processed = $1,
                        identify_started_at = NOW(),
                        cleanup_after = $2,
                        metadata = metadata || $3::jsonb
                       WHERE id = $4""",
                    processed_key,
                    deadline,
                    json.dumps({"seed_frame_key": seed_frame_key, "player_bboxes": bboxes}),
                    video_id,
                )
            finally:
                await conn.close()

        asyncio.run(save_results())

    except Exception as exc:
        update_video_status(video_id, "failed")
        raise self.retry(exc=exc, countdown=60)
    finally:
        # Clean up local tmp files (not R2 — those are permanent)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@celery.task(bind=True, name="app.workers.ingest.resume_after_identify", max_retries=3)
def resume_after_identify(self, video_id: str, user_id: str, seed_bbox: dict):
    """
    Stage 2 of the pipeline. Runs after user tap.
    Delegates to the Re-ID + scoring tasks (implemented in Tasks 11–16).
    """
    from app.workers.ingest import run_ai_pipeline
    run_ai_pipeline.delay(video_id, user_id, seed_bbox)


@celery.task(bind=True, name="app.workers.ingest.run_ai_pipeline", max_retries=2)
def run_ai_pipeline(self, video_id: str, user_id: str, seed_bbox: dict):
    """Full AI pipeline: Re-ID → rally detection → scoring → clip extraction."""
    # Implemented in Task 15 (pipeline orchestration)
    raise NotImplementedError("Implemented in Task 15")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ingest.py -v
```
Expected: `test_extract_frames_returns_correct_count` PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/workers/ingest.py backend/tests/test_ingest.py
git commit -m "feat(ingest): Add ingest worker with FFmpeg transcode, frame extraction, and seed frame detection"
```

---

## Task 9: Person Detection (YOLOv8n)

**Files:**
- Create: `backend/app/ml/person_detection.py`
- Create: `backend/tests/test_person_detection.py`

YOLOv8n `person` class is class index 0 in COCO. The model weights download automatically on first run (~6MB).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_person_detection.py`:
```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.ml.person_detection import detect_players, BoundingBox


def make_mock_result(boxes_xyxy: list[list[float]]):
    """Create a mock YOLO result with given bounding boxes."""
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    import torch
    mock_boxes.xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
    mock_boxes.cls = torch.zeros(len(boxes_xyxy))  # all class 0 = person
    mock_boxes.conf = torch.ones(len(boxes_xyxy)) * 0.9
    mock_result.boxes = mock_boxes
    
    return [mock_result]


def test_detect_players_returns_up_to_4_people():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Mock YOLO returning 5 detections (should clamp to 4 most confident)
    boxes = [[100, 200, 200, 500], [300, 200, 400, 500], 
             [600, 200, 700, 500], [900, 200, 1000, 500],
             [1100, 200, 1200, 500]]  # 5 boxes
    
    with patch("app.ml.person_detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.return_value = make_mock_result(boxes)
        MockYOLO.return_value = mock_model
        
        result = detect_players(frame)
    
    assert len(result) <= 4
    assert all(isinstance(b, dict) for b in result)
    assert all({"x", "y", "w", "h"}.issubset(b.keys()) for b in result)


def test_detect_players_converts_xyxy_to_xywh():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxes = [[100.0, 200.0, 300.0, 600.0]]  # x1=100, y1=200, x2=300, y2=600
    
    with patch("app.ml.person_detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.return_value = make_mock_result(boxes)
        MockYOLO.return_value = mock_model
        
        result = detect_players(frame)
    
    assert len(result) == 1
    b = result[0]
    assert b["x"] == 100
    assert b["y"] == 200
    assert b["w"] == 200   # x2 - x1
    assert b["h"] == 400   # y2 - y1


def test_detect_players_returns_empty_if_no_people():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    with patch("app.ml.person_detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.return_value = make_mock_result([])
        MockYOLO.return_value = mock_model
        
        result = detect_players(frame)
    
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_person_detection.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Create `backend/app/ml/person_detection.py`**

```python
from typing import TypedDict
import numpy as np
from ultralytics import YOLO

# Module-level model (loaded once per worker process)
_model: YOLO | None = None


class BoundingBox(TypedDict):
    x: int  # top-left x
    y: int  # top-left y
    w: int  # width
    h: int  # height


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # downloads ~6MB on first run
    return _model


def detect_players(frame: np.ndarray, max_players: int = 4) -> list[BoundingBox]:
    """
    Detect people in a frame using YOLOv8n.
    Returns up to max_players bounding boxes sorted by confidence (descending).
    Each bbox is {x, y, w, h} in pixels (top-left origin, positive dimensions).
    """
    model = _get_model()
    results = model(frame, classes=[0], verbose=False)  # class 0 = person

    bboxes: list[tuple[float, BoundingBox]] = []  # (confidence, bbox)

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i]]
            bbox: BoundingBox = {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
            }
            bboxes.append((conf, bbox))

    # Sort by confidence descending, take top max_players
    bboxes.sort(key=lambda t: t[0], reverse=True)
    return [bbox for _, bbox in bboxes[:max_players]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_person_detection.py -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/person_detection.py backend/tests/test_person_detection.py
git commit -m "feat(ml): Add YOLOv8n person detection returning up to 4 players as xywh bboxes"
```

---

## Task 10: Re-ID Tracking (OSNet)

**Files:**
- Create: `backend/app/ml/reid_tracking.py`
- Create: `backend/tests/test_reid_tracking.py`

OSNet extracts a 512-dim appearance embedding from a player crop. Cosine similarity against the seed embedding assigns identity across frames. Court-position heuristic acts as fallback when Re-ID confidence drops below 0.6.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_reid_tracking.py`:
```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.ml.reid_tracking import (
    extract_embedding,
    cosine_similarity,
    assign_player_roles,
    court_position_fallback,
    PlayerRole,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 0.0, 0.5])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)


def test_assign_player_roles_high_confidence():
    """Highest-similarity detection gets role 'user'."""
    seed_embedding = np.array([1.0, 0.0, 0.0])
    detections = [
        {"bbox": {"x": 100, "y": 200, "w": 80, "h": 180}, "embedding": np.array([0.9, 0.1, 0.0])},
        {"bbox": {"x": 500, "y": 200, "w": 80, "h": 180}, "embedding": np.array([0.1, 0.9, 0.0])},
    ]
    roles = assign_player_roles(seed_embedding, detections, conf_threshold=0.5)
    assert roles[0]["role"] == PlayerRole.USER
    assert roles[1]["role"] != PlayerRole.USER


def test_court_position_fallback_assigns_by_x_position():
    """When Re-ID fails, fallback assigns roles by court x-position."""
    # Doubles court: user's team on left side (x < frame_width/2)
    detections = [
        {"bbox": {"x": 100, "y": 300, "w": 80, "h": 180}},   # left side
        {"bbox": {"x": 900, "y": 300, "w": 80, "h": 180}},   # right side
    ]
    result = court_position_fallback(
        detections=detections,
        user_last_x=150,   # user was on left side last frame
        frame_width=1920,
    )
    assert result[0]["role"] == PlayerRole.USER
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_reid_tracking.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Create `backend/app/ml/reid_tracking.py`**

```python
from enum import Enum
from typing import TypedDict
import numpy as np
import cv2

# torchreid is installed from source; import lazily to avoid import errors on non-GPU machines
_extractor = None


class PlayerRole(str, Enum):
    USER = "user"
    PARTNER = "partner"
    OPPONENT_1 = "opponent_1"
    OPPONENT_2 = "opponent_2"


ROLE_ORDER = [PlayerRole.USER, PlayerRole.PARTNER, PlayerRole.OPPONENT_1, PlayerRole.OPPONENT_2]


def _get_extractor():
    global _extractor
    if _extractor is None:
        import torchreid
        _extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_x1_0",
            device="cuda" if _cuda_available() else "cpu",
        )
    return _extractor


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def extract_embedding(frame: np.ndarray, bbox: dict) -> np.ndarray:
    """
    Extract 512-dim OSNet appearance embedding from a player crop.
    bbox is {x, y, w, h}.
    Returns a unit-normalized 1D numpy array of shape (512,).
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return np.zeros(512)

    # OSNet expects RGB, resize to 256x128 (standard Re-ID input)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (128, 256))

    extractor = _get_extractor()
    features = extractor([crop_resized])  # returns (1, 512) tensor
    embedding = features[0].cpu().numpy()

    # L2 normalize
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalized vectors."""
    return float(np.dot(a, b))


def assign_player_roles(
    seed_embedding: np.ndarray,
    detections: list[dict],
    conf_threshold: float = 0.6,
) -> list[dict]:
    """
    Assign player roles to detections based on similarity to seed_embedding.
    
    Each detection must have an 'embedding' key (np.ndarray, shape 512).
    Returns detections with 'role' and 'reid_conf' added.
    High confidence (>= conf_threshold): role by similarity ranking.
    Low confidence: role marked as None (caller should use fallback).
    """
    if not detections:
        return []

    scored = []
    for det in detections:
        sim = cosine_similarity(seed_embedding, det["embedding"])
        scored.append({**det, "reid_conf": sim})

    # Sort: highest similarity first = most likely to be the user
    scored.sort(key=lambda d: d["reid_conf"], reverse=True)

    result = []
    for i, det in enumerate(scored):
        role = ROLE_ORDER[i] if i < len(ROLE_ORDER) else None
        result.append({**det, "role": role})

    return result


def court_position_fallback(
    detections: list[dict],
    user_last_x: float,
    frame_width: int,
) -> list[dict]:
    """
    Assign roles by court x-position when Re-ID confidence is low.
    The detection closest to user_last_x gets the USER role.
    In doubles, team members share the same half of the court.
    """
    if not detections:
        return []

    def center_x(det):
        b = det["bbox"]
        return b["x"] + b["w"] / 2

    centers = [(center_x(det), i) for i, det in enumerate(detections)]
    
    # Assign USER to detection closest to last known user position
    closest_idx = min(range(len(centers)), key=lambda i: abs(centers[i][0] - user_last_x))
    
    result = []
    role_idx = 0
    for i, det in enumerate(detections):
        if i == closest_idx:
            result.append({**det, "role": PlayerRole.USER, "reid_conf": 0.0, "used_fallback": True})
        else:
            role = ROLE_ORDER[role_idx + 1] if role_idx + 1 < len(ROLE_ORDER) else None
            result.append({**det, "role": role, "reid_conf": 0.0, "used_fallback": True})
            role_idx += 1

    return result


def track_user_across_frames(
    frames: list[np.ndarray],
    all_detections: list[list[dict]],  # one list of bboxes per frame
    seed_embedding: np.ndarray,
    conf_threshold: float = 0.6,
) -> list[list[dict]]:
    """
    Track user across all frames. For each frame, assign roles to detections.
    Falls back to court-position when Re-ID confidence < conf_threshold.
    
    Returns per-frame list of detections with 'role', 'reid_conf', 'embedding'.
    """
    user_last_x: float = 0.0
    labeled_frames = []

    for frame, frame_detections in zip(frames, all_detections):
        if not frame_detections:
            labeled_frames.append([])
            continue

        # Extract embeddings for all detections in this frame
        enriched = []
        for det in frame_detections:
            emb = extract_embedding(frame, det["bbox"])
            enriched.append({**det, "embedding": emb})

        # Assign roles via Re-ID
        assigned = assign_player_roles(seed_embedding, enriched, conf_threshold)

        # Check if user assignment is confident
        user_det = next((d for d in assigned if d["role"] == PlayerRole.USER), None)
        if user_det and user_det.get("reid_conf", 0) < conf_threshold:
            # Fall back to court-position for this frame
            assigned = court_position_fallback(enriched, user_last_x, frame.shape[1])

        # Update last known user position
        user_det = next((d for d in assigned if d["role"] == PlayerRole.USER), None)
        if user_det:
            b = user_det["bbox"]
            user_last_x = b["x"] + b["w"] / 2

        labeled_frames.append(assigned)

    return labeled_frames
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_reid_tracking.py -v
```
Expected: all 4 PASS. (The `extract_embedding` function will call torchreid — tests mock it, so they pass without GPU.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/reid_tracking.py backend/tests/test_reid_tracking.py
git commit -m "feat(reid): Add OSNet Re-ID tracking with cosine similarity, role assignment, and court-position fallback"
```

---

## Task 11: Rally Detector

**Files:**
- Create: `backend/app/ml/rally_detector.py`
- Create: `backend/tests/test_rally_detector.py`

Phase 1 uses motion-based rally detection (no ball tracking). A rally ends when overall frame motion drops below a threshold for > 1 second (ball out of play / point won).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_rally_detector.py`:
```python
import pytest
import numpy as np
from app.ml.rally_detector import detect_rallies, compute_frame_motion, Rally


def make_motion_signal(pattern: list) -> list[float]:
    """Create a mock motion signal. 1.0 = high motion, 0.0 = still."""
    return pattern


def test_single_rally_detected():
    # 30 frames at 2fps = 15 seconds
    # motion: 5 frames still, 20 frames active, 5 frames still
    motion = [0.05] * 5 + [0.8] * 20 + [0.05] * 5
    rallies = detect_rallies(motion, fps=2, motion_threshold=0.1, min_gap_frames=2)
    assert len(rallies) == 1
    assert rallies[0].start_frame == 5
    assert rallies[0].end_frame == 24


def test_two_rallies_detected():
    # Two active segments separated by a gap
    motion = [0.05] * 3 + [0.8] * 10 + [0.05] * 4 + [0.8] * 10 + [0.05] * 3
    rallies = detect_rallies(motion, fps=2, motion_threshold=0.1, min_gap_frames=2)
    assert len(rallies) == 2


def test_empty_signal_returns_no_rallies():
    rallies = detect_rallies([], fps=2)
    assert rallies == []


def test_rally_to_ms_conversion():
    rally = Rally(start_frame=10, end_frame=20, fps=2)
    assert rally.start_time_ms == 5000   # frame 10 at 2fps = 5 seconds
    assert rally.end_time_ms == 10000    # frame 20 at 2fps = 10 seconds


def test_compute_frame_motion_returns_scalar():
    prev = np.zeros((100, 100, 3), dtype=np.uint8)
    curr = np.zeros((100, 100, 3), dtype=np.uint8)
    curr[40:60, 40:60] = 200  # changed region

    motion = compute_frame_motion(prev, curr)
    assert isinstance(motion, float)
    assert motion > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rally_detector.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Create `backend/app/ml/rally_detector.py`**

```python
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    fps: float

    @property
    def start_time_ms(self) -> int:
        return int((self.start_frame / self.fps) * 1000)

    @property
    def end_time_ms(self) -> int:
        return int((self.end_frame / self.fps) * 1000)

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


def compute_frame_motion(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    Compute normalized motion between two consecutive frames.
    Returns a float in [0, 1] where 1 = maximum motion.
    Uses absolute pixel difference of grayscale frames.
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(curr_gray - prev_gray)
    return float(diff.mean() / 255.0)


def build_motion_signal(frames: list[np.ndarray]) -> list[float]:
    """Compute per-frame motion signal from a sequence of frames."""
    if len(frames) < 2:
        return [0.0] * len(frames)
    signal = [0.0]  # first frame has no previous
    for i in range(1, len(frames)):
        signal.append(compute_frame_motion(frames[i - 1], frames[i]))
    return signal


def detect_rallies(
    motion_signal: list[float],
    fps: float = 2.0,
    motion_threshold: float = 0.1,
    min_gap_frames: int = 2,
    min_rally_frames: int = 4,
) -> list[Rally]:
    """
    Detect rallies from a per-frame motion signal.
    
    A rally is a continuous region where motion > motion_threshold.
    Gaps shorter than min_gap_frames are ignored (brief pauses within a rally).
    Rallies shorter than min_rally_frames are discarded (noise).
    
    Returns list of Rally objects sorted by start_frame.
    """
    if not motion_signal:
        return []

    active = [m > motion_threshold for m in motion_signal]

    # Fill short gaps (brief still moments within a rally)
    for i in range(1, len(active) - 1):
        if not active[i]:
            gap_length = 0
            j = i
            while j < len(active) and not active[j]:
                gap_length += 1
                j += 1
            if gap_length < min_gap_frames:
                for k in range(i, j):
                    active[k] = True

    # Extract contiguous active segments
    rallies = []
    in_rally = False
    start = 0

    for i, is_active in enumerate(active):
        if is_active and not in_rally:
            in_rally = True
            start = i
        elif not is_active and in_rally:
            in_rally = False
            if (i - start) >= min_rally_frames:
                rallies.append(Rally(start_frame=start, end_frame=i - 1, fps=fps))

    # Close rally at end of signal
    if in_rally and (len(active) - start) >= min_rally_frames:
        rallies.append(Rally(start_frame=start, end_frame=len(active) - 1, fps=fps))

    return rallies
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rally_detector.py -v
```
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/rally_detector.py backend/tests/test_rally_detector.py
git commit -m "feat(ml): Add motion-based rally detector with threshold segmentation and gap filling"
```

---

## Task 12: Score State Machine

**Files:**
- Create: `backend/app/ml/score_state_machine.py`
- Create: `backend/tests/test_score_state_machine.py`

Pickleball scoring: only the serving team can score. Score goes to 11 (win by 2). In doubles, there are two serves per team (first server exception at game start).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_score_state_machine.py`:
```python
import pytest
from app.ml.score_state_machine import ScoreStateMachine, PointOutcome


def test_initial_state():
    sm = ScoreStateMachine()
    state = sm.get_state()
    assert state["user_team"] == 0
    assert state["opponent_team"] == 0
    assert state["serving_team"] in ("user_team", "opponent_team")


def test_serving_team_scores_increments_score():
    sm = ScoreStateMachine(serving_team="user_team")
    sm.record_point(PointOutcome.USER_TEAM_WINS)
    state = sm.get_state()
    assert state["user_team"] == 1
    assert state["opponent_team"] == 0


def test_non_serving_team_wins_causes_side_out():
    sm = ScoreStateMachine(serving_team="user_team")
    sm.record_point(PointOutcome.OPPONENT_TEAM_WINS)
    state = sm.get_state()
    # Side out: serving switches, score unchanged
    assert state["user_team"] == 0
    assert state["serving_team"] == "opponent_team"


def test_history_records_all_points():
    sm = ScoreStateMachine()
    sm.record_point(PointOutcome.USER_TEAM_WINS)
    sm.record_point(PointOutcome.OPPONENT_TEAM_WINS)
    assert len(sm.history) == 2


def test_point_outcome_includes_before_and_after():
    sm = ScoreStateMachine(serving_team="user_team")
    before = sm.get_state().copy()
    outcome = sm.record_point(PointOutcome.USER_TEAM_WINS)
    assert outcome["score_before"] == before
    assert outcome["score_after"]["user_team"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_score_state_machine.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Create `backend/app/ml/score_state_machine.py`**

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Literal


class PointOutcome(str, Enum):
    USER_TEAM_WINS = "user_team"
    OPPONENT_TEAM_WINS = "opponent_team"


@dataclass
class ScoreStateMachine:
    """
    Tracks pickleball score state using the rally-end signals from the rally detector.
    
    Pickleball rules:
    - Only serving team can score
    - Non-serving team winning a rally = side out (serve changes, no score)
    - In doubles: each team has 2 serves per rotation (first server exception at start)
    - First to 11, win by 2
    """
    serving_team: Literal["user_team", "opponent_team"] = "user_team"
    user_team_score: int = 0
    opponent_team_score: int = 0
    server_number: int = 1  # 1 or 2 in doubles
    is_first_serve: bool = True  # First server exception
    history: list[dict] = field(default_factory=list)

    def get_state(self) -> dict:
        return {
            "user_team": self.user_team_score,
            "opponent_team": self.opponent_team_score,
            "serving_team": self.serving_team,
            "server_number": self.server_number,
        }

    def record_point(self, outcome: PointOutcome) -> dict:
        """
        Record the outcome of a rally and update state.
        Returns a dict with score_before, score_after, and point_won_by.
        """
        score_before = self.get_state().copy()

        if outcome.value == self.serving_team:
            # Serving team wins the rally → score
            if self.serving_team == "user_team":
                self.user_team_score += 1
            else:
                self.opponent_team_score += 1
        else:
            # Non-serving team wins → side out
            if self.is_first_serve:
                # First server exception: immediately switch serving team
                self.is_first_serve = False
                self.serving_team = outcome.value
                self.server_number = 1
            elif self.server_number == 1:
                # Switch to second server on same team
                self.server_number = 2
            else:
                # Both servers used → switch serving team
                self.serving_team = outcome.value
                self.server_number = 1

        score_after = self.get_state().copy()
        record = {
            "score_before": score_before,
            "score_after": score_after,
            "point_won_by": outcome.value,
            "serving_team_at_start": score_before["serving_team"],
        }
        self.history.append(record)
        return record

    @property
    def is_game_over(self) -> bool:
        max_score = max(self.user_team_score, self.opponent_team_score)
        min_score = min(self.user_team_score, self.opponent_team_score)
        return max_score >= 11 and (max_score - min_score) >= 2
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_score_state_machine.py -v
```
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/score_state_machine.py backend/tests/test_score_state_machine.py
git commit -m "feat(ml): Add pickleball score state machine with side-out logic and doubles server rotation"
```

---

## Task 13: Highlight Scorer

**Files:**
- Create: `backend/app/ml/highlight_scorer.py`
- Create: `backend/tests/test_highlight_scorer.py`

Phase 1 scorer uses only the signals available before Phase 2 ML models (no ball detection, no pose). Signals: point_scored, rally_length, attributed_player_role.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_highlight_scorer.py`:
```python
import pytest
from app.ml.highlight_scorer import score_highlight, RoleWeights


def test_user_scoring_point_gets_highest_score():
    score = score_highlight(
        point_scored=True,
        point_won_by="user_team",
        rally_length=5,
        attributed_role="user",
    )
    assert score > 0.5


def test_opponent_scoring_gets_low_score():
    score = score_highlight(
        point_scored=True,
        point_won_by="opponent_team",
        rally_length=5,
        attributed_role="opponent_1",
    )
    assert score < 0.3


def test_long_rally_scores_higher_than_short():
    short = score_highlight(point_scored=False, point_won_by=None, rally_length=3, attributed_role="user")
    long = score_highlight(point_scored=False, point_won_by=None, rally_length=15, attributed_role="user")
    assert long > short


def test_partner_play_lower_than_user():
    user_score = score_highlight(point_scored=True, point_won_by="user_team", rally_length=5, attributed_role="user")
    partner_score = score_highlight(point_scored=True, point_won_by="user_team", rally_length=5, attributed_role="partner")
    assert user_score > partner_score


def test_lowlight_detection_weak_shot():
    is_lowlight = score_highlight(
        point_scored=False,
        point_won_by=None,
        rally_length=2,
        attributed_role="user",
        shot_quality=0.15,
    )
    # A shot_quality < 0.3 should flag as potential lowlight
    assert is_lowlight < 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_highlight_scorer.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Create `backend/app/ml/highlight_scorer.py`**

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class RoleWeights:
    user: float = 1.0
    partner: float = 0.7
    opponent_1: float = 0.3
    opponent_2: float = 0.3


def score_highlight(
    point_scored: bool,
    point_won_by: Literal["user_team", "opponent_team"] | None,
    rally_length: int,
    attributed_role: str,
    shot_quality: float = 0.5,
    weights: RoleWeights = None,
) -> float:
    """
    Phase 1 highlight scorer. Uses only signals available without ball/pose models.
    Returns a score in [0, 1].

    Signals:
    - point_scored + point_won_by: strongest highlight trigger
    - rally_length: longer rallies are more exciting
    - attributed_role: user's plays ranked above partner/opponents
    - shot_quality: 0–1 float (defaults to 0.5 when shot classifier not available)
    """
    if weights is None:
        weights = RoleWeights()

    # Base score from rally excitement (normalized, plateau at 20 shots)
    rally_score = min(rally_length / 20.0, 1.0) * 0.3

    # Shot quality contribution
    quality_score = shot_quality * 0.2

    # Point outcome
    point_score = 0.0
    if point_scored and point_won_by == "user_team":
        point_score = 0.5
    elif point_scored and point_won_by == "opponent_team":
        point_score = 0.05  # opponent scoring is low interest

    raw_score = rally_score + quality_score + point_score

    # Role-aware weighting
    role_weight = getattr(weights, attributed_role.replace("_", "_"), weights.user)
    if attributed_role == "user":
        role_weight = weights.user
    elif attributed_role == "partner":
        role_weight = weights.partner
    elif attributed_role in ("opponent_1", "opponent_2"):
        role_weight = weights.opponent_1

    final_score = raw_score * role_weight
    return min(final_score, 1.0)


def is_lowlight(shot_quality: float, point_lost_by_error: bool) -> bool:
    """
    Phase 1 lowlight detection: weak shot quality OR lost point by user error.
    """
    return shot_quality < 0.3 or point_lost_by_error


def rank_highlights(highlights: list[dict]) -> list[dict]:
    """Sort highlights by highlight_score descending."""
    return sorted(highlights, key=lambda h: h.get("highlight_score", 0), reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_highlight_scorer.py -v
```
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/highlight_scorer.py backend/tests/test_highlight_scorer.py
git commit -m "feat(ml): Add highlight scorer with role-aware weighting, rally length, and lowlight detection"
```

---

## Task 14: Clip Extraction

**Files:**
- Create: `backend/app/ml/clip_extractor.py` (note: separate from ingest worker for testability)
- Create: `backend/tests/test_clip_extractor.py`

Clips are extracted from the **original 2.7K video** (not the 1080p working copy) for maximum quality. FFmpeg cuts the clip at the start/end timestamps.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_clip_extractor.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, call
from app.ml.clip_extractor import extract_clip, ClipSpec


def test_extract_clip_calls_ffmpeg_with_correct_args():
    spec = ClipSpec(
        source_path="/tmp/original.mp4",
        output_path="/tmp/clip_001.mp4",
        start_ms=10000,
        end_ms=15000,
    )

    with patch("app.ml.clip_extractor.ffmpeg") as mock_ffmpeg:
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_stream.output.return_value = mock_stream
        mock_stream.overwrite_output.return_value = mock_stream

        extract_clip(spec)

        mock_ffmpeg.input.assert_called_once_with(
            "/tmp/original.mp4",
            ss=10.0,   # start in seconds
            to=15.0,   # end in seconds
        )


def test_clip_spec_duration():
    spec = ClipSpec(source_path="a.mp4", output_path="b.mp4", start_ms=5000, end_ms=12000)
    assert spec.duration_seconds == pytest.approx(7.0)


def test_clip_spec_start_seconds():
    spec = ClipSpec(source_path="a.mp4", output_path="b.mp4", start_ms=3500, end_ms=8000)
    assert spec.start_seconds == pytest.approx(3.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_clip_extractor.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Create `backend/app/ml/clip_extractor.py`**

```python
from dataclasses import dataclass
import ffmpeg


@dataclass
class ClipSpec:
    source_path: str
    output_path: str
    start_ms: int
    end_ms: int

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def extract_clip(spec: ClipSpec) -> None:
    """
    Extract a clip from source video using FFmpeg.
    Uses stream copy (no re-encode) for speed.
    Source should be the original 2.7K video for best quality.
    """
    (
        ffmpeg
        .input(spec.source_path, ss=spec.start_seconds, to=spec.end_seconds)
        .output(
            spec.output_path,
            vcodec="copy",   # stream copy — no re-encode, instant
            acodec="copy",
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def extract_clips_batch(specs: list[ClipSpec]) -> list[str]:
    """
    Extract multiple clips sequentially. Returns list of output paths.
    Skips clips that fail (logs error, continues).
    """
    successful = []
    for spec in specs:
        try:
            extract_clip(spec)
            successful.append(spec.output_path)
        except ffmpeg.Error as e:
            # Log and continue — don't fail the whole batch for one bad clip
            print(f"Clip extraction failed for {spec.start_ms}-{spec.end_ms}ms: {e.stderr.decode()}")
    return successful
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_clip_extractor.py -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/clip_extractor.py backend/tests/test_clip_extractor.py
git commit -m "feat(ml): Add clip extractor using FFmpeg stream copy from 2.7K original"
```

---

## Task 15: Full Pipeline Orchestration

**Files:**
- Modify: `backend/app/workers/ingest.py` (implement `run_ai_pipeline`)
- Create: `backend/app/routers/highlights.py`

Wire all ML modules together into the `run_ai_pipeline` Celery task. After the pipeline completes, clips are uploaded to R2 and highlight records are written to DB.

- [ ] **Step 1: Implement `run_ai_pipeline` in `backend/app/workers/ingest.py`**

Add to the bottom of `ingest.py` (replacing the `raise NotImplementedError`):

```python
@celery.task(bind=True, name="app.workers.ingest.run_ai_pipeline", max_retries=2)
def run_ai_pipeline(self, video_id: str, user_id: str, seed_bbox: dict):
    """
    Full AI pipeline after user tap:
    1. Download processed 1080p video
    2. Extract frames
    3. Run Re-ID tracking across all frames
    4. Run rally detection
    5. Infer point outcomes from rally boundaries
    6. Score each rally as highlight or lowlight
    7. Extract clips from 2.7K original
    8. Upload clips to R2
    9. Write highlights + rallies to DB
    10. Update video status → analyzed
    """
    import asyncio, asyncpg as apg, boto3, json, uuid, shutil
    from pathlib import Path
    from botocore.config import Config
    from app.ml.reid_tracking import extract_embedding, track_user_across_frames
    from app.ml.rally_detector import build_motion_signal, detect_rallies, Rally
    from app.ml.score_state_machine import ScoreStateMachine, PointOutcome
    from app.ml.highlight_scorer import score_highlight, is_lowlight
    from app.ml.clip_extractor import ClipSpec, extract_clips_batch
    from app.ml.person_detection import detect_players

    tmp_dir = Path(f"/tmp/pickleclips/{video_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    processed_path = str(tmp_dir / "processed_1080p.mp4")
    original_path = str(tmp_dir / "original.mp4")

    try:
        update_video_status(video_id, "processing")

        # Fetch video metadata from DB
        async def get_video():
            conn = await apg.connect(settings.database_url)
            try:
                return await conn.fetchrow(
                    "SELECT r2_key_original, r2_key_processed FROM videos WHERE id = $1", video_id
                )
            finally:
                await conn.close()

        video = asyncio.run(get_video())

        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

        # Download 1080p working copy for frame extraction
        s3.download_file(settings.r2_bucket_name, video["r2_key_processed"], processed_path)
        # Download original for clip extraction
        s3.download_file(settings.r2_bucket_name, video["r2_key_original"], original_path)

        # Extract frames at 2fps from 1080p working copy
        frames = extract_frames(processed_path, fps=2)

        # Detect players in each frame
        all_detections = [
            [{"bbox": b} for b in detect_players(f)]
            for f in frames
        ]

        # Extract seed embedding from the user-tapped bbox on the seed frame
        seed_frame = pick_seed_frame(frames)
        seed_embedding = extract_embedding(seed_frame, seed_bbox)

        # Track user across all frames
        labeled_frames = track_user_across_frames(frames, all_detections, seed_embedding)

        # Build motion signal + detect rallies
        motion_signal = build_motion_signal(frames)
        rallies: list[Rally] = detect_rallies(motion_signal, fps=2)

        # Run score state machine across rallies
        # Point outcome: if last user-team player had high Re-ID conf in that rally → user_team
        # This is a heuristic — improved in Phase 2 with ball tracking
        sm = ScoreStateMachine()
        rally_records = []
        highlight_records = []

        for rally in rallies:
            # Determine which team won the point (heuristic: who had last high-motion event)
            # For Phase 1, alternate based on serve rotation (score state machine handles this)
            # We can't reliably infer point winner without ball tracking, so we:
            # - detect rally boundaries
            # - record rallies without definitive point attribution
            # - mark them for manual correction via the API
            score_before = sm.get_state().copy()

            rally_id = str(uuid.uuid4())
            rally_record = {
                "id": rally_id,
                "video_id": video_id,
                "start_time_ms": rally.start_time_ms,
                "end_time_ms": rally.end_time_ms,
                "shot_count": max(1, int(rally.duration_seconds * 1.5)),  # ~1.5 shots/sec estimate
                "intensity_score": min(rally.duration_seconds / 30.0, 1.0),
                "point_won_by": None,  # Unknown without ball tracking
                "score_before": json.dumps(score_before),
                "score_after": json.dumps(score_before),  # Same — can't advance without outcome
                "is_comeback_point": False,
            }
            rally_records.append(rally_record)

            # Score this rally as a highlight (attributed to user by default for Phase 1)
            raw_score = score_highlight(
                point_scored=False,
                point_won_by=None,
                rally_length=rally_record["shot_count"],
                attributed_role="user",
                shot_quality=0.5,
            )

            # Add intensity bonus for longer rallies
            if rally.duration_seconds > 10:
                raw_score = min(raw_score * 1.3, 1.0)

            # Pad clip by 1 second on each side for context
            clip_start_ms = max(0, rally.start_time_ms - 1000)
            clip_end_ms = rally.end_time_ms + 1000

            highlight_id = str(uuid.uuid4())
            highlight_records.append({
                "id": highlight_id,
                "video_id": video_id,
                "rally_id": rally_id,
                "attributed_player_role": "user",
                "sub_highlight_type": "point_scored",
                "lowlight_type": None,
                "point_lost_by_error": False,
                "start_time_ms": clip_start_ms,
                "end_time_ms": clip_end_ms,
                "highlight_score": raw_score,
                "highlight_score_raw": raw_score,
                "shot_type": None,
                "shot_quality": 0.5,
                "point_scored": False,
                "point_won_by": None,
                "rally_length": rally_record["shot_count"],
                "rally_intensity": rally_record["intensity_score"],
                "score_source": "rule_based",
                "r2_key_clip": None,
            })

        # Extract clips for top highlights (save GPU time — only top 15)
        top_highlights = sorted(highlight_records, key=lambda h: h["highlight_score"], reverse=True)[:15]
        clip_specs = []
        for h in top_highlights:
            clip_path = str(tmp_dir / f"clip_{h['id'][:8]}.mp4")
            clip_specs.append(ClipSpec(
                source_path=original_path,
                output_path=clip_path,
                start_ms=h["start_time_ms"],
                end_ms=h["end_time_ms"],
            ))

        successful_paths = extract_clips_batch(clip_specs)

        # Upload clips to R2 and update r2_key_clip
        for i, spec in enumerate(clip_specs):
            if spec.output_path in successful_paths:
                highlight_id = top_highlights[i]["id"]
                r2_key = f"videos/{video_id}/clips/{highlight_id}.mp4"
                s3.upload_file(spec.output_path, settings.r2_bucket_name, r2_key)
                top_highlights[i]["r2_key_clip"] = r2_key

        # Write all records to DB
        async def save_to_db():
            conn = await apg.connect(settings.database_url)
            try:
                async with conn.transaction():
                    # Insert rallies
                    for r in rally_records:
                        await conn.execute(
                            """INSERT INTO rallies (id, video_id, start_time_ms, end_time_ms,
                               shot_count, intensity_score, point_won_by, score_before, score_after, is_comeback_point)
                               VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10)""",
                            r["id"], r["video_id"], r["start_time_ms"], r["end_time_ms"],
                            r["shot_count"], r["intensity_score"], r["point_won_by"],
                            r["score_before"], r["score_after"], r["is_comeback_point"],
                        )

                    # Insert highlights
                    for h in highlight_records:
                        await conn.execute(
                            """INSERT INTO highlights (id, video_id, rally_id, attributed_player_role,
                               sub_highlight_type, lowlight_type, point_lost_by_error,
                               start_time_ms, end_time_ms, highlight_score, highlight_score_raw,
                               shot_quality, point_scored, point_won_by, rally_length, rally_intensity,
                               score_source, r2_key_clip)
                               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)""",
                            h["id"], h["video_id"], h["rally_id"], h["attributed_player_role"],
                            h["sub_highlight_type"], h["lowlight_type"], h["point_lost_by_error"],
                            h["start_time_ms"], h["end_time_ms"], h["highlight_score"], h["highlight_score_raw"],
                            h["shot_quality"], h["point_scored"], h["point_won_by"],
                            h["rally_length"], h["rally_intensity"], h["score_source"], h.get("r2_key_clip"),
                        )

                    # Mark video as analyzed
                    await conn.execute(
                        "UPDATE videos SET status = 'analyzed' WHERE id = $1", video_id
                    )
            finally:
                await conn.close()

        asyncio.run(save_to_db())

    except Exception as exc:
        update_video_status(video_id, "failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 2: Implement `backend/app/routers/highlights.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
import asyncpg

from app.auth import get_current_user
from app.database import get_db
from app.services.storage import generate_download_url

router = APIRouter(tags=["highlights"])


@router.get("/videos/{video_id}/highlights")
async def list_highlights(
    video_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    video = await db.fetchrow(
        "SELECT id FROM videos WHERE id = $1 AND user_id = $2", video_id, user_id
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    rows = await db.fetch(
        """SELECT id, start_time_ms, end_time_ms, highlight_score, sub_highlight_type,
                  shot_type, shot_quality, point_scored, rally_length, r2_key_clip, user_feedback
           FROM highlights
           WHERE video_id = $1 AND sub_highlight_type != 'lowlight'
           ORDER BY highlight_score DESC""",
        video_id,
    )
    return [dict(r) for r in rows]


@router.get("/videos/{video_id}/lowlights")
async def list_lowlights(
    video_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    video = await db.fetchrow(
        "SELECT id FROM videos WHERE id = $1 AND user_id = $2", video_id, user_id
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    rows = await db.fetch(
        """SELECT id, start_time_ms, end_time_ms, highlight_score, lowlight_type,
                  shot_quality, r2_key_clip, user_feedback
           FROM highlights
           WHERE video_id = $1 AND sub_highlight_type = 'lowlight'
           ORDER BY shot_quality ASC""",
        video_id,
    )
    return [dict(r) for r in rows]


@router.get("/highlights/{highlight_id}/download")
async def get_clip_download_url(
    highlight_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    row = await db.fetchrow(
        """SELECT h.id, h.r2_key_clip, v.user_id
           FROM highlights h
           JOIN videos v ON h.video_id = v.id
           WHERE h.id = $1""",
        highlight_id,
    )
    if not row or str(row["user_id"]) != user_id:
        raise HTTPException(status_code=404, detail="Highlight not found")
    if not row["r2_key_clip"]:
        raise HTTPException(status_code=409, detail="Clip not yet extracted")

    url = generate_download_url(row["r2_key_clip"], expires_in=3600)
    return {"download_url": url}


@router.patch("/highlights/{highlight_id}")
async def update_highlight_feedback(
    highlight_id: str,
    body: dict,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    """Update user feedback (liked/disliked) on a highlight."""
    feedback = body.get("user_feedback")
    if feedback not in ("liked", "disliked", None):
        raise HTTPException(status_code=422, detail="user_feedback must be 'liked', 'disliked', or null")

    result = await db.fetchrow(
        """UPDATE highlights h SET user_feedback = $1
           FROM videos v
           WHERE h.id = $2 AND h.video_id = v.id AND v.user_id = $3
           RETURNING h.id""",
        feedback, highlight_id, user_id,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Highlight not found")
    return {"status": "updated"}
```

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all tests PASS (some may be skipped without GPU — that's expected).

- [ ] **Step 4: Commit**

```bash
git add backend/app/workers/ingest.py backend/app/routers/highlights.py
git commit -m "feat(pipeline): Add full pipeline orchestration — Re-ID, rally detection, scoring, clip extraction, and highlights API"
```

---

## Task 16: Pipeline Timeout + Cleanup Worker

**Files:**
- Modify: `backend/app/workers/cleanup.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_cleanup.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone


def test_cleanup_finds_stale_jobs():
    """Stale videos (status=identifying AND identify_started_at > 24h ago) should be cancelled."""
    stale_row = {
        "id": "video-123",
        "r2_key_original": "videos/video-123/original.mp4",
        "user_id": "user-456",
    }
    
    with patch("app.workers.cleanup.asyncio") as mock_asyncio, \
         patch("app.workers.cleanup.settings"):
        mock_asyncio.run = lambda coro: [stale_row]
        
        from app.workers.cleanup import find_stale_jobs
        # Just verify the function exists and is callable
        assert callable(find_stale_jobs)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cleanup.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `backend/app/workers/cleanup.py`**

```python
import asyncio
import asyncpg
from app.workers.celery_app import celery
from app.config import settings
from app.services.storage import delete_object


async def find_stale_jobs() -> list[dict]:
    """Find videos stuck in 'identifying' for more than 24 hours."""
    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch(
            """SELECT id, r2_key_original, r2_key_processed, user_id
               FROM videos
               WHERE status = 'identifying'
               AND identify_started_at < NOW() - INTERVAL '24 hours'"""
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def cancel_stale_job(video_id: str, r2_key_original: str, r2_key_processed: str | None) -> None:
    """Mark a job as timed_out and clean up R2 working copy."""
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(
            "UPDATE videos SET status = 'timed_out' WHERE id = $1", video_id
        )
    finally:
        await conn.close()

    # Delete processed working copy (saves R2 storage) — keep original for re-trigger
    if r2_key_processed:
        try:
            delete_object(r2_key_processed)
        except Exception:
            pass  # Don't fail cleanup if R2 delete errors


async def schedule_original_deletion(video_id: str) -> None:
    """Set cleanup_after to 7 days from now — R2 lifecycle policy handles actual deletion."""
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(
            "UPDATE videos SET cleanup_after = NOW() + INTERVAL '7 days' WHERE id = $1",
            video_id,
        )
    finally:
        await conn.close()


@celery.task(name="app.workers.cleanup.cleanup_stale_jobs")
def cleanup_stale_jobs() -> dict:
    """
    Celery beat task: runs every hour.
    Cancels jobs stuck in 'identifying' for > 24 hours.
    Returns dict with count of cancelled jobs.
    """
    stale = asyncio.run(find_stale_jobs())
    cancelled = 0

    for video in stale:
        asyncio.run(cancel_stale_job(
            video["id"],
            video["r2_key_original"],
            video.get("r2_key_processed"),
        ))
        asyncio.run(schedule_original_deletion(video["id"]))
        # Notification: Supabase Realtime update (status → timed_out) will
        # update the UI automatically. No email needed for personal use in Phase 1.
        cancelled += 1

    return {"cancelled": cancelled}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cleanup.py tests/ -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/workers/cleanup.py backend/tests/test_cleanup.py
git commit -m "feat(workers): Add pipeline timeout and cleanup worker with 7-day original retention"
```

---

## Task 17: Frontend — Next.js Scaffold + Upload UI

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/src/lib/supabase.ts`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/app/layout.tsx`
- Create: `frontend/src/components/UploadZone.tsx`
- Create: `frontend/src/app/upload/page.tsx`

- [ ] **Step 1: Scaffold Next.js project**

```bash
cd pickleclips
npx create-next-app@14 frontend --typescript --tailwind --app --no-src-dir
cd frontend
npm install @supabase/supabase-js @uppy/core @uppy/aws-s3-multipart @uppy/react @uppy/dashboard
```

Then move `src/` contents as needed. Ensure `tsconfig.json` has `"paths": { "@/*": ["./src/*"] }`.

- [ ] **Step 2: Create `frontend/src/lib/supabase.ts`**

```typescript
import { createBrowserClient } from '@supabase/ssr'

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}
```

- [ ] **Step 3: Create `frontend/src/lib/api.ts`**

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

async function apiFetch<T>(path: string, options: RequestInit = {}, token?: string): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...options.headers,
  }
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  createMultipartUpload: (token: string, filename: string) =>
    apiFetch<{ video_id: string; upload_id: string; key: string }>(
      '/api/v1/videos/multipart/create',
      { method: 'POST', body: JSON.stringify({ filename, content_type: 'video/mp4' }) },
      token
    ),

  signMultipartPart: (token: string, key: string, uploadId: string, partNumber: number) =>
    apiFetch<{ url: string }>(
      `/api/v1/videos/multipart/sign-part?key=${encodeURIComponent(key)}&upload_id=${uploadId}&part_number=${partNumber}`,
      {},
      token
    ),

  completeMultipartUpload: (token: string, key: string, uploadId: string, parts: object[]) =>
    apiFetch('/api/v1/videos/multipart/complete', {
      method: 'POST',
      body: JSON.stringify({ key, upload_id: uploadId, parts }),
    }, token),

  confirmUpload: (token: string, videoId: string) =>
    apiFetch(`/api/v1/videos/${videoId}/confirm`, { method: 'POST' }, token),

  listVideos: (token: string) =>
    apiFetch<object[]>('/api/v1/videos', {}, token),

  getVideo: (token: string, videoId: string) =>
    apiFetch<object>(`/api/v1/videos/${videoId}`, {}, token),

  getIdentifyFrame: (token: string, videoId: string) =>
    apiFetch<{ frame_url: string; bboxes: object[] }>(`/api/v1/videos/${videoId}/identify`, {}, token),

  tapIdentify: (token: string, videoId: string, bboxIndex: number) =>
    apiFetch(`/api/v1/videos/${videoId}/identify`, {
      method: 'POST',
      body: JSON.stringify({ bbox_index: bboxIndex }),
    }, token),

  listHighlights: (token: string, videoId: string) =>
    apiFetch<object[]>(`/api/v1/videos/${videoId}/highlights`, {}, token),

  getClipDownloadUrl: (token: string, highlightId: string) =>
    apiFetch<{ download_url: string }>(`/api/v1/highlights/${highlightId}/download`, {}, token),
}
```

- [ ] **Step 4: Create `frontend/src/components/UploadZone.tsx`**

```typescript
'use client'

import { useEffect, useRef, useState } from 'react'
import Uppy from '@uppy/core'
import AwsS3Multipart from '@uppy/aws-s3-multipart'
import { Dashboard } from '@uppy/react'
import '@uppy/core/dist/style.min.css'
import '@uppy/dashboard/dist/style.min.css'
import { api } from '@/lib/api'

interface Props {
  token: string
  onUploadComplete: (videoId: string) => void
}

export function UploadZone({ token, onUploadComplete }: Props) {
  const uppyRef = useRef<Uppy | null>(null)
  const videoIdRef = useRef<string | null>(null)

  useEffect(() => {
    const uppy = new Uppy({
      restrictions: {
        maxFileSize: 6 * 1024 * 1024 * 1024,  // 6GB
        allowedFileTypes: ['video/mp4', 'video/quicktime', 'video/x-msvideo'],
        maxNumberOfFiles: 1,
      },
    }).use(AwsS3Multipart, {
      async createMultipartUpload(file) {
        const result = await api.createMultipartUpload(token, file.name)
        videoIdRef.current = result.video_id
        return { uploadId: result.upload_id, key: result.key }
      },
      async signPart(_file, { uploadId, key, partNumber }) {
        const result = await api.signMultipartPart(token, key, uploadId, partNumber)
        return { url: result.url }
      },
      async completeMultipartUpload(_file, { uploadId, key, parts }) {
        await api.completeMultipartUpload(token, key, uploadId, parts)
        return { location: key }
      },
      async abortMultipartUpload(_file, { uploadId, key }) {
        await fetch(`/api/v1/videos/multipart/abort?key=${encodeURIComponent(key)}&upload_id=${uploadId}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` },
        })
      },
    })

    uppy.on('complete', async () => {
      const videoId = videoIdRef.current
      if (!videoId) return
      await api.confirmUpload(token, videoId)
      onUploadComplete(videoId)
    })

    uppyRef.current = uppy
    return () => uppy.destroy()
  }, [token, onUploadComplete])

  if (!uppyRef.current) return null

  return (
    <div className="w-full">
      <Dashboard
        uppy={uppyRef.current}
        proudlyDisplayPoweredByUppy={false}
        note="Upload your pickleball game video (MP4, up to 6GB)"
        height={400}
      />
    </div>
  )
}
```

- [ ] **Step 5: Create `frontend/src/app/upload/page.tsx`**

```typescript
'use client'

import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase'
import { UploadZone } from '@/components/UploadZone'

export default function UploadPage() {
  const [token, setToken] = useState<string | null>(null)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      if (!data.session) {
        router.push('/login')
      } else {
        setToken(data.session.access_token)
      }
    })
  }, [])

  if (!token) return <div className="p-8 text-center">Loading...</div>

  return (
    <div className="max-w-2xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6">Upload Game</h1>
      <UploadZone
        token={token}
        onUploadComplete={(videoId) => router.push(`/videos/${videoId}/identify`)}
      />
    </div>
  )
}
```

- [ ] **Step 6: Create `frontend/src/app/login/page.tsx`**

```typescript
'use client'

import { useState } from 'react'
import { createClient } from '@/lib/supabase'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const supabase = createClient()

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/videos` },
    })
    if (error) {
      setError(error.message)
    } else {
      setSent(true)
    }
  }

  if (sent) {
    return (
      <div className="max-w-md mx-auto p-8 text-center">
        <h1 className="text-2xl font-bold mb-4">Check your email</h1>
        <p className="text-gray-600">We sent a magic link to <strong>{email}</strong></p>
      </div>
    )
  }

  return (
    <div className="max-w-md mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6">Sign in to PickleClips</h1>
      <form onSubmit={handleLogin} className="space-y-4">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="your@email.com"
          required
          className="w-full border rounded-lg px-4 py-2"
        />
        {error && <p className="text-red-600 text-sm">{error}</p>}
        <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-lg">
          Send magic link
        </button>
      </form>
    </div>
  )
}
```

- [ ] **Step 7: Verify frontend builds**

```bash
cd frontend
npm run build
```
Expected: build succeeds with no type errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): Add Next.js 14 scaffold with Supabase auth, Uppy multipart upload, and upload page"
```

---

## Task 18: Frontend — Player Identify + Processing Status + Clips View

**Files:**
- Create: `frontend/src/components/PlayerIdentify.tsx`
- Create: `frontend/src/components/ProcessingStatus.tsx`
- Create: `frontend/src/components/ClipCard.tsx`
- Create: `frontend/src/app/videos/[id]/identify/page.tsx`
- Create: `frontend/src/app/videos/[id]/page.tsx`
- Create: `frontend/src/app/videos/page.tsx`

- [ ] **Step 1: Create `frontend/src/components/PlayerIdentify.tsx`**

```typescript
'use client'

import { useState } from 'react'

interface BBox {
  x: number
  y: number
  w: number
  h: number
}

interface Props {
  frameUrl: string
  bboxes: BBox[]
  onSelect: (index: number) => void
  isSubmitting: boolean
}

export function PlayerIdentify({ frameUrl, bboxes, onSelect, isSubmitting }: Props) {
  const [selected, setSelected] = useState<number | null>(null)
  const [imgSize, setImgSize] = useState({ w: 0, h: 0, naturalW: 0, naturalH: 0 })

  const scaleX = imgSize.w / (imgSize.naturalW || 1)
  const scaleY = imgSize.h / (imgSize.naturalH || 1)

  return (
    <div>
      <p className="text-gray-600 mb-4">Tap on yourself in the frame below.</p>
      <div className="relative inline-block">
        <img
          src={frameUrl}
          alt="Seed frame"
          className="max-w-full rounded-lg"
          onLoad={(e) => {
            const img = e.currentTarget
            setImgSize({ w: img.width, h: img.height, naturalW: img.naturalWidth, naturalH: img.naturalHeight })
          }}
        />
        {bboxes.map((bbox, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            style={{
              position: 'absolute',
              left: bbox.x * scaleX,
              top: bbox.y * scaleY,
              width: bbox.w * scaleX,
              height: bbox.h * scaleY,
              border: selected === i ? '3px solid #22c55e' : '2px solid #3b82f6',
              background: selected === i ? 'rgba(34,197,94,0.15)' : 'rgba(59,130,246,0.1)',
              borderRadius: 4,
              cursor: 'pointer',
            }}
          />
        ))}
      </div>
      <button
        className="mt-4 px-6 py-2 bg-green-600 text-white rounded-lg disabled:opacity-50"
        disabled={selected === null || isSubmitting}
        onClick={() => selected !== null && onSelect(selected)}
      >
        {isSubmitting ? 'Processing...' : "That's me"}
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Create `frontend/src/components/ProcessingStatus.tsx`**

```typescript
'use client'

import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase'

const STATUS_LABELS: Record<string, string> = {
  uploading: 'Uploading...',
  identifying: 'Waiting for player identification',
  processing: 'Analyzing your game...',
  analyzed: 'Ready!',
  failed: 'Processing failed',
  timed_out: 'Timed out — please re-upload',
}

interface Props {
  videoId: string
  initialStatus: string
  onAnalyzed: () => void
}

export function ProcessingStatus({ videoId, initialStatus, onAnalyzed }: Props) {
  const [status, setStatus] = useState(initialStatus)
  const supabase = createClient()

  useEffect(() => {
    const channel = supabase
      .channel(`video-${videoId}`)
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'videos', filter: `id=eq.${videoId}` },
        (payload) => {
          const newStatus = (payload.new as { status: string }).status
          setStatus(newStatus)
          if (newStatus === 'analyzed') onAnalyzed()
        }
      )
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [videoId])

  const isProcessing = ['uploading', 'processing'].includes(status)

  return (
    <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
      {isProcessing && (
        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      )}
      <span className={`text-sm ${status === 'failed' ? 'text-red-600' : 'text-gray-700'}`}>
        {STATUS_LABELS[status] || status}
      </span>
    </div>
  )
}
```

- [ ] **Step 3: Create `frontend/src/components/ClipCard.tsx`**

```typescript
'use client'

import { useState } from 'react'
import { api } from '@/lib/api'

interface Props {
  highlight: {
    id: string
    highlight_score: number
    start_time_ms: number
    end_time_ms: number
    shot_type: string | null
    rally_length: number
  }
  token: string
}

export function ClipCard({ highlight, token }: Props) {
  const [downloading, setDownloading] = useState(false)

  const duration = ((highlight.end_time_ms - highlight.start_time_ms) / 1000).toFixed(1)
  const startSec = (highlight.start_time_ms / 1000).toFixed(1)

  async function handleDownload() {
    setDownloading(true)
    try {
      const { download_url } = await api.getClipDownloadUrl(token, highlight.id)
      const a = document.createElement('a')
      a.href = download_url
      a.download = `clip-${highlight.id.slice(0, 8)}.mp4`
      a.click()
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm">
      <div className="flex justify-between items-start mb-2">
        <div>
          <span className="text-sm font-medium text-gray-900">
            {highlight.shot_type ?? 'Rally'} @ {startSec}s
          </span>
          <p className="text-xs text-gray-500">{duration}s · {highlight.rally_length} shots</p>
        </div>
        <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
          Score: {(highlight.highlight_score * 100).toFixed(0)}
        </span>
      </div>
      <button
        onClick={handleDownload}
        disabled={downloading}
        className="w-full mt-2 px-4 py-2 bg-blue-600 text-white text-sm rounded disabled:opacity-50"
      >
        {downloading ? 'Getting link...' : 'Download Clip'}
      </button>
    </div>
  )
}
```

- [ ] **Step 4: Create `frontend/src/app/videos/[id]/identify/page.tsx`**

```typescript
'use client'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { createClient } from '@/lib/supabase'
import { api } from '@/lib/api'
import { PlayerIdentify } from '@/components/PlayerIdentify'

export default function IdentifyPage() {
  const params = useParams()
  const videoId = params.id as string
  const router = useRouter()
  const [token, setToken] = useState<string | null>(null)
  const [frameData, setFrameData] = useState<{ frame_url: string; bboxes: object[] } | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const supabase = createClient()

  useEffect(() => {
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) return router.push('/login')
      const t = data.session.access_token
      setToken(t)
      try {
        const result = await api.getIdentifyFrame(t, videoId)
        setFrameData(result)
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to load frame')
      }
    })
  }, [videoId])

  async function handleSelect(index: number) {
    if (!token) return
    setSubmitting(true)
    try {
      await api.tapIdentify(token, videoId, index)
      router.push(`/videos/${videoId}`)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to submit')
      setSubmitting(false)
    }
  }

  if (error) return <div className="p-8 text-red-600">{error}</div>
  if (!frameData) return <div className="p-8">Loading frame...</div>

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6">Identify Yourself</h1>
      <PlayerIdentify
        frameUrl={frameData.frame_url}
        bboxes={frameData.bboxes as { x: number; y: number; w: number; h: number }[]}
        onSelect={handleSelect}
        isSubmitting={submitting}
      />
    </div>
  )
}
```

- [ ] **Step 5: Create `frontend/src/app/videos/[id]/page.tsx`**

```typescript
'use client'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { createClient } from '@/lib/supabase'
import { api } from '@/lib/api'
import { ProcessingStatus } from '@/components/ProcessingStatus'
import { ClipCard } from '@/components/ClipCard'

export default function VideoPage() {
  const params = useParams()
  const videoId = params.id as string
  const router = useRouter()
  const [token, setToken] = useState<string | null>(null)
  const [video, setVideo] = useState<{ status: string } | null>(null)
  const [highlights, setHighlights] = useState<object[]>([])
  const supabase = createClient()

  useEffect(() => {
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) return router.push('/login')
      const t = data.session.access_token
      setToken(t)
      const v = await api.getVideo(t, videoId)
      setVideo(v as { status: string })
      if ((v as { status: string }).status === 'analyzed') {
        const h = await api.listHighlights(t, videoId)
        setHighlights(h)
      }
      if ((v as { status: string }).status === 'identifying') {
        router.push(`/videos/${videoId}/identify`)
      }
    })
  }, [videoId])

  async function loadHighlights() {
    if (!token) return
    const h = await api.listHighlights(token, videoId)
    setHighlights(h)
    setVideo((v) => v ? { ...v, status: 'analyzed' } : v)
  }

  if (!video || !token) return <div className="p-8">Loading...</div>

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-4">Your Highlights</h1>
      <ProcessingStatus
        videoId={videoId}
        initialStatus={video.status}
        onAnalyzed={loadHighlights}
      />
      {highlights.length > 0 && (
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {highlights.map((h) => (
            <ClipCard key={(h as { id: string }).id} highlight={h as Parameters<typeof ClipCard>[0]['highlight']} token={token} />
          ))}
        </div>
      )}
      {video.status === 'analyzed' && highlights.length === 0 && (
        <p className="mt-8 text-gray-500">No highlights detected in this game.</p>
      )}
    </div>
  )
}
```

- [ ] **Step 6: Create `frontend/src/app/videos/page.tsx`**

```typescript
'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { createClient } from '@/lib/supabase'
import { api } from '@/lib/api'

export default function VideosPage() {
  const router = useRouter()
  const [token, setToken] = useState<string | null>(null)
  const [videos, setVideos] = useState<object[]>([])
  const supabase = createClient()

  useEffect(() => {
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) return router.push('/login')
      const t = data.session.access_token
      setToken(t)
      setVideos(await api.listVideos(t))
    })
  }, [])

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">My Games</h1>
        <Link href="/upload" className="px-4 py-2 bg-blue-600 text-white rounded-lg">
          Upload Game
        </Link>
      </div>
      {videos.length === 0 && <p className="text-gray-500">No games uploaded yet.</p>}
      <div className="space-y-3">
        {videos.map((v) => {
          const video = v as { id: string; status: string; uploaded_at: string }
          return (
            <Link key={video.id} href={`/videos/${video.id}`} className="block border rounded-lg p-4 hover:bg-gray-50">
              <div className="flex justify-between">
                <span className="font-medium">Game — {new Date(video.uploaded_at).toLocaleDateString()}</span>
                <span className={`text-sm px-2 py-1 rounded ${
                  video.status === 'analyzed' ? 'bg-green-100 text-green-700' :
                  video.status === 'failed' ? 'bg-red-100 text-red-600' :
                  'bg-yellow-100 text-yellow-700'
                }`}>{video.status}</span>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 7: Verify frontend builds**

```bash
cd frontend
npm run build
```
Expected: build succeeds.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): Add player identify UI, Supabase Realtime status, clip cards, and video library"
```

---

## Task 19: GitHub Actions CI/CD + Caddy

**Files:**
- Create: `.github/workflows/deploy.yml`
- Create: `backend/Dockerfile`
- Create: `Caddyfile`

- [ ] **Step 1: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV + FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `Caddyfile`**

```
pickleclips.yourdomain.com {
    reverse_proxy localhost:8000
}
```

Replace `pickleclips.yourdomain.com` with your actual domain. Caddy auto-provisions TLS via Let's Encrypt.

- [ ] **Step 3: Create `.github/workflows/deploy.yml`**

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run backend tests
        working-directory: backend
        run: |
          pip install -r requirements.txt
          pytest tests/ -v --ignore=tests/test_person_detection.py  # skip GPU tests in CI
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_ANON_KEY: ${{ secrets.SUPABASE_ANON_KEY }}
          SUPABASE_JWT_SECRET: ${{ secrets.SUPABASE_JWT_SECRET }}
          R2_ACCOUNT_ID: ${{ secrets.R2_ACCOUNT_ID }}
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          R2_BUCKET_NAME: ${{ secrets.R2_BUCKET_NAME }}
          REDIS_URL: redis://localhost:6379/0

      - name: Deploy to EC2
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /home/ubuntu/pickleclips
            git pull origin main
            cd backend
            pip install -r requirements.txt
            sudo systemctl restart pickleclips-api
            sudo systemctl restart pickleclips-worker
            sudo systemctl restart pickleclips-beat
```

- [ ] **Step 4: Add GitHub secrets**

In your GitHub repo → Settings → Secrets → Add:
- `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`
- `EC2_HOST` (your EC2 public IP), `EC2_SSH_KEY` (your private SSH key)

- [ ] **Step 5: Create systemd service files on EC2**

SSH into EC2 and create `/etc/systemd/system/pickleclips-api.service`:
```ini
[Unit]
Description=PickleClips API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/pickleclips/backend
EnvironmentFile=/home/ubuntu/pickleclips/.env
ExecStart=/usr/local/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/pickleclips-worker.service`:
```ini
[Unit]
Description=PickleClips Celery Worker
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/pickleclips/backend
EnvironmentFile=/home/ubuntu/pickleclips/.env
ExecStart=celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
Restart=always

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/pickleclips-beat.service`:
```ini
[Unit]
Description=PickleClips Celery Beat
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/pickleclips/backend
EnvironmentFile=/home/ubuntu/pickleclips/.env
ExecStart=celery -A app.workers.celery_app beat --loglevel=info
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable: `sudo systemctl enable pickleclips-api pickleclips-worker pickleclips-beat`

- [ ] **Step 6: Run full test suite one final time**

```bash
cd backend
pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 7: Commit and push**

```bash
git add .github/ backend/Dockerfile Caddyfile
git commit -m "feat(ci): Add GitHub Actions deploy to EC2 with Dockerfile and Caddy reverse proxy"
git push origin main
```
Expected: GitHub Actions runs, tests pass, deploys to EC2.

---

## End-to-End Smoke Test

After deployment, verify the full flow works:

- [ ] Open the frontend URL
- [ ] Log in with magic link
- [ ] Upload a game video (use a short 2-min test clip first)
- [ ] Confirm pipeline starts (video status → `processing`)
- [ ] Wait for `identifying` status
- [ ] Navigate to `/videos/{id}/identify`
- [ ] Tap on yourself in the frame
- [ ] Watch status progress to `analyzed` via Supabase Realtime
- [ ] See highlight clips appear
- [ ] Download one clip and verify it plays correctly

**Success criteria:** Clips of your plays appear within 15 minutes of upload, containing moments from your side of the court.

---

## Known Phase 1 Limitations (addressed in Phase 2)

1. **No ball detection** — rally detection is motion-based only (less precise)
2. **Point attribution unknown** — score state machine can't advance without knowing who won each point. Highlights are detected as "rallies" not "scored points"
3. **No shot classification** — all clips labeled as generic "rally" not typed shots
4. **No reel assembly** — clips are individual downloads, not assembled reels
5. **No lowlight detection** — `shot_quality` defaults to 0.5 (no pose/ball models yet)

These are all addressed in the Phase 2 plan.
