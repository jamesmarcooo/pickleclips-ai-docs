# Phase 3: ZIP Download + Optional Reel Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give users explicit control over their output after analysis completes. Two independent paths: download all extracted clips as a categorized ZIP archive, or trigger reel generation on demand. Neither path is automatic — both are user-initiated.

**Architecture:** The skeleton endpoints already exist in the codebase (`GET /videos/{id}/clips/download-zip` in `highlights.py`, `POST /videos/{id}/generate-reels` in `videos.py`). This phase hardens them (memory-safe streaming, idempotency), adds tests, and surfaces both actions as buttons in the video detail page. The automatic `trigger_auto_generated_reels` call was removed from `run_ai_pipeline` in a prior commit — reel generation is now fully on-demand.

**Tech Stack:** All Phase 2 stack. No new dependencies. Uses stdlib `zipfile` + `io.BytesIO` for ZIP streaming; `StreamingResponse` from FastAPI.

**Spec:** `docs/superpowers/specs/2026-04-05-pickleclips-architecture-design.md` §7 Phase 3

**Phase 2 plan:** `docs/superpowers/plans/2026-04-07-phase2-reels-shot-intelligence.md`

**Scope note:** Phase 2 deliverables (Re-ID tracking, ball detection, pose estimation, shot classification, reel assembly, reel API) are complete. The two Phase 3 endpoint skeletons are already in place. This plan completes, tests, and exposes them.

---

## File Map

```
pickleclips-ai/
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   │   ├── highlights.py       MODIFY — harden ZIP generator (chunked yield, allowZip64)
│   │   │   └── videos.py           MODIFY — harden generate-reels idempotency
│   │   └── workers/
│   │       └── reel_gen.py         MODIFY — skip existing reel types in trigger_auto_generated_reels
│   └── tests/
│       ├── test_highlights.py      CREATE — 4 tests for ZIP download endpoint
│       └── test_videos.py          MODIFY — 3 new generate-reels tests
├── frontend/
│   ├── app/
│   │   └── videos/
│   │       └── [id]/
│   │           └── page.tsx        MODIFY — add Download ZIP + Generate Reels buttons
│   └── lib/
│       └── api.ts                  MODIFY — downloadClipsZip + generateReels already added; verify
```

---

## Task 1: Harden ZIP Streaming (Memory Safety)

**Files:**
- Modify: `backend/app/routers/highlights.py`

**Reference:** Current skeleton at `highlights.py` lines 90–138. Problem: entire archive is buffered in `BytesIO` before yielding. Fix: yield in 64 KB chunks, add `allowZip64=True`.

- [ ] **Step 1: Replace `generate_zip()` inner function** (lines 118–132 of `highlights.py`)

Replace the current body with:

```python
def generate_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for row in rows:
            shot_type = row["shot_type"] or "unknown"
            filename = f"{shot_type}/{str(row['id'])}.mp4"
            try:
                obj = client.get_object(Bucket=settings.r2_bucket_name, Key=row["r2_key_clip"])
                zf.writestr(filename, obj["Body"].read())
            except Exception:
                pass  # skip clips that fail to download; do not abort the whole archive
    buf.seek(0)
    while chunk := buf.read(65536):
        yield chunk
```

- [ ] **Step 2: Verify**

```bash
grep -n "allowZip64" backend/app/routers/highlights.py   # → 1 match
grep -n "65536" backend/app/routers/highlights.py        # → 1 match
```

---

## Task 2: Harden `generate-reels` Idempotency

**Files:**
- Modify: `backend/app/workers/reel_gen.py`

**Reference:** `trigger_auto_generated_reels` at `reel_gen.py` lines 134–163. Currently inserts unconditionally — calling it twice creates duplicate reel rows.

- [ ] **Step 1: Add pre-check inside `_create_reels()` in `reel_gen.py`**

Replace the inner INSERT loop (inside `_create_reels`) with:

```python
for output_type in _AUTO_GENERATED_TYPES:
    existing = await conn.fetchrow(
        "SELECT id FROM reels WHERE video_id = $1 AND output_type = $2",
        video_id, output_type,
    )
    if existing:
        continue
    row = await conn.fetchrow(
        """INSERT INTO reels (user_id, video_id, output_type, format, auto_generated)
           VALUES ($1, $2, $3, 'horizontal', TRUE)
           RETURNING id""",
        user_id, video_id, output_type,
    )
    reel_ids[output_type] = str(row["id"])
```

- [ ] **Step 2: Verify**

```bash
grep -n "existing" backend/app/workers/reel_gen.py   # → at least 1 match
```

Calling `POST /videos/{id}/generate-reels` twice must produce exactly 4 reel rows, not 8.

---

## Task 3: Tests — ZIP Download Endpoint

**Files:**
- Create: `backend/tests/test_highlights.py`

**Reference:** Fixture pattern from `conftest.py` lines 54–65. Multi-call `side_effect` from `test_reels.py` lines 23–26. R2 client mock from `test_videos.py` lines 43–49.

- [ ] **Step 1: Create `test_highlights.py`**

```python
import io
import zipfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

TEST_VIDEO_ID = "00000000-0000-0000-0000-000000000099"
TEST_HIGHLIGHT_ID = "aaaaaaaa-0000-0000-0000-000000000001"

pytestmark = pytest.mark.asyncio


async def test_download_zip_returns_200_with_clips(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value={"id": TEST_VIDEO_ID})
    mock_db_connection.fetch = AsyncMock(return_value=[
        {"id": TEST_HIGHLIGHT_ID, "shot_type": "erne", "r2_key_clip": "videos/x/clips/a.mp4"},
    ])
    mock_r2 = MagicMock()
    mock_r2.get_object.return_value = {"Body": io.BytesIO(b"fakevideo")}
    with patch("app.routers.highlights.get_r2_client", return_value=mock_r2):
        res = await client.get(
            f"/api/v1/videos/{TEST_VIDEO_ID}/clips/download-zip",
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    assert "clips_" in res.headers["content-disposition"]
    # Verify ZIP is parseable and contains the expected file
    buf = io.BytesIO(res.content)
    with zipfile.ZipFile(buf) as zf:
        assert f"erne/{TEST_HIGHLIGHT_ID}.mp4" in zf.namelist()


async def test_download_zip_404_video_not_found(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value=None)
    res = await client.get(
        f"/api/v1/videos/{TEST_VIDEO_ID}/clips/download-zip",
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 404


async def test_download_zip_409_no_clips_yet(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value={"id": TEST_VIDEO_ID})
    mock_db_connection.fetch = AsyncMock(return_value=[])
    res = await client.get(
        f"/api/v1/videos/{TEST_VIDEO_ID}/clips/download-zip",
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 409
    assert "No clips" in res.json()["detail"]


async def test_download_zip_skips_failed_r2_fetch(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value={"id": TEST_VIDEO_ID})
    mock_db_connection.fetch = AsyncMock(return_value=[
        {"id": "aaaa0001-0000-0000-0000-000000000001", "shot_type": "drive", "r2_key_clip": "k1"},
        {"id": "aaaa0002-0000-0000-0000-000000000002", "shot_type": "dink",  "r2_key_clip": "k2"},
    ])
    mock_r2 = MagicMock()
    mock_r2.get_object.side_effect = [Exception("R2 error"), {"Body": io.BytesIO(b"ok")}]
    with patch("app.routers.highlights.get_r2_client", return_value=mock_r2):
        res = await client.get(
            f"/api/v1/videos/{TEST_VIDEO_ID}/clips/download-zip",
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert res.status_code == 200
    buf = io.BytesIO(res.content)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
    assert len(names) == 1   # only second clip present; first was skipped
    assert "dink/" in names[0]
```

- [ ] **Step 2: Run tests**

```bash
cd backend && pytest tests/test_highlights.py -v
```

All 4 tests must pass.

---

## Task 4: Tests — Generate Reels Endpoint

**Files:**
- Modify: `backend/tests/test_videos.py`

**Reference:** `sys.modules` patch pattern from `test_videos.py` lines 43–49. `side_effect` multi-call pattern from `test_reels.py` lines 23–26.

- [ ] **Step 1: Add 3 tests to `test_videos.py`**

```python
async def test_generate_reels_returns_202_when_analyzed(client, test_token, mock_db_connection):
    import sys
    mock_reel_gen = MagicMock()
    mock_reel_gen.trigger_auto_generated_reels = MagicMock()
    with patch.dict(sys.modules, {
        "app.workers.reel_gen": mock_reel_gen,
    }):
        mock_db_connection.fetchrow = AsyncMock(
            return_value={"id": "vid-001", "status": "analyzed"}
        )
        res = await client.post(
            "/api/v1/videos/vid-001/generate-reels",
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert res.status_code == 202
    assert res.json()["status"] == "queued"
    mock_reel_gen.trigger_auto_generated_reels.assert_called_once()


async def test_generate_reels_409_when_not_analyzed(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(
        return_value={"id": "vid-002", "status": "processing"}
    )
    res = await client.post(
        "/api/v1/videos/vid-002/generate-reels",
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 409
    assert "not analyzed" in res.json()["detail"]


async def test_generate_reels_404_video_not_found(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value=None)
    res = await client.post(
        "/api/v1/videos/vid-999/generate-reels",
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 404
```

- [ ] **Step 2: Run tests**

```bash
cd backend && pytest tests/test_videos.py -v -k "generate_reels"
```

All 3 tests must pass.

---

## Task 5: Frontend — Download ZIP Button

**Files:**
- Modify: `frontend/app/videos/[id]/page.tsx`

**Reference:** Token extraction at `page.tsx` lines 28–31. Single-clip download pattern in `ClipCard.tsx` lines 25–41 (create `<a>`, set href, `.click()`, revoke). `api.downloadClipsZip` already in `api.ts` lines 71–80.

- [ ] **Step 1: Add state**

After existing `useState` declarations in `page.tsx`:

```typescript
const [downloadingZip, setDownloadingZip] = useState(false)
```

- [ ] **Step 2: Add handler**

```typescript
async function handleDownloadZip() {
  if (!token) return
  setDownloadingZip(true)
  try {
    const blob = await api.downloadClipsZip(token, videoId)
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `clips_${videoId.slice(0, 8)}.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  } finally {
    setDownloadingZip(false)
  }
}
```

- [ ] **Step 3: Add button JSX** (after the highlights list section)

```tsx
<button
  onClick={handleDownloadZip}
  disabled={downloadingZip || video?.status !== 'analyzed'}
  className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-50"
>
  {downloadingZip ? 'Preparing ZIP…' : 'Download ZIP'}
</button>
```

- [ ] **Step 4: Verify**

```bash
grep -n "handleDownloadZip\|downloadingZip" frontend/app/videos/\[id\]/page.tsx
# → at least 3 matches each
```

---

## Task 6: Frontend — Generate Reels Button

**Files:**
- Modify: `frontend/app/videos/[id]/page.tsx`

**Reference:** Same state + handler pattern as Task 5. `api.generateReels` already in `api.ts`.

- [ ] **Step 1: Add state**

```typescript
const [generatingReels, setGeneratingReels] = useState(false)
const [reelsQueued, setReelsQueued] = useState(false)
```

- [ ] **Step 2: Add handler**

```typescript
async function handleGenerateReels() {
  if (!token) return
  setGeneratingReels(true)
  try {
    await api.generateReels(token, videoId)
    setReelsQueued(true)
  } finally {
    setGeneratingReels(false)
  }
}
```

- [ ] **Step 3: Add button JSX** (adjacent to Download ZIP button from Task 5)

```tsx
<button
  onClick={handleGenerateReels}
  disabled={generatingReels || reelsQueued || video?.status !== 'analyzed'}
  className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
>
  {reelsQueued ? 'Reels Queued ✓' : generatingReels ? 'Queuing…' : 'Generate Reels'}
</button>
```

- [ ] **Step 4: Verify**

```bash
grep -n "handleGenerateReels\|reelsQueued" frontend/app/videos/\[id\]/page.tsx
# → at least 2 matches each
```

---

## Task 7: CI Verification

**Files:**
- No changes — verification only

- [ ] **Step 1: Confirm numpy pre-install step is present**

```bash
grep -n "Pre-install numpy" .github/workflows/deploy.yml   # → 1 match
```

- [ ] **Step 2: Run full test suite**

```bash
cd backend
pip install "numpy==1.26.4"
pip install -r requirements.txt
pytest tests/ -v --ignore=tests/test_person_detection.py
```

- [ ] **Step 3: Confirm all 7 new Phase 3 tests pass**

```
tests/test_highlights.py::test_download_zip_returns_200_with_clips       PASSED
tests/test_highlights.py::test_download_zip_404_video_not_found          PASSED
tests/test_highlights.py::test_download_zip_409_no_clips_yet             PASSED
tests/test_highlights.py::test_download_zip_skips_failed_r2_fetch        PASSED
tests/test_videos.py::test_generate_reels_returns_202_when_analyzed      PASSED
tests/test_videos.py::test_generate_reels_409_when_not_analyzed          PASSED
tests/test_videos.py::test_generate_reels_404_video_not_found            PASSED
```

- [ ] **Step 4: Push to `main` — confirm GitHub Actions workflow goes green**

---

## Anti-Pattern Guards

- **Do not** buffer the entire ZIP archive before returning — use the chunked `while chunk := buf.read(65536)` generator from Task 1
- **Do not** skip `allowZip64=True` — without it, archives over 2 GB (high clip count) will corrupt silently
- **Do not** call `trigger_auto_generated_reels` without the existence check from Task 2 — double-calling creates duplicate reel rows
- **Do not** use `apiFetch` for the ZIP download in the frontend — it calls `.json()`, which will fail on binary data; use raw `fetch` + `.blob()` as in `api.ts` lines 71–80
- **Do not** skip `window.URL.revokeObjectURL()` after triggering the download — retained blob URLs leak memory across navigations
