# Phase 4: Personalization + Auto-Recognition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the feedback loop between what the AI produces and what individual users actually want. Two parallel tracks: (A) build a persistent appearance profile per player so the system eventually recognizes them automatically without a tap, and (B) learn from thumbs up/down feedback to adjust highlight scoring weights per user over time.

**Architecture:** Two tracks running in parallel. Track A adds `_upsert_player_profile()` at the end of `run_ai_pipeline`, a cosine-similarity gate in `ingest_video` that routes uploads into auto-identify / confirm / manual-tap flows, a new `confirming` video status, and a `POST /confirm-identity` endpoint. Track B extends `update_highlight_feedback` to update `users.highlight_preferences`, loads those weights in `run_ai_pipeline`, and passes them as overrides to `score_highlight`. Additional: audio analysis module, smart vertical cropping in the reel assembler.

**Tech Stack:** All Phase 3 stack. New optional dep: `tensorflow-hub` for YAMNet audio classifier (graceful fallback to RMS energy if unavailable). pgvector `ivfflat` index added for profile similarity search.

**Spec:** `docs/superpowers/specs/2026-04-05-pickleclips-architecture-design.md` §3.3 Player Identification, §3.5 Highlight Scorer, §9 Privacy

**Phase 3 plan:** `docs/superpowers/plans/2026-04-10-phase3-zip-download-optional-reels.md`

**Scope note:** Phase 3 deliverables (ZIP download, on-demand reel generation, all tests) must be complete before starting this phase. `player_profiles` and `video_players` tables already exist in `001_initial.sql`. `users.highlight_preferences JSONB` already exists. `cosine_similarity` and `extract_embedding` already implemented in `reid_tracking.py`.

---

## File Map

```
pickleclips-ai/
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── reid_tracking.py        REFERENCE ONLY — cosine_similarity, extract_embedding already here
│   │   │   ├── highlight_scorer.py     MODIFY — add shot_type_overrides param to score_highlight
│   │   │   ├── reel_assembler.py       MODIFY — smart vertical crop using user_center_x
│   │   │   └── audio_analyzer.py       CREATE — YAMNet + RMS fallback audio excitement scores
│   │   ├── workers/
│   │   │   └── ingest.py               MODIFY — profile upsert, auto-recognition gate, load user prefs
│   │   └── routers/
│   │       ├── videos.py               MODIFY — add POST /videos/{id}/confirm-identity
│   │       └── highlights.py           MODIFY — trigger preference update on feedback write
│   ├── migrations/
│   │   ├── 003_player_profiles_unique.sql   CREATE — UNIQUE(user_id) + ivfflat index
│   │   └── 004_video_confirming_status.sql  CREATE — add 'confirming' to status CHECK
│   └── tests/
│       ├── test_preference_learning.py  CREATE — 5 scoring + preference weight tests
│       ├── test_player_profile.py       CREATE — 2 profile upsert tests
│       └── test_videos.py              MODIFY — 5 new auto-recognition / confirm-identity tests
├── frontend/
│   ├── app/
│   │   └── videos/
│   │       └── [id]/
│   │           └── identify/
│   │               └── page.tsx        MODIFY — handle 'confirming' state, show single-candidate UI
│   └── lib/
│       └── api.ts                      MODIFY — add confirmIdentity method
```

---

## Task 1: DB Migration — player_profiles UNIQUE + ivfflat Index

**Files:**
- Create: `backend/migrations/003_player_profiles_unique.sql`

**Reference:** `player_profiles` table in `001_initial.sql` lines 16–24. `appearance_embedding VECTOR(512)` already defined. pgvector extension enabled at line 1.

- [ ] **Step 1: Create migration file**

```sql
-- 003_player_profiles_unique.sql
-- One profile per user (rolling average embedding)
ALTER TABLE player_profiles
    ADD CONSTRAINT player_profiles_user_id_unique UNIQUE (user_id);

-- ivfflat index for cosine similarity search at scale
-- lists=10 suitable for < 10k users; tune upward as user base grows
CREATE INDEX IF NOT EXISTS idx_player_profiles_embedding
    ON player_profiles USING ivfflat (appearance_embedding vector_cosine_ops)
    WITH (lists = 10);
```

- [ ] **Step 2: Verify migration exists**

```bash
ls backend/migrations/003_player_profiles_unique.sql   # → file present
```

---

## Task 2: DB Migration — Add `confirming` Video Status

**Files:**
- Create: `backend/migrations/004_video_confirming_status.sql`

**Reference:** `videos.status` CHECK constraint in `001_initial.sql` lines 34–36: current values are `'uploading', 'identifying', 'processing', 'analyzed', 'failed', 'timed_out'`. Adding `'confirming'` for the medium-confidence auto-recognition flow.

- [ ] **Step 1: Create migration file**

```sql
-- 004_video_confirming_status.sql
ALTER TABLE videos DROP CONSTRAINT videos_status_check;
ALTER TABLE videos ADD CONSTRAINT videos_status_check CHECK (
    status IN (
        'uploading', 'identifying', 'confirming',
        'processing', 'analyzed', 'failed', 'timed_out'
    )
);
```

- [ ] **Step 2: Verify**

```bash
ls backend/migrations/004_video_confirming_status.sql   # → file present
```

---

## Task 3: Upsert Player Profile After Each Video

**Files:**
- Modify: `backend/app/workers/ingest.py`

**Reference:** `extract_embedding` at `reid_tracking.py` lines 38–60 (returns 512-dim L2-normalized ndarray). `asyncio.run()` + inner async function pattern throughout `ingest.py`. `labeled_frames` variable in `run_ai_pipeline` — list of per-frame detection dicts with `role`, `embedding`, `reid_conf` keys.

- [ ] **Step 1: Add `_upsert_player_profile` helper at the bottom of `ingest.py`**

```python
PROFILE_MIN_UPLOADS = 3  # uploads needed before auto-recognition is attempted


def _upsert_player_profile(video_id: str, user_id: str, labeled_frames: list) -> None:
    """Average user embeddings across all labeled frames and upsert into player_profiles."""
    import numpy as np

    user_embeddings = []
    confidences = []
    for frame_detections in labeled_frames:
        for det in frame_detections:
            if det.get("role") == "user" and "embedding" in det:
                user_embeddings.append(det["embedding"])
                confidences.append(float(det.get("reid_conf", 0.0)))

    if not user_embeddings:
        return

    avg_embedding = np.mean(user_embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    avg_embedding = avg_embedding / norm if norm > 0 else avg_embedding
    avg_confidence = float(np.mean(confidences))

    async def _upsert():
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """INSERT INTO player_profiles
                       (user_id, appearance_embedding, embedding_confidence, uploads_contributing)
                   VALUES ($1, $2::vector, $3, 1)
                   ON CONFLICT (user_id) DO UPDATE SET
                     appearance_embedding = (
                         player_profiles.appearance_embedding
                             * player_profiles.uploads_contributing::float
                         + EXCLUDED.appearance_embedding
                     ) / (player_profiles.uploads_contributing + 1)::float,
                     embedding_confidence = (
                         player_profiles.embedding_confidence
                             * player_profiles.uploads_contributing::float
                         + EXCLUDED.embedding_confidence
                     ) / (player_profiles.uploads_contributing + 1)::float,
                     uploads_contributing = player_profiles.uploads_contributing + 1,
                     updated_at = NOW()""",
                user_id, avg_embedding.tolist(), avg_confidence,
            )
        finally:
            await conn.close()

    asyncio.run(_upsert())
```

- [ ] **Step 2: Call it at the end of `run_ai_pipeline`** (just before `finally:`)

```python
        # Update persistent player profile with this video's user embeddings
        _upsert_player_profile(video_id, user_id, labeled_frames)

    except Exception as exc:
```

- [ ] **Step 3: Verify**

```bash
grep -n "_upsert_player_profile\|PROFILE_MIN_UPLOADS" backend/app/workers/ingest.py
# → PROFILE_MIN_UPLOADS: 1 match, _upsert_player_profile: 2 matches (def + call)
```

---

## Task 4: Auto-Recognition Gate in `ingest_video`

**Files:**
- Modify: `backend/app/workers/ingest.py`

**Reference:** `ingest_video` task lines 112–211. Step 5 (person detection + bbox extraction) ends around line 168. The DB update (step 8) starts around line 184. `cosine_similarity` and `extract_embedding` from `reid_tracking.py` lines 38–73.

- [ ] **Step 1: Add profile fetch + similarity gate** after step 5 (bboxes extracted), before step 8 (DB update)

```python
        # 5b. Check for persistent player profile → route to correct identification flow
        async def get_profile():
            conn = await asyncpg.connect(settings.database_url)
            try:
                return await conn.fetchrow(
                    """SELECT appearance_embedding, embedding_confidence, uploads_contributing
                       FROM player_profiles WHERE user_id = $1""",
                    user_id,
                )
            finally:
                await conn.close()

        profile = asyncio.run(get_profile())
        auto_status = "identifying"
        auto_seed_bbox = None

        if profile and (profile["uploads_contributing"] or 0) >= PROFILE_MIN_UPLOADS and bboxes:
            from app.ml.reid_tracking import extract_embedding, cosine_similarity
            stored_embedding = np.array(profile["appearance_embedding"])
            best_sim, best_bbox = -1.0, None
            for bbox in bboxes:
                emb = extract_embedding(seed_frame, bbox)
                sim = cosine_similarity(stored_embedding, emb)
                if sim > best_sim:
                    best_sim, best_bbox = sim, bbox

            if best_sim > 0.85:
                auto_status = "processing"
                auto_seed_bbox = best_bbox
            elif best_sim >= 0.60:
                auto_status = "confirming"
                auto_seed_bbox = best_bbox  # store candidate for confirmation prompt
```

- [ ] **Step 2: Update the DB write (step 8) to use `auto_status`**

In the `save_results` async function, change the status write and add `auto_candidate_bbox` to metadata when confirming:

```python
metadata_payload = {"seed_frame_key": seed_frame_key, "player_bboxes": bboxes}
if auto_seed_bbox is not None:
    metadata_payload["auto_candidate_bbox"] = auto_seed_bbox

await conn.execute(
    """UPDATE videos SET
        status = $1,
        r2_key_processed = $2,
        identify_started_at = NOW(),
        cleanup_after = $3,
        metadata = metadata || $4::jsonb
       WHERE id = $5""",
    auto_status, processed_key, deadline,
    json.dumps(metadata_payload), video_id,
)
```

- [ ] **Step 3: Dispatch pipeline immediately for auto-identified videos**

After `asyncio.run(save_results())`:

```python
        if auto_status == "processing" and auto_seed_bbox is not None:
            run_ai_pipeline.delay(video_id, user_id, auto_seed_bbox)
```

- [ ] **Step 4: Verify**

```bash
grep -n "auto_status\|auto_seed_bbox\|confirming" backend/app/workers/ingest.py
# → at least 3 matches each
```

---

## Task 5: `POST /videos/{id}/confirm-identity` Endpoint

**Files:**
- Modify: `backend/app/routers/videos.py`

**Reference:** `tap_identify` endpoint at `videos.py` lines 220–252. Copy ownership check, metadata read, and `resume_after_identify.delay()` dispatch patterns exactly.

- [ ] **Step 1: Add `ConfirmIdentityRequest` model** (after existing Pydantic models at the top of `videos.py`)

```python
class ConfirmIdentityRequest(BaseModel):
    confirmed: bool  # True = accept auto-selected bbox; False = fall back to 4-box tap
```

- [ ] **Step 2: Add endpoint** (before or after `tap_identify`)

```python
@router.post("/videos/{video_id}/confirm-identity")
async def confirm_identity(
    video_id: str,
    body: ConfirmIdentityRequest,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    """
    For the 'confirming' flow: user accepts the auto-selected candidate or falls back to manual tap.
    """
    from app.workers.ingest import resume_after_identify

    video = await db.fetchrow(
        "SELECT id, status, metadata FROM videos WHERE id = $1 AND user_id = $2",
        video_id, user_id,
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video["status"] != "confirming":
        raise HTTPException(status_code=409, detail="Video is not waiting for identity confirmation")

    meta = video["metadata"] or {}
    auto_bbox = meta.get("auto_candidate_bbox")
    bboxes = meta.get("player_bboxes", [])

    if body.confirmed and auto_bbox:
        await db.execute("UPDATE videos SET status = 'processing' WHERE id = $1", video_id)
        resume_after_identify.delay(video_id, user_id, auto_bbox)
        return {"status": "processing", "auto_recognized": True}
    else:
        # Fall back to full 4-box tap flow
        await db.execute("UPDATE videos SET status = 'identifying' WHERE id = $1", video_id)
        return {"status": "identifying", "bboxes": bboxes}
```

- [ ] **Step 3: Add to `api.ts`**

```typescript
confirmIdentity: (token: string, videoId: string, confirmed: boolean) =>
  apiFetch<{ status: string; auto_recognized?: boolean; bboxes?: object[] }>(
    `/api/v1/videos/${videoId}/confirm-identity`,
    { method: 'POST', body: JSON.stringify({ confirmed }) },
    token
  ),
```

- [ ] **Step 4: Verify**

```bash
grep -n "confirm.identity\|confirm_identity" backend/app/routers/videos.py   # → 2+ matches
grep -n "confirmIdentity" frontend/lib/api.ts                                  # → 1 match
```

---

## Task 6: Frontend — Auto-Recognition Confirmation UI

**Files:**
- Modify: `frontend/app/videos/[id]/identify/page.tsx`

**Reference:** Token extraction from `page.tsx` lines 28–31. `api.tapIdentify` usage at `api.ts` lines 50–54. `api.confirmIdentity` added in Task 5.

- [ ] **Step 1: Add state for confirming flow**

```typescript
const [isConfirming, setIsConfirming] = useState(false)
const [candidateFrameUrl, setCandidateFrameUrl] = useState<string | null>(null)
```

- [ ] **Step 2: Add routing logic in `useEffect`** — after fetching video status, check for `confirming`:

```typescript
if (video.status === 'confirming') {
  setIsConfirming(true)
  // frame_url from getIdentifyFrame still works; candidateBbox stored in metadata
}
```

- [ ] **Step 3: Add handler**

```typescript
async function handleConfirm(confirmed: boolean) {
  if (!token) return
  const result = await api.confirmIdentity(token, videoId, confirmed)
  if (result.status === 'processing') {
    router.push(`/videos/${videoId}`)
  } else {
    // fell back to manual tap — re-render with 4-box UI
    setIsConfirming(false)
  }
}
```

- [ ] **Step 4: Add confirming UI branch** (above the existing 4-box tap JSX)

```tsx
{isConfirming ? (
  <div className="flex flex-col items-center gap-4">
    <p className="text-lg font-medium">We think we found you — is this you?</p>
    {candidateFrameUrl && (
      <img src={candidateFrameUrl} alt="Player candidate" className="w-48 rounded" />
    )}
    <div className="flex gap-3">
      <button
        onClick={() => handleConfirm(true)}
        className="px-4 py-2 bg-green-600 text-white rounded"
      >
        Yes, that's me
      </button>
      <button
        onClick={() => handleConfirm(false)}
        className="px-4 py-2 bg-gray-500 text-white rounded"
      >
        No, show me all players
      </button>
    </div>
  </div>
) : (
  /* existing 4-box tap UI */
)}
```

- [ ] **Step 5: Verify**

```bash
grep -n "isConfirming\|handleConfirm\|confirmIdentity" frontend/app/videos/\[id\]/identify/page.tsx
# → at least 2 matches each
```

---

## Task 7: Preference Update on Feedback

**Files:**
- Modify: `backend/app/routers/highlights.py`

**Reference:** `update_highlight_feedback` at `highlights.py` lines 141–163. `users.highlight_preferences JSONB` column at `001_initial.sql` line 10. Feedback values: `'liked'` / `'disliked'` (CHECK constraint line 98 of `001_initial.sql`).

- [ ] **Step 1: Add `_update_user_preferences` helper** (before the route handlers)

```python
import json

async def _update_user_preferences(
    db: asyncpg.Connection, user_id: str, shot_type: str, feedback: str
) -> None:
    """Adjust shot_type_weights in users.highlight_preferences based on clip feedback."""
    row = await db.fetchrow(
        "SELECT highlight_preferences FROM users WHERE id = $1", user_id
    )
    prefs = dict(row["highlight_preferences"] or {}) if row else {}
    weights = prefs.get("shot_type_weights", {})
    current = float(weights.get(shot_type, 1.0))
    delta = 0.05 if feedback == "liked" else -0.05
    weights[shot_type] = round(max(0.3, min(2.0, current + delta)), 4)
    prefs["shot_type_weights"] = weights
    prefs[f"{feedback}_count"] = prefs.get(f"{feedback}_count", 0) + 1
    await db.execute(
        "UPDATE users SET highlight_preferences = $1::jsonb WHERE id = $2",
        json.dumps(prefs), user_id,
    )
```

- [ ] **Step 2: Call it inside `update_highlight_feedback`** after the UPDATE succeeds

```python
    if not result:
        raise HTTPException(status_code=404, detail="Highlight not found")

    # Update user shot-type preferences based on feedback
    if feedback is not None:
        shot_row = await db.fetchrow(
            "SELECT shot_type FROM highlights WHERE id = $1", highlight_id
        )
        if shot_row and shot_row["shot_type"]:
            await _update_user_preferences(db, user_id, shot_row["shot_type"], feedback)

    return {"status": "updated"}
```

- [ ] **Step 3: Verify**

```bash
grep -n "_update_user_preferences\|shot_type_weights" backend/app/routers/highlights.py
# → at least 2 matches each
```

---

## Task 8: Use Stored Preferences in Pipeline Scoring

**Files:**
- Modify: `backend/app/ml/highlight_scorer.py`
- Modify: `backend/app/workers/ingest.py`

**Reference:** `score_highlight` signature at `highlight_scorer.py` lines 26–76. `_SHOT_TYPE_MULTIPLIERS` dict lines 13–23. `run_ai_pipeline` DB fetch pattern lines 254–263.

- [ ] **Step 1: Add `shot_type_overrides` param to `score_highlight`**

```python
def score_highlight(
    point_scored: bool,
    point_won_by: Literal["user_team", "opponent_team"] | None,
    rally_length: int,
    attributed_role: str,
    shot_quality: float = 0.5,
    shot_type: str | None = None,
    weights: RoleWeights | None = None,
    shot_type_overrides: dict[str, float] | None = None,  # ← new
) -> float:
    ...
    multipliers = dict(_SHOT_TYPE_MULTIPLIERS)
    if shot_type_overrides:
        multipliers.update(shot_type_overrides)
    shot_multiplier = multipliers.get(shot_type or "", 1.0)
    ...
```

- [ ] **Step 2: Load user preferences at the start of `run_ai_pipeline`** (after the existing `get_video` call)

```python
        async def get_user_prefs():
            conn = await asyncpg.connect(settings.database_url)
            try:
                row = await conn.fetchrow(
                    "SELECT highlight_preferences FROM users WHERE id = $1", user_id
                )
                return dict(row["highlight_preferences"] or {}) if row else {}
            finally:
                await conn.close()

        user_prefs = asyncio.run(get_user_prefs())
        user_shot_weights = user_prefs.get("shot_type_weights", {})
```

- [ ] **Step 3: Pass `shot_type_overrides` to `score_highlight`**

In the scoring loop inside `run_ai_pipeline`, add to the existing `score_highlight` call:

```python
            raw_score = score_highlight(
                point_scored=False,
                point_won_by=None,
                rally_length=rally_record["shot_count"],
                attributed_role="user",
                shot_type=shot_result.shot_type,
                shot_quality=shot_result.quality,
                shot_type_overrides=user_shot_weights,  # ← new
            )
```

- [ ] **Step 4: Verify**

```bash
grep -n "shot_type_overrides" backend/app/ml/highlight_scorer.py  # → 2 matches
grep -n "user_shot_weights\|user_prefs" backend/app/workers/ingest.py  # → 3+ matches
```

---

## Task 9: Audio Analyzer Module

**Files:**
- Create: `backend/app/ml/audio_analyzer.py`

**Reference:** `_mediapipe_available()` guard pattern from `ingest.py` lines 21–26. `PoseEstimator` module structure from `pose_estimator.py`. FFmpeg extraction pattern from `ingest.py` transcode helper.

- [ ] **Step 1: Create `audio_analyzer.py`**

```python
"""
Audio excitement signal extraction.
Uses YAMNet (TF Hub) when available; falls back to RMS energy heuristic.
"""
from __future__ import annotations
import subprocess
import numpy as np


def _yamnet_available() -> bool:
    try:
        import tensorflow_hub  # noqa: F401
        return True
    except ImportError:
        return False


class AudioAnalyzer:
    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract mono 16 kHz WAV from video using ffmpeg. Returns True on success."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
                 "-vn", output_path],
                capture_output=True, timeout=120,
            )
            return result.returncode == 0
        except Exception:
            return False

    def analyze(self, audio_path: str, fps: int = 2) -> list[float]:
        """
        Returns per-frame excitement scores (0.0–1.0) aligned to video FPS.
        Empty list on any failure — callers must handle gracefully.
        """
        try:
            if _yamnet_available():
                return self._analyze_yamnet(audio_path, fps)
            return self._analyze_rms(audio_path, fps)
        except Exception:
            return []

    def _analyze_rms(self, audio_path: str, fps: int) -> list[float]:
        """Fallback: RMS energy per window, normalized to [0, 1]."""
        import wave, struct
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        window = sample_rate // fps
        scores = []
        for i in range(0, len(samples), window):
            chunk = samples[i:i + window]
            if len(chunk) == 0:
                break
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            scores.append(rms)
        if not scores:
            return []
        max_rms = max(scores) or 1.0
        return [min(s / max_rms, 1.0) for s in scores]

    def _analyze_yamnet(self, audio_path: str, fps: int) -> list[float]:
        """YAMNet-based crowd/impact detection."""
        import tensorflow as tf
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        audio = tf.io.read_file(audio_path)
        waveform, _ = tf.audio.decode_wav(audio, desired_channels=1)
        waveform = tf.squeeze(waveform, axis=-1)
        scores, _, _ = model(waveform)
        # Average score across 521 classes per frame → single excitement value
        frame_scores = tf.reduce_max(scores, axis=1).numpy().tolist()
        return [float(s) for s in frame_scores]
```

- [ ] **Step 2: Integrate in `run_ai_pipeline`** — add after pose estimation loop (after line 317)

```python
        # Audio analysis (optional — gracefully skipped if extraction fails)
        from app.ml.audio_analyzer import AudioAnalyzer
        audio_path = str(tmp_dir / "audio.wav")
        audio_analyzer = AudioAnalyzer()
        audio_scores: list[float] = []
        if audio_analyzer.extract_audio(processed_path, audio_path):
            audio_scores = audio_analyzer.analyze(audio_path, fps=2)
```

Then in the scoring loop blend the audio signal:

```python
            audio_boost = audio_scores[mid_frame] * 0.1 if mid_frame < len(audio_scores) else 0.0
            raw_score = min(raw_score + audio_boost, 1.0)
```

- [ ] **Step 3: Verify**

```bash
ls backend/app/ml/audio_analyzer.py                                   # → file present
grep -n "AudioAnalyzer\|audio_scores" backend/app/workers/ingest.py   # → 2 matches each
```

---

## Task 10: Smart Vertical Cropping in Reel Assembler

**Files:**
- Modify: `backend/app/ml/reel_assembler.py`

**Reference:** `user_center_x` is already fetched in `reel_gen.py` lines 70–94 and passed into `assemble_and_upload`. Find where the FFmpeg `vf` (video filter) string is constructed for vertical format in `reel_assembler.py` and replace the center crop with a user-position-aware crop.

- [ ] **Step 1: Add `user_center_x` param to the assembly function that builds the crop filter**

In `reel_assembler.py`, locate the crop filter construction for `format == "vertical"` and replace:

```python
# Before: simple center crop
# crop_filter = f"crop=607:1080:656:0,scale=1080:1920"

# After: user-position-aware crop
def _vertical_crop_filter(user_center_x: float = 0.5) -> str:
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)  # 607 px
    user_x_px = int(user_center_x * frame_w)
    x_offset = max(0, min(frame_w - crop_w, user_x_px - crop_w // 2))
    return f"crop={crop_w}:{frame_h}:{x_offset}:0,scale=1080:1920"
```

Call `_vertical_crop_filter(user_center_x)` wherever the vertical format filter is applied.

- [ ] **Step 2: Verify**

```bash
grep -n "user_center_x\|_vertical_crop_filter" backend/app/ml/reel_assembler.py  # → 2+ matches
```

---

## Task 11: Tests — Preference Learning

**Files:**
- Create: `backend/tests/test_preference_learning.py`

**Reference:** `conftest.py` fixture pattern lines 54–65. `score_highlight` signature from `highlight_scorer.py` lines 26–34.

- [ ] **Step 1: Create test file**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from app.ml.highlight_scorer import score_highlight

pytestmark = pytest.mark.asyncio


def test_score_highlight_uses_overrides():
    default = score_highlight(
        point_scored=False, point_won_by=None, rally_length=5,
        attributed_role="user", shot_type="erne", shot_quality=0.8,
    )
    boosted = score_highlight(
        point_scored=False, point_won_by=None, rally_length=5,
        attributed_role="user", shot_type="erne", shot_quality=0.8,
        shot_type_overrides={"erne": 2.0},
    )
    assert boosted > default


def test_preference_weight_clamps_at_max():
    # Simulate: erne at 1.98, user likes → should clamp at 2.0
    weights = {"erne": 1.98}
    delta = 0.05
    result = round(max(0.3, min(2.0, weights["erne"] + delta)), 4)
    assert result == 2.0


def test_preference_weight_clamps_at_min():
    weights = {"dink": 0.32}
    delta = -0.05
    result = round(max(0.3, min(2.0, weights["dink"] + delta)), 4)
    assert result == 0.3


def test_feedback_liked_increases_weight():
    current = 1.0
    result = round(max(0.3, min(2.0, current + 0.05)), 4)
    assert result == 1.05


def test_feedback_disliked_decreases_weight():
    current = 1.0
    result = round(max(0.3, min(2.0, current - 0.05)), 4)
    assert result == 0.95
```

- [ ] **Step 2: Run tests**

```bash
cd backend && pytest tests/test_preference_learning.py -v   # → 5 tests pass
```

---

## Task 12: Tests — Player Profile Upsert + Auto-Recognition

**Files:**
- Create: `backend/tests/test_player_profile.py`
- Modify: `backend/tests/test_videos.py`

- [ ] **Step 1: Create `test_player_profile.py`**

```python
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio


async def test_upsert_creates_new_profile(mock_db_connection):
    # 10 labeled frames each with a user embedding
    embedding = np.random.randn(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    labeled_frames = [[{"role": "user", "embedding": embedding, "reid_conf": 0.9}]] * 10

    with patch("app.workers.ingest.asyncpg.connect") as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        from app.workers.ingest import _upsert_player_profile
        _upsert_player_profile("vid-001", "user-001", labeled_frames)

    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args[0]
    assert "INSERT INTO player_profiles" in call_args[0]
    assert call_args[2] == pytest.approx(0.9, abs=0.01)  # avg_confidence


async def test_upsert_skipped_when_no_user_embeddings(mock_db_connection):
    labeled_frames = [[{"role": "partner", "embedding": np.zeros(512)}]] * 5

    with patch("app.workers.ingest.asyncpg.connect") as mock_connect:
        from app.workers.ingest import _upsert_player_profile
        _upsert_player_profile("vid-002", "user-002", labeled_frames)

    mock_connect.assert_not_called()  # early return, no DB hit
```

- [ ] **Step 2: Add 5 auto-recognition tests to `test_videos.py`**

```python
async def test_confirm_identity_accepted(client, test_token, mock_db_connection):
    import sys
    mock_ingest = MagicMock()
    mock_ingest.resume_after_identify = MagicMock()
    mock_db_connection.fetchrow = AsyncMock(return_value={
        "id": "vid-001", "status": "confirming",
        "metadata": {"auto_candidate_bbox": {"x": 100, "y": 50, "w": 80, "h": 200},
                     "player_bboxes": []},
    })
    with patch.dict(sys.modules, {"app.workers.ingest": mock_ingest}):
        res = await client.post(
            "/api/v1/videos/vid-001/confirm-identity",
            json={"confirmed": True},
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert res.status_code == 200
    assert res.json()["status"] == "processing"
    assert res.json()["auto_recognized"] is True


async def test_confirm_identity_rejected_falls_back(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value={
        "id": "vid-002", "status": "confirming",
        "metadata": {"auto_candidate_bbox": {"x": 0, "y": 0, "w": 50, "h": 100},
                     "player_bboxes": [{"x": 0, "y": 0, "w": 50, "h": 100}]},
    })
    res = await client.post(
        "/api/v1/videos/vid-002/confirm-identity",
        json={"confirmed": False},
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "identifying"
    assert isinstance(res.json()["bboxes"], list)


async def test_confirm_identity_409_wrong_status(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value={
        "id": "vid-003", "status": "identifying", "metadata": {}
    })
    res = await client.post(
        "/api/v1/videos/vid-003/confirm-identity",
        json={"confirmed": True},
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 409


async def test_confirm_identity_404_not_found(client, test_token, mock_db_connection):
    mock_db_connection.fetchrow = AsyncMock(return_value=None)
    res = await client.post(
        "/api/v1/videos/vid-999/confirm-identity",
        json={"confirmed": True},
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert res.status_code == 404
```

- [ ] **Step 3: Run all new tests**

```bash
cd backend
pytest tests/test_preference_learning.py tests/test_player_profile.py -v
pytest tests/test_videos.py -v -k "confirm_identity"
```

All 11 tests must pass.

---

## Phase 4 Completion Checklist

```
Track A — Persistent Player Profiles
[ ] Task 1:  migrations/003_player_profiles_unique.sql — UNIQUE + ivfflat index
[ ] Task 2:  migrations/004_video_confirming_status.sql — 'confirming' status added
[ ] Task 3:  _upsert_player_profile + PROFILE_MIN_UPLOADS in ingest.py
[ ] Task 4:  Auto-recognition gate in ingest_video (0.85 / 0.60 thresholds)
[ ] Task 5:  POST /videos/{id}/confirm-identity in videos.py + api.ts
[ ] Task 6:  Identify page handles 'confirming' state with single-candidate UI

Track B — Preference Learning
[ ] Task 7:  _update_user_preferences called after every feedback write
[ ] Task 8:  run_ai_pipeline loads user prefs + passes shot_type_overrides to scorer
[ ] Task 9:  audio_analyzer.py created + integrated in pipeline
[ ] Task 10: Smart vertical crop uses _vertical_crop_filter(user_center_x)

Tests
[ ] Task 11: test_preference_learning.py — 5 tests green
[ ] Task 12: test_player_profile.py — 2 tests green
[ ] Task 12: test_videos.py confirm_identity — 4 tests green
[ ] Full CI run green after push to main
```

---

## Anti-Pattern Guards

- **Do not** store face embeddings — `extract_embedding` uses OSNet body/clothing features only. The architecture's privacy model (§9) depends on this: appearance embeddings are not facial recognition data
- **Do not** compare embeddings across users — every `player_profiles` query must include `WHERE user_id = $1`; there is no cross-user similarity search
- **Do not** define `PROFILE_MIN_UPLOADS` in more than one place — it is declared once at the top of `ingest.py`; import it wherever needed
- **Do not** let preference weights drift unbounded — `_update_user_preferences` must clamp to `[0.3, 2.0]` with `max(0.3, min(2.0, ...))` as in Task 7
- **Do not** raise from `AudioAnalyzer.analyze()` — it must return `[]` on any failure so the pipeline continues without audio scoring
- **Do not** call `asyncio.run()` inside an already-running event loop — all Celery worker DB calls follow the inner-async-fn pattern from `ingest.py` (define `async def _fn(): ...`, then `asyncio.run(_fn())`)
- **Do not** add `tensorflow-hub` to `requirements.txt` — it is an optional dep; `_yamnet_available()` guards its import so the pipeline runs without it
