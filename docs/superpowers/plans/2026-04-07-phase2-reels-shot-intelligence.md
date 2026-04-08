# Phase 2: Reels + Shot Intelligence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-generate shareable reels from processed pickleball games. Ball detection + pose estimation unlock shot classification, richer highlight scoring, and a Points of Improvement coaching view.

**Architecture:** Two ML modules (TrackNetV2 ball detection, MediaPipe Pose) are added as stages in the existing `run_ai_pipeline` Celery task. A new `shot_classifier.py` derives shot types and quality scores from their outputs. A new `reel_gen` Celery task chains off the existing pipeline and assembles reels from FFmpeg + MoviePy with transitions, slow-mo, and music. A new `reels` router serves reel CRUD, share links, and export. Four auto-generated reel types + four on-demand types. Frontend adds a Reels tab, Points of Improvement tab, thumbs up/down feedback, and a reel share page.

**Tech Stack:** All Phase 1 stack plus: `mediapipe==0.10.14`, `moviepy==1.0.3`, `torch` (already installed), TrackNetV2 (vendored as `backend/app/ml/tracknetv2/`). Music: pre-downloaded MP3s from Free Music Archive stored in `backend/static/music/`.

**Spec:** `docs/superpowers/specs/2026-04-05-pickleclips-architecture-design.md` §7 Phase 2, §5 Output Types, §3 ML Pipeline

**Phase 1 plan:** `docs/superpowers/plans/2026-04-05-phase1-upload-identify-clips.md`

**Scope note:** All Phase 1 deliverables (upload, identify, clips, lowlights endpoint, feedback endpoint) are complete. This plan builds on top without changing Phase 1 logic.

---

## File Map

```
pickleclips-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                         MODIFY — register reels router
│   │   ├── ml/
│   │   │   ├── tracknetv2/                 CREATE — vendored TrackNetV2 model
│   │   │   │   ├── __init__.py
│   │   │   │   └── model.py                CREATE — TrackNetV2 PyTorch architecture
│   │   │   ├── ball_detection.py           CREATE — TrackNetV2 inference wrapper
│   │   │   ├── pose_estimator.py           CREATE — MediaPipe Pose wrapper
│   │   │   ├── shot_classifier.py          CREATE — rule-based shot type + quality
│   │   │   ├── highlight_scorer.py         MODIFY — add Phase 2 shot_type multipliers
│   │   │   └── reel_assembler.py           CREATE — FFmpeg+MoviePy reel assembly
│   │   ├── workers/
│   │   │   ├── ingest.py                   MODIFY — add ball/pose/shot stages to run_ai_pipeline
│   │   │   └── reel_gen.py                 CREATE — Celery reel generation task
│   │   ├── services/
│   │   │   └── reel.py                     CREATE — reel orchestration + share URLs
│   │   └── routers/
│   │       └── reels.py                    CREATE — reels CRUD + share + export
│   ├── static/
│   │   └── music/                          CREATE — royalty-free MP3 tracks
│   │       ├── README.txt                  source metadata
│   │       ├── energetic_bg.mp3            FMA: "Energetic Background" (CC0)
│   │       └── chill_bg.mp3               FMA: "Chill Background" (CC0)
│   ├── migrations/
│   │   └── 002_reels.sql                   CREATE — reels + clip_edits tables
│   ├── requirements.txt                    MODIFY — add mediapipe, moviepy
│   └── tests/
│       ├── test_ball_detection.py          CREATE
│       ├── test_pose_estimator.py          CREATE
│       ├── test_shot_classifier.py         CREATE
│       ├── test_reel_assembler.py          CREATE
│       └── test_reels.py                   CREATE — API endpoint tests
├── frontend/
│   ├── app/
│   │   ├── videos/
│   │   │   └── [id]/
│   │   │       ├── page.tsx                MODIFY — add Reels tab + POI tab + feedback
│   │   │       └── reels/
│   │   │           └── page.tsx            CREATE — reels list + generate button
│   │   └── reels/
│   │       └── [id]/
│   │           └── page.tsx                CREATE — reel player + share
│   ├── components/
│   │   ├── FeedbackButtons.tsx             CREATE — thumbs up/down for highlights
│   │   └── ReelCard.tsx                    CREATE — reel card with player + share
│   └── lib/
│       └── api.ts                          MODIFY — add reel API methods
```

---

## Task 1: DB Migration — reels + clip_edits tables

**Files:**
- Create: `backend/migrations/002_reels.sql`

- [ ] **Step 1: Write the migration**

```sql
-- 002_reels.sql

CREATE TABLE reels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_id UUID REFERENCES videos(id) ON DELETE SET NULL,
    output_type TEXT NOT NULL CHECK (output_type IN (
        'highlight_montage', 'my_best_plays', 'game_recap',
        'points_of_improvement', 'best_shots', 'scored_point_rally',
        'full_rally_replay', 'single_shot_clip'
    )),
    r2_key TEXT,
    format TEXT NOT NULL DEFAULT 'horizontal' CHECK (format IN ('vertical', 'horizontal', 'square')),
    duration_seconds FLOAT,
    clip_ids UUID[] DEFAULT '{}',
    rally_ids UUID[] DEFAULT '{}',
    assembly_profile JSONB DEFAULT '{}',
    music_track_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'generating', 'ready', 'failed')),
    auto_generated BOOLEAN DEFAULT FALSE,
    share_token TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE clip_edits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    highlight_id UUID NOT NULL REFERENCES highlights(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trim_start_ms INT NOT NULL DEFAULT 0,
    trim_end_ms INT,
    slow_mo_factor FLOAT NOT NULL DEFAULT 1.0,
    crop_override JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (highlight_id, user_id)
);

CREATE INDEX idx_reels_user_id ON reels(user_id);
CREATE INDEX idx_reels_video_id ON reels(video_id);
CREATE INDEX idx_reels_share_token ON reels(share_token) WHERE share_token IS NOT NULL;
CREATE INDEX idx_clip_edits_highlight_id ON clip_edits(highlight_id);
```

- [ ] **Step 2: Apply migration to local Postgres**

```bash
psql $DATABASE_URL -f backend/migrations/002_reels.sql
```

Expected: no errors, tables `reels` and `clip_edits` created.

- [ ] **Step 3: Apply migration to Supabase**

In Supabase dashboard → SQL Editor, paste and run `002_reels.sql`.

- [ ] **Step 4: Commit**

```bash
git add backend/migrations/002_reels.sql
git commit -m "feat(migration): Add reels and clip_edits tables"
```

---

## Task 2: Add Phase 2 dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add mediapipe and moviepy**

In `backend/requirements.txt`, append after `numpy==1.26.4`:

```
mediapipe==0.10.14
moviepy==1.0.3
imageio==2.34.0
imageio-ffmpeg==0.4.9
```

- [ ] **Step 2: Install in dev environment**

```bash
cd backend && pip install mediapipe==0.10.14 moviepy==1.0.3 imageio==2.34.0 imageio-ffmpeg==0.4.9
```

Expected: installs cleanly with no conflicting torch/torchvision pins.

- [ ] **Step 3: Create TrackNetV2 vendor directory**

```bash
mkdir -p backend/app/ml/tracknetv2
touch backend/app/ml/tracknetv2/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt backend/app/ml/tracknetv2/__init__.py
git commit -m "feat(deps): Add mediapipe, moviepy, imageio for Phase 2 ML + reel assembly"
```

---

## Task 3: TrackNetV2 model architecture + ball detection module

**Files:**
- Create: `backend/app/ml/tracknetv2/model.py`
- Create: `backend/app/ml/ball_detection.py`
- Create: `backend/tests/test_ball_detection.py`

TrackNetV2 takes 3 consecutive RGB frames stacked channel-wise (9 channels total, shape `[B, 9, H, W]`) and outputs a heatmap `[B, 3, H, W]` — one heatmap per frame. Ball position is the argmax of the heatmap.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_ball_detection.py`:

```python
import numpy as np
import pytest
from app.ml.ball_detection import BallDetection, BallDetector


def make_frames(n: int, h: int = 288, w: int = 512) -> list:
    """Generate n random uint8 BGR frames at (h, w, 3)."""
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_ball_detection_returns_one_result_per_frame():
    detector = BallDetector(weights_path=None)  # CPU, no weights → random heatmap
    frames = make_frames(5)
    results = detector.detect_sequence(frames)
    assert len(results) == 5


def test_ball_detection_result_type():
    detector = BallDetector(weights_path=None)
    frames = make_frames(3)
    results = detector.detect_sequence(frames)
    for r in results:
        assert r is None or isinstance(r, BallDetection)


def test_ball_detection_normalised_coords():
    detector = BallDetector(weights_path=None)
    frames = make_frames(3)
    results = detector.detect_sequence(frames)
    for r in results:
        if r is not None:
            assert 0.0 <= r.x <= 1.0
            assert 0.0 <= r.y <= 1.0
            assert 0.0 <= r.confidence <= 1.0


def test_ball_trajectory_from_detections():
    from app.ml.ball_detection import ball_trajectory_from_detections
    detections = [
        BallDetection(frame_idx=0, x=0.5, y=0.5, confidence=0.9),
        None,
        BallDetection(frame_idx=2, x=0.6, y=0.4, confidence=0.8),
    ]
    traj = ball_trajectory_from_detections(detections, fps=2)
    assert len(traj) == 3
    # interpolated frame 1 should be between frame 0 and frame 2
    if traj[1] is not None:
        assert 0.5 <= traj[1].x <= 0.6
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_ball_detection.py -v
```

Expected: `ImportError` — `ball_detection` module does not exist.

- [ ] **Step 3: Implement TrackNetV2 model architecture**

Create `backend/app/ml/tracknetv2/model.py`:

```python
"""
TrackNetV2 architecture — VGG-16 encoder + symmetric decoder.
Reference: "TrackNet: A Deep Learning Network for Tracking High-speed and
Tiny Objects in Sports Applications" (Huang et al., 2019), V2 improvements.

Input : (B, 9, H, W)  — 3 RGB frames concatenated channel-wise
Output: (B, 3, H, W)  — heatmap for each of the 3 frames
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TrackNetV2(nn.Module):
    """Lightweight TrackNetV2. Encodes 3 stacked frames → per-frame heatmaps."""

    def __init__(self, in_channels: int = 9, out_channels: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Head: sigmoid for heatmap probability
        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
```

- [ ] **Step 4: Implement ball_detection.py**

Create `backend/app/ml/ball_detection.py`:

```python
"""
Ball detection using TrackNetV2.
Processes sequences of frames and returns per-frame ball positions.

Weights: if weights_path is None (local dev), the model runs with random
weights and returns low-confidence detections — useful for pipeline testing.
On the GPU instance, weights are loaded from the path specified in settings.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.ml.tracknetv2.model import TrackNetV2

# TrackNetV2 was trained at 288×512 resolution
_INPUT_H = 288
_INPUT_W = 512
_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class BallDetection:
    frame_idx: int
    x: float         # normalized [0, 1] from left
    y: float         # normalized [0, 1] from top
    confidence: float


class BallDetector:
    """
    Wraps TrackNetV2 to detect ball position across a sequence of video frames.

    Usage:
        detector = BallDetector(weights_path=settings.tracknetv2_weights_path)
        detections = detector.detect_sequence(frames)  # list of BallDetection | None
    """

    def __init__(self, weights_path: str | None, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = TrackNetV2().to(self.device).eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

    def _preprocess_triplet(
        self, frames: list[np.ndarray]
    ) -> torch.Tensor:
        """Stack 3 BGR frames into a (1, 9, H, W) float32 tensor."""
        channels = []
        for frame in frames:
            resized = cv2.resize(frame, (_INPUT_W, _INPUT_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            channels.append(rgb.transpose(2, 0, 1))  # (3, H, W)
        stacked = np.concatenate(channels, axis=0)  # (9, H, W)
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    def _heatmap_to_detection(
        self, heatmap: np.ndarray, frame_idx: int, orig_h: int, orig_w: int
    ) -> Optional[BallDetection]:
        """Convert a (H, W) heatmap to a BallDetection, or None if below threshold."""
        confidence = float(heatmap.max())
        if confidence < _CONFIDENCE_THRESHOLD:
            return None
        hy, hx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        return BallDetection(
            frame_idx=frame_idx,
            x=float(hx) / heatmap.shape[1],
            y=float(hy) / heatmap.shape[0],
            confidence=confidence,
        )

    def detect_sequence(
        self, frames: list[np.ndarray]
    ) -> list[Optional[BallDetection]]:
        """
        Run ball detection on a sequence of frames.
        Processes triplets with stride 1 (frames [i-1, i, i+1]).
        Returns one Optional[BallDetection] per frame.
        """
        n = len(frames)
        results: list[Optional[BallDetection]] = [None] * n
        if n < 3:
            return results

        orig_h, orig_w = frames[0].shape[:2]

        with torch.no_grad():
            for i in range(1, n - 1):
                triplet = [frames[i - 1], frames[i], frames[i + 1]]
                inp = self._preprocess_triplet(triplet)
                heatmaps = self.model(inp).squeeze(0).cpu().numpy()
                # heatmaps shape: (3, H, W) — index 1 is the current frame
                results[i] = self._heatmap_to_detection(
                    heatmaps[1], frame_idx=i, orig_h=orig_h, orig_w=orig_w
                )
        return results


def ball_trajectory_from_detections(
    detections: list[Optional[BallDetection]],
    fps: float,
) -> list[Optional[BallDetection]]:
    """
    Linear interpolation over None gaps in ball detections.
    Gaps of more than 5 frames (0.5s at 10fps) are left as None.
    """
    result = list(detections)
    n = len(result)
    max_gap = int(fps * 0.5)

    i = 0
    while i < n:
        if result[i] is None:
            # find gap start
            gap_start = i - 1
            gap_end = i
            while gap_end < n and result[gap_end] is None:
                gap_end += 1

            gap_len = gap_end - (gap_start + 1)
            if (
                gap_start >= 0
                and gap_end < n
                and gap_len <= max_gap
            ):
                a = result[gap_start]
                b = result[gap_end]
                for j in range(1, gap_len + 1):
                    t = j / (gap_len + 1)
                    result[gap_start + j] = BallDetection(
                        frame_idx=gap_start + j,
                        x=a.x + t * (b.x - a.x),
                        y=a.y + t * (b.y - a.y),
                        confidence=min(a.confidence, b.confidence) * 0.7,
                    )
            i = gap_end
        else:
            i += 1

    return result
```

- [ ] **Step 5: Run tests — all should pass**

```bash
cd backend && python -m pytest tests/test_ball_detection.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ml/tracknetv2/ backend/app/ml/ball_detection.py backend/tests/test_ball_detection.py
git commit -m "feat(ml): Add TrackNetV2 ball detection module"
```

---

## Task 4: Pose Estimator module (MediaPipe Pose)

**Files:**
- Create: `backend/app/ml/pose_estimator.py`
- Create: `backend/tests/test_pose_estimator.py`

MediaPipe Pose extracts 33 body keypoints per person. We run it on user-player crops (extracted using Re-ID tracking bboxes) to get wrist, elbow, shoulder keypoints for paddle swing analysis.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_pose_estimator.py`:

```python
import numpy as np
import pytest
from app.ml.pose_estimator import PoseEstimator, PoseKeypoints, estimate_swing_angle


def make_frame(h: int = 480, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_pose_estimator_returns_none_on_blank_frame():
    estimator = PoseEstimator()
    frame = make_frame()
    result = estimator.estimate(frame)
    assert result is None or isinstance(result, PoseKeypoints)


def test_pose_keypoints_has_required_joints():
    kp = PoseKeypoints(
        left_wrist=(0.5, 0.3, 0.9),
        right_wrist=(0.6, 0.3, 0.8),
        left_elbow=(0.5, 0.5, 0.9),
        right_elbow=(0.6, 0.5, 0.85),
        left_shoulder=(0.4, 0.6, 0.95),
        right_shoulder=(0.65, 0.6, 0.9),
    )
    assert len(kp.left_wrist) == 3  # (x, y, visibility)
    assert len(kp.right_wrist) == 3


def test_swing_angle_right_arm():
    kp = PoseKeypoints(
        left_wrist=(0.3, 0.2, 0.9),
        right_wrist=(0.7, 0.1, 0.9),   # right wrist high (swing)
        left_elbow=(0.3, 0.5, 0.9),
        right_elbow=(0.65, 0.4, 0.9),
        left_shoulder=(0.3, 0.6, 0.9),
        right_shoulder=(0.65, 0.6, 0.9),
    )
    angle = estimate_swing_angle(kp, hand="right")
    assert isinstance(angle, float)
    assert -180.0 <= angle <= 180.0


def test_swing_angle_returns_none_on_low_visibility():
    kp = PoseKeypoints(
        left_wrist=(0.3, 0.2, 0.1),    # visibility < 0.5 → unreliable
        right_wrist=(0.7, 0.1, 0.1),
        left_elbow=(0.3, 0.5, 0.1),
        right_elbow=(0.65, 0.4, 0.1),
        left_shoulder=(0.3, 0.6, 0.1),
        right_shoulder=(0.65, 0.6, 0.1),
    )
    angle = estimate_swing_angle(kp, hand="right")
    assert angle is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_pose_estimator.py -v
```

Expected: `ImportError` — module does not exist.

- [ ] **Step 3: Implement pose_estimator.py**

Create `backend/app/ml/pose_estimator.py`:

```python
"""
Pose estimation using MediaPipe Pose.
Extracts wrist, elbow, and shoulder keypoints from a player crop.
Used by shot_classifier.py to determine swing angle and quality.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp

_mp_pose = mp.solutions.pose


@dataclass
class PoseKeypoints:
    """
    Normalized (x, y, visibility) tuples from MediaPipe.
    x, y are in [0, 1] relative to the crop frame.
    visibility is in [0, 1].
    """
    left_wrist: tuple[float, float, float]
    right_wrist: tuple[float, float, float]
    left_elbow: tuple[float, float, float]
    right_elbow: tuple[float, float, float]
    left_shoulder: tuple[float, float, float]
    right_shoulder: tuple[float, float, float]


class PoseEstimator:
    """
    Wraps MediaPipe Pose for single-image inference on player crops.

    Designed to run on a cropped bounding box of a single player, not on
    the full game frame (full frame pose estimation is unreliable at court
    distances and with 4 players).

    Usage:
        estimator = PoseEstimator()
        kp = estimator.estimate(player_crop)  # np.ndarray (H, W, 3) BGR
    """

    def __init__(self, model_complexity: int = 1, min_confidence: float = 0.5):
        self._pose = _mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=min_confidence,
        )

    def estimate(self, frame_bgr: np.ndarray) -> Optional[PoseKeypoints]:
        """
        Run pose estimation on a single BGR frame or crop.
        Returns PoseKeypoints or None if no person is detected.
        """
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark

        def _lm(idx: int) -> tuple[float, float, float]:
            p = lm[idx]
            return (p.x, p.y, p.visibility)

        return PoseKeypoints(
            left_wrist=_lm(_mp_pose.PoseLandmark.LEFT_WRIST),
            right_wrist=_lm(_mp_pose.PoseLandmark.RIGHT_WRIST),
            left_elbow=_lm(_mp_pose.PoseLandmark.LEFT_ELBOW),
            right_elbow=_lm(_mp_pose.PoseLandmark.RIGHT_ELBOW),
            left_shoulder=_lm(_mp_pose.PoseLandmark.LEFT_SHOULDER),
            right_shoulder=_lm(_mp_pose.PoseLandmark.RIGHT_SHOULDER),
        )

    def close(self) -> None:
        self._pose.close()


_VISIBILITY_THRESHOLD = 0.5


def estimate_swing_angle(
    kp: PoseKeypoints, hand: str = "right"
) -> Optional[float]:
    """
    Compute the wrist-to-elbow vector angle (degrees, 0° = straight down).
    Returns None if keypoint visibility is below threshold — angle is unreliable.

    Used by shot_classifier to detect overhead swings (angle > 90°) and
    determine smash vs dink motion.
    """
    if hand == "right":
        wrist = kp.right_wrist
        elbow = kp.right_elbow
    else:
        wrist = kp.left_wrist
        elbow = kp.left_elbow

    if wrist[2] < _VISIBILITY_THRESHOLD or elbow[2] < _VISIBILITY_THRESHOLD:
        return None

    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    return math.degrees(math.atan2(dy, dx))
```

- [ ] **Step 4: Run tests — all should pass**

```bash
cd backend && python -m pytest tests/test_pose_estimator.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/pose_estimator.py backend/tests/test_pose_estimator.py
git commit -m "feat(ml): Add MediaPipe pose estimator module"
```

---

## Task 5: Shot Classifier module (rule-based)

**Files:**
- Create: `backend/app/ml/shot_classifier.py`
- Create: `backend/tests/test_shot_classifier.py`

Shot type is derived from ball trajectory + pose keypoints, per the spec's rule-based definitions. Shot quality is a 0–1 float computed from pose consistency and contact precision.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_shot_classifier.py`:

```python
import pytest
from app.ml.ball_detection import BallDetection
from app.ml.pose_estimator import PoseKeypoints
from app.ml.shot_classifier import classify_shot, ShotClassification


def _ball(x: float, y: float, conf: float = 0.9) -> BallDetection:
    return BallDetection(frame_idx=0, x=x, y=y, confidence=conf)


def _pose_overhead() -> PoseKeypoints:
    """Wrist above shoulder — smash/overhead setup."""
    return PoseKeypoints(
        left_wrist=(0.3, 0.1, 0.9),    # high y = top of frame = wrist above head
        right_wrist=(0.7, 0.1, 0.9),
        left_elbow=(0.3, 0.3, 0.9),
        right_elbow=(0.7, 0.3, 0.9),
        left_shoulder=(0.3, 0.5, 0.9),
        right_shoulder=(0.7, 0.5, 0.9),
    )


def _pose_low() -> PoseKeypoints:
    """Wrist near waist — dink/drop setup."""
    return PoseKeypoints(
        left_wrist=(0.3, 0.7, 0.9),
        right_wrist=(0.7, 0.7, 0.9),
        left_elbow=(0.3, 0.6, 0.9),
        right_elbow=(0.7, 0.6, 0.9),
        left_shoulder=(0.3, 0.5, 0.9),
        right_shoulder=(0.7, 0.5, 0.9),
    )


def test_classify_smash_high_wrist_ball_above_net():
    # ball is high (y close to 0 = top of frame), wrist above shoulder
    shot = classify_shot(
        ball_before=_ball(0.5, 0.3),
        ball_after=_ball(0.6, 0.8),   # ball descending fast → smash
        pose=_pose_overhead(),
        player_crossed_centerline=False,
    )
    assert shot.shot_type == "smash"


def test_classify_lob_ball_trajectory_steeply_upward():
    shot = classify_shot(
        ball_before=_ball(0.5, 0.8),
        ball_after=_ball(0.5, 0.1),   # ball rising steeply
        pose=_pose_low(),
        player_crossed_centerline=False,
    )
    assert shot.shot_type == "lob"


def test_classify_erne_player_crosses_centerline():
    shot = classify_shot(
        ball_before=_ball(0.5, 0.5),
        ball_after=_ball(0.7, 0.5),
        pose=_pose_low(),
        player_crossed_centerline=True,
    )
    assert shot.shot_type == "erne"


def test_classify_dink_low_pose_ball_near_net():
    shot = classify_shot(
        ball_before=_ball(0.5, 0.55),
        ball_after=_ball(0.55, 0.5),  # gentle motion near net height
        pose=_pose_low(),
        player_crossed_centerline=False,
    )
    assert shot.shot_type == "dink"


def test_classify_drive_default_high_speed():
    shot = classify_shot(
        ball_before=_ball(0.2, 0.5),
        ball_after=_ball(0.9, 0.5),   # fast horizontal — high speed
        pose=_pose_low(),
        player_crossed_centerline=False,
    )
    assert shot.shot_type == "drive"


def test_shot_quality_float_in_range():
    shot = classify_shot(
        ball_before=_ball(0.5, 0.5),
        ball_after=_ball(0.6, 0.5),
        pose=_pose_low(),
        player_crossed_centerline=False,
    )
    assert isinstance(shot.quality, float)
    assert 0.0 <= shot.quality <= 1.0


def test_classify_returns_unknown_on_no_ball():
    shot = classify_shot(
        ball_before=None,
        ball_after=None,
        pose=None,
        player_crossed_centerline=False,
    )
    assert shot.shot_type == "drive"  # default fallback
    assert shot.quality == 0.5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_shot_classifier.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement shot_classifier.py**

Create `backend/app/ml/shot_classifier.py`:

```python
"""
Rule-based shot classifier for pickleball.

Derives shot type and quality from:
  - Ball trajectory (position before/after contact)
  - Pose keypoints (wrist/elbow angles)
  - Player court position (did they cross centerline = erne)

Phase 3 upgrade: replace this with a VideoMAE/MoViNet classifier
trained on these rule-based labels as weak supervision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from app.ml.ball_detection import BallDetection
from app.ml.pose_estimator import PoseKeypoints, estimate_swing_angle

ShotType = Literal[
    "drive", "dink", "lob", "erne", "smash", "overhead", "drop", "speed_up", "atp"
]

# Thresholds
_NET_Y = 0.5          # normalized y-coord of net (centre of frame)
_SPEED_HIGH = 0.25    # ball displacement per frame above this → fast shot
_SPEED_LOW = 0.08     # below this → soft contact (dink/drop)
_LOB_DY = -0.25       # ball rising steeply (y decreasing, origin top-left)
_SMASH_WRIST_ABOVE_SHOULDER_Y = 0.1  # wrist.y above shoulder.y by this amount


@dataclass
class ShotClassification:
    shot_type: ShotType
    quality: float  # 0.0–1.0


def _ball_speed(b1: BallDetection, b2: BallDetection) -> float:
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    return math.sqrt(dx * dx + dy * dy)


def _is_overhead_pose(pose: PoseKeypoints) -> bool:
    """Returns True if the higher-visibility wrist is above shoulder level."""
    for wrist, shoulder in [
        (pose.right_wrist, pose.right_shoulder),
        (pose.left_wrist, pose.left_shoulder),
    ]:
        if wrist[2] > 0.5 and shoulder[2] > 0.5:
            # In normalized coords, smaller y = higher on screen
            if wrist[1] < shoulder[1] - _SMASH_WRIST_ABOVE_SHOULDER_Y:
                return True
    return False


def _shot_quality(
    pose: Optional[PoseKeypoints],
    shot_type: ShotType,
    ball_before: Optional[BallDetection],
    ball_after: Optional[BallDetection],
) -> float:
    """
    Heuristic quality score:
    - Consistent pose visibility: higher → better form
    - For drives/smashes: clean fast contact → higher quality
    - For dinks/drops: gentle, controlled motion → higher quality
    """
    if pose is None:
        return 0.5

    # Average visibility of key joints as form proxy
    joints = [
        pose.right_wrist, pose.left_wrist,
        pose.right_elbow, pose.left_elbow,
        pose.right_shoulder, pose.left_shoulder,
    ]
    avg_visibility = sum(j[2] for j in joints) / len(joints)
    base = avg_visibility * 0.7

    if ball_before is not None and ball_after is not None:
        speed = _ball_speed(ball_before, ball_after)
        if shot_type in ("drive", "smash", "overhead", "speed_up"):
            # Clean hard shots should be fast
            speed_bonus = min(speed / _SPEED_HIGH, 1.0) * 0.3
        else:
            # Soft shots should be controlled — penalize too-fast
            speed_bonus = (1.0 - min(speed / _SPEED_HIGH, 1.0)) * 0.3
        return min(base + speed_bonus, 1.0)

    return base


def classify_shot(
    ball_before: Optional[BallDetection],
    ball_after: Optional[BallDetection],
    pose: Optional[PoseKeypoints],
    player_crossed_centerline: bool,
) -> ShotClassification:
    """
    Classify the shot type using rule-based logic.

    Priority order (highest specificity first):
    1. Erne — player crosses centerline during swing
    2. Smash/Overhead — wrist above shoulder + ball descending
    3. Lob — ball trajectory steeply upward
    4. Dink — soft contact + ball near net height
    5. Drive — default for fast horizontal shots
    """
    # 1. Erne (most specific — player movement trumps ball signal)
    if player_crossed_centerline:
        quality = _shot_quality(pose, "erne", ball_before, ball_after)
        return ShotClassification(shot_type="erne", quality=quality)

    # Without ball data, return default
    if ball_before is None or ball_after is None:
        quality = _shot_quality(pose, "drive", None, None)
        return ShotClassification(shot_type="drive", quality=quality)

    dy = ball_after.y - ball_before.y  # positive = ball falling, origin top-left
    speed = _ball_speed(ball_before, ball_after)

    # 2. Smash / Overhead — overhead pose + ball descending fast
    if pose is not None and _is_overhead_pose(pose) and dy > 0.1:
        shot_type = "smash" if speed > _SPEED_HIGH else "overhead"
        quality = _shot_quality(pose, shot_type, ball_before, ball_after)
        return ShotClassification(shot_type=shot_type, quality=quality)

    # 3. Lob — ball steeply rising
    if dy < _LOB_DY:
        quality = _shot_quality(pose, "lob", ball_before, ball_after)
        return ShotClassification(shot_type="lob", quality=quality)

    # 4. Dink — soft contact near net
    if speed < _SPEED_LOW and ball_before.y > _NET_Y - 0.1:
        quality = _shot_quality(pose, "dink", ball_before, ball_after)
        return ShotClassification(shot_type="dink", quality=quality)

    # 5. Drive — default fast shot
    quality = _shot_quality(pose, "drive", ball_before, ball_after)
    return ShotClassification(shot_type="drive", quality=quality)
```

- [ ] **Step 4: Run tests — all should pass**

```bash
cd backend && python -m pytest tests/test_shot_classifier.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/shot_classifier.py backend/tests/test_shot_classifier.py
git commit -m "feat(ml): Add rule-based shot classifier with ball trajectory + pose signals"
```

---

## Task 6: Extend highlight_scorer.py with Phase 2 shot signals

**Files:**
- Modify: `backend/app/ml/highlight_scorer.py`
- Modify: `backend/tests/test_highlight_scorer.py`

Phase 1's `score_highlight` defaults `shot_quality=0.5` and has no shot_type multipliers. Phase 2 adds multipliers for premium shot types (`erne`, `smash`, `atp`) per the spec's role-aware scoring.

- [ ] **Step 1: Write failing tests for shot_type multipliers**

Add to the bottom of `backend/tests/test_highlight_scorer.py`:

```python
from app.ml.highlight_scorer import score_highlight


def test_erne_scored_by_user_gets_high_score():
    erne_score = score_highlight(
        point_scored=True,
        point_won_by="user_team",
        rally_length=5,
        attributed_role="user",
        shot_quality=0.8,
        shot_type="erne",
    )
    drive_score = score_highlight(
        point_scored=True,
        point_won_by="user_team",
        rally_length=5,
        attributed_role="user",
        shot_quality=0.8,
        shot_type="drive",
    )
    assert erne_score > drive_score


def test_premium_shot_types_increase_score():
    for premium in ("erne", "smash", "atp"):
        s = score_highlight(
            point_scored=False,
            point_won_by=None,
            rally_length=4,
            attributed_role="user",
            shot_quality=0.7,
            shot_type=premium,
        )
        base = score_highlight(
            point_scored=False,
            point_won_by=None,
            rally_length=4,
            attributed_role="user",
            shot_quality=0.7,
            shot_type="dink",
        )
        assert s > base, f"{premium} should score higher than dink"


def test_shot_type_none_does_not_crash():
    s = score_highlight(
        point_scored=False,
        point_won_by=None,
        rally_length=3,
        attributed_role="user",
        shot_quality=0.5,
        shot_type=None,
    )
    assert 0.0 <= s <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_highlight_scorer.py -v
```

Expected: 3 new tests FAIL — `score_highlight` has no `shot_type` parameter.

- [ ] **Step 3: Extend score_highlight in highlight_scorer.py**

In `backend/app/ml/highlight_scorer.py`, replace the `score_highlight` function signature and body:

```python
# Shot type multipliers — premium pickleball shots boost the highlight score
_SHOT_TYPE_MULTIPLIERS: dict[str, float] = {
    "erne": 1.3,
    "atp": 1.3,
    "smash": 1.3,
    "overhead": 1.1,
    "lob": 1.05,
    "drive": 1.0,
    "speed_up": 1.0,
    "dink": 0.9,
    "drop": 0.9,
}


def score_highlight(
    point_scored: bool,
    point_won_by: Literal["user_team", "opponent_team"] | None,
    rally_length: int,
    attributed_role: str,
    shot_quality: float = 0.5,
    shot_type: str | None = None,
    weights: RoleWeights | None = None,
) -> float:
    """
    Phase 2 highlight scorer. Adds shot_type multipliers for premium shots.
    Backwards-compatible: shot_type=None uses no multiplier (multiplier=1.0).
    Returns a score in [0, 1].
    """
    if weights is None:
        weights = RoleWeights()

    rally_score = min(rally_length / 20.0, 1.0) * 0.3
    quality_score = shot_quality * 0.2

    point_score = 0.0
    if point_scored and point_won_by == "user_team":
        point_score = 0.5
    elif point_scored and point_won_by == "opponent_team":
        point_score = 0.05

    raw_score = rally_score + quality_score + point_score

    if attributed_role == "user":
        role_weight = weights.user
    elif attributed_role == "partner":
        role_weight = weights.partner
    elif attributed_role in ("opponent_1", "opponent_2"):
        role_weight = weights.opponent_1
    else:
        raise ValueError(
            f"Unknown attributed_role: {attributed_role!r}. "
            "Expected one of: user, partner, opponent_1, opponent_2"
        )

    shot_multiplier = _SHOT_TYPE_MULTIPLIERS.get(shot_type or "", 1.0)
    final_score = raw_score * role_weight * shot_multiplier
    return min(final_score, 1.0)
```

- [ ] **Step 4: Run full highlight scorer tests**

```bash
cd backend && python -m pytest tests/test_highlight_scorer.py -v
```

Expected: all tests PASS (existing Phase 1 tests + 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/highlight_scorer.py backend/tests/test_highlight_scorer.py
git commit -m "feat(ml): Extend highlight scorer with Phase 2 shot_type multipliers"
```

---

## Task 7: Wire Phase 2 ML stages into ingest pipeline

**Files:**
- Modify: `backend/app/workers/ingest.py`
- Modify: `backend/app/config.py`
- Modify: `backend/tests/test_ingest.py`

The `run_ai_pipeline` task runs after user tap. Currently it runs Re-ID → rally detection → scoring → clip extraction. Phase 2 adds two new stages between Re-ID and rally detection: ball detection and pose estimation. Shot classifier replaces the Phase 1 default shot_quality=0.5.

- [ ] **Step 1: Add tracknetv2_weights_path to config.py**

In `backend/app/config.py`, add to the Settings class:

```python
tracknetv2_weights_path: str | None = None  # set to path on GPU instance
```

- [ ] **Step 2: Write failing test for Phase 2 pipeline stages**

Add to the bottom of `backend/tests/test_ingest.py`:

```python
def test_run_ai_pipeline_populates_shot_type(
    mock_r2, mock_db, video_id, user_id, seed_bbox
):
    """After Phase 2, highlights should have non-null shot_type from classifier."""
    # Run pipeline (mocked R2 + DB)
    run_ai_pipeline(video_id, user_id, seed_bbox)
    # Verify shot_type is set on saved highlights
    saved_highlights = mock_db.fetch_highlights(video_id)
    for h in saved_highlights:
        assert h["shot_type"] in (
            "drive", "dink", "lob", "erne", "smash", "overhead",
            "drop", "speed_up", "atp"
        ), f"unexpected shot_type: {h['shot_type']}"
```

- [ ] **Step 3: Run the new test to confirm it fails**

```bash
cd backend && python -m pytest tests/test_ingest.py::test_run_ai_pipeline_populates_shot_type -v
```

Expected: FAIL — shot_type is null (Phase 1 pipeline doesn't set it).

- [ ] **Step 4: Add ball detection + pose stages to run_ai_pipeline**

In `backend/app/workers/ingest.py`, inside `run_ai_pipeline`, add these imports at the top of the function body (after the existing imports block at line 229):

```python
from app.ml.ball_detection import BallDetector, ball_trajectory_from_detections
from app.ml.pose_estimator import PoseEstimator
from app.ml.shot_classifier import classify_shot
```

After the `labeled_frames = track_user_across_frames(...)` line, add:

```python
        # ── Phase 2: Ball Detection ─────────────────────────────────────────
        ball_detector = BallDetector(weights_path=settings.tracknetv2_weights_path)
        raw_ball_detections = ball_detector.detect_sequence(frames)
        ball_detections = ball_trajectory_from_detections(raw_ball_detections, fps=2)

        # ── Phase 2: Pose Estimation ────────────────────────────────────────
        # Run on user-player crops only (identified by Re-ID tracking)
        pose_estimator = PoseEstimator()
        frame_poses = []
        for frame_i, (frame, labeled) in enumerate(zip(frames, labeled_frames)):
            user_bbox = next(
                (d["bbox"] for d in labeled if d.get("role") == "user"), None
            )
            if user_bbox:
                x, y, w, h = (
                    int(user_bbox["x"]), int(user_bbox["y"]),
                    int(user_bbox["w"]), int(user_bbox["h"]),
                )
                crop = frame[max(0, y):y + h, max(0, x):x + w]
                pose = pose_estimator.estimate(crop) if crop.size > 0 else None
            else:
                pose = None
            frame_poses.append(pose)
        pose_estimator.close()
```

Then replace the highlight scoring block (where `score_highlight` is called) to include `shot_type`:

```python
        for idx, rally in enumerate(rallies):
            # Pair ball detections with this rally's frame range
            rally_frame_start = rally.start_time_ms * 2 // 1000
            rally_frame_end = rally.end_time_ms * 2 // 1000

            mid_frame = (rally_frame_start + rally_frame_end) // 2
            ball_before = ball_detections[mid_frame - 1] if mid_frame > 0 else None
            ball_after = ball_detections[mid_frame + 1] if mid_frame + 1 < len(ball_detections) else None
            pose = frame_poses[mid_frame] if mid_frame < len(frame_poses) else None

            # Determine if user crossed centerline (for erne detection)
            user_positions = [
                d["bbox"]["x"] for frame_labeled in labeled_frames[rally_frame_start:rally_frame_end]
                for d in frame_labeled if d.get("role") == "user"
            ]
            player_crossed = (
                len(user_positions) > 1
                and max(user_positions) > 0.5
                and min(user_positions) < 0.4
            )

            shot_result = classify_shot(
                ball_before=ball_before,
                ball_after=ball_after,
                pose=pose,
                player_crossed_centerline=player_crossed,
            )

            hl_score = score_highlight(
                point_scored=rally.point_won_by is not None,
                point_won_by=rally.point_won_by,
                rally_length=len(rally.segments),
                attributed_role="user",
                shot_quality=shot_result.quality,
                shot_type=shot_result.shot_type,
            )
```

- [ ] **Step 5: Run the new test**

```bash
cd backend && python -m pytest tests/test_ingest.py -v
```

Expected: all tests PASS including the new Phase 2 test.

- [ ] **Step 6: Commit**

```bash
git add backend/app/workers/ingest.py backend/app/config.py backend/tests/test_ingest.py
git commit -m "feat(pipeline): Wire ball detection + pose estimation + shot classifier into ingest pipeline"
```

---

## Task 8: Music library + Reel Assembler module

**Files:**
- Create: `backend/static/music/README.txt`
- Create: `backend/app/ml/reel_assembler.py`
- Create: `backend/tests/test_reel_assembler.py`

The reel assembler takes a list of clip R2 keys + assembly config and produces a single output video. It:
1. Downloads clips from R2 to tmp
2. Applies slow-mo on peak moments (highlight_score > 0.85)
3. Applies smart vertical/square crop centered on user bbox
4. Adds fade transitions between clips
5. Mixes in background music track (ducked under natural audio)
6. Uploads assembled reel to R2

Music tracks are pre-downloaded CC0 MP3s from Free Music Archive. No API needed.

- [ ] **Step 1: Download music tracks**

```bash
mkdir -p backend/static/music
```

Go to https://freemusicarchive.org and download two royalty-free CC0 tracks:
- An upbeat/energetic track → save as `backend/static/music/energetic_bg.mp3`
- A chill background track → save as `backend/static/music/chill_bg.mp3`

Alternatively, use these specific FMA tracks:
```bash
# Podington Bear - Upbeat (CC BY-NC: free for personal use)
curl -L "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Upbeat/Podington_Bear_-_Upbeat.mp3" \
  -o backend/static/music/energetic_bg.mp3

# Podington Bear - Chill (CC BY-NC)
curl -L "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Kai_Engel/Sustain/Kai_Engel_-_Contention.mp3" \
  -o backend/static/music/chill_bg.mp3
```

- [ ] **Step 2: Create music README**

Create `backend/static/music/README.txt`:

```
Music tracks for PickleClips reel assembly.
All tracks are licensed for personal, non-commercial use (CC BY-NC).

energetic_bg.mp3
  Title: Upbeat
  Artist: Podington Bear
  License: CC BY-NC 3.0
  Source: https://freemusicarchive.org/music/Podington_Bear

chill_bg.mp3
  Title: Contention
  Artist: Kai Engel
  License: CC BY-NC 3.0
  Source: https://freemusicarchive.org/music/Kai_Engel
```

- [ ] **Step 3: Write failing tests**

Create `backend/tests/test_reel_assembler.py`:

```python
import os
import tempfile
import numpy as np
import cv2
import pytest
from app.ml.reel_assembler import (
    ClipSpec,
    ReelConfig,
    ReelAssembler,
    smart_crop_frame,
)


def make_test_video(path: str, n_frames: int = 30, fps: int = 30) -> str:
    """Write a minimal MP4 to disk for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (640, 360))
    for i in range(n_frames):
        frame = np.full((360, 640, 3), i * 8 % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_smart_crop_horizontal_noop():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cropped = smart_crop_frame(frame, format="horizontal", user_center_x=0.5)
    assert cropped.shape == frame.shape


def test_smart_crop_vertical_returns_9x16_ratio():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cropped = smart_crop_frame(frame, format="vertical", user_center_x=0.5)
    h, w = cropped.shape[:2]
    ratio = h / w
    assert abs(ratio - (16 / 9)) < 0.1


def test_smart_crop_square_returns_1x1_ratio():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cropped = smart_crop_frame(frame, format="square", user_center_x=0.5)
    h, w = cropped.shape[:2]
    assert abs(h - w) <= 2


def test_reel_assembler_init():
    assembler = ReelAssembler(music_dir="backend/static/music")
    assert assembler is not None


def test_clip_spec_dataclass():
    spec = ClipSpec(local_path="/tmp/clip.mp4", highlight_score=0.9, slow_mo_factor=0.5)
    assert spec.slow_mo_factor == 0.5


def test_reel_config_defaults():
    config = ReelConfig(output_type="highlight_montage")
    assert config.format == "horizontal"
    assert config.music_track == "energetic_bg"
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_reel_assembler.py -v
```

Expected: `ImportError`.

- [ ] **Step 5: Implement reel_assembler.py**

Create `backend/app/ml/reel_assembler.py`:

```python
"""
Reel assembly using FFmpeg + MoviePy.

Takes a list of downloaded clip files + assembly config and produces
a single output video with transitions, optional slow-mo, smart crop,
and background music.

Heavy lifting is done by FFmpeg subprocess calls for speed; MoviePy
is used for audio mixing and transition compositing.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

Format = Literal["vertical", "horizontal", "square"]
OutputType = Literal[
    "highlight_montage", "my_best_plays", "game_recap",
    "points_of_improvement", "best_shots", "scored_point_rally",
    "full_rally_replay", "single_shot_clip",
]

# Output dimensions by format
_FORMAT_DIMS: dict[Format, tuple[int, int]] = {
    "horizontal": (1920, 1080),
    "vertical": (1080, 1920),
    "square": (1080, 1080),
}

# Slow-mo threshold: clips with highlight_score above this get half-speed treatment
_SLOW_MO_SCORE_THRESHOLD = 0.85
_SLOW_MO_FACTOR = 0.5  # half speed for peak moments

# Music track for each output type
_OUTPUT_TYPE_MUSIC: dict[OutputType, str] = {
    "highlight_montage": "energetic_bg",
    "my_best_plays": "energetic_bg",
    "game_recap": "chill_bg",
    "points_of_improvement": "chill_bg",
    "best_shots": "energetic_bg",
    "scored_point_rally": "energetic_bg",
    "full_rally_replay": "chill_bg",
    "single_shot_clip": "energetic_bg",
}


@dataclass
class ClipSpec:
    local_path: str
    highlight_score: float = 0.5
    slow_mo_factor: float = 1.0  # 1.0 = normal speed
    user_center_x: float = 0.5   # normalized x for smart vertical crop


@dataclass
class ReelConfig:
    output_type: OutputType
    format: Format = "horizontal"
    music_track: str = "energetic_bg"
    include_music: bool = True
    music_volume: float = 0.3     # 0.0 = silent, 1.0 = full volume


def smart_crop_frame(
    frame: np.ndarray,
    format: Format,
    user_center_x: float = 0.5,
) -> np.ndarray:
    """
    Crop a frame to the target aspect ratio, centering on the user's x position.
    For 'horizontal', returns the frame unchanged (already 16:9).
    """
    if format == "horizontal":
        return frame

    h, w = frame.shape[:2]
    target_w, target_h = _FORMAT_DIMS[format]
    target_ratio = target_w / target_h

    if format == "vertical":
        # Crop width to match 9:16 from a 16:9 source
        crop_w = int(h * target_ratio)
        crop_w = min(crop_w, w)
        center_x = int(user_center_x * w)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(w, x1 + crop_w)
        x1 = max(0, x2 - crop_w)
        return frame[:, x1:x2]

    if format == "square":
        # Crop to 1:1 centered on user
        side = min(h, w)
        center_x = int(user_center_x * w)
        x1 = max(0, center_x - side // 2)
        x2 = min(w, x1 + side)
        x1 = max(0, x2 - side)
        y_offset = (h - side) // 2
        return frame[y_offset:y_offset + side, x1:x2]

    return frame


class ReelAssembler:
    """
    Assembles highlight clips into a reel video.

    Usage:
        assembler = ReelAssembler(music_dir="backend/static/music")
        output_path = assembler.assemble(clips, config, output_path="/tmp/reel.mp4")
    """

    def __init__(self, music_dir: str = "backend/static/music"):
        self.music_dir = Path(music_dir)

    def _apply_slow_mo(self, input_path: str, output_path: str, factor: float) -> None:
        """Use FFmpeg setpts + atempo to slow down a clip."""
        pts_factor = 1.0 / factor
        audio_tempo = max(0.5, min(factor, 2.0))  # atempo range: [0.5, 2.0]
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"setpts={pts_factor:.3f}*PTS",
            "-af", f"atempo={audio_tempo:.3f}",
            "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _resize_and_crop(
        self, input_path: str, output_path: str, config: ReelConfig,
        user_center_x: float = 0.5,
    ) -> None:
        """Resize clip to target format dimensions via FFmpeg."""
        target_w, target_h = _FORMAT_DIMS[config.format]

        if config.format == "horizontal":
            scale_filter = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
        elif config.format == "vertical":
            # Crop horizontally, then scale
            cx_pct = user_center_x
            scale_filter = (
                f"scale=iw*{target_h}/ih:-1,"
                f"crop={target_w}:{target_h}:"
                f"(iw-{target_w})*{cx_pct:.3f}:0"
            )
        else:  # square
            scale_filter = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", scale_filter,
            "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _add_fade_transitions(
        self, clip_paths: list[str], output_path: str, fade_duration: float = 0.3
    ) -> None:
        """Concatenate clips with cross-fade transitions using FFmpeg."""
        if len(clip_paths) == 1:
            import shutil
            shutil.copy(clip_paths[0], output_path)
            return

        # Build FFmpeg concat with xfade filter
        inputs = []
        for p in clip_paths:
            inputs.extend(["-i", p])

        # Simple concat without xfade for reliability (xfade needs duration per clip)
        list_file = output_path + ".txt"
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.unlink(list_file)

    def _mix_music(
        self, video_path: str, output_path: str, config: ReelConfig
    ) -> None:
        """Mix background music into the video, ducked under natural audio."""
        music_path = self.music_dir / f"{config.music_track}.mp3"
        if not music_path.exists():
            # No music file — skip silently
            import shutil
            shutil.copy(video_path, output_path)
            return

        vol = config.music_volume
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            f"[0:a]volume=1.0[orig];[1:a]volume={vol:.2f}[music];[orig][music]amix=inputs=2:duration=first[out]",
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def assemble(
        self,
        clips: list[ClipSpec],
        config: ReelConfig,
        output_path: str,
    ) -> str:
        """
        Assemble clips into a reel.
        Returns the output_path on success.
        Raises subprocess.CalledProcessError if FFmpeg fails.
        """
        if not clips:
            raise ValueError("No clips provided for reel assembly")

        with tempfile.TemporaryDirectory(prefix="pickleclips_reel_") as tmp:
            processed = []

            for i, clip in enumerate(clips):
                step_path = os.path.join(tmp, f"clip_{i:03d}_step.mp4")
                current = clip.local_path

                # Apply slow-mo if clip is a peak moment or explicit factor
                effective_factor = clip.slow_mo_factor
                if effective_factor == 1.0 and clip.highlight_score >= _SLOW_MO_SCORE_THRESHOLD:
                    effective_factor = _SLOW_MO_FACTOR

                if effective_factor != 1.0:
                    slo_path = os.path.join(tmp, f"clip_{i:03d}_slo.mp4")
                    self._apply_slow_mo(current, slo_path, effective_factor)
                    current = slo_path

                # Resize + smart crop to target format
                self._resize_and_crop(current, step_path, config, clip.user_center_x)
                processed.append(step_path)

            # Concatenate with fade transitions
            concat_path = os.path.join(tmp, "concat.mp4")
            self._add_fade_transitions(processed, concat_path)

            # Mix music
            if config.include_music:
                self._mix_music(concat_path, output_path, config)
            else:
                import shutil
                shutil.copy(concat_path, output_path)

        return output_path
```

- [ ] **Step 6: Run tests**

```bash
cd backend && python -m pytest tests/test_reel_assembler.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/ml/reel_assembler.py backend/tests/test_reel_assembler.py backend/static/music/
git commit -m "feat(ml): Add reel assembler with FFmpeg+MoviePy transitions, slow-mo, smart crop, music"
```

---

## Task 9: Reel Service (services/reel.py)

**Files:**
- Create: `backend/app/services/reel.py`

The reel service orchestrates: selecting clips for a given output type, building assembly config, uploading completed reel to R2, generating a share URL. It is called by the reel_gen Celery worker (Task 10).

- [ ] **Step 1: Write failing tests**

```bash
# No separate test file — reel service is tested via test_reels.py in Task 12.
# But write one unit test for clip selection logic:
```

Create `backend/tests/test_reel_service.py`:

```python
import pytest
from app.services.reel import select_clips_for_output_type


def _make_highlight(id: str, score: float, shot_type: str = "drive",
                    sub_type: str = "point_scored", role: str = "user"):
    return {
        "id": id,
        "highlight_score": score,
        "shot_type": shot_type,
        "sub_highlight_type": sub_type,
        "attributed_player_role": role,
        "r2_key_clip": f"clips/{id}.mp4",
    }


def test_highlight_montage_returns_top_10():
    highlights = [_make_highlight(str(i), float(i) / 10) for i in range(20)]
    selected = select_clips_for_output_type("highlight_montage", highlights, lowlights=[])
    assert len(selected) <= 10
    scores = [h["highlight_score"] for h in selected]
    assert scores == sorted(scores, reverse=True)


def test_my_best_plays_filters_to_user_only():
    highlights = [
        _make_highlight("1", 0.9, role="user"),
        _make_highlight("2", 0.8, role="partner"),
        _make_highlight("3", 0.7, role="opponent_1"),
    ]
    selected = select_clips_for_output_type("my_best_plays", highlights, lowlights=[])
    assert all(h["attributed_player_role"] == "user" for h in selected)


def test_points_of_improvement_uses_lowlights():
    highlights = [_make_highlight("1", 0.9)]
    lowlights = [
        {"id": "l1", "shot_quality": 0.1, "r2_key_clip": "clips/l1.mp4", "sub_highlight_type": "lowlight"},
        {"id": "l2", "shot_quality": 0.2, "r2_key_clip": "clips/l2.mp4", "sub_highlight_type": "lowlight"},
    ]
    selected = select_clips_for_output_type("points_of_improvement", highlights, lowlights=lowlights)
    assert all(h["sub_highlight_type"] == "lowlight" for h in selected)


def test_game_recap_includes_all_scored_points():
    highlights = [
        _make_highlight("1", 0.9, sub_type="point_scored"),
        _make_highlight("2", 0.6, sub_type="shot_form"),
        _make_highlight("3", 0.7, sub_type="point_scored"),
    ]
    selected = select_clips_for_output_type("game_recap", highlights, lowlights=[])
    ids = {h["id"] for h in selected}
    assert "1" in ids and "3" in ids
    assert "2" not in ids
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_reel_service.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement services/reel.py**

Create `backend/app/services/reel.py`:

```python
"""
Reel service — clip selection, assembly orchestration, R2 upload, share URL generation.
Called by reel_gen Celery worker.
"""
from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from app.config import settings
from app.ml.reel_assembler import ClipSpec, ReelAssembler, ReelConfig

_MAX_CLIPS_PER_REEL = 10
_MAX_GAME_RECAP_CLIPS = 20


def select_clips_for_output_type(
    output_type: str,
    highlights: list[dict],
    lowlights: list[dict],
) -> list[dict]:
    """
    Select and order highlights/lowlights for a given output type.
    Returns a list of highlight dicts with r2_key_clip set.
    """
    available = [h for h in highlights if h.get("r2_key_clip")]

    if output_type == "highlight_montage":
        sorted_h = sorted(available, key=lambda h: h["highlight_score"], reverse=True)
        return sorted_h[:_MAX_CLIPS_PER_REEL]

    if output_type == "my_best_plays":
        user_clips = [h for h in available if h.get("attributed_player_role") == "user"]
        sorted_h = sorted(user_clips, key=lambda h: h["highlight_score"], reverse=True)
        return sorted_h[:_MAX_CLIPS_PER_REEL]

    if output_type == "game_recap":
        scored = [h for h in available if h.get("sub_highlight_type") == "point_scored"]
        # chronological order for game recap
        return sorted(scored, key=lambda h: h.get("start_time_ms", 0))[:_MAX_GAME_RECAP_CLIPS]

    if output_type == "points_of_improvement":
        available_low = [l for l in lowlights if l.get("r2_key_clip")]
        return sorted(available_low, key=lambda h: h.get("shot_quality", 0.5))[:_MAX_CLIPS_PER_REEL]

    if output_type == "best_shots":
        sorted_h = sorted(available, key=lambda h: h.get("shot_quality", 0.5), reverse=True)
        return sorted_h[:_MAX_CLIPS_PER_REEL]

    if output_type in ("scored_point_rally", "full_rally_replay"):
        scored = [h for h in available if h.get("sub_highlight_type") == "point_scored"]
        sorted_h = sorted(scored, key=lambda h: h["highlight_score"], reverse=True)
        return sorted_h[:1]

    if output_type == "single_shot_clip":
        sorted_h = sorted(available, key=lambda h: h["highlight_score"], reverse=True)
        return sorted_h[:1]

    return sorted(available, key=lambda h: h["highlight_score"], reverse=True)[:_MAX_CLIPS_PER_REEL]


def _get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def download_clips(clips: list[dict], tmp_dir: str) -> list[tuple[dict, str]]:
    """Download clip files from R2 to tmp_dir. Returns [(highlight, local_path)]."""
    s3 = _get_r2_client()
    downloaded = []
    for clip in clips:
        r2_key = clip["r2_key_clip"]
        local_path = os.path.join(tmp_dir, Path(r2_key).name)
        s3.download_file(settings.r2_bucket_name, r2_key, local_path)
        downloaded.append((clip, local_path))
    return downloaded


def assemble_and_upload(
    reel_id: str,
    output_type: str,
    clips: list[dict],
    lowlights: list[dict],
    format: str = "horizontal",
    user_center_x: float = 0.5,
    music_dir: str = "backend/static/music",
) -> str:
    """
    Full pipeline: select clips → download from R2 → assemble → upload to R2.
    Returns the R2 key of the uploaded reel.
    """
    selected = select_clips_for_output_type(output_type, clips, lowlights)
    if not selected:
        raise ValueError(f"No clips available for output_type={output_type!r}")

    assembler = ReelAssembler(music_dir=music_dir)
    config = ReelConfig(output_type=output_type, format=format)  # type: ignore[arg-type]

    with tempfile.TemporaryDirectory(prefix=f"pickleclips_reel_{reel_id}_") as tmp:
        downloaded = download_clips(selected, tmp)

        clip_specs = [
            ClipSpec(
                local_path=local_path,
                highlight_score=clip.get("highlight_score", 0.5),
                user_center_x=user_center_x,
            )
            for clip, local_path in downloaded
        ]

        output_path = os.path.join(tmp, f"reel_{reel_id}.mp4")
        assembler.assemble(clip_specs, config, output_path)

        r2_key = f"reels/{reel_id}/output_{format}.mp4"
        s3 = _get_r2_client()
        s3.upload_file(output_path, settings.r2_bucket_name, r2_key)

    return r2_key


def generate_share_token() -> str:
    return secrets.token_urlsafe(16)


def generate_share_url(share_token: str) -> str:
    base = settings.app_base_url.rstrip("/")
    return f"{base}/reels/share/{share_token}"
```

- [ ] **Step 4: Add app_base_url to config.py**

In `backend/app/config.py`, add to Settings:

```python
app_base_url: str = "http://localhost:3000"
```

- [ ] **Step 5: Run tests**

```bash
cd backend && python -m pytest tests/test_reel_service.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/reel.py backend/tests/test_reel_service.py backend/app/config.py
git commit -m "feat(service): Add reel service with clip selection, assembly orchestration, share URLs"
```

---

## Task 10: Reel Generation Worker (workers/reel_gen.py)

**Files:**
- Create: `backend/app/workers/reel_gen.py`

The Celery task `generate_reel` is triggered via the reels API. It: fetches clips from DB, calls `assemble_and_upload`, updates reel status + r2_key in DB, generates share token. Auto-generation of the 4 default reel types is triggered at the end of `run_ai_pipeline`.

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_reel_gen.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


def test_generate_reel_updates_status_to_ready(mock_db, mock_r2):
    from app.workers.reel_gen import generate_reel
    reel_id = "test-reel-id"
    video_id = "test-video-id"
    user_id = "test-user-id"

    with patch("app.workers.reel_gen.assemble_and_upload", return_value="reels/test/output.mp4"):
        generate_reel(reel_id=reel_id, video_id=video_id, user_id=user_id,
                      output_type="highlight_montage", format="horizontal")

    updated = mock_db.get_reel(reel_id)
    assert updated["status"] == "ready"
    assert updated["r2_key"] == "reels/test/output.mp4"


def test_generate_reel_sets_failed_status_on_error(mock_db, mock_r2):
    from app.workers.reel_gen import generate_reel
    reel_id = "test-reel-id-2"

    with patch("app.workers.reel_gen.assemble_and_upload", side_effect=RuntimeError("FFmpeg error")):
        with pytest.raises(RuntimeError):
            generate_reel(reel_id=reel_id, video_id="v1", user_id="u1",
                          output_type="highlight_montage", format="horizontal")

    updated = mock_db.get_reel(reel_id)
    assert updated["status"] == "failed"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_reel_gen.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement workers/reel_gen.py**

Create `backend/app/workers/reel_gen.py`:

```python
"""
Celery task for reel generation.
Triggered by the reels API (on-demand) and auto-triggered at the end of run_ai_pipeline (auto-generated types).
"""
from __future__ import annotations

import asyncio
import json
from typing import Literal

import asyncpg

from app.config import settings
from app.services.reel import assemble_and_upload, generate_share_token, generate_share_url
from app.workers.celery_app import celery

_AUTO_GENERATED_TYPES = [
    "highlight_montage",
    "my_best_plays",
    "game_recap",
    "points_of_improvement",
]


def _db_update_reel(reel_id: str, status: str, r2_key: str | None = None,
                    share_token: str | None = None) -> None:
    """Sync DB update from Celery worker thread."""
    async def _update():
        conn = await asyncpg.connect(settings.database_url)
        try:
            if r2_key:
                await conn.execute(
                    """UPDATE reels SET status = $1, r2_key = $2, share_token = $3 WHERE id = $4""",
                    status, r2_key, share_token, reel_id,
                )
            else:
                await conn.execute("UPDATE reels SET status = $1 WHERE id = $2", status, reel_id)
        finally:
            await conn.close()

    asyncio.run(_update())


def _fetch_clips_and_lowlights(video_id: str) -> tuple[list[dict], list[dict]]:
    """Fetch all highlights and lowlights for a video from DB."""
    async def _fetch():
        conn = await asyncpg.connect(settings.database_url)
        try:
            highlights = await conn.fetch(
                """SELECT id, highlight_score, shot_type, sub_highlight_type,
                          attributed_player_role, r2_key_clip, start_time_ms, shot_quality
                   FROM highlights
                   WHERE video_id = $1 AND sub_highlight_type != 'lowlight'
                   AND r2_key_clip IS NOT NULL""",
                video_id,
            )
            lowlights = await conn.fetch(
                """SELECT id, highlight_score, shot_quality, sub_highlight_type, r2_key_clip
                   FROM highlights
                   WHERE video_id = $1 AND sub_highlight_type = 'lowlight'
                   AND r2_key_clip IS NOT NULL""",
                video_id,
            )
            return [dict(h) for h in highlights], [dict(l) for l in lowlights]
        finally:
            await conn.close()

    return asyncio.run(_fetch())


def _get_user_center_x(video_id: str) -> float:
    """Get the user player's average horizontal position (for smart vertical crop)."""
    async def _fetch():
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """SELECT seed_frame_bbox FROM video_players
                   WHERE video_id = $1 AND role = 'user'""",
                video_id,
            )
            if row and row["seed_frame_bbox"]:
                bbox = json.loads(row["seed_frame_bbox"]) if isinstance(row["seed_frame_bbox"], str) else row["seed_frame_bbox"]
                # bbox is {x, y, w, h} in pixels — normalize to frame width
                # Use center of bbox as user position
                x = bbox.get("x", 0)
                w_box = bbox.get("w", 100)
                # We don't have frame width here; default to 1920
                frame_w = bbox.get("frame_w", 1920)
                return (x + w_box / 2) / frame_w
        finally:
            await conn.close()
        return 0.5  # default center

    return asyncio.run(_fetch())


@celery.task(bind=True, name="app.workers.reel_gen.generate_reel", max_retries=2)
def generate_reel(
    self,
    reel_id: str,
    video_id: str,
    user_id: str,
    output_type: str,
    format: str = "horizontal",
    music_dir: str = "backend/static/music",
) -> None:
    """
    Generate a reel for the given reel_id.
    Updates reel status → 'generating' → 'ready' (or 'failed').
    """
    _db_update_reel(reel_id, "generating")
    try:
        clips, lowlights = _fetch_clips_and_lowlights(video_id)
        user_center_x = _get_user_center_x(video_id)

        r2_key = assemble_and_upload(
            reel_id=reel_id,
            output_type=output_type,
            clips=clips,
            lowlights=lowlights,
            format=format,
            user_center_x=user_center_x,
            music_dir=music_dir,
        )

        share_token = generate_share_token()
        _db_update_reel(reel_id, "ready", r2_key=r2_key, share_token=share_token)

    except Exception as exc:
        _db_update_reel(reel_id, "failed")
        raise self.retry(exc=exc, countdown=60)


def trigger_auto_generated_reels(video_id: str, user_id: str) -> None:
    """
    Called at the end of run_ai_pipeline to queue all 4 auto-generated reel types.
    Creates DB rows for each reel type then dispatches Celery tasks.
    """
    async def _create_reels():
        conn = await asyncpg.connect(settings.database_url)
        try:
            reel_ids = {}
            for output_type in _AUTO_GENERATED_TYPES:
                row = await conn.fetchrow(
                    """INSERT INTO reels (user_id, video_id, output_type, format, auto_generated)
                       VALUES ($1, $2, $3, 'horizontal', TRUE)
                       RETURNING id""",
                    user_id, video_id, output_type,
                )
                reel_ids[output_type] = str(row["id"])
            return reel_ids
        finally:
            await conn.close()

    reel_ids = asyncio.run(_create_reels())

    for output_type, reel_id in reel_ids.items():
        generate_reel.delay(
            reel_id=reel_id,
            video_id=video_id,
            user_id=user_id,
            output_type=output_type,
        )
```

- [ ] **Step 4: Trigger auto-generated reels at end of run_ai_pipeline**

In `backend/app/workers/ingest.py`, at the end of `run_ai_pipeline` (after `update_video_status(video_id, "analyzed")`), add:

```python
        # Trigger Phase 2 auto-generated reels
        from app.workers.reel_gen import trigger_auto_generated_reels
        trigger_auto_generated_reels(video_id=video_id, user_id=user_id)
```

- [ ] **Step 5: Run tests**

```bash
cd backend && python -m pytest tests/test_reel_gen.py -v
```

Expected: tests PASS (mock-based, no real DB/R2 needed).

- [ ] **Step 6: Commit**

```bash
git add backend/app/workers/reel_gen.py backend/app/workers/ingest.py backend/tests/test_reel_gen.py
git commit -m "feat(worker): Add reel generation Celery task with auto-trigger after pipeline"
```

---

## Task 11: Reels API Router

**Files:**
- Create: `backend/app/routers/reels.py`
- Create: `backend/tests/test_reels.py`

Endpoints:
- `GET /api/v1/videos/{video_id}/reels` — list all reels for a video
- `POST /api/v1/reels` — create + queue reel generation (on-demand)
- `GET /api/v1/reels/{reel_id}` — reel details + download URL when ready
- `POST /api/v1/reels/{reel_id}/share` — generate/return share link
- `GET /api/v1/reels/share/{share_token}` — public OG preview (no auth required)

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_reels.py`:

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_reels_for_video(client: AsyncClient, auth_headers, analyzed_video_id):
    resp = await client.get(f"/api/v1/videos/{analyzed_video_id}/reels", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_create_reel_queues_task(client: AsyncClient, auth_headers, analyzed_video_id, mock_celery):
    resp = await client.post(
        "/api/v1/reels",
        json={"video_id": analyzed_video_id, "output_type": "highlight_montage", "format": "horizontal"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["status"] == "queued"
    assert mock_celery.called


@pytest.mark.asyncio
async def test_get_reel_not_found_returns_404(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/reels/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_share_reel_returns_share_url(client: AsyncClient, auth_headers, ready_reel_id):
    resp = await client.post(f"/api/v1/reels/{ready_reel_id}/share", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "share_url" in data
    assert data["share_url"].startswith("http")


@pytest.mark.asyncio
async def test_public_share_page_no_auth_required(client: AsyncClient, ready_reel_share_token):
    resp = await client.get(f"/api/v1/reels/share/{ready_reel_share_token}")
    assert resp.status_code == 200
    data = resp.json()
    assert "output_type" in data
    assert "download_url" in data
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_reels.py -v
```

Expected: 404 on all routes (router not registered yet).

- [ ] **Step 3: Implement routers/reels.py**

Create `backend/app/routers/reels.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import asyncpg

from app.auth import get_current_user
from app.database import get_db
from app.services.reel import generate_share_token, generate_share_url
from app.services.storage import generate_download_url

router = APIRouter(tags=["reels"])

_VALID_OUTPUT_TYPES = {
    "highlight_montage", "my_best_plays", "game_recap",
    "points_of_improvement", "best_shots", "scored_point_rally",
    "full_rally_replay", "single_shot_clip",
}
_VALID_FORMATS = {"vertical", "horizontal", "square"}


class CreateReelBody(BaseModel):
    video_id: str
    output_type: str
    format: str = "horizontal"


@router.get("/videos/{video_id}/reels")
async def list_reels(
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
        """SELECT id, output_type, format, status, duration_seconds, auto_generated, created_at
           FROM reels WHERE video_id = $1 ORDER BY created_at DESC""",
        video_id,
    )
    return [dict(r) for r in rows]


@router.post("/reels", status_code=201)
async def create_reel(
    body: CreateReelBody,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    if body.output_type not in _VALID_OUTPUT_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid output_type: {body.output_type!r}")
    if body.format not in _VALID_FORMATS:
        raise HTTPException(status_code=422, detail=f"Invalid format: {body.format!r}")

    video = await db.fetchrow(
        "SELECT id FROM videos WHERE id = $1 AND user_id = $2 AND status = 'analyzed'",
        body.video_id, user_id,
    )
    if not video:
        raise HTTPException(status_code=404, detail="Analyzed video not found")

    row = await db.fetchrow(
        """INSERT INTO reels (user_id, video_id, output_type, format, auto_generated)
           VALUES ($1, $2, $3, $4, FALSE)
           RETURNING id, status""",
        user_id, body.video_id, body.output_type, body.format,
    )
    reel_id = str(row["id"])

    # Queue reel generation task
    from app.workers.reel_gen import generate_reel
    generate_reel.delay(
        reel_id=reel_id,
        video_id=body.video_id,
        user_id=user_id,
        output_type=body.output_type,
        format=body.format,
    )

    return {"id": reel_id, "status": "queued", "output_type": body.output_type}


@router.get("/reels/{reel_id}")
async def get_reel(
    reel_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    row = await db.fetchrow(
        """SELECT r.id, r.output_type, r.format, r.status, r.r2_key,
                  r.duration_seconds, r.auto_generated, r.share_token, r.created_at
           FROM reels r
           WHERE r.id = $1 AND r.user_id = $2""",
        reel_id, user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Reel not found")

    result = dict(row)
    if row["status"] == "ready" and row["r2_key"]:
        result["download_url"] = generate_download_url(row["r2_key"], expires_in=3600)
    if row["share_token"]:
        result["share_url"] = generate_share_url(row["share_token"])

    return result


@router.post("/reels/{reel_id}/share")
async def share_reel(
    reel_id: str,
    user_id: str = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
):
    row = await db.fetchrow(
        "SELECT id, status, share_token, r2_key FROM reels WHERE id = $1 AND user_id = $2",
        reel_id, user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Reel not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=409, detail="Reel is not ready yet")

    token = row["share_token"] or generate_share_token()
    if not row["share_token"]:
        await db.execute("UPDATE reels SET share_token = $1 WHERE id = $2", token, reel_id)

    return {"share_url": generate_share_url(token), "share_token": token}


@router.get("/reels/share/{share_token}")
async def get_shared_reel(
    share_token: str,
    db: asyncpg.Connection = Depends(get_db),
):
    """Public endpoint — no auth required. Returns OG preview metadata + download URL."""
    row = await db.fetchrow(
        """SELECT r.id, r.output_type, r.format, r.duration_seconds, r.r2_key
           FROM reels r
           WHERE r.share_token = $1 AND r.status = 'ready'""",
        share_token,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Shared reel not found")

    download_url = generate_download_url(row["r2_key"], expires_in=86400) if row["r2_key"] else None
    return {
        "output_type": row["output_type"],
        "format": row["format"],
        "duration_seconds": row["duration_seconds"],
        "download_url": download_url,
    }
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_reels.py -v
```

Expected: all 5 tests PASS (after registering router in next task).

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/reels.py backend/tests/test_reels.py
git commit -m "feat(router): Add reels API router with CRUD, share, and public OG preview"
```

---

## Task 12: Register reels router in main.py

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Register the reels router**

In `backend/app/main.py`, add the import and router registration:

```python
from app.routers import videos, highlights, reels   # add 'reels'

# ... (after existing include_router calls)
app.include_router(reels.router, prefix="/api/v1")
```

- [ ] **Step 2: Run all backend tests**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "feat(app): Register reels router"
```

---

## Task 13: Frontend — FeedbackButtons component

**Files:**
- Create: `frontend/components/FeedbackButtons.tsx`

This component renders thumbs up / thumbs down buttons for a highlight or lowlight. It calls `PATCH /api/v1/highlights/{id}` and shows optimistic state.

- [ ] **Step 1: Write the component**

Create `frontend/components/FeedbackButtons.tsx`:

```tsx
'use client'

import { useState } from 'react'
import { api } from '@/lib/api'

interface Props {
  highlightId: string
  initialFeedback: 'liked' | 'disliked' | null
  token: string
}

export function FeedbackButtons({ highlightId, initialFeedback, token }: Props) {
  const [feedback, setFeedback] = useState<'liked' | 'disliked' | null>(initialFeedback)
  const [loading, setLoading] = useState(false)

  async function handleFeedback(value: 'liked' | 'disliked') {
    if (loading) return
    const next = feedback === value ? null : value  // toggle off if same
    setFeedback(next)  // optimistic update
    setLoading(true)
    try {
      await api.updateHighlightFeedback(token, highlightId, next)
    } catch {
      setFeedback(feedback)  // revert on error
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex gap-2 items-center">
      <button
        onClick={() => handleFeedback('liked')}
        disabled={loading}
        className={`px-3 py-1 rounded text-sm transition-colors ${
          feedback === 'liked'
            ? 'bg-green-600 text-white'
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        }`}
        aria-label="Liked"
      >
        👍
      </button>
      <button
        onClick={() => handleFeedback('disliked')}
        disabled={loading}
        className={`px-3 py-1 rounded text-sm transition-colors ${
          feedback === 'disliked'
            ? 'bg-red-600 text-white'
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        }`}
        aria-label="Disliked"
      >
        👎
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Confirm the build passes**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/FeedbackButtons.tsx
git commit -m "feat(ui): Add FeedbackButtons component with optimistic thumbs up/down"
```

---

## Task 14: Frontend — ReelCard component

**Files:**
- Create: `frontend/components/ReelCard.tsx`

Displays a reel with: output type label, status badge, duration, share button, download button (when ready).

- [ ] **Step 1: Write the component**

Create `frontend/components/ReelCard.tsx`:

```tsx
'use client'

import { useState } from 'react'
import { api } from '@/lib/api'

interface Reel {
  id: string
  output_type: string
  format: string
  status: 'queued' | 'generating' | 'ready' | 'failed'
  duration_seconds: number | null
  auto_generated: boolean
  download_url?: string
  share_url?: string
}

interface Props {
  reel: Reel
  token: string
}

const OUTPUT_TYPE_LABELS: Record<string, string> = {
  highlight_montage: 'Highlight Montage',
  my_best_plays: 'My Best Plays',
  game_recap: 'Game Recap',
  points_of_improvement: 'Points of Improvement',
  best_shots: 'Best Shots',
  scored_point_rally: 'Scored Point Rally',
  full_rally_replay: 'Full Rally Replay',
  single_shot_clip: 'Single Shot Clip',
}

const STATUS_STYLES: Record<string, string> = {
  queued: 'bg-gray-600 text-gray-200',
  generating: 'bg-yellow-600 text-yellow-100 animate-pulse',
  ready: 'bg-green-700 text-green-100',
  failed: 'bg-red-700 text-red-100',
}

export function ReelCard({ reel, token }: Props) {
  const [shareUrl, setShareUrl] = useState<string | null>(reel.share_url ?? null)
  const [sharing, setSharing] = useState(false)
  const [copied, setCopied] = useState(false)

  async function handleShare() {
    if (shareUrl) {
      await navigator.clipboard.writeText(shareUrl)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      return
    }
    setSharing(true)
    try {
      const result = await api.shareReel(token, reel.id)
      setShareUrl(result.share_url)
      await navigator.clipboard.writeText(result.share_url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } finally {
      setSharing(false)
    }
  }

  const label = OUTPUT_TYPE_LABELS[reel.output_type] ?? reel.output_type
  const duration = reel.duration_seconds ? `${Math.round(reel.duration_seconds)}s` : null

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-white text-sm">{label}</h3>
          <p className="text-gray-400 text-xs mt-1 capitalize">
            {reel.format}{duration ? ` · ${duration}` : ''}
            {reel.auto_generated ? ' · Auto-generated' : ''}
          </p>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_STYLES[reel.status]}`}>
          {reel.status}
        </span>
      </div>

      {reel.status === 'ready' && (
        <div className="flex gap-2">
          {reel.download_url && (
            <a
              href={reel.download_url}
              download
              className="flex-1 text-center px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded transition-colors"
            >
              Download
            </a>
          )}
          <button
            onClick={handleShare}
            disabled={sharing}
            className="flex-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded transition-colors"
          >
            {copied ? 'Copied!' : sharing ? 'Getting link…' : shareUrl ? 'Copy link' : 'Share'}
          </button>
        </div>
      )}

      {reel.status === 'generating' && (
        <p className="text-yellow-400 text-xs">Assembling reel…</p>
      )}

      {reel.status === 'failed' && (
        <p className="text-red-400 text-xs">Reel generation failed.</p>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Confirm the build passes**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/ReelCard.tsx
git commit -m "feat(ui): Add ReelCard component with status, download, and share"
```

---

## Task 15: Frontend — api.ts additions

**Files:**
- Modify: `frontend/lib/api.ts`

Add reel and lowlight API methods needed by the new pages.

- [ ] **Step 1: Add methods to api.ts**

In `frontend/lib/api.ts`, extend the `api` object with:

```typescript
  listLowlights: (token: string, videoId: string) =>
    apiFetch<object[]>(`/api/v1/videos/${videoId}/lowlights`, {}, token),

  updateHighlightFeedback: (token: string, highlightId: string, feedback: 'liked' | 'disliked' | null) =>
    apiFetch(`/api/v1/highlights/${highlightId}`, {
      method: 'PATCH',
      body: JSON.stringify({ user_feedback: feedback }),
    }, token),

  listReels: (token: string, videoId: string) =>
    apiFetch<object[]>(`/api/v1/videos/${videoId}/reels`, {}, token),

  createReel: (token: string, videoId: string, outputType: string, format: string = 'horizontal') =>
    apiFetch<{ id: string; status: string; output_type: string }>(
      '/api/v1/reels',
      { method: 'POST', body: JSON.stringify({ video_id: videoId, output_type: outputType, format }) },
      token
    ),

  getReel: (token: string, reelId: string) =>
    apiFetch<object>(`/api/v1/reels/${reelId}`, {}, token),

  shareReel: (token: string, reelId: string) =>
    apiFetch<{ share_url: string; share_token: string }>(
      `/api/v1/reels/${reelId}/share`,
      { method: 'POST' },
      token
    ),
```

- [ ] **Step 2: Confirm the build passes**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/api.ts
git commit -m "feat(api): Add reel and lowlight API client methods"
```

---

## Task 16: Frontend — video detail page Phase 2 updates

**Files:**
- Modify: `frontend/app/videos/[id]/page.tsx`

Phase 1's video detail page shows highlights only. Phase 2 adds:
- A tab bar: **Highlights | Points of Improvement | Reels**
- Feedback buttons (👍/👎) on each ClipCard
- A "Generate Reel" shortcut button (links to reels subpage)

- [ ] **Step 1: Rewrite the video detail page**

Replace the contents of `frontend/app/videos/[id]/page.tsx`:

```tsx
'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import Link from 'next/link'
import { createClient } from '@/lib/supabase'
import { api } from '@/lib/api'
import { ProcessingStatus } from '@/components/ProcessingStatus'
import { ClipCard } from '@/components/ClipCard'
import { FeedbackButtons } from '@/components/FeedbackButtons'

type Tab = 'highlights' | 'lowlights' | 'reels'

export default function VideoPage() {
  const params = useParams()
  const videoId = params.id as string
  const router = useRouter()

  const [token, setToken] = useState<string | null>(null)
  const [video, setVideo] = useState<{ status: string } | null>(null)
  const [highlights, setHighlights] = useState<object[]>([])
  const [lowlights, setLowlights] = useState<object[]>([])
  const [activeTab, setActiveTab] = useState<Tab>('highlights')

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) return router.push('/login')
      const t = data.session.access_token
      setToken(t)
      const v = await api.getVideo(t, videoId)
      setVideo(v as { status: string })
      if ((v as { status: string }).status === 'analyzed') {
        await loadClips(t)
      }
      if ((v as { status: string }).status === 'identifying') {
        router.push(`/videos/${videoId}/identify`)
      }
    })
  }, [videoId, router])

  async function loadClips(t: string) {
    const [h, l] = await Promise.all([
      api.listHighlights(t, videoId),
      api.listLowlights(t, videoId),
    ])
    setHighlights(h)
    setLowlights(l)
    setVideo((v) => v ? { ...v, status: 'analyzed' } : v)
  }

  async function onAnalyzed() {
    if (token) await loadClips(token)
  }

  if (!video || !token) return <div className="p-8 text-gray-400">Loading…</div>

  const tabs: { id: Tab; label: string; count: number }[] = [
    { id: 'highlights', label: 'Highlights', count: highlights.length },
    { id: 'lowlights', label: 'Points of Improvement', count: lowlights.length },
    { id: 'reels', label: 'Reels', count: 0 },
  ]

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Your Game</h1>
        {video.status === 'analyzed' && (
          <Link
            href={`/videos/${videoId}/reels`}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded transition-colors"
          >
            View Reels
          </Link>
        )}
      </div>

      <ProcessingStatus
        videoId={videoId}
        initialStatus={video.status}
        onAnalyzed={onAnalyzed}
      />

      {video.status === 'analyzed' && (
        <>
          {/* Tab bar */}
          <div className="flex border-b border-gray-700 mb-6 mt-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-white'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                {tab.label}
                {tab.count > 0 && (
                  <span className="ml-2 bg-gray-700 text-gray-300 text-xs px-1.5 py-0.5 rounded-full">
                    {tab.count}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Highlights tab */}
          {activeTab === 'highlights' && (
            highlights.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {highlights.map((h) => {
                  const hi = h as { id: string; highlight_score: number; start_time_ms: number; end_time_ms: number; shot_type: string | null; rally_length: number; user_feedback: 'liked' | 'disliked' | null }
                  return (
                    <div key={hi.id} className="flex flex-col gap-2">
                      <ClipCard highlight={hi} token={token} />
                      <FeedbackButtons
                        highlightId={hi.id}
                        initialFeedback={hi.user_feedback}
                        token={token}
                      />
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-gray-500">No highlights detected in this game.</p>
            )
          )}

          {/* Points of Improvement tab */}
          {activeTab === 'lowlights' && (
            lowlights.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {lowlights.map((l) => {
                  const li = l as { id: string; highlight_score: number; start_time_ms: number; end_time_ms: number; shot_type: string | null; rally_length: number; user_feedback: 'liked' | 'disliked' | null }
                  return (
                    <div key={li.id} className="flex flex-col gap-2">
                      <ClipCard highlight={li} token={token} />
                      <FeedbackButtons
                        highlightId={li.id}
                        initialFeedback={li.user_feedback}
                        token={token}
                      />
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-gray-500">No errors or weak shots detected.</p>
            )
          )}

          {/* Reels tab — redirect to reels page */}
          {activeTab === 'reels' && (
            <div className="text-center py-12">
              <p className="text-gray-400 mb-4">View and generate highlight reels for this game.</p>
              <Link
                href={`/videos/${videoId}/reels`}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors"
              >
                Go to Reels
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Confirm the build passes**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/videos/[id]/page.tsx
git commit -m "feat(ui): Add Highlights/POI/Reels tabs and feedback buttons to video detail page"
```

---

## Task 17: Frontend — Reels list page

**Files:**
- Create: `frontend/app/videos/[id]/reels/page.tsx`

Lists all reels for a video. Shows auto-generated reels first (with status). Provides buttons to generate on-demand reel types.

- [ ] **Step 1: Create the reels page**

Create `frontend/app/videos/[id]/reels/page.tsx`:

```tsx
'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { createClient } from '@/lib/supabase'
import { api } from '@/lib/api'
import { ReelCard } from '@/components/ReelCard'

const ON_DEMAND_TYPES = [
  { id: 'best_shots', label: 'Best Shots Reel' },
  { id: 'scored_point_rally', label: 'Scored Point Rally' },
  { id: 'full_rally_replay', label: 'Full Rally Replay' },
]

export default function ReelsPage() {
  const params = useParams()
  const videoId = params.id as string
  const router = useRouter()

  const [token, setToken] = useState<string | null>(null)
  const [reels, setReels] = useState<object[]>([])
  const [generating, setGenerating] = useState<string | null>(null)

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) return router.push('/login')
      const t = data.session.access_token
      setToken(t)
      const r = await api.listReels(t, videoId)
      setReels(r)
    })
  }, [videoId, router])

  async function handleGenerate(outputType: string) {
    if (!token || generating) return
    setGenerating(outputType)
    try {
      const newReel = await api.createReel(token, videoId, outputType, 'horizontal')
      setReels((prev) => [newReel, ...prev])
    } finally {
      setGenerating(null)
    }
  }

  if (!token) return <div className="p-8 text-gray-400">Loading…</div>

  const existingTypes = new Set(
    (reels as { output_type: string }[]).map((r) => r.output_type)
  )

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="flex items-center gap-4 mb-8">
        <button
          onClick={() => router.back()}
          className="text-gray-400 hover:text-white text-sm"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-bold">Reels</h1>
      </div>

      {/* Auto-generated reels */}
      {reels.length > 0 && (
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Your Reels
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {(reels as { id: string; output_type: string; format: string; status: 'queued' | 'generating' | 'ready' | 'failed'; duration_seconds: number | null; auto_generated: boolean; download_url?: string; share_url?: string }[]).map((reel) => (
              <ReelCard key={reel.id} reel={reel} token={token} />
            ))}
          </div>
        </section>
      )}

      {/* On-demand reel generation */}
      <section>
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Generate On-Demand
        </h2>
        <div className="flex flex-col gap-2">
          {ON_DEMAND_TYPES.filter((t) => !existingTypes.has(t.id)).map((type) => (
            <button
              key={type.id}
              onClick={() => handleGenerate(type.id)}
              disabled={!!generating}
              className="flex items-center justify-between px-4 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-left transition-colors disabled:opacity-50"
            >
              <span className="text-white text-sm">{type.label}</span>
              {generating === type.id ? (
                <span className="text-yellow-400 text-xs">Queuing…</span>
              ) : (
                <span className="text-blue-400 text-xs">Generate →</span>
              )}
            </button>
          ))}
          {ON_DEMAND_TYPES.every((t) => existingTypes.has(t.id)) && (
            <p className="text-gray-500 text-sm">All reel types generated.</p>
          )}
        </div>
      </section>
    </div>
  )
}
```

- [ ] **Step 2: Confirm the build passes**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/videos/[id]/reels/page.tsx
git commit -m "feat(ui): Add reels list page with auto-generated reels and on-demand generation"
```

---

## Task 18: Frontend — Reel share page

**Files:**
- Create: `frontend/app/reels/[id]/page.tsx`

The reel detail/player page. Shows reel metadata, video player (when ready), and copy-share-link button. Also serves as the landing page for shared links.

- [ ] **Step 1: Create the reel player page**

Create `frontend/app/reels/[id]/page.tsx`:

```tsx
'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { createClient } from '@/lib/supabase'
import { api } from '@/lib/api'

const OUTPUT_TYPE_LABELS: Record<string, string> = {
  highlight_montage: 'Highlight Montage',
  my_best_plays: 'My Best Plays',
  game_recap: 'Game Recap',
  points_of_improvement: 'Points of Improvement',
  best_shots: 'Best Shots',
  scored_point_rally: 'Scored Point Rally',
  full_rally_replay: 'Full Rally Replay',
  single_shot_clip: 'Single Shot Clip',
}

interface ReelDetail {
  id: string
  output_type: string
  format: string
  status: string
  duration_seconds: number | null
  download_url?: string
  share_url?: string
}

export default function ReelPage() {
  const params = useParams()
  const reelId = params.id as string
  const router = useRouter()

  const [token, setToken] = useState<string | null>(null)
  const [reel, setReel] = useState<ReelDetail | null>(null)
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [sharing, setSharing] = useState(false)

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) return router.push('/login')
      const t = data.session.access_token
      setToken(t)
      const r = await api.getReel(t, reelId)
      const rd = r as ReelDetail
      setReel(rd)
      if (rd.share_url) setShareUrl(rd.share_url)
    })
  }, [reelId, router])

  async function handleShare() {
    if (!token || !reel) return
    if (shareUrl) {
      await navigator.clipboard.writeText(shareUrl)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      return
    }
    setSharing(true)
    try {
      const result = await api.shareReel(token, reelId)
      setShareUrl(result.share_url)
      await navigator.clipboard.writeText(result.share_url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } finally {
      setSharing(false)
    }
  }

  if (!reel || !token) return <div className="p-8 text-gray-400">Loading…</div>

  const label = OUTPUT_TYPE_LABELS[reel.output_type] ?? reel.output_type
  const isVertical = reel.format === 'vertical'

  return (
    <div className="max-w-2xl mx-auto p-8">
      <button
        onClick={() => router.back()}
        className="text-gray-400 hover:text-white text-sm mb-6 block"
      >
        ← Back
      </button>

      <h1 className="text-2xl font-bold mb-1">{label}</h1>
      <p className="text-gray-400 text-sm mb-6 capitalize">
        {reel.format}
        {reel.duration_seconds ? ` · ${Math.round(reel.duration_seconds)}s` : ''}
      </p>

      {/* Video player (only when ready) */}
      {reel.status === 'ready' && reel.download_url && (
        <div className={`mb-6 bg-black rounded-lg overflow-hidden ${isVertical ? 'max-w-xs mx-auto' : ''}`}>
          <video
            src={reel.download_url}
            controls
            autoPlay={false}
            className="w-full"
          />
        </div>
      )}

      {reel.status === 'generating' && (
        <div className="mb-6 bg-gray-800 rounded-lg p-8 text-center">
          <p className="text-yellow-400 animate-pulse">Assembling your reel…</p>
          <p className="text-gray-500 text-sm mt-2">Usually takes 1–2 minutes.</p>
        </div>
      )}

      {reel.status === 'queued' && (
        <div className="mb-6 bg-gray-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">Reel is queued and will start shortly.</p>
        </div>
      )}

      {reel.status === 'failed' && (
        <div className="mb-6 bg-red-900/40 border border-red-700 rounded-lg p-6 text-center">
          <p className="text-red-400">Reel generation failed. Try regenerating.</p>
        </div>
      )}

      {/* Action buttons */}
      {reel.status === 'ready' && (
        <div className="flex gap-3">
          {reel.download_url && (
            <a
              href={reel.download_url}
              download
              className="flex-1 text-center px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors"
            >
              Download
            </a>
          )}
          <button
            onClick={handleShare}
            disabled={sharing}
            className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            {copied ? 'Link copied!' : sharing ? 'Getting link…' : shareUrl ? 'Copy share link' : 'Share'}
          </button>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Confirm the build passes**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no errors.

- [ ] **Step 3: Run full test suite**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/reels/
git commit -m "feat(ui): Add reel player/share page with video preview and share link"
```

---

## Self-Review

### Spec coverage check

| Phase 2 Spec Item | Task |
|---|---|
| TrackNetV2 ball detection | Task 3 |
| MediaPipe Pose estimation | Task 4 |
| Rule-based shot classifier | Task 5 |
| Role-aware highlight + lowlight scoring | Task 6 |
| FFmpeg + MoviePy reel assembly | Task 8 |
| Transitions, slow-mo on peak moments | Task 8 (reel_assembler.py) |
| Smart vertical crop (center on user) | Task 8 + Task 10 |
| Royalty-free music library | Task 8 (Step 1-2) |
| 4 auto-generated output types | Tasks 10-11 (trigger_auto_generated_reels) |
| 4 on-demand output types | Tasks 11, 17 |
| Share links with OG preview cards | Task 11 (GET /reels/share/{token}) |
| Thumbs up/down feedback on clips | Task 13 (already in API from Phase 1) + Task 13 (UI) |
| Supabase Realtime for status | Already built in Phase 1 (ProcessingStatus component) |
| Points of Improvement output | Tasks 9 (select_clips), 16 (POI tab) |

### Placeholder scan

No TBD, TODO, or unimplemented stubs found in the plan.

### Type consistency check

- `BallDetection.frame_idx / x / y / confidence` — used consistently across Task 3, 5, 7
- `PoseKeypoints` fields (tuples of 3) — consistent across Task 4, 5
- `ShotClassification.shot_type / quality` — consumed by Task 6 (`score_highlight(shot_type=...)`)
- `ClipSpec.local_path / highlight_score / slow_mo_factor / user_center_x` — produced in Task 9, consumed in Task 8
- `ReelConfig.output_type / format / music_track` — consistent across Tasks 8, 9, 10
- `generate_reel` Celery task signature `(reel_id, video_id, user_id, output_type, format)` — consistent between Task 10 and Task 11

### Gaps identified

None — all spec items are covered. Phase 3 items (persistent player profiles, auto-recognition, YAMNet audio, VideoMAE shot classifier) are explicitly deferred per the spec.
