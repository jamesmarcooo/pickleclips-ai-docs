# PickleClips — AI-Powered Pickleball Highlights Platform

## Architecture Design Document

---

## 1. Product Vision

PickleClips lets pickleball players upload full game videos (up to 2.7K resolution), then uses AI to automatically identify highlights — winning shots, rallies, aces, and key plays — and generates shareable reels optimized for Instagram, TikTok, and YouTube Shorts.

### Core User Flow

```
Upload Video → Player Identification → AI Analysis → Highlight Detection → Clip Extraction → Reel Assembly → Share/Download
```

### Key Features

- Upload full-length doubles game videos (2.7K, ~15–20 minutes per game)
- **Player identification** — tap on yourself once, AI tracks you for the rest of the game (and eventually recognizes you automatically)
- AI-powered highlight detection trained on pickleball-specific patterns
- Personalized "highlight style" that learns what *your* best plays look like
- Auto-generated reels with transitions, slow-mo, and music
- Manual clip trimming and reel editing
- One-tap sharing to social platforms
- Web app (launch) + mobile app (phase 2)

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│                                                                      │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│   │  Web App      │    │  Mobile App       │    │  Share Widget    │   │
│   │  (Next.js)    │    │  (React Native)   │    │  (Embed/OG)     │   │
│   └──────┬───────┘    └────────┬─────────┘    └────────┬────────┘   │
└──────────┼─────────────────────┼──────────────────────┼─────────────┘
           │                     │                      │
           ▼                     ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY LAYER                             │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  API Gateway (AWS API Gateway / Kong)                         │   │
│   │  — Auth, rate limiting, request routing                       │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      APPLICATION SERVICES                            │
│                                                                      │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐  │
│   │  Auth       │  │  Upload     │  │  Highlight  │  │  Reel        │  │
│   │  Service    │  │  Service    │  │  Service    │  │  Service     │  │
│   └────────────┘  └────────────┘  └────────────┘  └─────────────┘  │
│                                                                      │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐  │
│   │  User       │  │  Social     │  │  Player ID  │  │  Notification │  │
│   │  Service    │  │  Service    │  │  Service    │  │  Service      │  │
│   └────────────┘  └────────────┘  └────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       AI / ML PIPELINE                                │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                    Video Ingestion                            │   │
│   │  Transcode → Scene Detection → Frame Extraction               │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │               Player Identification                           │   │
│   │  User tap seed → Person Detection → Re-ID Tracking            │   │
│   │  → Persistent Profile Matching (after 3+ uploads)             │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                  Analysis Models                              │   │
│   │                                                               │   │
│   │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │   │
│   │  │ Court & Ball │  │ Player Pose  │  │ Score Detection    │  │   │
│   │  │ Detection    │  │ Estimation   │  │ (OCR on overlay)   │  │   │
│   │  └─────────────┘  └──────────────┘  └────────────────────┘  │   │
│   │                                                               │   │
│   │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │   │
│   │  │ Shot Class.  │  │ Rally        │  │ Audio Analysis     │  │   │
│   │  │ (type/qual.) │  │ Tracking     │  │ (crowd/paddle)     │  │   │
│   │  └─────────────┘  └──────────────┘  └────────────────────┘  │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                 Highlight Scorer                              │   │
│   │  Combines signals → ranks moments → user preference model    │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                 Reel Generator                                │   │
│   │  Clip extraction → transitions → slow-mo → music → export    │   │
│   └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA & STORAGE                                │
│                                                                      │
│   ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐   │
│   │  S3 / R2    │  │  PostgreSQL │  │  Redis    │  │  Pinecone /   │   │
│   │  (video +   │  │  (metadata, │  │  (cache,  │  │  Qdrant       │   │
│   │   clips)    │  │   users)    │  │   jobs)   │  │  (embeddings) │   │
│   └────────────┘  └────────────┘  └──────────┘  └──────────────┘   │
│                                                                      │
│   ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│   │  CDN (CloudFront/Bunny)  │    │  Message Queue (SQS/BullMQ)  │   │
│   │  — clip & reel delivery  │    │  — async job orchestration    │   │
│   └──────────────────────────┘    └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### 3.1 Upload Pipeline

A typical 15–20 minute doubles game at 2.7K (30–45 Mbps bitrate) produces a **3–5 GB file**. Manageable, but still requires a robust upload strategy.

**Upload Flow:**

1. Client requests a pre-signed upload URL from the Upload Service
2. Client uploads directly to S3 using multipart upload (bypasses the API server)
3. S3 event triggers the ingestion pipeline via SQS
4. Upload Service tracks progress via WebSocket and updates the user in real-time

**Key Decisions:**

- **Resumable uploads** — use the tus protocol (tus.io) so users can resume interrupted uploads on spotty connections
- **Client-side compression** — optional pre-upload transcode to H.265 to reduce upload size by ~40%, with a fallback to server-side transcode
- **Chunk size** — 10 MB chunks for multipart upload, balancing reliability and throughput
- **Storage tiers** — originals go to S3 Standard, then lifecycle-policy to S3 Glacier after 30 days; processed clips stay in S3 Standard

### 3.2 Video Ingestion & Preprocessing

Once the raw video lands in S3, an ingestion worker picks it up.

**Pipeline Steps:**

1. **Transcode** — FFmpeg normalizes to H.264 1080p (working copy) + preserves 2.7K original for final clip extraction. GPU-accelerated via NVENC on EC2 g5 instances.
2. **Scene Detection** — PySceneDetect splits the video into logical segments (rallies, timeouts, between-point breaks).
3. **Frame Extraction** — Sample keyframes at 2 FPS for model inference (keeps GPU cost manageable while capturing all shot events).
4. **Audio Extraction** — Separate audio track for crowd/paddle-hit analysis.

**Infrastructure:** AWS Batch or Kubernetes Jobs with GPU nodes. Each video spawns a directed acyclic graph (DAG) of tasks managed by a lightweight orchestrator (Temporal.io or AWS Step Functions).

### 3.3 Player Identification Pipeline (Doubles-Aware)

In a doubles game there are 4 players on the court. The system needs to know *which one is the user* so it can generate highlights of their plays, not their opponents'.

#### Stage 1 — MVP: User Tap-to-Identify

After upload, the system detects all 4 players in a representative frame using YOLOv8 person detection and presents the frame to the user with bounding boxes around each player. The user taps on themselves. This "seed frame" anchors all downstream tracking.

**UX Flow:**

```
Upload completes → system picks a clear mid-game frame →
shows 4 bounding boxes → user taps "That's me" →
seed identity stored → processing begins
```

#### Stage 2 — In-Game Re-ID Tracking

Once the user taps on themselves, the system extracts an **appearance embedding** for that player and tracks them across the entire video.

- **Model:** FastReID or OSNet (lightweight Re-ID model optimized for sports)
- **Features extracted:** jersey/shirt color histogram, body proportions, gear (hat, sunglasses, paddle color), height relative to other players
- **Court-position heuristic:** In doubles, players stay on their side of the net and generally occupy left or right positions. When the Re-ID model confidence drops (e.g., during a scramble at the net), the system falls back to court-position logic: "Player A was on the near-left side and hasn't crossed to the far-right"
- **Output:** Per-frame player assignment — each detected person is labeled as `user`, `partner`, `opponent_1`, or `opponent_2`

**Handling edge cases:**

- **Side switches between games:** Detected via score reset + players crossing the net. Re-ID re-anchors positions.
- **Similar clothing:** If two players wear nearly identical outfits, the system flags low confidence and may ask the user for a second tap on a later frame.
- **Camera angle changes:** If the camera moves significantly (e.g., tripod knocked), scene-change detection triggers a Re-ID re-anchor.

#### Stage 3 — Cross-Video Persistent Profiles

After **3–5 uploads**, the system has enough appearance data to build a persistent player profile.

- **Player embedding store:** Each user gets a rolling average embedding vector stored in Qdrant, updated after each video
- **Auto-recognition flow:** On new upload, the system runs Re-ID against the user's stored embedding on the seed frame. If confidence > 0.85, skip the tap step entirely. If 0.6–0.85, show a confirmation: "Is this you?" with a single tap to confirm. Below 0.6, fall back to manual tap.
- **Partner/opponent recognition:** If the user's frequent doubles partner also uses PickleClips, the system can cross-reference: "You and @sarah were in the same game — want to share this reel with her?"

#### Stage 4 — Fully Automatic (No Interaction)

At maturity, the system recognizes the user with high confidence across different courts, lighting conditions, and outfits by combining appearance embeddings with contextual signals (upload device, usual playing time, frequent courts via GPS metadata).

#### How Player ID Feeds Into Highlights

Once the system knows which player is the user, the highlight scorer applies **role-aware weighting:**

```
if action.player == user:
    if action.scored_point:
        highlight_score *= 1.5    # user's winning shots are top priority
    if action.shot_type in [erne, ATP, smash]:
        highlight_score *= 1.3    # user's spectacular shots boosted
elif action.player == partner:
    highlight_score *= 0.7        # partner's plays are secondary
elif action.player in [opponent_1, opponent_2]:
    if action.scored_point:
        highlight_score *= 0.3    # opponent scoring = less interesting
    if action.rally_length > 8:
        highlight_score *= 0.8    # but long rallies are still exciting
```

### 3.4 AI/ML Models — The Highlight Brain

This is the core IP of the platform. Multiple specialized models feed into a unified highlight scorer.

#### Model 1: Court & Ball Detection

- **What:** Detects court boundaries, ball position, and ball trajectory per frame
- **Architecture:** YOLOv8 fine-tuned on pickleball court/ball datasets
- **Output:** Ball position (x, y), court keypoints, ball speed estimate
- **Why it matters:** Ball trajectory is essential for classifying shot types and determining if a ball is in/out

#### Model 2: Player Pose Estimation

- **What:** Tracks player body positions and paddle movement
- **Architecture:** MediaPipe Pose or RTMPose (lightweight, runs at 30+ FPS)
- **Output:** 33 body keypoints per player per frame, paddle angle and swing velocity
- **Why it matters:** Swing quality, footwork, and positioning are what separate a "good" shot from a routine one

#### Model 3: Score Detection (OCR)

- **What:** Reads the scoreboard overlay (if present) or infers score from game flow
- **Architecture:** PaddleOCR for scoreboard text + rule-based state machine for score tracking
- **Output:** Current score, serving team, point attribution
- **Why it matters:** "Scored a point" is a primary highlight trigger

#### Model 4: Shot Classifier

- **What:** Classifies each shot by type and quality
- **Architecture:** Temporal CNN or Video Transformer (TimeSformer) operating on 2-second clips
- **Inputs:** Ball trajectory + pose keypoints + raw frames
- **Output:** Shot type (drive, dink, lob, erne, ATP, drop, smash) + quality score (0–1)
- **Why it matters:** An erne or ATP is almost always a highlight; a routine dink usually isn't

#### Model 5: Rally Tracker

- **What:** Segments continuous play into individual rallies and tracks rally length/intensity
- **Architecture:** Rule-based on ball detection + pose data, with a lightweight LSTM for edge cases
- **Output:** Rally start/end timestamps, rally length, intensity score
- **Why it matters:** Long rallies with escalating intensity are inherently exciting

#### Model 6: Audio Analyzer

- **What:** Detects crowd reactions, paddle impact sounds, and player celebrations
- **Architecture:** Audio classification model (PANNs or YAMNet) fine-tuned on pickleball audio
- **Output:** Crowd excitement score, paddle-hit intensity, verbal reaction timestamps
- **Why it matters:** Crowd eruptions and "ohhh!" moments are strong highlight signals even when the visual model is uncertain

### 3.5 Highlight Scorer — Personalized Ranking

All model outputs converge into the Highlight Scorer, which ranks every moment in the video.

**Scoring Formula (base):**

```
highlight_score = (
    w1 × shot_quality +
    w2 × shot_rarity +
    w3 × point_scored +
    w4 × rally_intensity +
    w5 × audio_excitement +
    w6 × player_reaction
)
```

**Personalization Layer:**

- Each user has a preference vector stored in the vector DB
- On first upload, use default weights (biased toward points scored and rare shots)
- After user feedback (thumbs up/down on clips, manual clip edits), update weights via online learning
- Over time the system learns: "This user loves dink battles more than power smashes" or "This user only wants their own winning shots"

**Feedback Loop:**

```
User watches clips → thumbs up/down or edits → feedback stored →
preference model updated → next video uses updated weights
```

### 3.6 Sub-Highlights, Output Types & Reel Generation

#### Sub-Highlight Classification

Every detected moment in a game falls into one of two sub-highlight categories. These are the atomic building blocks that get assembled into the various output types.

**Type A — Shot Form (technique-based):**
A single shot that demonstrates good or exceptional technique, regardless of whether the rally was won. Detected by combining pose estimation (swing mechanics) with ball trajectory (speed, placement, spin).

Examples: clean forehand drive, well-placed backhand, controlled dink, erne, ATP (around-the-post), speed-up, drop shot from transition zone, overhead smash, lob winner.

**Type B — Point Scored (outcome-based):**
Any rally where the user's team wins the point. The full rally (serve to point conclusion) is captured, not just the final shot. Detected by combining score OCR with rally tracking.

These two types overlap — a rally can contain great shot form AND end in a scored point, making it a high-value highlight that ranks well in multiple output types.

#### Output Taxonomy

The Reel Service can generate **11 distinct output types** from the same analyzed game. Each uses a different filter and assembly strategy on the same pool of sub-highlights.

**Highlight Outputs (Auto-Generated After Every Game)**

| # | Output Type | Description | Source Clips | Typical Duration |
|---|---|---|---|---|
| 1 | **Highlight Montage** | Best-of compilation mixing both sub-highlight types — the flagship reel. Top N moments by highlight score, interleaved for pacing. | Top Type A + Type B, ranked by score | 30–90 sec |
| 2 | **Scored Point Rally** | A single full rally shown end-to-end, from serve to point won. One output per scored point (user selects which to share). | Type B only, full rally | 10–45 sec each |
| 3 | **Best Shots Reel** | Pure shot form compilation — cleanest drives, nastiest dinks, ATPs. No score context needed, just technique on display. | Type A only, ranked by shot_quality | 20–60 sec |
| 4 | **Full Rally Replay** | Any long or intense rally regardless of who won the point. Great for showing endurance, consistency, and competitive play. | Rallies where rally_length > 8 OR rally_intensity > 0.7 | 15–60 sec each |
| 5 | **Single Shot Clip** | One isolated shot trimmed tight with optional slow-mo. Perfect for stories or single-post sharing. | Any Type A clip, individually | 3–8 sec |
| 6 | **Game Recap** | Condensed version of the entire game showing only score-changing moments. Like a sports broadcast highlight package. | All Type B, chronological, 1 clip per point | 2–5 min |

**Player-Specific Outputs**

| # | Output Type | Description | Source Clips | Typical Duration |
|---|---|---|---|---|
| 7 | **My Best Plays** | Everything attributed to the user only (not partner), filtered by highlight score threshold. "This is what *I* did today." | Type A + B where attributed_player == user | 30–90 sec |
| 8 | **Partnership Highlights** | Plays involving both user and partner — setup + finish combos, great team defense, coordinated net play. | Rallies where both user and partner contribute key shots | 20–60 sec |

**Fun / Social Outputs**

| # | Output Type | Description | Source Clips | Typical Duration |
|---|---|---|---|---|
| 9 | **Fails & Bloopers** | The inverse of highlights: mishits, net clips, whiffed shots, own errors. Players love sharing these. | Clips where shot_quality < 0.2 AND attributed_player == user | 15–45 sec |
| 10 | **Side-by-Side Progress** | Same shot type compared across weeks/months showing improvement. "Your forehand drive in January vs. now." | Type A clips matched by shot_type across multiple videos | 10–30 sec |

**Comeback Reel (Conditional)**

| # | Output Type | Description | Source Clips | Typical Duration |
|---|---|---|---|---|
| 11 | **Comeback Reel** | Auto-detected when user's team was down significantly and rallied back. Pulls the turning-point moments with score overlay showing the deficit shrinking. | Type B from the comeback arc, with score context overlays | 30–90 sec |

#### Output Generation Logic

```
After highlight scoring completes:

1. ALWAYS auto-generate:
   ├── Highlight Montage (top 8–12 moments)
   ├── Game Recap (all scored points, chronological)
   └── My Best Plays (user-only, top 6–10 moments)

2. GENERATE IF conditions met:
   ├── Comeback Reel → only if score deficit ≥ 4 points then won
   ├── Partnership Highlights → only if partner actions detected
   ├── Fails & Bloopers → only if ≥ 3 low-quality user shots found
   └── Side-by-Side Progress → only if ≥ 2 prior games with same shot type

3. AVAILABLE on-demand (user taps to generate):
   ├── Scored Point Rally (per individual rally)
   ├── Full Rally Replay (per individual rally)
   ├── Single Shot Clip (per individual shot)
   └── Best Shots Reel (custom shot type filter)
```

#### Reel Assembly Pipeline (FFmpeg + MoviePy)

Each output type has its own assembly profile:

| Output Type | Clip Ordering | Transitions | Slow-Mo | Music | Text Overlay |
|---|---|---|---|---|---|
| Highlight Montage | By score (best first) | Cross-dissolve + whip-pan | 0.5x on peak moments | Upbeat, beat-synced | Shot type + score |
| Scored Point Rally | Chronological (single rally) | None (continuous) | 0.5x on winning shot only | Optional ambient | Score throughout |
| Best Shots Reel | By shot quality | Quick cuts (0.2s) | 0.3x on contact frame | High energy | Shot type label |
| Full Rally Replay | Chronological (single rally) | None (continuous) | None (real-time) | None (keep game audio) | Rally length counter |
| Single Shot Clip | N/A (one clip) | Fade in/out | 0.5x with ramp | None | Shot type + speed |
| Game Recap | Chronological | Cross-dissolve (0.3s) | 0.5x on final shot per rally | Background music | Running score |
| My Best Plays | By score | Cross-dissolve | 0.5x on peak | Upbeat | Shot type + "MY PLAY" |
| Partnership Highlights | Chronological | Cross-dissolve | 0.5x on combos | Upbeat | "TEAM PLAY" label |
| Fails & Bloopers | Chronological | Jump cuts | 1x (real speed is funnier) | Comedy/lighthearted | Emoji overlays |
| Side-by-Side | Split screen | Sync on contact frame | 0.5x synchronized | Motivational | Date labels + improvement % |
| Comeback Reel | Chronological (comeback arc) | Cross-dissolve + flash | 0.5x on turning point | Building intensity | Score deficit counter |

**Export formats for all types:** 9:16 vertical (TikTok/Reels/Shorts), 16:9 horizontal (YouTube), 1:1 square (Instagram feed)

### 3.7 User-Facing Features

**Dashboard:**
- Upload queue with real-time progress
- Video library with AI-detected highlights timeline
- **Output gallery** — after processing, shows auto-generated reels (Highlight Montage, Game Recap, My Best Plays) plus conditional reels (Comeback, Bloopers) if detected
- **Rally browser** — scroll through all rallies with score context, tap to generate Scored Point Rally or Full Rally Replay
- **Shot browser** — all detected shots by type, tap any for a Single Shot Clip
- Per-clip controls: trim, keep, discard, adjust slow-mo, change crop
- Reel editor: drag-and-drop clips, reorder, change music, add text
- **Progress tracker** — Side-by-Side Progress view across games for each shot type
- Sharing: direct post to Instagram/TikTok (via their APIs) or download

**Notification System:**
- WebSocket for real-time processing status
- Push notification (mobile) when reel is ready
- Email digest of weekly highlight stats

---

## 4. Data Model (Simplified)

```
users
├── id (UUID)
├── email
├── display_name
├── avatar_url
├── highlight_preferences (JSONB — personalized weights)
├── subscription_tier (free | pro | team)
└── created_at

player_profiles
├── id (UUID)
├── user_id → users.id
├── appearance_embedding (VECTOR — rolling avg from Re-ID model)
├── embedding_confidence (float — improves with more uploads)
├── uploads_contributing (int — count of videos used to build profile)
├── metadata (JSONB — shirt colors seen, gear, height estimate)
└── updated_at

videos
├── id (UUID)
├── user_id → users.id
├── s3_key_original
├── s3_key_processed
├── duration_seconds
├── resolution
├── status (uploading | identifying | processing | analyzed | failed)
├── metadata (JSONB — fps, codec, file_size)
└── uploaded_at

video_players
├── id (UUID)
├── video_id → videos.id
├── role (user | partner | opponent_1 | opponent_2)
├── player_profile_id → player_profiles.id (nullable — linked if recognized)
├── seed_frame_bbox (JSONB — {x, y, w, h} from user tap or auto-detect)
├── appearance_embedding (VECTOR — per-video snapshot)
├── tracking_confidence (float — avg Re-ID confidence across video)
└── created_at

highlights
├── id (UUID)
├── video_id → videos.id
├── attributed_player_role (user | partner | opponent_1 | opponent_2)
├── sub_highlight_type (shot_form | point_scored | both)
├── start_time_ms
├── end_time_ms
├── highlight_score (float)
├── highlight_score_raw (float — before role-aware weighting)
├── shot_type (enum — drive, dink, lob, erne, atp, drop, smash, overhead, speed_up)
├── shot_quality (float — 0 to 1, used for best shots AND fails detection)
├── point_scored (boolean)
├── point_won_by (user_team | opponent_team | null)
├── rally_id (UUID — groups shots belonging to the same rally)
├── rally_length (int — number of shots in the rally)
├── rally_intensity (float)
├── model_outputs (JSONB — full breakdown)
├── user_feedback (liked | disliked | null)
└── created_at

rallies
├── id (UUID)
├── video_id → videos.id
├── start_time_ms
├── end_time_ms
├── shot_count (int)
├── intensity_score (float)
├── point_won_by (user_team | opponent_team | null)
├── score_before (JSONB — {user_team: 5, opponent_team: 3, server: "user_team"})
├── score_after (JSONB)
├── is_comeback_point (boolean — part of a deficit recovery arc)
└── created_at

reels
├── id (UUID)
├── user_id → users.id
├── video_id → videos.id (nullable — null for cross-video outputs like side-by-side)
├── output_type (enum — see output taxonomy below)
├── s3_key
├── format (vertical | horizontal | square)
├── duration_seconds
├── clip_ids (UUID[])
├── rally_ids (UUID[] — for rally-based outputs)
├── assembly_profile (JSONB — transitions, slow_mo, music, overlays used)
├── music_track_id
├── status (queued | generating | ready | failed)
├── auto_generated (boolean — true if system created it, false if user requested)
├── share_url
└── created_at

-- output_type enum values:
-- highlight_montage, scored_point_rally, best_shots_reel,
-- full_rally_replay, single_shot_clip, game_recap,
-- my_best_plays, partnership_highlights,
-- fails_and_bloopers, side_by_side_progress, comeback_reel

clip_edits
├── id (UUID)
├── highlight_id → highlights.id
├── user_id → users.id
├── trim_start_ms
├── trim_end_ms
├── slow_mo_factor
├── crop_override (JSONB)
└── updated_at
```

---

## 5. Infrastructure & Cost Architecture

### Compute Tiers

| Component | Instance Type | Scaling Strategy |
|---|---|---|
| API Services | AWS Fargate / ECS | Auto-scale on request count |
| Video Transcode | EC2 g5.xlarge (GPU) | Spot instances, queue-based |
| ML Inference | EC2 g5.2xlarge or SageMaker | Batch inference, scale to zero |
| Reel Generation | EC2 c6i.2xlarge (CPU) | Queue-based, spot instances |

### Storage Cost Estimates (per 1,000 users uploading 4 games/month, ~4 GB per game)

| Storage Type | Monthly Volume | Est. Cost/Month |
|---|---|---|
| S3 Standard (clips, reels) | ~500 GB | ~$12 |
| S3 Glacier (originals after 30d) | ~16 TB cumulative | ~$7 |
| CDN egress (viewing/sharing) | ~3 TB | ~$90 |
| GPU compute (transcode + inference) | ~800 GPU-hours | ~$640 (spot) |

**Per-video processing cost: ~$0.16** — enables a generous free tier.

### Key Cost Optimizations

- **Spot instances** for all async GPU work (70% cost reduction)
- **Scale-to-zero** inference endpoints when no jobs are queued
- **Tiered processing** — free users get 720p reels, pro users get full 2.7K
- **Aggressive caching** — CDN caches all generated reels and clips
- **Lifecycle policies** — auto-archive originals, auto-delete abandoned uploads

---

## 6. Tech Stack Summary

| Layer | Technology |
|---|---|
| Web Frontend | Next.js 14, TypeScript, Tailwind, Zustand |
| Mobile App | React Native (Expo), shared business logic |
| API | **Python (FastAPI)** — async, unified with ML stack |
| Auth | Clerk or Auth0 (social login + magic links) |
| Database | PostgreSQL (Supabase or RDS) + pgvector for embeddings |
| Cache / Queue | Redis + BullMQ |
| Object Storage | AWS S3 (or Cloudflare R2 for egress savings) |
| CDN | CloudFront or Bunny CDN |
| Video Processing | FFmpeg, MoviePy |
| ML Framework | PyTorch, Ultralytics (YOLO), MediaPipe |
| Player Re-ID | FastReID / OSNet (player tracking + persistent profiles) |
| ML Serving | AWS SageMaker or Triton Inference Server |
| Orchestration | Temporal.io (workflow engine for video pipeline) |
| Vector DB | Qdrant (user preference + player appearance embeddings) |
| Monitoring | Datadog or Grafana + Prometheus |
| CI/CD | GitHub Actions → AWS ECR → ECS/K8s |

---

## 7. Development Phases

### Phase 1 — MVP (Weeks 1–8)

**Goal:** Upload → identify yourself → auto-detect highlights → download clips

- Web app with upload and video library
- Basic transcode pipeline (FFmpeg)
- **Player tap-to-identify** — post-upload frame with bounding boxes, user taps on themselves
- Court + ball detection model (YOLOv8 fine-tuned)
- Person detection (YOLOv8) to track all 4 players
- Simple highlight scoring (ball speed + point-scored signals, filtered to user's plays only)
- Clip extraction at detected timestamps
- Manual download of individual clips

### Phase 2 — Smart Reels + Player Tracking (Weeks 9–14)

**Goal:** Auto-generated reels with music, transitions, and robust player tracking

- **Re-ID tracking model** — track the identified user across the full video using appearance embeddings
- **Court-position fallback** — when Re-ID confidence drops, use doubles positioning logic
- Shot classifier model (drive, dink, smash, etc.)
- Role-aware highlight scoring (boost user's plays, deprioritize opponent's)
- Reel assembly pipeline with transitions and slow-mo
- Music library integration with beat-sync
- Text overlays (shot type, score)
- Share links with OG preview cards
- User feedback system (thumbs up/down on clips)

### Phase 3 — Output Controls: ZIP Download + Optional Reel Generation (Weeks 15–17)

**Goal:** Give users full control over their output — download raw clips or opt into reel generation on demand

#### 3.a — ZIP Clip Download

Users can download all extracted clips as a single ZIP archive, organized into subfolders by shot type (e.g. `drive/`, `dink/`, `erne/`, `smash/`). No reel generation required — the raw clips are immediately useful for manual editing, social posting, or archiving.

- `GET /api/v1/videos/{video_id}/clips/download-zip` — streams a ZIP with category subfolders
- Available as soon as the video reaches `analyzed` status
- Authenticated, per-user download only
- Frontend: "Download ZIP" button on the video detail page

#### 3.b — On-Demand Reel Generation

Reel generation (previously auto-triggered at the end of Phase 2 pipeline) is now fully optional and user-initiated. After analysis completes, users choose whether to generate reels.

- `POST /api/v1/videos/{video_id}/generate-reels` — triggers the standard auto-generated reel set
- Idempotent: skips reel types already queued or generated for the video
- Users can also request individual reel types via `POST /api/v1/reels` (unchanged)
- Frontend: "Generate Reels" button on the video detail page, separate from ZIP download

#### Output Decision Flow (Updated)

```
Analysis complete (video status → 'analyzed')
    │
    ├── Option A: Download ZIP of raw clips
    │   └── GET /videos/{id}/clips/download-zip
    │       → clips/<shot_type>/<clip_id>.mp4 in ZIP archive
    │
    └── Option B: Generate Reels (on-demand)
        └── POST /videos/{id}/generate-reels
            → queues: highlight_montage, my_best_plays,
                      game_recap, points_of_improvement
            → user can also request individual types via POST /reels
```

Both options are independent — users can download the ZIP, generate reels, or do both.

### Phase 4 — Personalization + Auto-Recognition (Weeks 18–24)

**Goal:** AI learns your highlight preferences and recognizes you automatically

- **Persistent player profiles** — after 3–5 uploads, build rolling appearance embedding
- **Auto-recognition** — skip the tap step when confidence > 0.85, one-tap confirm at 0.6–0.85
- **Partner/opponent linking** — "You played with @sarah — share this reel?"
- Player pose estimation model
- Audio analysis model
- Personalized highlight scoring with online learning
- Smart vertical cropping (center on the user using tracked position)
- Reel templates (e.g., "Best of the Month", "Tournament Recap")

### Phase 5 — Mobile + Social (Weeks 25–32)

**Goal:** Full mobile app + direct social posting

- React Native app with camera upload
- Direct posting to Instagram Reels, TikTok, YouTube Shorts
- Player tagging and team features
- Public profile pages with highlight reels
- Push notifications when reels are ready

### Phase 6 — Advanced (Weeks 33+)

**Goal:** Community and advanced AI features

- Multi-camera angle support
- Live game analysis (real-time highlight detection)
- Player performance analytics dashboard
- Community features (follow players, highlight feeds)
- Coaching tools (swing analysis, positioning heatmaps)

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Upload size (~4 GB per game) | Slow on weak connections | Resumable uploads (tus), optional client-side compress |
| ML model accuracy on varied courts | Bad highlight detection | Start with controlled dataset, add user feedback loop |
| GPU costs spiral with growth | Margin erosion | Spot instances, batching, tiered processing by plan |
| Video processing takes too long | Poor UX | Target 8–12 min per game; show progressive results |
| Music licensing complexity | Legal risk | Use royalty-free library only (Epidemic Sound API or similar) |
| Mobile app camera quality varies | Inconsistent detection | Minimum quality gate + guidance overlay during recording |
| Player Re-ID fails (similar outfits) | Wrong player's highlights | Court-position fallback + ask user for second tap when confidence is low |
| 4-player tracking confusion at net | Swapped player IDs mid-rally | Use paddle-side dominance + Re-ID re-anchor after each point break |

---

## 9. Security & Privacy

- All videos encrypted at rest (S3 SSE-S3) and in transit (TLS 1.3)
- Pre-signed URLs expire after 1 hour
- Users can delete all their data (GDPR-compliant cascade delete)
- No video content is used for model training without explicit opt-in consent
- Face detection model runs locally to offer face-blur option for non-consenting players
- API rate limiting per user tier to prevent abuse

### Player Identification Privacy

- **Appearance embeddings are not facial recognition** — the Re-ID model uses body shape, clothing, and gear features, not biometric face data. This avoids most facial recognition regulations (BIPA, GDPR Art. 9).
- **Embeddings are stored per-user, not globally** — the system only matches a player against their *own* stored profile, never against a global database of all users. There is no "who is this stranger?" capability.
- **Opponent data is ephemeral** — opponent appearance embeddings extracted during a game are used only for in-video tracking and are discarded after processing. They are never stored persistently or linked to any identity.
- **Partner linking is opt-in only** — cross-referencing between users ("You played with @sarah") requires both users to have accounts and to have enabled partner discovery in their settings.
- **User can delete their player profile** — deleting the profile purges all stored appearance embeddings and resets the system to tap-to-identify mode.

---

## 10. Processing Timeline (Per Game)

Based on a typical 15–20 minute doubles game at 2.7K:

```
┌─────────────────────────────────────────────────────────────────┐
│  0:00 ─── Upload complete, job enters queue                     │
│                                                                  │
│  0:00–1:30  Transcode to 1080p working copy (GPU, NVENC)        │
│  1:30–2:00  Scene detection + frame extraction (2,400 frames)   │
│  2:00–2:30  Person detection — locate all 4 players             │
│                                                                  │
│  ── PAUSE: Wait for user tap (or auto-recognize) ──             │
│  (If auto-recognized, no pause — continues immediately)         │
│                                                                  │
│  2:30–4:00  Re-ID tracking — label user across all frames       │
│  4:00–6:00  Court/ball detection (YOLOv8, batch inference)      │
│  6:00–7:30  Pose estimation (MediaPipe, all 4 players)          │
│  7:30–8:00  Score detection (OCR on scoreboard frames)          │
│  8:00–9:00  Shot classification + rally tracking                │
│  9:00–9:30  Audio analysis (crowd/paddle)                       │
│  9:30–10:00 Highlight scoring + role-aware weighting            │
│ 10:00–11:00 Clip extraction from 2.7K original                  │
│ 11:00–12:00 Reel assembly (transitions, slow-mo, music)         │
│                                                                  │
│  ~12:00 ─── Reel ready, push notification sent                  │
└─────────────────────────────────────────────────────────────────┘
```

**Total: ~8–12 minutes** (excluding user tap wait time). With auto-recognition enabled, the entire pipeline runs unattended. The user can upload a game and have their highlight reel by the time they finish their next game.

**Progressive delivery:** The system can surface individual highlight clips as they're detected (starting around the 10-minute mark) before the full reel is assembled, so users see results faster.

---

## 11. API Design (Key Endpoints)

### Upload & Video Management

```
POST   /api/v1/videos/upload-url       → returns pre-signed S3 URL
POST   /api/v1/videos/{id}/confirm     → triggers ingestion pipeline
GET    /api/v1/videos                   → list user's videos
GET    /api/v1/videos/{id}              → video details + processing status
DELETE /api/v1/videos/{id}              → delete video + all derived data
```

### Player Identification

```
GET    /api/v1/videos/{id}/identify     → returns frame image with 4 bounding boxes
POST   /api/v1/videos/{id}/identify     → { bbox_index: 2 } — user taps player #2
GET    /api/v1/player-profile           → user's persistent player profile
DELETE /api/v1/player-profile           → purge appearance embeddings, reset to tap mode
PUT    /api/v1/player-profile/settings  → { auto_recognize: true, partner_discovery: true }
```

### Highlights & Reels

```
GET    /api/v1/videos/{id}/highlights          → ranked list of detected highlights
GET    /api/v1/videos/{id}/rallies             → all rallies with scores and intensity
PATCH  /api/v1/highlights/{id}                 → update feedback (liked/disliked) or trim
GET    /api/v1/videos/{id}/reels               → all generated reels for this video
POST   /api/v1/reels                           → generate reel (body: { video_id, output_type, format, clip_ids? })
GET    /api/v1/reels/{id}                      → reel details + download/share URLs
POST   /api/v1/reels/{id}/share                → generate share link with OG metadata
GET    /api/v1/reels/{id}/export/{format}      → download in specific format (vertical/horizontal/square)
POST   /api/v1/reels/side-by-side              → cross-video comparison (body: { shot_type, video_ids[] })
```

**Output type values for POST /api/v1/reels:**

```
highlight_montage        — auto: yes  — best-of compilation
scored_point_rally       — auto: no   — single rally, requires rally_id
best_shots_reel          — auto: no   — user taps to generate
full_rally_replay        — auto: no   — single rally, requires rally_id
single_shot_clip         — auto: no   — requires highlight_id
game_recap               — auto: yes  — all scored points chronological
my_best_plays            — auto: yes  — user-only highlights
partnership_highlights   — auto: cond — only if partner detected
fails_and_bloopers       — auto: cond — only if ≥3 low-quality user shots
comeback_reel            — auto: cond — only if deficit ≥4 then won
side_by_side_progress    — auto: cond — only if ≥2 prior games with same shot type
```

### Real-Time Updates

```
WS     /api/v1/ws/processing/{video_id}        → live status updates during pipeline
```

**WebSocket message types:**

```json
{ "event": "stage_complete", "stage": "transcode", "progress": 0.15 }
{ "event": "identify_ready", "frame_url": "...", "bboxes": [...] }
{ "event": "highlight_found", "highlight_id": "...", "preview_url": "..." }
{ "event": "reel_ready", "reel_id": "...", "download_url": "..." }
```

---

## 12. Monitoring & Observability

### Key Metrics to Track

**Pipeline Health:**
- Videos processed per hour
- Average processing time per game (target: <12 min)
- Pipeline failure rate by stage
- Queue depth and wait time

**ML Model Performance:**
- Re-ID tracking accuracy (measured via user corrections — "that's not me" taps)
- Auto-recognition success rate (target: >90% after 5 uploads)
- Highlight precision — % of surfaced highlights that users keep vs. discard
- Shot classification accuracy (validated against user feedback)

**User Engagement:**
- Upload-to-first-view time
- Highlight keep/discard ratio
- Reel share rate
- Return upload rate (users who upload a second game)

**Infrastructure:**
- GPU utilization and spot instance interruption rate
- S3 storage growth trajectory
- CDN cache hit ratio
- API latency p50/p95/p99

### Alerting Thresholds

```
Pipeline failure rate > 5%          → page on-call
Processing time > 20 min            → warning
Re-ID accuracy < 80%                → investigate model drift
Queue depth > 100 videos            → auto-scale GPU fleet
Spot interruption rate > 30%        → fall back to on-demand
```

---

*Document version: 3.0 — March 2026 (updated: sub-highlight types, 11 output types with assembly profiles, rallies table, expanded API and dashboard)*
