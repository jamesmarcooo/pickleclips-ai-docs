# PickleClips — Refined Architecture Design

**Date:** 2026-04-05
**Version:** 4.0 (refined from v3.0)
**Context:** Solo developer, personal project first, backend/infra background, no ML experience

---

## 1. Product Vision

PickleClips lets pickleball players upload full game videos (up to 2.7K resolution), then uses AI to automatically identify highlights and lowlights — winning shots, rallies, key plays, and errors worth improving — and generates shareable reels optimized for Instagram, TikTok, and YouTube Shorts.

### Core User Flow

```
Upload Video → Player Identification → AI Analysis → Highlight + Lowlight Detection → Clip Extraction → Reel Assembly → Share/Download
```

### Key Features (scoped to personal use first)

- Upload full-length doubles game videos (2.7K, ~15–20 minutes per game)
- **Player identification** — tap on yourself once, AI tracks you for the rest of the game
- AI-powered highlight detection using pretrained models
- **Lowlights / Points of Improvement** — coaching-oriented view of errors and weak plays
- Auto-generated reels with transitions, slow-mo, and music
- Manual clip trimming and reel editing
- Download and share to social platforms
- Web app (launch) + mobile app (Phase 4 only)

---

## 2. Architecture

### Guiding Principle

This is a solo-developer personal project. Every technology choice optimizes for **operational simplicity** and **free tier maximization** first, with a clear upgrade path if the project grows into a real product.

Removed from v3.0: BullMQ (Node.js, wrong runtime), Temporal.io (too heavy to operate solo), AWS Step Functions, API Gateway/Kong, Qdrant, Pinecone, SageMaker, Triton, 8 microservices, Auth0/Clerk, CloudFront/Bunny CDN, Datadog/Grafana+Prometheus, React Native (Phase 4 only).

### High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT LAYER                       │
│   ┌─────────────────────┐   ┌─────────────────────┐ │
│   │   Web App (Next.js) │   │  Share Widget (OG)  │ │
│   └──────────┬──────────┘   └──────────┬──────────┘ │
└──────────────┼──────────────────────────┼────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────┐
│         FastAPI Monolith (single deployment)         │
│                                                      │
│  routers/          workers/          services/       │
│  ├── videos.py     ├── ingest.py     ├── storage.py  │
│  ├── highlights.py ├── player_id.py  ├── reel.py     │
│  ├── reels.py      ├── ml_pipeline.py└── notify.py  │
│  └── auth.py       └── reel_gen.py                  │
│                                                      │
│  Caddy (TLS termination + reverse proxy)             │
└─────────────┬───────────────────┬───────────────────┘
              │                   │
      ┌───────▼──────┐    ┌───────▼──────┐
      │   Supabase   │    │    Redis      │
      │  PostgreSQL  │    │  (Upstash,   │
      │  + pgvector  │    │  job queue + │
      │  + Auth      │    │  cache)      │
      │  + Realtime  │    └───────┬──────┘
      └──────────────┘            │
                          ┌───────▼──────┐
┌─────────────────────────│    Celery    │──────────────┐
│  GPU PROCESSING LAYER   │   Workers   │               │
│                         └─────────────┘               │
│  EC2 g5.xlarge spot (spins up per job, shuts down)    │
│  ├── FFmpeg (transcode + clip extraction)             │
│  ├── YOLOv8n (person detection)                       │
│  ├── OSNet/torchreid (Re-ID tracking)                 │
│  ├── TrackNetV2 (ball detection — Phase 2)            │
│  ├── MediaPipe Pose (Phase 2)                         │
│  └── FFmpeg + MoviePy (reel assembly)                 │
└───────────────────────┬───────────────────────────────┘
                        │
              ┌─────────▼─────────┐
              │  Cloudflare R2    │
              │  raw videos       │
              │  processed clips  │
              │  generated reels  │
              └───────────────────┘
```

### Tech Stack — All Decisions Resolved

| Layer | Technology | Reason |
|---|---|---|
| Web Frontend | Next.js 14 + TypeScript + Tailwind | Keep as-is |
| Frontend Hosting | Vercel (free for personal projects) | No EC2 needed for frontend |
| API | FastAPI monolith | 8 microservices → impossible solo; same code structure, zero infra overhead |
| Auth | Supabase Auth | Free, built-in to DB, magic links + social login |
| Database | Supabase PostgreSQL + pgvector | Managed, free tier, pgvector replaces Qdrant + Pinecone entirely |
| Job Queue | Celery + Redis (Upstash) | Python-native; BullMQ removed (Node.js); Temporal removed (heavy) |
| Video Storage | Cloudflare R2 | Free egress; S3-compatible API; built-in CDN |
| App Hosting | EC2 t3.micro + Caddy | Free tier (year 1), Caddy handles TLS automatically |
| GPU Compute | EC2 g5.xlarge spot | Spin up per job, shut down when idle (~$0.15/game) |
| ML Serving | Direct on GPU instance | No SageMaker or Triton — run inference directly, batch jobs only |
| Monitoring | Sentry (free tier) + structured logs | Not Datadog ($$$) or self-hosted Grafana |
| CI/CD | GitHub Actions → EC2 | Simple deploy to single EC2 instance |

### Local Development

All services run locally via Docker Compose. GPU work runs on CPU locally (slower, but functional for pipeline logic testing). For full-speed ML testing, spin up a g5.xlarge spot instance on-demand.

```
Local (fast iteration)     → API, frontend, DB, job queue logic
Dev EC2 spot (on-demand)   → full ML pipeline at real speed
Production EC2             → your actual games
```

### Free Tier Map

| Service | Free Tier | Notes |
|---|---|---|
| Supabase | 500MB DB, 50k MAU, Realtime included | Store metadata only; videos go to R2 |
| Cloudflare R2 | 10GB storage, free egress always | Delete originals after processing |
| EC2 t3.micro | 750 hrs/month (12 months) | App server; t3.medium after free year (~$30/mo) |
| EC2 g5.xlarge spot | ~$0.36–0.60/hr | ~$0.15/game; only pay when processing |
| Upstash Redis | 10k commands/day, 256MB | Fine for personal use |
| Sentry | 5k errors/month | Never hit for personal use |
| GitHub Actions | 2,000 min/month | ~50–100 deploys |
| Vercel | Unlimited personal, 100GB bandwidth | Free forever for personal projects |

**Estimated monthly cost (personal use): ~$5–20/month**

---

## 3. ML Pipeline Strategy

### Principle

Run pretrained models first. Fine-tune only when real footage proves a model fails at a specific task. All 6 models are sequenced by build order — do not build them all at once.

### Score Detection

Videos have **no scoreboard overlay** (raw camera footage). PaddleOCR is not used. The primary score detection approach is a **rule-based state machine**:

- **Phase 1 (no ball detection):** Detect rally end via frame-level motion analysis — when overall pixel motion drops below a threshold for >1 second, a rally has ended. Point attribution inferred from court-side of last high-motion event.
- **Phase 2+ (with TrackNetV2):** Detect rally end when ball leaves frame or stops moving. Infer point attribution from which player's side ball last landed, or who hit out. More accurate.
- Track score progression: state machine updates score after each rally end
- `score_source` field records `'rule_based'` on all highlights

### Model Sequence

| # | Model | Purpose | Pretrained? | Phase |
|---|---|---|---|---|
| 1 | YOLOv8n | Person detection (4 players, bounding boxes for tap-to-identify) | Yes — works well out of box | 1 |
| 2 | OSNet (torchreid) | Re-ID tracking — identify user across all frames | Yes — ~75–85%, court-position fallback handles gaps | 1 |
| 3 | Rule-based state machine | Score/rally detection (no OCR) | N/A | 1 |
| 4 | TrackNetV2 | Ball detection + trajectory | Partial — may need fine-tune on pickleball footage | 2 |
| 5 | MediaPipe Pose | Body keypoints + swing mechanics | Yes — works well, Google-maintained | 2 |
| 6 | YAMNet | Audio analysis (crowd, paddle impact) | Partial — defer to Phase 3 | 3 |

### Shot Classifier

No pretrained model exists for pickleball shot types. Approach:

- **Phase 1–2:** Rule-based classifier derived from existing model outputs:
  - `erne` = player crosses centerline during swing
  - `smash` = paddle above head + ball above net height
  - `lob` = ball trajectory steeply upward
  - `dink` = soft contact + low ball speed near net
  - `drive` = high ball speed, everything else
- **Phase 3:** Train lightweight VideoMAE/MoViNet classifier on own footage. Use rule-based labels as weak supervision. ~500 labeled clips → ~80%+ accuracy on personal footage. Training time: ~2 hours on g5.xlarge (~$1).

### Fine-Tuning Guide (when needed)

**Ball detection (most likely to need fine-tuning):**
1. Export 200–400 frames from own footage
2. Label ball positions with CVAT (free, cloud or self-hosted)
3. Train YOLOv8 for 50 epochs on g5.xlarge (~$1 GPU cost)
4. Evaluate on held-out frames from a different game

**Re-ID tracking (if identity swaps are frequent):**
1. Export player crops from own games (100–200 crops of yourself)
2. Fine-tune OSNet on your appearance specifically
3. This pays off the most — you're matching against yourself specifically

**Shot classifier (Phase 3):**
1. Auto-label clips using rule-based classifier
2. Manually correct ~20% of labels
3. Train VideoMAE on labeled 2-second clips
4. Evaluate precision/recall per shot type

---

## 4. Player Identification Pipeline

Unchanged from v3.0 — this design is solid.

### Stage 1 — Tap to Identify (MVP)

After upload, YOLOv8n detects all 4 players in a representative mid-game frame. User is shown the frame with 4 bounding boxes and taps on themselves. This seed identity anchors all downstream tracking.

### Stage 2 — Re-ID Tracking

OSNet extracts an appearance embedding from the tapped crop and tracks the user across all frames using cosine similarity. Court-position heuristic (doubles positioning logic) acts as fallback when Re-ID confidence drops.

Output: per-frame player labels — `user`, `partner`, `opponent_1`, `opponent_2`.

### Stage 3 — Persistent Profiles (Phase 3)

After 3–5 uploads, build a rolling average appearance embedding per user stored in pgvector. Auto-recognition thresholds:
- `confidence > 0.85` → skip tap entirely
- `0.60–0.85` → show "Is this you?" one-tap confirm
- `< 0.60` → fall back to manual tap

### Role-Aware Highlight Scoring

```python
if action.player == user:
    if action.scored_point:
        highlight_score *= 1.5
    if action.shot_type in ['erne', 'atp', 'smash']:
        highlight_score *= 1.3
elif action.player == partner:
    highlight_score *= 0.7
elif action.player in ['opponent_1', 'opponent_2']:
    if action.scored_point:
        highlight_score *= 0.3
    if action.rally_length > 8:
        highlight_score *= 0.8
```

### Lowlight Scoring

```python
# Phase 2 (shot_quality available after ball detection + pose estimation)
if action.player == user:
    if action.shot_quality < 0.3:
        lowlight_score += shot_weakness_weight
    if action.point_lost_by_error:
        lowlight_score += lost_point_weight
# positioning analysis deferred to Phase 3 (requires pose history across rally)
```

### Pipeline Timeout & Cleanup

Jobs stuck in `WAITING_FOR_TAP` state are cleaned up by a Celery beat scheduler:

```
Job enters WAITING_FOR_TAP
→ Celery beat checks every hour for stale jobs
→ If no tap after 24 hours → cancel + notify user
→ Cleanup worker deletes: working copy from R2, extracted frames, partial state
→ Original video retained for 7 days (user can re-trigger)
→ Original deleted after 7 days if no action (R2 lifecycle policy)
```

---

## 5. Output Types (12 Total)

### Sub-Highlight Classifications

- **Type A — Shot Form:** Single shot demonstrating technique, regardless of rally outcome
- **Type B — Point Scored:** Full rally where user's team wins the point
- **Type C — Lowlight:** Errors, weak shots, lost points — coaching-oriented (NEW)

### Output Type Taxonomy

| # | Output Type | Auto-Generate | Phase | Source |
|---|---|---|---|---|
| 1 | Highlight Montage | Always | 2 | Top Type A + B by score |
| 2 | My Best Plays | Always | 2 | Type A + B where player == user |
| 3 | Game Recap | Always | 2 | All scored points, chronological |
| 4 | Points of Improvement | Always | 2 | Type C — shot_quality < 0.3 OR point_lost_by_user_error (NEW) |
| 5 | Best Shots Reel | On-demand | 2 | Type A ranked by shot_quality |
| 6 | Scored Point Rally | On-demand | 2 | Type B, single rally |
| 7 | Full Rally Replay | On-demand | 2 | Rally length > 8 OR intensity > 0.7 |
| 8 | Single Shot Clip | On-demand | 2 | Any highlight_id |
| 9 | Partnership Highlights | Conditional | 3 | Rallies where user + partner both contribute |
| 10 | Fails & Bloopers | Conditional | 3 | shot_quality < 0.2, comedy framing |
| 11 | Side-by-Side Progress | Conditional | 3 | Same shot_type across ≥2 videos |
| 12 | Comeback Reel | Conditional | 3 | Deficit ≥4 then won |

**Lowlights vs Fails & Bloopers distinction:**

| | Points of Improvement | Fails & Bloopers |
|---|---|---|
| Purpose | Self-coaching | Fun sharing |
| Framing | Neutral, analytical | Comedy, emoji overlays |
| Source | Errors + lost points | Whiffs + mishits only |
| Overlay | Shot quality score + error type | 😬 🤦 emoji overlays |
| Threshold | shot_quality < 0.3 | shot_quality < 0.2 (worst of worst) |

---

## 6. Data Model

Changes from v3.0 are marked **[NEW]** or **[CHANGED]**.

```sql
users
├── id (UUID)
├── email
├── display_name
├── avatar_url
├── highlight_preferences (JSONB)
├── subscription_tier (free | pro | team)
└── created_at

player_profiles
├── id (UUID)
├── user_id → users.id
├── appearance_embedding (VECTOR — pgvector, rolling avg from Re-ID)
├── embedding_confidence (float)
├── uploads_contributing (int)
├── metadata (JSONB)
└── updated_at

videos
├── id (UUID)
├── user_id → users.id
├── r2_key_original          -- [CHANGED: s3_key → r2_key]
├── r2_key_processed
├── duration_seconds
├── resolution
├── status (uploading | identifying | processing | analyzed | failed | timed_out)  -- [CHANGED: added timed_out]
├── identify_started_at (TIMESTAMPTZ NULLABLE)  -- [NEW: for 24hr timeout]
├── cleanup_after (TIMESTAMPTZ NULLABLE)         -- [NEW: R2 cleanup scheduling]
├── metadata (JSONB)
└── uploaded_at

video_players
├── id (UUID)
├── video_id → videos.id
├── role (user | partner | opponent_1 | opponent_2)
├── player_profile_id → player_profiles.id (nullable)
├── seed_frame_bbox (JSONB)
├── appearance_embedding (VECTOR)
├── tracking_confidence (float)
└── created_at

highlights
├── id (UUID)
├── video_id → videos.id
├── attributed_player_role (user | partner | opponent_1 | opponent_2)
├── sub_highlight_type (shot_form | point_scored | lowlight | both)  -- [CHANGED: added lowlight]
├── lowlight_type (unforced_error | positioning | weak_shot | lost_point | NULL)  -- [NEW]
├── point_lost_by_error (BOOLEAN DEFAULT FALSE)  -- [NEW]
├── start_time_ms
├── end_time_ms
├── highlight_score (float)
├── highlight_score_raw (float)
├── shot_type (enum: drive | dink | lob | erne | atp | drop | smash | overhead | speed_up)
├── shot_quality (float 0–1)
├── point_scored (boolean)
├── point_won_by (user_team | opponent_team | null)
├── rally_id (UUID)
├── rally_length (int)
├── rally_intensity (float)
├── score_source (ocr | rule_based | manual) DEFAULT 'rule_based'  -- [NEW: replaces OCR assumption]
├── model_outputs (JSONB)
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
├── score_before (JSONB)
├── score_after (JSONB)
├── is_comeback_point (boolean)
└── created_at

reels
├── id (UUID)
├── user_id → users.id
├── video_id → videos.id (nullable — null for cross-video outputs)
├── output_type (enum — see output taxonomy, includes points_of_improvement)  -- [CHANGED: added value]
├── r2_key           -- [CHANGED: s3_key → r2_key]
├── format (vertical | horizontal | square)
├── duration_seconds
├── clip_ids (UUID[])
├── rally_ids (UUID[])
├── assembly_profile (JSONB)
├── music_track_id
├── status (queued | generating | ready | failed)
├── auto_generated (boolean)
├── share_url
└── created_at

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

## 7. Development Phases

### Phase 1 — Upload → Identify → Clips (~4–6 weeks)

**Goal:** Upload a game, tap yourself, get downloadable individual clips.

**Infrastructure:**
- Docker Compose for local dev (FastAPI + PostgreSQL + Redis)
- Supabase project (DB + Auth)
- Cloudflare R2 bucket + presigned upload URLs
- EC2 t3.micro for app (free tier)
- Celery + Upstash Redis for async jobs
- GitHub Actions deploy pipeline
- Caddy reverse proxy + TLS

**Product:**
- Upload flow with resumable tus protocol
- FFmpeg transcode to 1080p working copy
- YOLOv8n person detection → seed frame with 4 bboxes
- User tap-to-identify UX
- OSNet Re-ID tracking across full video
- Court-position fallback for low-confidence frames
- Rule-based rally detector (no OCR)
- Rule-based score state machine
- Simple highlight scorer (point scored + rally length)
- Clip extraction from 2.7K original via FFmpeg
- Download individual clips
- Pipeline timeout + cleanup (24hr, Celery beat)

**Success criteria:** Upload own game → tap yourself → clips of YOUR plays appear in under 15 minutes.

**Output types:** Individual clips only, no assembled reels.

---

### Phase 2 — Reels + Shot Intelligence (~4–6 weeks)

**Goal:** Auto-generated shareable reels. Shot classification unlocks richer highlights.

**ML additions:**
- TrackNetV2 ball detection (fine-tune if needed)
- MediaPipe Pose estimation
- Rule-based shot classifier (ball trajectory + pose keypoints)
- Role-aware highlight + lowlight scoring

**Product:**
- FFmpeg + MoviePy reel assembly pipeline
- 4 auto-generated output types: Highlight Montage, My Best Plays, Game Recap, **Points of Improvement**
- On-demand output types: Best Shots Reel, Scored Point Rally, Full Rally Replay, Single Shot Clip
- Transitions, slow-mo on peak moments
- Smart vertical crop (center on user)
- Royalty-free music library (Free Music Archive — free, no API needed, downloadable MP3s)
- Share links with OG preview cards
- Thumbs up/down feedback on clips
- Supabase Realtime for live pipeline status (replaces custom WebSocket server)

**Success criteria:** Upload game → get a shareable highlight reel you'd post on Instagram. Points of Improvement view shows errors worth working on.

---

### Phase 3 — Personalization + Auto-Recognition

**Triggered after:** 5+ games of own footage processed.

- Persistent player profiles (rolling embedding, pgvector)
- Auto-recognition flow (skip tap when confidence > 0.85)
- Personalized highlight scoring with online learning from feedback
- YAMNet audio analysis
- ML shot classifier trained on own footage (replaces rule-based)
- Remaining conditional output types: Partnership Highlights, Fails & Bloopers, Side-by-Side Progress, Comeback Reel
- Partner/opponent recognition (opt-in)

---

### Phase 4 — Open to Others + Mobile

**Only after Phase 1–2 prove the concept on own games.**

- Stripe billing (subscription_tier enforcement)
- React Native mobile app (Expo)
- Direct social posting (Instagram Reels, TikTok, YouTube Shorts APIs)
- Public profile pages
- Scale GPU fleet for multiple concurrent users
- Migrate Supabase → RDS if needed (unlikely before 1k+ users)

---

## 8. API Design

### Upload & Video Management

```
POST   /api/v1/videos/upload-url         → returns presigned R2 URL
POST   /api/v1/videos/{id}/confirm       → triggers ingestion pipeline
GET    /api/v1/videos                    → list user's videos
GET    /api/v1/videos/{id}               → video details + processing status
DELETE /api/v1/videos/{id}               → delete video + all derived data
```

### Player Identification

```
GET    /api/v1/videos/{id}/identify      → frame image with 4 bounding boxes
POST   /api/v1/videos/{id}/identify      → { bbox_index: 2 } — user taps player #2
GET    /api/v1/player-profile            → user's persistent player profile
DELETE /api/v1/player-profile            → purge embeddings, reset to tap mode
PUT    /api/v1/player-profile/settings   → { auto_recognize: true }
```

### Highlights, Lowlights & Reels

```
GET    /api/v1/videos/{id}/highlights    → ranked highlights
GET    /api/v1/videos/{id}/lowlights     → ranked lowlights (Points of Improvement)  [NEW]
GET    /api/v1/videos/{id}/rallies       → all rallies with scores
PATCH  /api/v1/highlights/{id}           → update feedback or trim
GET    /api/v1/videos/{id}/reels         → all generated reels
POST   /api/v1/reels                     → generate reel (output_type, format, clip_ids?)
GET    /api/v1/reels/{id}                → reel details + download/share URLs
POST   /api/v1/reels/{id}/share          → generate share link with OG metadata
GET    /api/v1/reels/{id}/export/{fmt}   → download (vertical/horizontal/square)
POST   /api/v1/reels/side-by-side        → cross-video comparison
```

### Real-Time Updates (Supabase Realtime)

Subscribe to `videos` table changes filtered by `video_id`. Status transitions broadcast automatically — no custom WebSocket server needed.

Status progression: `uploading → identifying → processing → analyzed`

---

## 9. Processing Pipeline

```
0:00     Upload complete, job enters queue
0:00–1:30  FFmpeg transcode to 1080p working copy (GPU, NVENC)
1:30–2:00  Scene detection + frame extraction (2 FPS sample)
2:00–2:30  YOLOv8n person detection — locate all 4 players

── PAUSE: Wait for user tap (or auto-recognize in Phase 3) ──
   Timeout: 24 hours → cancel + cleanup

2:30–4:00  OSNet Re-ID tracking — label user across all frames
4:00–6:00  TrackNetV2 ball detection (Phase 2+)
6:00–7:30  MediaPipe Pose estimation (Phase 2+)
7:30–8:30  Rule-based shot classification + rally tracking
8:30–9:00  Highlight + lowlight scoring (role-aware weighting)
9:00–10:00 Clip extraction from 2.7K original
10:00–11:00 Reel assembly (4 auto-generated types in Phase 2)

~12:00   Reels ready, Supabase Realtime update triggers UI
```

**Target: ~8–12 minutes** (excluding user tap wait time).

**Progressive delivery:** Surface individual clips as detected (~10-minute mark) before full reel assembly completes.

---

## 10. Infrastructure & Cost

### Compute

| Component | Instance | Strategy |
|---|---|---|
| API + Celery | EC2 t3.micro (free yr 1) / t3.medium after | Always-on |
| GPU Pipeline | EC2 g5.xlarge spot | Spin up per job, shut down when idle |
| Frontend | Vercel (free) | CDN-deployed Next.js |

### Storage Cost (personal use, ~4 games/month)

| Type | Monthly Volume | Cost |
|---|---|---|
| R2 (clips + reels) | ~2 GB | Free (under 10GB limit) |
| R2 (originals, delete after 7d) | ~4 GB transient | Free |
| GPU compute | ~4 game jobs × ~25 min | ~$0.60/month spot |

**Total: ~$5–20/month for personal use.**

---

## 11. Security & Privacy

- All videos encrypted at rest (R2 SSE) and in transit (TLS 1.3 via Caddy)
- Presigned R2 URLs expire after 1 hour
- GDPR-compliant cascade delete — users can delete all their data
- No video content used for model training without explicit opt-in
- **Appearance embeddings are not facial recognition** — Re-ID uses body shape, clothing, gear. Not biometric face data. Avoids BIPA/GDPR Art. 9.
- Embeddings stored per-user only — no global matching database
- Opponent data ephemeral — discarded after processing, never stored persistently
- Partner linking opt-in only (Phase 3)
- Rate limiting per user tier

---

## 12. Monitoring & Observability

**Sentry (free tier):**
- Error tracking on API + Celery workers
- Performance monitoring for slow pipeline stages

**Structured logging:**
- Pipeline stage timing logged per job
- Alert if processing time > 20 minutes (CloudWatch log metric)

**Key metrics to watch:**
- Processing time per game (target: < 12 min)
- Re-ID accuracy (measured via "that's not me" corrections)
- Highlight keep/discard ratio (thumbs up/down)
- Pipeline failure rate by stage

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Re-ID fails (similar outfits) | Wrong player's highlights | Court-position fallback + ask user for second tap on low confidence |
| Ball detection poor on pickleball | Weak shot classification | Start with TrackNetV2; fine-tune with 200 labeled frames if needed |
| Score state machine inaccurate (no overlay) | Wrong point attribution | Accept lower accuracy initially; add manual correction UI |
| R2 free tier exceeded | Storage cost | Delete originals after 7 days; processed clips after 30 days |
| GPU spot interruption | Job fails mid-pipeline | Celery task retry on interruption; checkpoint after each stage |
| Supabase free tier DB size | Needs upgrade | Metadata only; no video/blob data in Supabase |
| Upload fails on slow connection | Bad UX | tus resumable upload protocol handles reconnects |

---

*Document version: 4.0 — 2026-04-05*
*Refined from v3.0 (March 2026) for solo developer personal project context*
*Key changes: Option A stack, monolith, Supabase + R2, Celery, no OCR (rule-based score), lowlights as 4th auto-generated output type, pipeline timeout/cleanup, all tech decisions resolved*
