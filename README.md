# Dance Pose Extraction MVP

Extract reference pose landmark data from a dance video, then practice along with that reference using the phone’s front camera. The backend validates videos, runs MediaPipe pose detection, saves pose JSON and reference media, and supports live frame capture during practice. The mobile app has two modes: **Reference Video Mode** (upload and extract) and **Practice Mode** (dance along and review post-practice analysis).

> **Developer note:** Reference video extraction is the stable base feature. Practice Mode builds on saved references without changing the extraction pipeline unless explicitly required.

## Reference Video Mode

1. Select a dance video from the phone gallery.
2. Extract pose data (`POST /extract-pose`).
3. Backend saves:
   - **pose JSON** under `backend/outputs/`
   - **reference video media** under `backend/reference_media/` for music playback in Practice Mode

The app shows reference quality (`excellent` → `failed`), warnings, and a compact JSON summary. Poor videos are marked honestly instead of being treated as good references.

## Practice Mode

1. Load a saved reference.
2. Allow camera permission.
3. Stand far enough back for full body visibility.
4. Run full-body check.
5. Press **Start Practice**.
6. Wait for countdown.
7. Music starts (from the saved reference video).
8. Dance until the timer ends.
9. View post-practice analysis.

During practice, a visual pose overlay confirms live detection. No live coaching or scores are shown until the session ends.

### Important notes

- The practice timer matches the reference video duration.
- Live poses are compared to the reference video timestamp one-to-one.
- No live coaching is shown yet.
- Analysis appears after the session ends.
- Full body must be visible for accurate comparison.
- Better lighting improves pose detection.
- Keep the phone stable.
- Use the front camera only.

## What this MVP does

- **Reference Video Mode:** select a dance video, upload, extract per-frame pose landmarks, and save JSON + reference media
- **Practice Mode:** load a saved reference, run a full-body check, practice with reference music, capture live poses, and show post-practice analysis
- Validate file type, size, duration, resolution, and basic readability (OpenCV)
- Extract poses with MediaPipe Pose Landmarker (Tasks API) and conservative lighting preprocessing variants
- Save pose JSON with metadata, quality metrics, and usability checks
- List saved references (`GET /references`) and stream reference video for playback
- Extract live pose from camera snapshots during practice (`POST /extract-live-pose-frame`)
- Show reference quality, warnings, practice analysis, and honest limitations in the app

## What this MVP does not do

- Live coaching or real-time scoring during practice
- DTW / time-warping alignment between live and reference motion
- Final or authoritative dance scoring
- Authentication, database, or cloud storage
- Export/share flows or advanced UI

## Architecture

| Layer | Stack |
|-------|--------|
| Mobile | Expo React Native, `expo-image-picker`, `expo-camera`, `expo-video` |
| Backend | Python FastAPI |
| Pose | MediaPipe Pose Landmarker (Tasks API, VIDEO + IMAGE modes) |
| Video | OpenCV |
| Output | JSON in `backend/outputs/`; reference video copy in `backend/reference_media/` |

## Repository layout

```
danceCoachTwo/
├── mobile/          # Expo app
├── backend/         # FastAPI + pose pipeline
│   ├── app/
│   ├── outputs/          # Generated pose JSON (runtime)
│   ├── reference_media/  # Saved reference videos for Practice Mode playback
│   ├── uploads/          # Temporary uploads (deleted after processing)
│   └── scripts/     # CLI test script
└── README.md
```

---

## Backend setup

From the project root:

```bash
cd backend
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

The pose model downloads automatically on first extraction to `backend/models/`.

## Backend run command

From `backend/` with the virtual environment activated:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Use `--host 0.0.0.0` so phones on the same Wi‑Fi can reach the API.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Backend test script command

Test extraction without the mobile app (from `backend/`):

```bash
python scripts/test_extract_pose.py path/to/your/video.mp4
```

Optional custom output name:

```bash
python scripts/test_extract_pose.py path/to/video.mp4 --output-name test_output.json
```

---

## Expo setup

From the project root:

```bash
cd mobile
npm install
```

## Expo run command (development)

From `mobile/`:

```bash
npx expo start
```

## Install the full app on your phone (not Expo Go)

See **[mobile/INSTALL.md](mobile/INSTALL.md)** for step-by-step instructions.

**Android (Windows, USB):** install Android Studio, enable USB debugging, set `BACKEND_URL` in `App.tsx`, then:

```bash
cd mobile
npx expo run:android
```

**Android (APK, no USB build tools):** use EAS Build — `eas build --platform android --profile preview`, download and install the APK.

## BACKEND_URL for Android phone testing

Edit the constant at the top of `mobile/App.tsx`:

```typescript
// For Android physical device testing, localhost will not work. Use the
// computer's LAN IP address, for example:
// http://192.168.x.x:8000
const BACKEND_URL = "http://192.168.1.42:8000";
```

| Device | Typical `BACKEND_URL` |
|--------|------------------------|
| iOS Simulator | `http://127.0.0.1:8000` |
| Android Emulator | `http://10.0.2.2:8000` |
| Physical phone (same Wi‑Fi) | `http://<your-computer-LAN-IP>:8000` |

Find your LAN IP: `ipconfig` (Windows) or `ifconfig` / System Settings (macOS).

The backend must be running with `--host 0.0.0.0`. Phone and computer must be on the same network.

---

## Recommended video recording conditions

- Full body visible from head to feet.
- Dancer centered in frame.
- Camera stable, not handheld if possible.
- Bright, even lighting.
- Avoid strong backlighting.
- Avoid very dark clothing against a dark background.
- Avoid extreme side angles.
- Avoid recording too close to the dancer.
- Avoid cutting off feet, hands, or head.
- Prefer 720p or 1080p.
- Prefer 24–60 FPS.
- Keep video under 3 minutes for MVP testing (backend max duration: 180 seconds).

## Known limitations

### Reference extraction

- Pose extraction may fail with heavy occlusion.
- Pose extraction may be poor in very dark videos.
- Fast spins, motion blur, and loose clothing can reduce landmark accuracy.
- Extreme camera angles can reduce full-body visibility.

### Practice Mode

- This is not final dance scoring.
- No DTW / time-warping yet.
- If the user starts late or early relative to the music, the analysis will reflect that.
- Frame sampling is low-frequency for stability.
- Fast motion can reduce detection quality.
- Live poses are compared to reference timestamps one-to-one; timing drift affects results.

---

## MVP completion checklist

- [x] User selects a dance video on mobile
- [x] App uploads to `POST /extract-pose` (`multipart/form-data`, field `file`)
- [x] Backend validates video before expensive processing
- [x] Backend extracts poses with preprocessing variants for difficult lighting
- [x] Backend saves pose JSON and reference video media; uploads are deleted after processing
- [x] App shows metadata, quality, and warnings
- [x] Poor videos are marked `poor` or `failed` in `reference_quality`, not hidden
- [x] Practice Mode loads saved references and plays reference music
- [x] Full-body check before practice; countdown and duration-matched timer
- [x] Live pose capture during practice with post-session analysis

## More detail

- [backend/README.md](backend/README.md) — API endpoints and configuration
- [mobile/README.md](mobile/README.md) — app flow and troubleshooting
