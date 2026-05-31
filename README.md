# Dance Pose Extraction MVP

Extract reference pose landmark data from a dance video uploaded from your phone. The backend validates the video, runs MediaPipe pose detection with lighting preprocessing variants, saves JSON output, and reports honest quality metrics. The mobile app uploads the video and shows results and warnings.

## What this MVP does

- Select a dance video from the phone gallery (Expo React Native)
- Upload the video to a local FastAPI backend (`POST /extract-pose`)
- Validate file type, size, duration, resolution, and basic readability (OpenCV)
- Extract per-frame pose landmarks (MediaPipe Pose Landmarker, with legacy fallback)
- Try multiple conservative frame preprocessing variants per frame for difficult lighting
- Save pose JSON under `backend/outputs/` with metadata, quality metrics, and usability checks
- Show reference quality (`excellent` → `failed`), warnings, and a compact JSON summary in the app
- Reject or warn on poor videos instead of pretending they are good references

## What this MVP does not do

- Live camera tracking or real-time coaching
- Comparing two dances, scoring, or DTW
- Authentication, database, or cloud storage
- Export/share flows or advanced UI
- Dance coaching feedback

## Architecture

| Layer | Stack |
|-------|--------|
| Mobile | Expo React Native, `expo-image-picker` |
| Backend | Python FastAPI |
| Pose | MediaPipe Pose Landmarker (Tasks API, VIDEO mode) |
| Video | OpenCV |
| Output | JSON (`video_metadata`, `quality`, `pose_frames`) |

## Repository layout

```
danceCoachTwo/
├── mobile/          # Expo app
├── backend/         # FastAPI + pose pipeline
│   ├── app/
│   ├── outputs/     # Generated JSON (runtime)
│   ├── uploads/     # Temporary uploads (deleted after processing)
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

- Pose extraction may fail with heavy occlusion.
- Pose extraction may be poor in very dark videos.
- Fast spins, motion blur, and loose clothing can reduce landmark accuracy.
- Extreme camera angles can reduce full-body visibility.
- This MVP extracts reference pose data only. It does not compare dances yet.

---

## MVP completion checklist

- [x] User selects a dance video on mobile
- [x] App uploads to `POST /extract-pose` (`multipart/form-data`, field `file`)
- [x] Backend validates video before expensive processing
- [x] Backend extracts poses with preprocessing variants for difficult lighting
- [x] Backend saves pose JSON safely; uploads are deleted after processing
- [x] App shows metadata, quality, and warnings
- [x] Poor videos are marked `poor` or `failed` in `reference_quality`, not hidden

## More detail

- [backend/README.md](backend/README.md) — API endpoints and configuration
- [mobile/README.md](mobile/README.md) — app flow and troubleshooting
