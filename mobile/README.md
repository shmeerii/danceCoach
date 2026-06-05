# Mobile (Expo)

Expo app with two modes:

- **Reference Video Mode** — pick a gallery video, upload to the backend, view extraction results and warnings.
- **Practice Mode** — load a saved reference, use the front camera, run a full-body check, practice with reference music, and review post-practice analysis.

See the [project README](../README.md) for full setup, practice flow, and **BACKEND_URL** for Android physical device testing.

## Configure backend URL

Edit `BACKEND_URL` at the top of `App.tsx`:

```typescript
// For Android physical device testing, localhost will not work. Use the
// computer's LAN IP address, for example:
// http://192.168.x.x:8000
const BACKEND_URL = "http://192.168.1.42:8000";
```

## Run

```bash
npm install
npx expo start
```

## Reference Video Mode flow

1. Read recording tips
2. **Select Video** — gallery only (resets previous results after a new pick)
3. **Process Video** — upload + extract (buttons disabled while busy)
4. View results, warnings, and optional **View Pose JSON Summary**

## Practice Mode flow

1. Load a saved reference
2. Allow camera permission
3. Stand far enough back for full body visibility
4. Run full-body check
5. Press **Start Practice**
6. Wait for countdown
7. Music starts; dance until the timer ends
8. View post-practice analysis

During practice, a pose overlay shows live detection only — no scores or coaching until the session ends.

## Status messages (Reference Video Mode)

- Preparing video...
- Uploading video...
- Extracting pose data...
- Processing complete.

## Failure message

If upload or extraction fails, the app shows a Wi‑Fi/backend hint. Technical details are logged to the Metro/console log.

## Out of scope

Final dance scoring, DTW/time-warping, live coaching, export/share.
