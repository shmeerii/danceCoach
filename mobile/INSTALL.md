# Install the full app on your phone (not Expo Go)

You have two main options on Windows:

| Method | Best for | Needs |
|--------|----------|--------|
| **A. USB install (local build)** | Fast iteration, Android phone + cable | Android Studio, USB debugging |
| **B. EAS cloud build (APK file)** | Install APK without Android Studio on PC | Free Expo account, same Wi‑Fi for backend |

Before any build, set **`BACKEND_URL`** in `App.tsx` to your PC’s LAN IP (not `localhost`):

```typescript
const BACKEND_URL = "http://192.168.1.42:8000";
```

You must **rebuild the app** if you change this URL.

---

## Prerequisites (both methods)

1. **Backend running on your PC** (phone and PC on same Wi‑Fi):

   ```powershell
   cd c:\Users\shahm\danceCoachTwo\backend
   .\.venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Find your PC IP** (`ipconfig` → Wi‑Fi IPv4, e.g. `192.168.1.42`).

3. **Test from the phone browser:** `http://192.168.1.42:8000/health` → `{"status":"ok"}`.

4. **Install dependencies** in `mobile/`:

   ```powershell
   cd c:\Users\shahm\danceCoachTwo\mobile
   npm install
   ```

---

## Method A — Install directly on Android (USB)

### One-time setup

1. Install [Android Studio](https://developer.android.com/studio).
2. In Android Studio: **SDK Manager** → install **Android SDK** and **Platform Tools**.
3. On your phone: **Settings → Developer options → USB debugging** (on).
4. Connect phone by USB; accept the debugging prompt.

### Install Expo core packages (first time only)

If `npx expo run:android` fails with **`expo-asset` cannot be found**:

```powershell
cd c:\Users\shahm\danceCoachTwo\mobile
npx expo install expo-asset expo-constants expo-font
```

### Build and install

```powershell
cd c:\Users\shahm\danceCoachTwo\mobile
npx expo run:android
```

First run downloads native dependencies and can take **10–20+ minutes**. When it finishes, the app **Dance Pose Extractor** is installed on the phone.

Later runs are faster. The app is a normal installed app (no Expo Go).

### Run again after code changes

```powershell
npx expo run:android
```

Or start Metro separately:

```powershell
npx expo start
```

Then open the installed app (development build connects to Metro when on the same network).

---

## Method B — Build an APK and install manually (EAS)

Good if you do not want Android Studio, or you want to share the APK.

### One-time setup

1. Create a free account at [expo.dev](https://expo.dev).
2. Install EAS CLI:

   ```powershell
   npm install -g eas-cli
   eas login
   ```

3. Link the project (from `mobile/`):

   ```powershell
   cd c:\Users\shahm\danceCoachTwo\mobile
   eas build:configure
   ```

   (If prompted, accept defaults; `eas.json` is already in the repo.)

### Build APK in the cloud

```powershell
cd c:\Users\shahm\danceCoachTwo\mobile
npm run build:android
```

Or:

```powershell
eas build --platform android --profile preview
```

Wait for the build on expo.dev (often **10–20 minutes**). When done, open the build page and **download the `.apk`**.

### Install APK on the phone

1. Copy the APK to the phone (USB, email, Drive, etc.).
2. Open the file on the phone.
3. Allow **Install from unknown sources** if Android asks.
4. Install **Dance Pose Extractor**.

---

## iPhone (full app, no Expo Go)

You need either:

- A **Mac** with Xcode: `npx expo run:ios` with the phone connected, or  
- **EAS Build** (cloud): `eas build --platform ios --profile preview`  
  - Apple Developer account required for device install outside TestFlight  
  - Simplest for testers: [TestFlight](https://developer.apple.com/testflight/) via EAS Submit (extra setup)

On Windows only, use EAS for iOS or borrow a Mac for a local build.

---

## After install

1. Open **Dance Pose Extractor** on the phone.
2. Confirm `BACKEND_URL` in the build matches your PC IP.
3. PC backend must be running on the same Wi‑Fi.
4. **Select Video** → **Process Video**.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App can’t reach backend | Same Wi‑Fi; `BACKEND_URL` uses PC IP; firewall allows port 8000 |
| Changed `BACKEND_URL` | Rebuild/reinstall the app |
| `expo run:android` no devices | USB debugging on; run `adb devices` |
| `expo-asset` cannot be found | Run `npx expo install expo-asset expo-constants expo-font`, then `npx expo run:android` again |
| Gradle BUILD SUCCESSFUL then Metro error | Native build OK; fix missing packages above and re-run |
| EAS build fails | Run `eas build:configure`; check expo.dev build logs |
| Cleartext HTTP blocked | Already enabled via `usesCleartextTraffic` in `app.json` |

---

## Quick reference

```powershell
# Backend (keep running)
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Android USB install
cd mobile
npx expo run:android

# Android APK via cloud
cd mobile
eas build --platform android --profile preview
```
