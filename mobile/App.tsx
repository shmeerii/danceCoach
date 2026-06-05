import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import {
  ActivityIndicator,
  Alert,
  AppState,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import Constants from "expo-constants";
import * as ImagePicker from "expo-image-picker";
import type { ImagePickerAsset } from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { useVideoPlayer, VideoView } from "expo-video";
import { NativeModules } from "react-native";
import {
  analyzePracticeSession,
  type PracticeAnalysisResult,
} from "./src/utils/practiceAnalysis";
import { PoseOverlay } from "./src/components/PoseOverlay";
import type { PoseLandmarkInput } from "./src/utils/poseNormalization";

const BACKEND_PORT = 8000;

/** Override in .env: EXPO_PUBLIC_BACKEND_URL=http://192.168.1.42:8000 */
function hostFromDevUri(uri: string | undefined): string | null {
  if (!uri) {
    return null;
  }
  const withoutScheme = uri.replace(/^[a-z][a-z0-9+.-]*:\/\//i, "");
  const host = withoutScheme.split(/[:/]/)[0];
  if (!host || host === "localhost" || host === "127.0.0.1") {
    return null;
  }
  return host;
}

function getMetroDevHost(): string | null {
  const uris = [
    Constants.expoConfig?.hostUri,
    Constants.expoGoConfig?.debuggerHost,
    (Constants.manifest as { debuggerHost?: string } | null)?.debuggerHost,
  ];
  for (const uri of uris) {
    const host = hostFromDevUri(uri);
    if (host) {
      return host;
    }
  }

  const scriptURL: string | undefined =
    NativeModules.SourceCode?.getConstants?.()?.scriptURL ??
    (NativeModules as { SourceCode?: { scriptURL?: string } }).SourceCode
      ?.scriptURL;
  const match = scriptURL?.match(/https?:\/\/([^:/]+)/);
  if (
    match?.[1] &&
    match[1] !== "localhost" &&
    match[1] !== "127.0.0.1"
  ) {
    return match[1];
  }

  return null;
}

function getConfiguredBackendUrl(): string | null {
  const fromEnv = process.env.EXPO_PUBLIC_BACKEND_URL?.trim();
  if (fromEnv) {
    return fromEnv.replace(/\/$/, "");
  }

  const fromExtra = Constants.expoConfig?.extra?.backendUrl;
  if (typeof fromExtra === "string" && fromExtra.trim()) {
    return fromExtra.trim().replace(/\/$/, "");
  }

  return null;
}

const BACKEND_PROBE_TIMEOUT_MS = 4000;
let resolvedBackendUrl: string | null = null;
let backendUrlResolvePromise: Promise<{ url: string; reachable: boolean }> | null =
  null;

function resetBackendUrlCache(): void {
  resolvedBackendUrl = null;
  backendUrlResolvePromise = null;
}

function collectBackendUrlCandidates(): string[] {
  const seen = new Set<string>();
  const candidates: string[] = [];
  const add = (url: string | null | undefined) => {
    if (!url?.trim()) {
      return;
    }
    const normalized = url.trim().replace(/\/$/, "");
    if (seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    candidates.push(normalized);
  };

  const devHost = getMetroDevHost();
  if (devHost) {
    add(`http://${devHost}:${BACKEND_PORT}`);
  }

  add(getConfiguredBackendUrl());

  if (Platform.OS === "android" && Constants.isDevice === false) {
    add(`http://10.0.2.2:${BACKEND_PORT}`);
  }

  if (!Constants.isDevice) {
    add(`http://127.0.0.1:${BACKEND_PORT}`);
  }

  return candidates;
}

function getBackendUrlFallback(): string {
  const candidates = collectBackendUrlCandidates();
  if (candidates.length > 0) {
    return candidates[0];
  }

  if (Platform.OS === "android" && Constants.isDevice === false) {
    return `http://10.0.2.2:${BACKEND_PORT}`;
  }

  if (!Constants.isDevice) {
    return `http://127.0.0.1:${BACKEND_PORT}`;
  }

  return getConfiguredBackendUrl() ?? `http://127.0.0.1:${BACKEND_PORT}`;
}

function getBackendUrl(): string {
  return resolvedBackendUrl ?? getBackendUrlFallback();
}

async function probeBackendUrl(baseUrl: string): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(
    () => controller.abort(),
    BACKEND_PROBE_TIMEOUT_MS,
  );
  try {
    const response = await fetch(`${baseUrl}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function resolveBackendUrl(): Promise<{ url: string; reachable: boolean }> {
  if (resolvedBackendUrl) {
    return { url: resolvedBackendUrl, reachable: true };
  }

  if (!backendUrlResolvePromise) {
    backendUrlResolvePromise = (async () => {
      for (const candidate of collectBackendUrlCandidates()) {
        if (await probeBackendUrl(candidate)) {
          resolvedBackendUrl = candidate;
          return { url: candidate, reachable: true };
        }
      }

      const fallback = getBackendUrlFallback();
      return { url: fallback, reachable: false };
    })();
  }

  return backendUrlResolvePromise;
}

function useResolvedBackendUrl() {
  const [backendUrl, setBackendUrl] = useState(() => getBackendUrl());
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);

  const refreshBackendUrl = useCallback(async () => {
    resetBackendUrlCache();
    const { url, reachable } = await resolveBackendUrl();
    setBackendUrl(url);
    setBackendReachable(reachable);
  }, []);

  useEffect(() => {
    void refreshBackendUrl();
    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState === "active") {
        void refreshBackendUrl();
      }
    });
    return () => subscription.remove();
  }, [refreshBackendUrl]);

  return { backendUrl, backendReachable, refreshBackendUrl };
}

function showBackendUnreachableAlert(backend: string): void {
  Alert.alert(
    "Cannot reach backend",
    `Tried ${backend}. On your PC run backend\\start-server.ps1, confirm phone and PC share Wi-Fi, and open ${backend}/health in the phone browser. If that fails, run backend\\allow-firewall.ps1 as Administrator. After changing mobile/.env, run npx expo run:android again.`,
  );
}

const FAILURE_MESSAGE =
  "Could not process the video. Check that the backend is running and that your phone and computer are on the same Wi-Fi network.";

type AppStatus =
  | "idle"
  | "video_selected"
  | "preparing"
  | "uploading"
  | "extracting"
  | "complete"
  | "failed";

type SelectedVideo = {
  uri: string;
  fileName: string | null;
  durationSeconds: number | null;
  fileSizeBytes: number | null;
  mimeType: string | null;
};

type LowConfidenceSegment = {
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
};

type QualityPayload = {
  pose_detection_percentage?: number;
  average_visibility?: number | null;
  full_body_visibility_percentage?: number;
  reference_quality?: string;
  lighting_warnings?: string[];
  video_warnings?: string[];
  body_size_warning?: string | null;
  edge_cutoff_warning?: string | null;
  angle_warning?: string | null;
  low_confidence_segments?: LowConfidenceSegment[];
  user_guidance?: string[];
};

type VideoMetadataPayload = {
  fps?: number;
  duration_seconds?: number;
  width?: number;
  height?: number;
  frame_count?: number;
  processed_frames?: number;
  pose_detected_frames?: number;
  frame_stride?: number;
};

type ExtractionResponse = {
  success: boolean;
  output_filename: string;
  output_path: string;
  video_metadata: VideoMetadataPayload;
  quality: QualityPayload;
  summary: {
    processed_frames: number;
    pose_detected_frames: number;
    pose_detection_percentage: number;
    reference_quality: string;
  };
  warnings?: string[];
};

type PoseFramePayload = {
  frame_index: number;
  timestamp_seconds: number;
  pose_detected: boolean;
  detection_variant: string | null;
  landmarks?: unknown[];
};

type OutputJsonPayload = {
  video_metadata?: VideoMetadataPayload;
  quality?: QualityPayload;
  pose_frames?: PoseFramePayload[];
};

type PoseJsonSummary = {
  video_metadata: VideoMetadataPayload;
  quality: QualityPayload;
  pose_frame_count: number;
  first_detected_frame_index: number | null;
  first_detected_timestamp: number | null;
  first_detected_landmark_count: number | null;
  first_detected_variant: string | null;
};

/** Saved reference row from GET /references (Practice Mode only). */
type CatalogReference = {
  reference_id: string;
  json_filename: string;
  json_url: string;
  reference_video_filename: string | null;
  reference_video_url: string | null;
  duration_seconds: number | null;
  created_at: string | null;
  reference_quality: string | null;
  pose_detection_percentage: number | null;
};

type ReferencesResponse = {
  references: CatalogReference[];
};

/** Loaded reference for Practice Mode — pose JSON stored but not rendered. */
type PracticeLoadedReference = {
  referenceId: string;
  jsonFilename: string;
  videoUrl: string;
  durationSeconds: number;
  referenceQuality: string;
  poseDetectionPercentage: number;
  poseJson: OutputJsonPayload;
};

type PoseLandmark = {
  name: string;
  visibility?: number | null;
};

type LivePoseFrameResponse = {
  success: boolean;
  pose_detected: boolean;
  timestamp_server: number;
  processing_time_ms?: number;
  image_metadata: { width: number; height: number };
  detection_variant: string | null;
  landmarks: PoseLandmark[];
  world_landmarks: unknown[];
  quality: {
    average_visibility: number | null;
    full_body_visible: boolean;
    full_body_visibility_score: number;
    warnings: string[];
  };
};

type PracticeSessionState =
  | "no_reference"
  | "reference_loaded"
  | "camera_ready"
  | "checking_full_body"
  | "ready_to_start"
  | "countdown"
  | "practicing"
  | "analyzing"
  | "complete"
  | "failed";

type PracticeSessionSummary = {
  referenceDurationSeconds: number;
  elapsedSeconds: number;
  livePoseFrames: PracticeLivePoseFrame[];
  completedFully: boolean;
};

type PracticeLivePoseFrame = {
  practice_elapsed_seconds: number;
  client_timestamp: number;
  pose_detected: boolean;
  full_body_visible: boolean;
  landmarks: unknown[];
  world_landmarks: unknown[];
  quality: Record<string, unknown>;
  latency_ms: number;
  skipped_reason: string | null;
};

type LatestLivePoseOverlay = {
  pose_detected: boolean;
  landmarks: PoseLandmarkInput[];
};

type ReferenceMusicLoadState = "loading" | "ready" | "error";

type PracticeMusicPlaybackState = "idle" | "starting" | "playing" | "error";

type PracticeMusicPlayerHandle = {
  playFromStart: () => Promise<void>;
  stopAndReset: () => void;
};
const FULL_BODY_FRAME_COUNT = 3;
const FULL_BODY_PASS_MIN = 2;
const FULL_BODY_FRAME_INTERVAL_MS = 500;
const PRACTICE_COUNTDOWN_START = 5;
const PRACTICE_COUNTDOWN_TICK_MS = 1000;
const PRACTICE_TIMER_TICK_MS = 250;
const LIVE_FRAME_INTERVAL_MS = 700;
const LIVE_FRAME_JPEG_QUALITY = 0.55;
const MAX_CONSECUTIVE_LIVE_POSE_FAILURES = 5;
const MUSIC_PLAYBACK_START_TIMEOUT_MS = 5000;

const CAMERA_SESSION_STATES: PracticeSessionState[] = [
  "reference_loaded",
  "camera_ready",
  "checking_full_body",
  "ready_to_start",
  "countdown",
  "practicing",
  "failed",
];

function isCameraSessionState(state: PracticeSessionState): boolean {
  return CAMERA_SESSION_STATES.includes(state);
}

function meetsPracticeStartRequirements(
  loadedReference: PracticeLoadedReference | null,
  musicLoadState: ReferenceMusicLoadState,
  fullBodyPassed: boolean,
): boolean {
  if (!loadedReference || !fullBodyPassed) {
    return false;
  }
  if (musicLoadState !== "ready") {
    return false;
  }
  if (
    Number.isNaN(loadedReference.durationSeconds) ||
    loadedReference.durationSeconds <= 0
  ) {
    return false;
  }
  return Boolean(loadedReference.videoUrl?.trim());
}

function resolveReadySessionState(
  loadedReference: PracticeLoadedReference | null,
  musicLoadState: ReferenceMusicLoadState,
  fullBodyPassed: boolean,
): "camera_ready" | "ready_to_start" {
  return meetsPracticeStartRequirements(
    loadedReference,
    musicLoadState,
    fullBodyPassed,
  )
    ? "ready_to_start"
    : "camera_ready";
}

const LANDMARK_VISIBILITY_THRESHOLD = 0.5;
const LOW_AVERAGE_VISIBILITY_THRESHOLD = 0.45;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isLandmarkVisible(landmark: PoseLandmark | undefined): boolean {
  if (!landmark) {
    return false;
  }
  const visibility = landmark.visibility;
  if (visibility === null || visibility === undefined) {
    return true;
  }
  return visibility >= LANDMARK_VISIBILITY_THRESHOLD;
}

function analyzeFrameGuidance(payload: LivePoseFrameResponse): {
  anklesMissing: boolean;
  shouldersOrHipsMissing: boolean;
  lowVisibility: boolean;
  noBody: boolean;
} {
  if (!payload.pose_detected) {
    return {
      anklesMissing: false,
      shouldersOrHipsMissing: false,
      lowVisibility: false,
      noBody: true,
    };
  }

  const byName = new Map(payload.landmarks.map((landmark) => [landmark.name, landmark]));
  const anklesMissing =
    !isLandmarkVisible(byName.get("left_ankle")) ||
    !isLandmarkVisible(byName.get("right_ankle"));
  const shouldersMissing =
    !isLandmarkVisible(byName.get("left_shoulder")) ||
    !isLandmarkVisible(byName.get("right_shoulder"));
  const hipsMissing =
    !isLandmarkVisible(byName.get("left_hip")) ||
    !isLandmarkVisible(byName.get("right_hip"));
  const averageVisibility = payload.quality.average_visibility;
  const lowVisibility =
    averageVisibility !== null &&
    averageVisibility !== undefined &&
    averageVisibility < LOW_AVERAGE_VISIBILITY_THRESHOLD;

  return {
    anklesMissing,
    shouldersOrHipsMissing: shouldersMissing || hipsMissing,
    lowVisibility,
    noBody: false,
  };
}

function buildFullBodyGuidance(frames: LivePoseFrameResponse[]): string[] {
  let anklesMissing = false;
  let shouldersOrHipsMissing = false;
  let lowVisibility = false;
  let noBodyFrames = 0;

  for (const frame of frames) {
    const flags = analyzeFrameGuidance(frame);
    if (flags.noBody) {
      noBodyFrames += 1;
    }
    if (flags.anklesMissing) {
      anklesMissing = true;
    }
    if (flags.shouldersOrHipsMissing) {
      shouldersOrHipsMissing = true;
    }
    if (flags.lowVisibility) {
      lowVisibility = true;
    }
  }

  const messages: string[] = [];
  if (anklesMissing) {
    messages.push("Move farther back so your feet are visible.");
  }
  if (shouldersOrHipsMissing) {
    messages.push("Center your body in the camera.");
  }
  if (lowVisibility) {
    messages.push("Use brighter lighting and keep the camera stable.");
  }
  if (messages.length === 0) {
    if (noBodyFrames === frames.length) {
      messages.push("No body detected. Step into view and improve lighting.");
    } else {
      messages.push(
        "Move back until your head, arms, legs, and feet are visible.",
      );
    }
  }
  return messages;
}

function getFullBodyOverlayMessage(
  sessionState: PracticeSessionState,
  checkFrameIndex: number,
): string {
  if (sessionState === "checking_full_body" && checkFrameIndex >= 1) {
    return `Checking frame ${checkFrameIndex} of ${FULL_BODY_FRAME_COUNT}`;
  }
  if (sessionState === "ready_to_start") {
    return "Full body check passed";
  }
  if (sessionState === "failed") {
    return "Full body check failed";
  }
  return "Check full body visibility before starting.";
}

const RECORDING_TIPS = [
  "Record the full body from head to feet.",
  "Use bright lighting.",
  "Keep the camera stable.",
  "Avoid extreme side angles.",
  "Avoid dark backgrounds with dark clothing.",
  "Keep the dancer mostly centered.",
];

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "Unknown";
  }
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function formatMatchScore(score: number | null): string {
  if (score === null) {
    return "Not enough data";
  }
  return `${score}/100`;
}

function describeOverallMatch(score: number | null): string {
  if (score === null) {
    return "Not enough clear frames to review your overall match.";
  }
  if (score >= 75) {
    return "You matched the reference well overall.";
  }
  if (score >= 50) {
    return "You matched parts of the reference in several moments.";
  }
  return "Your practice highlights useful moments to compare with the reference.";
}

function describeRegionMatch(
  region: "arms" | "legs" | "torso",
  score: number | null,
): string {
  if (score === null) {
    return `Not enough data to review your ${region}.`;
  }
  if (score >= 75) {
    return `Your ${region} were close to the reference.`;
  }
  if (score >= 50) {
    return `Your ${region} matched the reference in several moments.`;
  }
  return `Your ${region} need more attention.`;
}

const PRACTICE_ANALYSIS_TIPS = [
  "Keep full body visible.",
  "Improve lighting.",
  "Move farther back if feet are missing.",
  "Keep timing with the music.",
  "Try again with the same reference.",
] as const;

const MIN_USABLE_COMPARISON_FRAMES_FOR_SCORE = 5;

const INSUFFICIENT_PRACTICE_DATA_TIPS = [
  "Make sure your full body is visible.",
  "Use better lighting.",
  "Complete more of the dance.",
] as const;

function resolveReferenceDurationSeconds(
  poseJson: OutputJsonPayload,
  catalogDuration: number | null,
): number {
  const jsonDuration = poseJson.video_metadata?.duration_seconds;
  if (
    jsonDuration !== null &&
    jsonDuration !== undefined &&
    !Number.isNaN(Number(jsonDuration)) &&
    Number(jsonDuration) > 0
  ) {
    return Number(jsonDuration);
  }
  const catalog = Number(catalogDuration);
  if (!Number.isNaN(catalog) && catalog > 0) {
    return catalog;
  }
  return 0;
}

function getElapsedPracticeSeconds(
  startTimeMs: number | null,
  maxSeconds?: number | null,
): number {
  if (startTimeMs === null) {
    return 0;
  }
  let elapsed = Math.max(0, (Date.now() - startTimeMs) / 1000);
  if (
    maxSeconds !== null &&
    maxSeconds !== undefined &&
    maxSeconds > 0
  ) {
    elapsed = Math.min(elapsed, maxSeconds);
  }
  return elapsed;
}

function getRemainingPracticeSeconds(
  elapsedSeconds: number,
  referenceDurationSeconds: number,
): number {
  if (referenceDurationSeconds <= 0) {
    return 0;
  }
  return Math.max(0, referenceDurationSeconds - elapsedSeconds);
}

function formatFileSize(bytes: number | null): string {
  if (bytes === null || bytes <= 0) {
    return "Unknown";
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatResolution(
  width: number | undefined,
  height: number | undefined,
): string {
  if (width && height) {
    return `${width} x ${height}`;
  }
  return "Unknown";
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Unknown";
  }
  return `${value}%`;
}

function formatVisibility(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Unknown";
  }
  return value.toFixed(2);
}

function joinBackendUrl(path: string): string {
  const trimmed = path.trim();
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed;
  }
  const base = getBackendUrl().replace(/\/$/, "");
  return `${base}${trimmed.startsWith("/") ? trimmed : `/${trimmed}`}`;
}

function isPracticeSelectableReference(ref: CatalogReference): boolean {
  if (!ref.json_url?.trim() || !ref.reference_video_url?.trim()) {
    return false;
  }
  const duration = ref.duration_seconds;
  return (
    duration !== null &&
    duration !== undefined &&
    !Number.isNaN(Number(duration)) &&
    Number(duration) > 0
  );
}

function parseTechnicalError(status: number, bodyText: string): string {
  try {
    const data = JSON.parse(bodyText);
    if (typeof data.detail === "string") {
      return data.detail;
    }
    if (data.detail && typeof data.detail === "object") {
      if (typeof data.detail.message === "string") {
        return data.detail.message;
      }
      if (Array.isArray(data.detail.errors) && data.detail.errors.length > 0) {
        return data.detail.errors.join(" ");
      }
      return JSON.stringify(data.detail);
    }
    if (typeof data.message === "string") {
      return data.message;
    }
    return JSON.stringify(data);
  } catch {
    return bodyText || `HTTP ${status}`;
  }
}

function collectBackendWarnings(quality: QualityPayload): string[] {
  const warnings: string[] = [];
  const seen = new Set<string>();

  const add = (message: string | null | undefined) => {
    if (!message || seen.has(message)) {
      return;
    }
    seen.add(message);
    warnings.push(message);
  };

  for (const list of [quality.lighting_warnings, quality.video_warnings]) {
    if (list) {
      for (const item of list) {
        add(item);
      }
    }
  }

  add(quality.body_size_warning);
  add(quality.edge_cutoff_warning);
  add(quality.angle_warning);

  const segments = quality.low_confidence_segments ?? [];
  if (segments.length === 1) {
    const segment = segments[0];
    add(
      `Low confidence section: frames ${segment.start_frame}–${segment.end_frame} ` +
        `(${formatDuration(segment.start_time)}–${formatDuration(segment.end_time)}).`,
    );
  } else if (segments.length > 1) {
    add(
      `${segments.length} low confidence sections where pose detection failed or was weak.`,
    );
    for (const segment of segments.slice(0, 3)) {
      add(
        `  Frames ${segment.start_frame}–${segment.end_frame} ` +
          `(${formatDuration(segment.start_time)}–${formatDuration(segment.end_time)})`,
      );
    }
    if (segments.length > 3) {
      add(`  …and ${segments.length - 3} more section(s).`);
    }
  }

  return warnings;
}

function buildPoseJsonSummary(data: OutputJsonPayload): PoseJsonSummary {
  const poseFrames = data.pose_frames ?? [];
  const firstDetected = poseFrames.find((frame) => frame.pose_detected);

  return {
    video_metadata: data.video_metadata ?? {},
    quality: data.quality ?? {},
    pose_frame_count: poseFrames.length,
    first_detected_frame_index: firstDetected?.frame_index ?? null,
    first_detected_timestamp: firstDetected?.timestamp_seconds ?? null,
    first_detected_landmark_count: firstDetected?.landmarks?.length ?? null,
    first_detected_variant: firstDetected?.detection_variant ?? null,
  };
}

function assetToSelectedVideo(asset: ImagePickerAsset): SelectedVideo {
  let durationSeconds: number | null = null;
  if (typeof asset.duration === "number") {
    durationSeconds =
      asset.duration > 1000 ? asset.duration / 1000 : asset.duration;
  }

  return {
    uri: asset.uri,
    fileName: asset.fileName ?? null,
    durationSeconds,
    fileSizeBytes: asset.fileSize ?? null,
    mimeType: asset.mimeType ?? null,
  };
}

function isImagePickerResult(value: unknown): value is ImagePicker.ImagePickerResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "canceled" in value &&
    !("code" in value)
  );
}

/** Android native returns null (no pick) or a single result — not always an array. */
function normalizePendingPickerResults(
  raw: unknown,
): (ImagePicker.ImagePickerResult | ImagePicker.ImagePickerErrorResult)[] {
  if (raw == null) {
    return [];
  }
  if (Array.isArray(raw)) {
    return raw.filter(
      (entry): entry is
        | ImagePicker.ImagePickerResult
        | ImagePicker.ImagePickerErrorResult => entry != null,
    );
  }
  if (typeof raw === "object" && raw !== null) {
    if ("canceled" in raw || "code" in raw) {
      return [
        raw as
          | ImagePicker.ImagePickerResult
          | ImagePicker.ImagePickerErrorResult,
      ];
    }
  }
  return [];
}

async function recoverPendingVideoSelection(): Promise<SelectedVideo | null> {
  if (Platform.OS !== "android") {
    return null;
  }

  let raw: unknown;
  try {
    raw = await ImagePicker.getPendingResultAsync();
  } catch {
    return null;
  }

  const pending = normalizePendingPickerResults(raw);
  for (const entry of pending) {
    if (!isImagePickerResult(entry)) {
      console.error("[image-picker] pending error:", entry);
      continue;
    }
    if (entry.canceled || !entry.assets?.length) {
      continue;
    }
    return assetToSelectedVideo(entry.assets[0]);
  }
  return null;
}

function getStatusMessage(status: AppStatus): string | null {
  switch (status) {
    case "preparing":
      return "Preparing video...";
    case "uploading":
      return "Uploading video...";
    case "extracting":
      return "Extracting pose data...";
    case "complete":
      return "Processing complete.";
    case "idle":
      return "Ready to select a video.";
    case "video_selected":
      return "Video selected. Tap Process Video when ready.";
    case "failed":
      return null;
    default:
      return null;
  }
}

/** Reference video upload (stable). Do not change for live camera unless the shared API contract changes. */
async function uploadVideo(
  video: SelectedVideo,
  onUploading: () => void,
  onExtracting: () => void,
): Promise<ExtractionResponse> {
  const { url: backendBase, reachable } = await resolveBackendUrl();
  if (!reachable) {
    showBackendUnreachableAlert(backendBase);
    throw new Error("BACKEND_UNREACHABLE");
  }

  return new Promise((resolve, reject) => {
    const formData = new FormData();
    const name = video.fileName ?? "dance_video.mp4";
    const type = video.mimeType ?? "video/mp4";

    formData.append("file", {
      uri: video.uri,
      name,
      type,
    } as unknown as Blob);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${backendBase}/extract-pose`);
    xhr.timeout = 15 * 60 * 1000;

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        if (event.loaded >= event.total) {
          onExtracting();
        } else {
          onUploading();
        }
      } else {
        onUploading();
      }
    };

    xhr.upload.onload = () => {
      onExtracting();
    };

    xhr.onload = () => {
      const text = xhr.responseText ?? "";
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(text) as ExtractionResponse);
        } catch (parseError) {
          reject(
            new Error(
              `Invalid JSON response: ${
                parseError instanceof Error ? parseError.message : "parse error"
              }`,
            ),
          );
        }
        return;
      }
      reject(
        new Error(parseTechnicalError(xhr.status, text)),
      );
    };

    xhr.onerror = () => {
      reject(new Error("Network request failed."));
    };

    xhr.ontimeout = () => {
      reject(new Error("Request timed out."));
    };

    xhr.send(formData);
  });
}

type RootScreen = "mode_select" | "reference" | "practice";

function ModeSelectorScreen({
  onSelectReference,
  onSelectPractice,
}: {
  onSelectReference: () => void;
  onSelectPractice: () => void;
}) {
  return (
    <View style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Dance Coach</Text>
        <Text style={styles.instruction}>Choose how you want to use the app:</Text>

        <Pressable style={styles.button} onPress={onSelectReference}>
          <Text style={styles.buttonText}>Reference Video Mode</Text>
        </Pressable>

        <Pressable
          style={[styles.button, styles.buttonSecondary]}
          onPress={onSelectPractice}
        >
          <Text style={styles.buttonText}>Practice Mode</Text>
        </Pressable>

        <View style={styles.modeHintBox}>
          <Text style={styles.modeHintText}>
            Reference Video Mode uploads a dance video and extracts pose data for
            use as a reference. Practice Mode is for dancing along once a
            reference is ready.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

/** Hidden expo-video player for reference audio in Practice Mode only. */
const PracticeReferenceMusicPlayer = forwardRef<
  PracticeMusicPlayerHandle,
  {
    videoUrl: string;
    onLoadStateChange: (state: ReferenceMusicLoadState) => void;
    onPlaybackError?: () => void;
    onPlayToEnd?: () => void;
  }
>(function PracticeReferenceMusicPlayer(
  { videoUrl, onLoadStateChange, onPlaybackError, onPlayToEnd },
  ref,
) {
  const player = useVideoPlayer(videoUrl, (instance) => {
    instance.loop = false;
    instance.muted = false;
    instance.volume = 1;
  });

  useImperativeHandle(
    ref,
    () => ({
      playFromStart() {
        return new Promise<void>((resolve, reject) => {
          if (player.status === "error") {
            reject(new Error("Reference music is not ready to play."));
            return;
          }

          let settled = false;
          const cleanup = (
            playingSubscription: { remove: () => void },
            statusSubscription: { remove: () => void },
          ) => {
            playingSubscription.remove();
            statusSubscription.remove();
          };

          const timeoutId = setTimeout(() => {
            if (settled) {
              return;
            }
            settled = true;
            cleanup(playingSubscription, statusSubscription);
            reject(new Error("Reference music did not start in time."));
          }, MUSIC_PLAYBACK_START_TIMEOUT_MS);

          const playingSubscription = player.addListener(
            "playingChange",
            ({ isPlaying }) => {
              if (settled || !isPlaying) {
                return;
              }
              settled = true;
              clearTimeout(timeoutId);
              cleanup(playingSubscription, statusSubscription);
              resolve();
            },
          );

          const statusSubscription = player.addListener(
            "statusChange",
            ({ status, error }) => {
              if (settled || status !== "error") {
                return;
              }
              settled = true;
              clearTimeout(timeoutId);
              cleanup(playingSubscription, statusSubscription);
              reject(
                new Error(error?.message ?? "Reference music playback failed."),
              );
            },
          );

          player.currentTime = 0;
          player.play();

          if (player.playing) {
            settled = true;
            clearTimeout(timeoutId);
            cleanup(playingSubscription, statusSubscription);
            resolve();
          }
        });
      },
      stopAndReset() {
        try {
          player.pause();
          player.currentTime = 0;
        } catch {
          // Native player may already be released during teardown.
        }
      },
    }),
    [player],
  );

  useEffect(() => {
    onLoadStateChange("loading");

    const reportStatus = (status: string) => {
      if (status === "readyToPlay") {
        onLoadStateChange("ready");
      } else if (status === "error") {
        onLoadStateChange("error");
      }
    };

    reportStatus(player.status);

    const statusSubscription = player.addListener("statusChange", ({ status }) => {
      reportStatus(status);
      if (status === "error") {
        onPlaybackError?.();
      }
    });

    const playToEndSubscription = player.addListener("playToEnd", () => {
      onPlayToEnd?.();
    });

    return () => {
      statusSubscription.remove();
      playToEndSubscription.remove();
    };
  }, [player, onLoadStateChange, onPlaybackError, onPlayToEnd]);

  return (
    <VideoView
      player={player}
      style={styles.practiceHiddenVideo}
      nativeControls={false}
      contentFit="contain"
      allowsFullscreen={false}
      allowsPictureInPicture={false}
    />
  );
});

function PracticeModeScreen({ onBack }: { onBack: () => void }) {
  const [showReferenceList, setShowReferenceList] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selectableReferences, setSelectableReferences] = useState<
    CatalogReference[]
  >([]);
  const [loadingReferenceId, setLoadingReferenceId] = useState<string | null>(
    null,
  );
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadedReference, setLoadedReference] =
    useState<PracticeLoadedReference | null>(null);
  const [sessionState, setSessionState] =
    useState<PracticeSessionState>("no_reference");

  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const [checkFrameIndex, setCheckFrameIndex] = useState(0);
  const [fullBodyGuidance, setFullBodyGuidance] = useState<string[]>([]);
  const [fullBodyCheckError, setFullBodyCheckError] = useState<string | null>(
    null,
  );
  const [countdownValue, setCountdownValue] = useState<number | null>(null);
  const countdownTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [musicLoadState, setMusicLoadState] =
    useState<ReferenceMusicLoadState>("loading");
  const [practiceElapsedSeconds, setPracticeElapsedSeconds] = useState(0);
  const [sampledFrameCount, setSampledFrameCount] = useState(0);
  const [practiceMusicPlaybackState, setPracticeMusicPlaybackState] =
    useState<PracticeMusicPlaybackState>("idle");
  const [practiceSessionError, setPracticeSessionError] = useState<string | null>(
    null,
  );
  const [completedSessionSummary, setCompletedSessionSummary] =
    useState<PracticeSessionSummary | null>(null);
  const [practiceAnalysisResult, setPracticeAnalysisResult] =
    useState<PracticeAnalysisResult | null>(null);
  const [latestLivePose, setLatestLivePose] =
    useState<LatestLivePoseOverlay | null>(null);
  const musicPlayerRef = useRef<PracticeMusicPlayerHandle>(null);
  const practiceTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const practiceStartTimeRef = useRef<number | null>(null);
  const referenceDurationSecondsRef = useRef(0);
  const livePoseCollectionAbortRef = useRef(false);
  const livePoseCollectionRunningRef = useRef(false);
  const livePoseCollectionIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const livePoseRequestInFlightRef = useRef(false);
  const consecutiveLivePoseFailuresRef = useRef(0);
  const livePoseFramesRef = useRef<PracticeLivePoseFrame[]>([]);
  const fullBodyPassedRef = useRef(false);
  const completePracticeSessionRef = useRef<() => void>(() => {});
  const startLivePoseCollectionRef = useRef<() => void>(() => {});
  const startPracticeSessionRef = useRef<() => void>(() => {});
  const sessionEndHandledRef = useRef(false);
  const sessionStateRef = useRef<PracticeSessionState>("no_reference");
  const abortPracticeForMusicErrorRef = useRef<(message: string) => void>(() => {});
  const abortPracticeForLivePoseFailureRef = useRef<(message: string) => void>(
    () => {},
  );

  const isLoadingReference = loadingReferenceId !== null;
  const { backendUrl, backendReachable } = useResolvedBackendUrl();
  const canStartPractice =
    sessionState === "ready_to_start" &&
    meetsPracticeStartRequirements(
      loadedReference,
      musicLoadState,
      fullBodyPassedRef.current,
    );

  function clearCountdownTimer() {
    if (countdownTimerRef.current !== null) {
      clearTimeout(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
  }

  function clearPracticeTimer() {
    if (practiceTimerRef.current !== null) {
      clearInterval(practiceTimerRef.current);
      practiceTimerRef.current = null;
    }
  }

  function clearLivePoseCollectionTimer() {
    if (livePoseCollectionIntervalRef.current !== null) {
      clearInterval(livePoseCollectionIntervalRef.current);
      livePoseCollectionIntervalRef.current = null;
    }
  }

  function clearLivePoseFrames() {
    livePoseFramesRef.current = [];
    setSampledFrameCount(0);
    setLatestLivePose(null);
  }

  function clearStoredPracticeResults() {
    setCompletedSessionSummary(null);
    setPracticeAnalysisResult(null);
    setPracticeSessionError(null);
  }

  function resetFullBodyCheckState() {
    fullBodyPassedRef.current = false;
    setCheckFrameIndex(0);
    setFullBodyGuidance([]);
    setFullBodyCheckError(null);
  }

  function resetPracticeAttemptUI() {
    setCountdownValue(null);
    setPracticeElapsedSeconds(0);
    setPracticeMusicPlaybackState("idle");
  }

  function appendLivePoseFrame(frame: PracticeLivePoseFrame) {
    livePoseFramesRef.current.push(frame);
    setSampledFrameCount(livePoseFramesRef.current.length);
    setLatestLivePose({
      pose_detected: frame.pose_detected,
      landmarks: frame.landmarks as PoseLandmarkInput[],
    });
  }

  function stopLivePoseCollection() {
    livePoseCollectionAbortRef.current = true;
    livePoseCollectionRunningRef.current = false;
    clearLivePoseCollectionTimer();
  }

  function resetLivePoseCollection() {
    livePoseCollectionAbortRef.current = false;
    livePoseCollectionRunningRef.current = false;
    livePoseRequestInFlightRef.current = false;
    consecutiveLivePoseFailuresRef.current = 0;
    clearLivePoseCollectionTimer();
    clearLivePoseFrames();
  }

  function stopAllSessionResources() {
    clearCountdownTimer();
    clearPracticeTimer();
    stopLivePoseCollection();
    clearLivePoseFrames();
    musicPlayerRef.current?.stopAndReset();
    practiceStartTimeRef.current = null;
    referenceDurationSecondsRef.current = 0;
    sessionEndHandledRef.current = false;
    setPracticeMusicPlaybackState("idle");
  }

  function leavePracticeMode() {
    stopAllSessionResources();
    resetPracticeAttemptUI();
    clearStoredPracticeResults();
    onBack();
  }

  const finalizeSession = useCallback(() => {
    setSessionState("complete");
  }, []);

  const enterAnalyzing = useCallback((completedFully: boolean) => {
    if (sessionEndHandledRef.current) {
      return;
    }
    sessionEndHandledRef.current = true;

    const durationSeconds = referenceDurationSecondsRef.current;
    const startTimeMs = practiceStartTimeRef.current;
    const elapsedSeconds =
      startTimeMs === null
        ? 0
        : getElapsedPracticeSeconds(startTimeMs, durationSeconds);

    clearPracticeTimer();
    clearCountdownTimer();
    stopLivePoseCollection();
    musicPlayerRef.current?.stopAndReset();
    practiceStartTimeRef.current = null;
    referenceDurationSecondsRef.current = 0;
    setPracticeMusicPlaybackState("idle");

    setPracticeElapsedSeconds(elapsedSeconds);
    setCompletedSessionSummary({
      referenceDurationSeconds: durationSeconds,
      elapsedSeconds,
      livePoseFrames: [...livePoseFramesRef.current],
      completedFully,
    });
    setSessionState("analyzing");
  }, []);

  const resetForRetry = useCallback(() => {
    stopAllSessionResources();
    clearStoredPracticeResults();
    resetFullBodyCheckState();
    resetPracticeAttemptUI();
    setCameraReady(false);
    setSessionState("camera_ready");
  }, []);

  const abortPracticeForMusicError = useCallback((message: string) => {
    if (sessionStateRef.current !== "practicing") {
      return;
    }
    if (sessionEndHandledRef.current) {
      return;
    }
    sessionEndHandledRef.current = true;

    clearPracticeTimer();
    clearCountdownTimer();
    stopLivePoseCollection();
    musicPlayerRef.current?.stopAndReset();
    practiceStartTimeRef.current = null;
    referenceDurationSecondsRef.current = 0;
    setPracticeMusicPlaybackState("error");
    setPracticeSessionError(message);
    setSessionState("failed");
  }, []);

  const abortPracticeForLivePoseFailure = useCallback((message: string) => {
    if (sessionStateRef.current !== "practicing") {
      return;
    }
    if (sessionEndHandledRef.current) {
      return;
    }
    sessionEndHandledRef.current = true;

    clearPracticeTimer();
    clearCountdownTimer();
    stopLivePoseCollection();
    musicPlayerRef.current?.stopAndReset();
    practiceStartTimeRef.current = null;
    referenceDurationSecondsRef.current = 0;
    setPracticeMusicPlaybackState("idle");
    setPracticeSessionError(message);
    setSessionState("failed");
  }, []);

  const completePracticeSession = useCallback(() => {
    enterAnalyzing(true);
  }, [enterAnalyzing]);

  const failPracticeStart = useCallback((message: string) => {
    clearPracticeTimer();
    stopLivePoseCollection();
    musicPlayerRef.current?.stopAndReset();
    practiceStartTimeRef.current = null;
    referenceDurationSecondsRef.current = 0;
    setPracticeMusicPlaybackState("idle");
    setPracticeSessionError(message);
    setSessionState(
      fullBodyPassedRef.current ? "ready_to_start" : "camera_ready",
    );
  }, []);

  const startPracticeSession = useCallback(async () => {
    if (!loadedReference) {
      failPracticeStart("No reference loaded.");
      return;
    }

    if (!loadedReference.videoUrl?.trim()) {
      failPracticeStart("Reference media is missing.");
      return;
    }

    if (musicLoadState !== "ready") {
      failPracticeStart("Reference music is not ready.");
      return;
    }

    const durationSeconds = loadedReference.durationSeconds;
    if (durationSeconds <= 0) {
      failPracticeStart("Reference duration is invalid.");
      return;
    }

    if (!fullBodyPassedRef.current) {
      failPracticeStart("Run the full-body check before starting practice.");
      return;
    }

    resetLivePoseCollection();
    sessionEndHandledRef.current = false;
    referenceDurationSecondsRef.current = durationSeconds;
    practiceStartTimeRef.current = null;
    setPracticeElapsedSeconds(0);
    setSampledFrameCount(0);
    setCompletedSessionSummary(null);
    setPracticeAnalysisResult(null);
    setPracticeSessionError(null);
    setPracticeMusicPlaybackState("starting");
    clearPracticeTimer();

    try {
      await musicPlayerRef.current?.playFromStart();
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[practice] music failed to start:", technical);
      musicPlayerRef.current?.stopAndReset();
      setPracticeMusicPlaybackState("idle");
      setPracticeSessionError(
        "Reference music could not start. Check your connection and try again.",
      );
      setSessionState("ready_to_start");
      return;
    }

    practiceStartTimeRef.current = Date.now();
    setPracticeMusicPlaybackState("playing");
    startLivePoseCollectionRef.current();
    practiceTimerRef.current = setInterval(() => {
      const startTimeMs = practiceStartTimeRef.current;
      const referenceDuration = referenceDurationSecondsRef.current;
      if (startTimeMs === null || referenceDuration <= 0) {
        return;
      }

      const elapsedSeconds = getElapsedPracticeSeconds(
        startTimeMs,
        referenceDuration,
      );
      setSampledFrameCount(livePoseFramesRef.current.length);
      if (elapsedSeconds >= referenceDuration) {
        setPracticeElapsedSeconds(referenceDuration);
        completePracticeSessionRef.current();
        return;
      }
      setPracticeElapsedSeconds(elapsedSeconds);
    }, PRACTICE_TIMER_TICK_MS);
  }, [loadedReference, musicLoadState, failPracticeStart]);

  useEffect(() => {
    completePracticeSessionRef.current = completePracticeSession;
  }, [completePracticeSession]);

  useEffect(() => {
    startPracticeSessionRef.current = () => {
      void startPracticeSession();
    };
  }, [startPracticeSession]);

  useEffect(() => {
    sessionStateRef.current = sessionState;
  }, [sessionState]);

  useEffect(() => {
    abortPracticeForMusicErrorRef.current = abortPracticeForMusicError;
  }, [abortPracticeForMusicError]);

  useEffect(() => {
    abortPracticeForLivePoseFailureRef.current = abortPracticeForLivePoseFailure;
  }, [abortPracticeForLivePoseFailure]);

  const handleMusicPlaybackError = useCallback(() => {
    if (sessionStateRef.current !== "practicing") {
      return;
    }
    if (practiceStartTimeRef.current === null) {
      return;
    }
    abortPracticeForMusicErrorRef.current(
      "Reference music stopped unexpectedly. Your practice session was ended to keep timing accurate.",
    );
  }, []);

  const handleMusicPlayToEnd = useCallback(() => {
    if (sessionStateRef.current !== "practicing") {
      return;
    }
    if (sessionEndHandledRef.current) {
      return;
    }
    completePracticeSessionRef.current();
  }, []);

  useEffect(() => {
    if (sessionState !== "analyzing") {
      return;
    }
    if (!loadedReference || !completedSessionSummary) {
      return;
    }

    const analysis = analyzePracticeSession({
      referencePoseFrames: loadedReference.poseJson.pose_frames ?? [],
      livePoseFrames: completedSessionSummary.livePoseFrames,
      referenceDurationSeconds:
        completedSessionSummary.referenceDurationSeconds,
    });

    setPracticeAnalysisResult(analysis);
    finalizeSession();
  }, [
    sessionState,
    loadedReference,
    completedSessionSummary,
    finalizeSession,
  ]);

  useEffect(() => {
    if (
      sessionState !== "camera_ready" ||
      !fullBodyPassedRef.current ||
      !loadedReference
    ) {
      return;
    }
    if (
      meetsPracticeStartRequirements(
        loadedReference,
        musicLoadState,
        fullBodyPassedRef.current,
      )
    ) {
      setSessionState("ready_to_start");
    }
  }, [sessionState, musicLoadState, loadedReference]);

  useEffect(() => {
    void requestCameraPermission();
  }, [requestCameraPermission]);

  useEffect(() => {
    return () => {
      stopAllSessionResources();
    };
  }, []);

  useEffect(() => {
    fullBodyPassedRef.current = false;
    setCameraReady(false);
    setCheckFrameIndex(0);
    setFullBodyGuidance([]);
    setFullBodyCheckError(null);
    setCountdownValue(null);
    setMusicLoadState("loading");
    setPracticeElapsedSeconds(0);
    setCompletedSessionSummary(null);
    setPracticeAnalysisResult(null);
    resetLivePoseCollection();
    sessionEndHandledRef.current = false;

    if (!loadedReference) {
      setSessionState("no_reference");
      return;
    }

    setSessionState("reference_loaded");
  }, [loadedReference?.referenceId]);

  useEffect(() => {
    if (sessionState !== "countdown" || countdownValue === null) {
      if (sessionState !== "countdown") {
        clearCountdownTimer();
      }
      return;
    }

    countdownTimerRef.current = setTimeout(() => {
      setCountdownValue((current) => {
        if (current === null) {
          return null;
        }
        if (current <= 1) {
          setSessionState("practicing");
          startPracticeSessionRef.current();
          return null;
        }
        return current - 1;
      });
    }, PRACTICE_COUNTDOWN_TICK_MS);

    return () => {
      clearCountdownTimer();
    };
  }, [sessionState, countdownValue]);

  async function captureAndSendLivePoseFrame(): Promise<LivePoseFrameResponse> {
    if (!cameraRef.current) {
      throw new Error("Camera is not available.");
    }

    const photo = await cameraRef.current.takePictureAsync({
      quality: LIVE_FRAME_JPEG_QUALITY,
      imageType: "jpg",
    });

    if (!photo?.uri) {
      throw new Error("Could not capture a camera frame.");
    }

    const formData = new FormData();
    formData.append("file", {
      uri: photo.uri,
      name: "practice_frame.jpg",
      type: "image/jpeg",
    } as unknown as Blob);

    const response = await fetch(`${backendUrl}/extract-live-pose-frame`, {
      method: "POST",
      body: formData,
    });
    const bodyText = await response.text();

    if (!response.ok) {
      throw new Error(parseTechnicalError(response.status, bodyText));
    }

    try {
      return JSON.parse(bodyText) as LivePoseFrameResponse;
    } catch (parseError) {
      throw new Error(
        parseError instanceof Error
          ? parseError.message
          : "Invalid response from pose check",
      );
    }
  }

  const processLivePoseCapture = useCallback(async () => {
    if (livePoseCollectionAbortRef.current) {
      return;
    }

    const startTimeMs = practiceStartTimeRef.current;
    const referenceDuration = referenceDurationSecondsRef.current;
    if (startTimeMs === null || referenceDuration <= 0) {
      return;
    }

    const practiceElapsedSeconds = getElapsedPracticeSeconds(
      startTimeMs,
      referenceDuration,
    );
    if (practiceElapsedSeconds >= referenceDuration) {
      return;
    }

    if (livePoseRequestInFlightRef.current) {
      return;
    }

    livePoseRequestInFlightRef.current = true;
    const clientTimestamp = Date.now();
    const elapsedAtCapture = getElapsedPracticeSeconds(
      startTimeMs,
      referenceDuration,
    );
    const requestStartedAt = Date.now();

    try {
      const payload = await captureAndSendLivePoseFrame();
      const latencyMs = Date.now() - requestStartedAt;

      if (livePoseCollectionAbortRef.current) {
        return;
      }

      consecutiveLivePoseFailuresRef.current = 0;
      appendLivePoseFrame({
        practice_elapsed_seconds: elapsedAtCapture,
        client_timestamp: clientTimestamp,
        pose_detected: payload.pose_detected,
        full_body_visible: payload.quality?.full_body_visible ?? false,
        landmarks: payload.landmarks ?? [],
        world_landmarks: payload.world_landmarks ?? [],
        quality: (payload.quality ?? {}) as Record<string, unknown>,
        latency_ms: latencyMs,
        skipped_reason: null,
      });
    } catch (error) {
      const latencyMs = Date.now() - requestStartedAt;
      if (livePoseCollectionAbortRef.current) {
        return;
      }

      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[practice] live pose frame failed:", technical);

      appendLivePoseFrame({
        practice_elapsed_seconds: elapsedAtCapture,
        client_timestamp: clientTimestamp,
        pose_detected: false,
        full_body_visible: false,
        landmarks: [],
        world_landmarks: [],
        quality: {},
        latency_ms: latencyMs,
        skipped_reason: "request_failed",
      });

      consecutiveLivePoseFailuresRef.current += 1;
      if (
        consecutiveLivePoseFailuresRef.current >=
        MAX_CONSECUTIVE_LIVE_POSE_FAILURES
      ) {
        abortPracticeForLivePoseFailureRef.current(
          "Practice stopped because live pose extraction stopped responding.",
        );
      }
    } finally {
      livePoseRequestInFlightRef.current = false;
    }
  }, [backendUrl]);

  const startLivePoseCollection = useCallback(() => {
    clearLivePoseFrames();
    livePoseCollectionAbortRef.current = false;
    livePoseCollectionRunningRef.current = true;
    livePoseRequestInFlightRef.current = false;
    consecutiveLivePoseFailuresRef.current = 0;
    clearLivePoseCollectionTimer();

    livePoseCollectionIntervalRef.current = setInterval(() => {
      if (livePoseCollectionAbortRef.current) {
        clearLivePoseCollectionTimer();
        return;
      }

      const startTimeMs = practiceStartTimeRef.current;
      const referenceDuration = referenceDurationSecondsRef.current;
      if (startTimeMs === null || referenceDuration <= 0) {
        return;
      }

      const elapsedSeconds = getElapsedPracticeSeconds(
        startTimeMs,
        referenceDuration,
      );
      if (elapsedSeconds >= referenceDuration) {
        clearLivePoseCollectionTimer();
        return;
      }

      void processLivePoseCapture();
    }, LIVE_FRAME_INTERVAL_MS);

    void processLivePoseCapture();
  }, [processLivePoseCapture]);

  useEffect(() => {
    startLivePoseCollectionRef.current = startLivePoseCollection;
  }, [startLivePoseCollection]);

  async function handleCheckFullBody() {
    if (
      !cameraRef.current ||
      !cameraReady ||
      (sessionState !== "camera_ready" && sessionState !== "failed")
    ) {
      return;
    }

    setFullBodyCheckError(null);
    setFullBodyGuidance([]);
    clearCountdownTimer();
    setCountdownValue(null);
    setSessionState("checking_full_body");
    setCheckFrameIndex(0);

    const frameResults: LivePoseFrameResponse[] = [];
    let fullBodyPassCount = 0;

    try {
      for (let frame = 1; frame <= FULL_BODY_FRAME_COUNT; frame += 1) {
        setCheckFrameIndex(frame);
        const payload = await captureAndSendLivePoseFrame();
        frameResults.push(payload);

        if (payload.pose_detected && payload.quality?.full_body_visible) {
          fullBodyPassCount += 1;
        }

        if (frame < FULL_BODY_FRAME_COUNT) {
          await delay(FULL_BODY_FRAME_INTERVAL_MS);
        }
      }

      const passed = fullBodyPassCount >= FULL_BODY_PASS_MIN;
      fullBodyPassedRef.current = passed;

      if (passed) {
        setFullBodyGuidance([]);
        setSessionState(
          resolveReadySessionState(
            loadedReference,
            musicLoadState,
            true,
          ),
        );
      } else {
        setFullBodyGuidance(buildFullBodyGuidance(frameResults));
        setSessionState("failed");
      }
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[practice] full body check failed:", technical);
      fullBodyPassedRef.current = false;
      setCheckFrameIndex(0);
      setSessionState("failed");
      setFullBodyCheckError(
        "Could not check full body visibility. Check the backend and try again.",
      );
    }
  }

  function handleStartPractice() {
    if (sessionState !== "ready_to_start") {
      return;
    }
    if (
      !meetsPracticeStartRequirements(
        loadedReference,
        musicLoadState,
        fullBodyPassedRef.current,
      )
    ) {
      return;
    }
    setPracticeSessionError(null);
    setPracticeMusicPlaybackState("idle");
    clearCountdownTimer();
    setSessionState("countdown");
    setCountdownValue(PRACTICE_COUNTDOWN_START);
  }

  function handleCancelCountdown() {
    clearCountdownTimer();
    setCountdownValue(null);
    setSessionState("ready_to_start");
  }

  function handleStopPractice() {
    if (sessionState !== "practicing") {
      return;
    }
    enterAnalyzing(false);
  }

  function handleLeavePractice() {
    leavePracticeMode();
  }

  function handleTryAgain() {
    resetForRetry();
  }

  function handleChooseDifferentReference() {
    stopAllSessionResources();
    clearStoredPracticeResults();
    resetFullBodyCheckState();
    resetPracticeAttemptUI();
    setLoadedReference(null);
    setCameraReady(false);
    setLoadError(null);
    setShowReferenceList(true);
    setSessionState("no_reference");
    void handleLoadReference();
  }

  function handleLeaveFromSummary() {
    leavePracticeMode();
  }

  function handleCameraReady() {
    setCameraReady(true);
    setSessionState((current) =>
      current === "reference_loaded" ? "camera_ready" : current,
    );
  }

  async function handleLoadReference() {
    setShowReferenceList(true);
    setCatalogLoading(true);
    setCatalogError(null);
    setLoadError(null);

    try {
      const response = await fetch(`${backendUrl}/references`);
      const bodyText = await response.text();

      if (!response.ok) {
        throw new Error(parseTechnicalError(response.status, bodyText));
      }

      let payload: ReferencesResponse;
      try {
        payload = JSON.parse(bodyText) as ReferencesResponse;
      } catch (parseError) {
        throw new Error(
          parseError instanceof Error
            ? parseError.message
            : "Invalid JSON from /references",
        );
      }

      const references = Array.isArray(payload.references)
        ? payload.references
        : [];
      setSelectableReferences(references.filter(isPracticeSelectableReference));
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[practice] /references failed:", technical);
      setCatalogError(
        "Could not load saved references. Check that the backend is running.",
      );
      setSelectableReferences([]);
    } finally {
      setCatalogLoading(false);
    }
  }

  async function handleSelectReference(item: CatalogReference) {
    if (!isPracticeSelectableReference(item)) {
      return;
    }

    setLoadingReferenceId(item.reference_id);
    setLoadError(null);

    const jsonUrl = joinBackendUrl(item.json_url);
    const videoUrl = joinBackendUrl(item.reference_video_url!);

    try {
      const response = await fetch(jsonUrl);
      const bodyText = await response.text();

      if (!response.ok) {
        throw new Error(parseTechnicalError(response.status, bodyText));
      }

      let poseJson: OutputJsonPayload;
      try {
        poseJson = JSON.parse(bodyText) as OutputJsonPayload;
      } catch (parseError) {
        throw new Error(
          parseError instanceof Error
            ? parseError.message
            : "Invalid reference JSON",
        );
      }

      const durationSeconds = resolveReferenceDurationSeconds(
        poseJson,
        item.duration_seconds,
      );
      if (durationSeconds <= 0) {
        throw new Error("Reference duration is missing or invalid.");
      }

      const quality = poseJson.quality ?? {};
      const referenceQuality =
        quality.reference_quality ?? item.reference_quality ?? "Unknown";
      const posePctRaw =
        quality.pose_detection_percentage ?? item.pose_detection_percentage;
      const poseDetectionPercentage =
        posePctRaw !== null &&
        posePctRaw !== undefined &&
        !Number.isNaN(Number(posePctRaw))
          ? Number(posePctRaw)
          : 0;

      setLoadedReference({
        referenceId: item.reference_id,
        jsonFilename: item.json_filename,
        videoUrl,
        durationSeconds,
        referenceQuality: String(referenceQuality),
        poseDetectionPercentage,
        poseJson,
      });
      setShowReferenceList(false);
      setSessionState("reference_loaded");
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[practice] load reference failed:", technical);
      setLoadError("Could not load the selected reference. Try again.");
    } finally {
      setLoadingReferenceId(null);
    }
  }

  if (sessionState === "analyzing" && loadedReference) {
    return (
      <View style={styles.container}>
        <StatusBar style="dark" />
        <View style={styles.practiceAnalyzingContainer}>
          <ActivityIndicator size="large" color="#1a4d8f" />
          <Text style={styles.practiceAnalyzingTitle}>Analyzing session</Text>
          <Text style={styles.practiceAnalyzingText}>
            Comparing your practice to the reference…
          </Text>
        </View>
      </View>
    );
  }

  if (
    sessionState === "complete" &&
    loadedReference &&
    completedSessionSummary &&
    practiceAnalysisResult
  ) {
    const analysis = practiceAnalysisResult;
    const stoppedEarly = !completedSessionSummary.completedFully;
    const hasEnoughDataForScores =
      analysis.usable_comparison_frames >= MIN_USABLE_COMPARISON_FRAMES_FOR_SCORE;
    const baseTips = new Set<string>(PRACTICE_ANALYSIS_TIPS);
    const analysisTips = hasEnoughDataForScores
      ? [
          ...PRACTICE_ANALYSIS_TIPS,
          ...analysis.tips.filter((tip) => !baseTips.has(tip)),
        ]
      : [...INSUFFICIENT_PRACTICE_DATA_TIPS];

    return (
      <View style={styles.container}>
        <StatusBar style="dark" />
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <Text style={styles.title}>Practice Analysis</Text>
          <Text style={styles.instruction}>
            A friendly review of how your practice compared to the reference.
            This is not a final score.
          </Text>

          {stoppedEarly && (
            <View style={styles.practiceAnalysisNotice}>
              <Text style={styles.referenceListItemMeta}>
                Practice stopped early. Analysis is based only on the part you
                completed.
              </Text>
            </View>
          )}

          {!hasEnoughDataForScores ? (
            <>
              <View style={styles.summaryBox}>
                <Text style={styles.sectionTitle}>Analysis</Text>
                <Text style={styles.referenceListItemMeta}>
                  Not enough usable pose data for analysis.
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  Usable comparison frames: {analysis.usable_comparison_frames}
                </Text>
                {analysis.skipped_frames.partial_body > 0 && (
                  <Text style={styles.referenceListItemMeta}>
                    Frames skipped (body not fully visible):{" "}
                    {analysis.skipped_frames.partial_body}
                  </Text>
                )}
              </View>

              <View style={styles.summaryBox}>
                <Text style={styles.sectionTitle}>Tips</Text>
                {analysisTips.map((tip) => (
                  <Text key={tip} style={styles.practiceAnalysisTip}>
                    • {tip}
                  </Text>
                ))}
              </View>
            </>
          ) : (
            <>
              <View style={styles.summaryBox}>
                <Text style={styles.sectionTitle}>Match overview</Text>
                <Text style={styles.practiceAnalysisScore}>
                  Overall match: {formatMatchScore(analysis.overall_score)}
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  {describeOverallMatch(analysis.overall_score)}
                </Text>

                <Text style={styles.practiceAnalysisScore}>
                  Arms: {formatMatchScore(analysis.arms_score)}
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  {describeRegionMatch("arms", analysis.arms_score)}
                </Text>

                <Text style={styles.practiceAnalysisScore}>
                  Legs: {formatMatchScore(analysis.legs_score)}
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  {describeRegionMatch("legs", analysis.legs_score)}
                </Text>

                <Text style={styles.practiceAnalysisScore}>
                  Torso: {formatMatchScore(analysis.torso_score)}
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  {describeRegionMatch("torso", analysis.torso_score)}
                </Text>
              </View>

              <View style={styles.summaryBox}>
                <Text style={styles.sectionTitle}>Frames reviewed</Text>
                <Text style={styles.referenceListItemMeta}>
                  Usable comparison frames: {analysis.usable_comparison_frames}
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  Frames skipped (body not fully visible):{" "}
                  {analysis.skipped_frames.partial_body}
                </Text>
                {analysis.skipped_frames.partial_body > 0 && (
                  <Text style={styles.referenceListItemMeta}>
                    Some frames were skipped because your full body was not
                    visible.
                  </Text>
                )}
                <Text style={styles.referenceListItemMeta}>
                  Best moment:{" "}
                  {analysis.best_moment
                    ? `${formatDuration(analysis.best_moment.time_seconds)} (${analysis.best_moment.score}/100)`
                    : "Not enough data"}
                </Text>
                <Text style={styles.referenceListItemMeta}>
                  Needs work moment:{" "}
                  {analysis.needs_work_moment
                    ? `${formatDuration(analysis.needs_work_moment.time_seconds)} (${analysis.needs_work_moment.score}/100)`
                    : "Not enough data"}
                </Text>
              </View>

              <View style={styles.summaryBox}>
                <Text style={styles.sectionTitle}>Tips</Text>
                {analysisTips.map((tip) => (
                  <Text key={tip} style={styles.practiceAnalysisTip}>
                    • {tip}
                  </Text>
                ))}
              </View>
            </>
          )}

          <Pressable style={styles.button} onPress={handleTryAgain}>
            <Text style={styles.buttonText}>Try Again</Text>
          </Pressable>

          <Pressable
            style={[styles.button, styles.buttonOutline]}
            onPress={handleChooseDifferentReference}
          >
            <Text style={styles.buttonOutlineText}>Choose Different Reference</Text>
          </Pressable>

          <Pressable
            style={[styles.button, styles.buttonOutline, styles.backButton]}
            onPress={handleLeaveFromSummary}
          >
            <Text style={styles.buttonOutlineText}>Back</Text>
          </Pressable>
        </ScrollView>
      </View>
    );
  }

  if (loadedReference && isCameraSessionState(sessionState)) {
    const showCamera = cameraPermission?.granted === true;

    return (
      <View style={styles.practiceCameraContainer}>
        <StatusBar style="light" />
        <PracticeReferenceMusicPlayer
          key={loadedReference.referenceId}
          ref={musicPlayerRef}
          videoUrl={loadedReference.videoUrl}
          onLoadStateChange={setMusicLoadState}
          onPlaybackError={handleMusicPlaybackError}
          onPlayToEnd={handleMusicPlayToEnd}
        />
        {showCamera ? (
          <>
            <CameraView
              ref={cameraRef}
              style={styles.practiceCameraPreview}
              facing="front"
              onCameraReady={handleCameraReady}
            />
            {sessionState === "practicing" && (
              <PoseOverlay
                poseDetected={latestLivePose?.pose_detected ?? false}
                landmarks={latestLivePose?.landmarks ?? []}
              />
            )}
            {!cameraReady && (
              <View style={styles.practiceCameraLoading}>
                <ActivityIndicator size="large" color="#fff" />
                <Text style={styles.practiceCameraLoadingText}>
                  Starting camera…
                </Text>
              </View>
            )}
          </>
        ) : (
          <View style={styles.practiceCameraPermission}>
            {cameraPermission === null ? (
              <ActivityIndicator size="large" color="#1a4d8f" />
            ) : (
              <>
                <Text style={styles.practicePermissionText}>
                  Camera access is required for Practice Mode.
                </Text>
                <Pressable
                  style={styles.button}
                  onPress={() => void requestCameraPermission()}
                >
                  <Text style={styles.buttonText}>Allow camera</Text>
                </Pressable>
              </>
            )}
          </View>
        )}

        {sessionState === "countdown" && countdownValue !== null && (
          <View style={styles.practiceCountdownOverlay} pointerEvents="box-none">
            <Text style={styles.practiceCountdownNumber}>{countdownValue}</Text>
            <Text style={styles.practiceCountdownMessage}>
              Get ready. Music starts after the countdown.
            </Text>
          </View>
        )}

        {sessionState === "practicing" && (
          <View style={styles.practiceTimerOverlay} pointerEvents="box-none">
            <Text style={styles.practiceTimerLabel}>Practice</Text>
            <Text style={styles.practiceTimerValue}>
              Elapsed: {formatDuration(practiceElapsedSeconds)}
            </Text>
            <Text style={styles.practiceTimerMeta}>
              Total: {formatDuration(loadedReference.durationSeconds)}
            </Text>
            <Text style={styles.practiceTimerMeta}>
              Remaining:{" "}
              {formatDuration(
                getRemainingPracticeSeconds(
                  practiceElapsedSeconds,
                  loadedReference.durationSeconds,
                ),
              )}
            </Text>
            <Text
              style={[
                styles.practiceTimerMeta,
                practiceMusicPlaybackState === "playing" &&
                  styles.practiceMusicStatusOk,
                practiceMusicPlaybackState === "error" &&
                  styles.practiceMusicStatusWarn,
              ]}
            >
              Music:{" "}
              {practiceMusicPlaybackState === "starting"
                ? "Starting…"
                : practiceMusicPlaybackState === "playing"
                  ? "Playing"
                  : practiceMusicPlaybackState === "error"
                    ? "Error"
                    : "Idle"}
            </Text>
            <Text style={styles.practiceTimerMeta}>
              Sampled frames: {sampledFrameCount}
            </Text>
          </View>
        )}

        <View style={styles.practiceCameraOverlay} pointerEvents="box-none">
          {sessionState !== "countdown" && sessionState !== "practicing" && (
            <>
              <Text style={styles.practiceOverlayInstruction}>
                Stand far enough back so your full body is visible.
              </Text>
              <Text style={styles.practiceOverlayDuration}>
                Reference duration:{" "}
                {formatDuration(loadedReference.durationSeconds)}
              </Text>
              {musicLoadState === "loading" && (
                <Text style={styles.practiceMusicStatus}>
                  Loading reference music…
                </Text>
              )}
              {musicLoadState === "ready" && (
                <Text
                  style={[
                    styles.practiceMusicStatus,
                    styles.practiceMusicStatusOk,
                  ]}
                >
                  Music ready
                </Text>
              )}
              {musicLoadState === "error" && (
                <>
                  <Text
                    style={[
                      styles.practiceMusicStatus,
                      styles.practiceMusicStatusWarn,
                    ]}
                  >
                    Music could not be loaded
                  </Text>
                  <Text style={styles.practiceMusicError}>
                    Could not load reference music. Try extracting the reference
                    video again.
                  </Text>
                </>
              )}
              {practiceSessionError && (
                <Text style={styles.practiceMusicError}>{practiceSessionError}</Text>
              )}
            </>
          )}
          {cameraReady &&
            sessionState !== "countdown" &&
            sessionState !== "practicing" && (
            <>
              <Text
                style={[
                  styles.practiceFullBodyStatus,
                  sessionState === "ready_to_start" &&
                    styles.practiceFullBodyStatusOk,
                  sessionState === "failed" &&
                    styles.practiceFullBodyStatusWarn,
                ]}
              >
                {getFullBodyOverlayMessage(sessionState, checkFrameIndex)}
              </Text>
              {fullBodyGuidance.length > 0 && (
                <View style={styles.practiceGuidanceBox}>
                  {fullBodyGuidance.map((message) => (
                    <Text key={message} style={styles.practiceGuidanceText}>
                      • {message}
                    </Text>
                  ))}
                </View>
              )}
            </>
          )}
        </View>

        <View style={styles.practiceCameraFooter}>
          {sessionState === "practicing" ? (
            <>
              <Pressable
                style={[styles.button, styles.buttonSecondary]}
                onPress={handleStopPractice}
              >
                <Text style={styles.buttonText}>Stop Practice</Text>
              </Pressable>
              <Pressable
                style={[
                  styles.button,
                  styles.buttonOutline,
                  styles.practiceBackOnCamera,
                ]}
                onPress={handleLeavePractice}
              >
                <Text style={styles.buttonOutlineOnDarkText}>Leave</Text>
              </Pressable>
            </>
          ) : sessionState === "countdown" ? (
            <Pressable
              style={[styles.button, styles.buttonOutline, styles.practiceBackOnCamera]}
              onPress={handleCancelCountdown}
            >
              <Text style={styles.buttonOutlineOnDarkText}>Cancel</Text>
            </Pressable>
          ) : (
            <>
              <Pressable
                style={[
                  styles.button,
                  styles.buttonSecondary,
                  (!cameraReady || sessionState === "checking_full_body") &&
                    styles.buttonDisabled,
                ]}
                onPress={() => void handleCheckFullBody()}
                disabled={
                  !cameraReady || sessionState === "checking_full_body"
                }
              >
                {sessionState === "checking_full_body" ? (
                  <View style={styles.loadingRowInline}>
                    <ActivityIndicator color="#fff" />
                    <Text style={styles.buttonText}>Checking…</Text>
                  </View>
                ) : (
                  <Text style={styles.buttonText}>Check Full Body</Text>
                )}
              </Pressable>

              {fullBodyCheckError && (
                <Text style={styles.practiceFooterError}>{fullBodyCheckError}</Text>
              )}

              {practiceSessionError && (
                <Text style={styles.practiceFooterError}>{practiceSessionError}</Text>
              )}

              {musicLoadState === "error" && (
                <Text style={styles.practiceFooterError}>
                  Could not load reference music. Try extracting the reference
                  video again.
                </Text>
              )}

              <Pressable
                style={[
                  styles.button,
                  !canStartPractice && styles.buttonDisabled,
                ]}
                onPress={handleStartPractice}
                disabled={!canStartPractice}
              >
                <Text style={styles.buttonText}>Start Practice</Text>
              </Pressable>

              <Pressable
                style={[
                  styles.button,
                  styles.buttonOutline,
                  styles.practiceBackOnCamera,
                ]}
                onPress={handleLeavePractice}
                disabled={
                  isLoadingReference || sessionState === "checking_full_body"
                }
              >
                <Text style={styles.buttonOutlineOnDarkText}>Back</Text>
              </Pressable>
            </>
          )}
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Practice Mode</Text>
        <Text style={styles.instruction}>Load a reference first.</Text>

        <Pressable
          style={[
            styles.button,
            (catalogLoading || isLoadingReference) && styles.buttonDisabled,
          ]}
          onPress={() => void handleLoadReference()}
          disabled={catalogLoading || isLoadingReference}
        >
          {catalogLoading ? (
            <View style={styles.loadingRowInline}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.buttonText}>Loading references…</Text>
            </View>
          ) : (
            <Text style={styles.buttonText}>Load Reference</Text>
          )}
        </Pressable>

        {showReferenceList && (
          <View style={styles.referenceListBox}>
            <Text style={styles.sectionTitle}>Saved references</Text>
            {catalogLoading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator size="large" color="#1a4d8f" />
              </View>
            )}
            {catalogError && (
              <Text style={styles.errorText}>{catalogError}</Text>
            )}
            {!catalogLoading && !catalogError && (
              <>
                {selectableReferences.length === 0 ? (
                  <Text style={styles.practiceListEmpty}>
                    No references with pose JSON and saved video are available
                    yet. Extract a reference in Reference Video Mode first.
                  </Text>
                ) : (
                  selectableReferences.map((item) => {
                    const isLoading =
                      loadingReferenceId === item.reference_id;
                    return (
                      <Pressable
                        key={item.reference_id}
                        style={[
                          styles.referenceListItem,
                          (isLoadingReference && !isLoading) &&
                            styles.referenceListItemDisabled,
                        ]}
                        onPress={() => void handleSelectReference(item)}
                        disabled={isLoadingReference}
                      >
                        <Text style={styles.referenceListItemTitle}>
                          {item.json_filename}
                        </Text>
                        <Text style={styles.referenceListItemMeta}>
                          Duration: {formatDuration(item.duration_seconds)}
                        </Text>
                        <Text style={styles.referenceListItemMeta}>
                          Quality: {item.reference_quality ?? "Unknown"}
                        </Text>
                        <Text style={styles.referenceListItemMeta}>
                          Pose detection:{" "}
                          {formatPercent(item.pose_detection_percentage)}
                        </Text>
                        {isLoading && (
                          <ActivityIndicator
                            size="small"
                            color="#1a4d8f"
                            style={styles.referenceListItemSpinner}
                          />
                        )}
                      </Pressable>
                    );
                  })
                )}
              </>
            )}
          </View>
        )}

        {loadError && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{loadError}</Text>
          </View>
        )}

        <Pressable
          style={[styles.button, styles.buttonOutline, styles.backButton]}
          onPress={onBack}
          disabled={isLoadingReference}
        >
          <Text style={styles.buttonOutlineText}>Back</Text>
        </Pressable>

        <Text style={styles.backendNote}>
          Backend: {backendUrl}
          {backendReachable === false ? " (unreachable)" : ""}
        </Text>
      </ScrollView>
    </View>
  );
}

/** Reference upload/extract UI — stable; do not merge state with Practice Mode. */
function ReferenceVideoModeScreen() {
  const [status, setStatus] = useState<AppStatus>("idle");
  const [video, setVideo] = useState<SelectedVideo | null>(null);
  const [result, setResult] = useState<ExtractionResponse | null>(null);
  const [backendWarnings, setBackendWarnings] = useState<string[]>([]);
  const [jsonSummary, setJsonSummary] = useState<PoseJsonSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const processingLock = useRef(false);
  const pendingRecoveryChecked = useRef(false);
  const { backendUrl, backendReachable } = useResolvedBackendUrl();

  function applySelectedVideo(nextVideo: SelectedVideo) {
    resetResults();
    setVideo(nextVideo);
    setStatus("video_selected");
  }

  async function tryRecoverPendingSelection() {
    const recovered = await recoverPendingVideoSelection();
    if (recovered) {
      applySelectedVideo(recovered);
    }
  }

  useEffect(() => {
    if (pendingRecoveryChecked.current) {
      return;
    }
    pendingRecoveryChecked.current = true;
    void tryRecoverPendingSelection();
  }, []);

  useEffect(() => {
    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState === "active") {
        void tryRecoverPendingSelection();
      }
    });
    return () => subscription.remove();
  }, []);

  const isProcessing =
    status === "preparing" ||
    status === "uploading" ||
    status === "extracting";

  const canSelectVideo = !isProcessing;
  const canProcessVideo =
    video !== null &&
    !isProcessing &&
    (status === "video_selected" || status === "complete" || status === "failed");

  const canViewJsonSummary =
    status === "complete" && result !== null && !isProcessing && !summaryLoading;

  const showRecordingTips =
    status === "idle" ||
    status === "video_selected" ||
    status === "complete" ||
    status === "failed";

  const statusMessage = getStatusMessage(status);

  function resetResults() {
    setResult(null);
    setBackendWarnings([]);
    setJsonSummary(null);
    setSummaryError(null);
    setSummaryLoading(false);
  }

  async function handleViewJsonSummary() {
    if (!result?.output_filename) {
      return;
    }

    setSummaryLoading(true);
    setSummaryError(null);
    setJsonSummary(null);

    const url = `${backendUrl}/outputs/${encodeURIComponent(result.output_filename)}`;

    try {
      const response = await fetch(url);
      const bodyText = await response.text();

      if (!response.ok) {
        throw new Error(
          parseTechnicalError(response.status, bodyText),
        );
      }

      let payload: OutputJsonPayload;
      try {
        payload = JSON.parse(bodyText) as OutputJsonPayload;
      } catch (parseError) {
        throw new Error(
          parseError instanceof Error
            ? parseError.message
            : "Invalid JSON in output file",
        );
      }

      setJsonSummary(buildPoseJsonSummary(payload));
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[outputs] technical error:", technical);
      setSummaryError(
        "Could not load the pose JSON summary. Check that the backend is still running.",
      );
    } finally {
      setSummaryLoading(false);
    }
  }

  async function handleSelectVideo() {
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert(
          "Permission needed",
          "Please allow access to your photo library to select a dance video.",
        );
        return;
      }

      const pickerResult = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["videos"],
        allowsEditing: false,
      });

      if (!pickerResult.canceled && pickerResult.assets?.length) {
        applySelectedVideo(assetToSelectedVideo(pickerResult.assets[0]));
        return;
      }

      const recovered = await recoverPendingVideoSelection();
      if (recovered) {
        applySelectedVideo(recovered);
        return;
      }

      if (!pickerResult.canceled) {
        Alert.alert(
          "Could not use video",
          "The selected file could not be loaded. Try another video, or pick one saved in your gallery rather than a cloud-only copy.",
        );
      }
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[image-picker] selection failed:", technical);

      const recovered = await recoverPendingVideoSelection();
      if (recovered) {
        applySelectedVideo(recovered);
        return;
      }

      Alert.alert(
        "Could not open video",
        "Something went wrong while loading the video. Try again, or restart the app if this keeps happening.",
      );
    }
  }

  async function handleProcessVideo() {
    if (!video || processingLock.current) {
      return;
    }

    processingLock.current = true;
    resetResults();
    setStatus("preparing");

    try {
      await new Promise((resolve) => setTimeout(resolve, 300));
      setStatus("uploading");

      const response = await uploadVideo(
        video,
        () => setStatus("uploading"),
        () => setStatus("extracting"),
      );

      setResult(response);
      const fromQuality = collectBackendWarnings(response.quality ?? {});
      const fromApi = response.warnings ?? [];
      const merged: string[] = [];
      const seen = new Set<string>();
      for (const message of [...fromApi, ...fromQuality]) {
        if (message && !seen.has(message)) {
          seen.add(message);
          merged.push(message);
        }
      }
      setBackendWarnings(merged);
      setStatus("complete");
    } catch (error) {
      const technical =
        error instanceof Error ? error.message : String(error);
      console.error("[extract-pose] technical error:", technical);
      if (error instanceof Error && error.stack) {
        console.error("[extract-pose] stack:", error.stack);
      }
      if (
        technical !== "BACKEND_UNREACHABLE" &&
        (technical.includes("Network request failed") ||
          technical.includes("Network request timed out"))
      ) {
        showBackendUnreachableAlert(backendUrl);
      }
      setStatus("failed");
    } finally {
      processingLock.current = false;
    }
  }

  const meta = result?.video_metadata ?? {};
  const quality = result?.quality ?? {};
  const summary = result?.summary;

  return (
    <View style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Dance Pose Extractor MVP</Text>

        <Text style={styles.instruction}>
          Select a dance video. The app will extract body pose data so it can
          later be used as a reference dance video.
        </Text>

        {showRecordingTips && (
          <View style={styles.tipsBox}>
            <Text style={styles.sectionTitle}>Recording tips</Text>
            {RECORDING_TIPS.map((tip) => (
              <Text key={tip} style={styles.tipItem}>
                • {tip}
              </Text>
            ))}
          </View>
        )}

        <Pressable
          style={[styles.button, !canSelectVideo && styles.buttonDisabled]}
          onPress={handleSelectVideo}
          disabled={!canSelectVideo}
        >
          <Text style={styles.buttonText}>Select Video</Text>
        </Pressable>

        <Pressable
          style={[
            styles.button,
            styles.buttonSecondary,
            !canProcessVideo && styles.buttonDisabled,
          ]}
          onPress={handleProcessVideo}
          disabled={!canProcessVideo}
        >
          <Text style={styles.buttonText}>Process Video</Text>
        </Pressable>

        {video && (
          <View style={styles.infoBox}>
            <Text style={styles.sectionTitle}>Selected video</Text>
            <Text style={styles.infoLine}>
              File name: {video.fileName ?? "Unknown"}
            </Text>
            <Text style={styles.infoLine} selectable>
              URI: {video.uri}
            </Text>
            <Text style={styles.infoLine}>
              Duration: {formatDuration(video.durationSeconds)}
            </Text>
            <Text style={styles.infoLine}>
              File size: {formatFileSize(video.fileSizeBytes)}
            </Text>
          </View>
        )}

        <View style={styles.statusBox}>
          <Text style={styles.sectionTitle}>Status</Text>
          {isProcessing && (
            <View style={styles.loadingRow}>
              <ActivityIndicator size="large" color="#1a4d8f" />
            </View>
          )}
          {statusMessage && (
            <Text
              style={[
                styles.statusMessage,
                status === "complete" && styles.statusMessageSuccess,
              ]}
            >
              {statusMessage}
            </Text>
          )}
        </View>

        {status === "failed" && (
          <View style={styles.errorBox}>
            <Text style={styles.sectionTitle}>Could not process</Text>
            <Text style={styles.errorText}>{FAILURE_MESSAGE}</Text>
          </View>
        )}

        {status === "complete" && result && summary && (
          <View style={styles.resultBox}>
            <Text style={styles.sectionTitle}>Extraction complete</Text>
            <Text style={styles.infoLine}>
              Reference quality: {summary.reference_quality}
            </Text>
            <Text style={styles.infoLine}>
              Processed frames: {summary.processed_frames}
            </Text>
            <Text style={styles.infoLine}>
              Pose detected frames: {summary.pose_detected_frames}
            </Text>
            <Text style={styles.infoLine}>
              Pose detection:{" "}
              {formatPercent(summary.pose_detection_percentage)}
            </Text>
            <Text style={styles.infoLine}>
              Average visibility:{" "}
              {formatVisibility(quality.average_visibility)}
            </Text>
            <Text style={styles.infoLine}>
              Full body visibility:{" "}
              {formatPercent(quality.full_body_visibility_percentage)}
            </Text>
            <Text style={styles.infoLine}>
              FPS: {meta.fps ?? "Unknown"}
            </Text>
            <Text style={styles.infoLine}>
              Duration: {formatDuration(meta.duration_seconds)}
            </Text>
            <Text style={styles.infoLine}>
              Resolution: {formatResolution(meta.width, meta.height)}
            </Text>
            <Text style={styles.infoLine}>
              Output JSON: {result.output_filename}
            </Text>

            <Pressable
              style={[
                styles.button,
                styles.buttonOutline,
                !canViewJsonSummary && styles.buttonDisabled,
              ]}
              onPress={handleViewJsonSummary}
              disabled={!canViewJsonSummary}
            >
              {summaryLoading ? (
                <View style={styles.loadingRowInline}>
                  <ActivityIndicator color="#1a4d8f" />
                  <Text style={styles.buttonOutlineText}>Loading summary…</Text>
                </View>
              ) : (
                <Text style={styles.buttonOutlineText}>
                  View Pose JSON Summary
                </Text>
              )}
            </Pressable>
          </View>
        )}

        {summaryError && (
          <View style={styles.errorBox}>
            <Text style={styles.sectionTitle}>Summary error</Text>
            <Text style={styles.errorText}>{summaryError}</Text>
          </View>
        )}

        {jsonSummary && (
          <View style={styles.summaryBox}>
            <Text style={styles.sectionTitle}>Pose JSON summary</Text>
            <Text style={styles.summarySubtitle}>Video metadata</Text>
            <Text style={styles.infoLine}>
              FPS: {jsonSummary.video_metadata.fps ?? "Unknown"}
            </Text>
            <Text style={styles.infoLine}>
              Duration:{" "}
              {formatDuration(jsonSummary.video_metadata.duration_seconds)}
            </Text>
            <Text style={styles.infoLine}>
              Resolution:{" "}
              {formatResolution(
                jsonSummary.video_metadata.width,
                jsonSummary.video_metadata.height,
              )}
            </Text>
            <Text style={styles.infoLine}>
              Processed frames:{" "}
              {jsonSummary.video_metadata.processed_frames ?? "Unknown"}
            </Text>
            <Text style={styles.infoLine}>
              Pose detected frames:{" "}
              {jsonSummary.video_metadata.pose_detected_frames ?? "Unknown"}
            </Text>
            <Text style={styles.infoLine}>
              Frame stride:{" "}
              {jsonSummary.video_metadata.frame_stride ?? "Unknown"}
            </Text>

            <Text style={styles.summarySubtitle}>Quality metrics</Text>
            <Text style={styles.infoLine}>
              Reference quality:{" "}
              {jsonSummary.quality.reference_quality ?? "Unknown"}
            </Text>
            <Text style={styles.infoLine}>
              Pose detection:{" "}
              {formatPercent(jsonSummary.quality.pose_detection_percentage)}
            </Text>
            <Text style={styles.infoLine}>
              Average visibility:{" "}
              {formatVisibility(jsonSummary.quality.average_visibility)}
            </Text>
            <Text style={styles.infoLine}>
              Full body visibility:{" "}
              {formatPercent(jsonSummary.quality.full_body_visibility_percentage)}
            </Text>
            <Text style={styles.infoLine}>
              Low confidence sections:{" "}
              {jsonSummary.quality.low_confidence_segments?.length ?? 0}
            </Text>

            <Text style={styles.summarySubtitle}>Pose frames (summary only)</Text>
            <Text style={styles.infoLine}>
              Total pose frames in file: {jsonSummary.pose_frame_count}
            </Text>
            <Text style={styles.infoLine}>
              First detected frame index:{" "}
              {jsonSummary.first_detected_frame_index ?? "None"}
            </Text>
            <Text style={styles.infoLine}>
              First detected timestamp:{" "}
              {jsonSummary.first_detected_timestamp !== null
                ? formatDuration(jsonSummary.first_detected_timestamp)
                : "None"}
            </Text>
            <Text style={styles.infoLine}>
              Landmarks in first detected frame:{" "}
              {jsonSummary.first_detected_landmark_count ?? "None"}
            </Text>
            <Text style={styles.infoLine}>
              Detection variant:{" "}
              {jsonSummary.first_detected_variant ?? "None"}
            </Text>
            <Text style={styles.summaryNote}>
              Full per-frame landmarks are stored in the JSON file but not shown
              here to keep the app responsive.
            </Text>
          </View>
        )}

        {status === "complete" && backendWarnings.length > 0 && (
          <View style={styles.warningBox}>
            <Text style={styles.sectionTitle}>Warnings</Text>
            <Text style={styles.warningIntro}>
              The video was processed, but quality checks found possible issues:
            </Text>
            {backendWarnings.map((warning) => (
              <Text key={warning} style={styles.warningItem}>
                • {warning}
              </Text>
            ))}
          </View>
        )}

        <Text style={styles.backendNote}>
          Backend: {backendUrl}
          {backendReachable === false ? " (unreachable)" : ""}
        </Text>
      </ScrollView>
    </View>
  );
}

export default function App() {
  const [screen, setScreen] = useState<RootScreen>("mode_select");

  if (screen === "mode_select") {
    return (
      <ModeSelectorScreen
        onSelectReference={() => setScreen("reference")}
        onSelectPractice={() => setScreen("practice")}
      />
    );
  }

  if (screen === "practice") {
    return <PracticeModeScreen onBack={() => setScreen("mode_select")} />;
  }

  return <ReferenceVideoModeScreen />;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f7f7f7",
  },
  scrollContent: {
    padding: 20,
    paddingTop: 56,
    paddingBottom: 40,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111",
    marginBottom: 12,
  },
  instruction: {
    fontSize: 16,
    lineHeight: 22,
    color: "#333",
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#111",
    marginBottom: 8,
  },
  tipsBox: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  tipItem: {
    fontSize: 14,
    lineHeight: 20,
    color: "#444",
    marginBottom: 4,
  },
  button: {
    backgroundColor: "#1a4d8f",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    marginBottom: 10,
  },
  buttonSecondary: {
    backgroundColor: "#2d6a4f",
  },
  buttonOutline: {
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "#1a4d8f",
    marginTop: 12,
    marginBottom: 0,
  },
  buttonOutlineText: {
    color: "#1a4d8f",
    fontSize: 16,
    fontWeight: "600",
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  infoBox: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  infoLine: {
    fontSize: 14,
    color: "#333",
    marginBottom: 6,
  },
  statusBox: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#e0e0e0",
    minHeight: 72,
  },
  statusMessage: {
    fontSize: 16,
    color: "#1a4d8f",
    fontWeight: "600",
    marginTop: 8,
  },
  statusMessageSuccess: {
    color: "#2d6a4f",
  },
  loadingRow: {
    alignItems: "flex-start",
    marginBottom: 4,
  },
  loadingRowInline: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  errorBox: {
    backgroundColor: "#fdecea",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#f5c2c0",
  },
  errorText: {
    fontSize: 14,
    color: "#8a1c1c",
    lineHeight: 22,
  },
  resultBox: {
    backgroundColor: "#eaf4ea",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#b7ddb7",
  },
  summaryBox: {
    backgroundColor: "#eef4fb",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#b8cfe8",
  },
  summarySubtitle: {
    fontSize: 15,
    fontWeight: "600",
    color: "#1a4d8f",
    marginTop: 10,
    marginBottom: 6,
  },
  summaryNote: {
    fontSize: 12,
    color: "#555",
    marginTop: 10,
    lineHeight: 18,
    fontStyle: "italic",
  },
  warningBox: {
    backgroundColor: "#fff8e6",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#f0d78c",
  },
  warningIntro: {
    fontSize: 14,
    color: "#5c4a00",
    marginBottom: 8,
    lineHeight: 20,
  },
  warningItem: {
    fontSize: 14,
    color: "#5c4a00",
    lineHeight: 20,
    marginBottom: 6,
  },
  backendNote: {
    marginTop: 20,
    fontSize: 12,
    color: "#666",
  },
  modeHintBox: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  modeHintText: {
    fontSize: 14,
    lineHeight: 20,
    color: "#444",
  },
  backButton: {
    marginTop: 24,
    alignSelf: "flex-start",
    paddingHorizontal: 24,
  },
  practiceLoadedBox: {
    backgroundColor: "#eef4fb",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#b8cfe8",
  },
  referenceListBox: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 14,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  referenceListItem: {
    paddingVertical: 12,
    paddingHorizontal: 10,
    marginTop: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#c5d4e8",
    backgroundColor: "#f8fafc",
  },
  referenceListItemDisabled: {
    opacity: 0.5,
  },
  referenceListItemTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: "#111",
    marginBottom: 4,
  },
  referenceListItemMeta: {
    fontSize: 13,
    color: "#444",
    lineHeight: 18,
  },
  referenceListItemSpinner: {
    marginTop: 8,
  },
  practiceListEmpty: {
    fontSize: 14,
    lineHeight: 20,
    color: "#555",
  },
  practiceAnalyzingContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    backgroundColor: "#f7f7f7",
  },
  practiceAnalyzingTitle: {
    marginTop: 16,
    fontSize: 20,
    fontWeight: "700",
    color: "#111",
  },
  practiceAnalyzingText: {
    marginTop: 8,
    fontSize: 16,
    lineHeight: 22,
    color: "#444",
    textAlign: "center",
  },
  practiceAnalysisScore: {
    marginTop: 10,
    fontSize: 16,
    fontWeight: "700",
    color: "#1a4d8f",
  },
  practiceAnalysisTip: {
    fontSize: 14,
    lineHeight: 22,
    color: "#444",
    marginTop: 6,
  },
  practiceAnalysisNotice: {
    backgroundColor: "#fff8e6",
    borderRadius: 8,
    padding: 14,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#e8d48a",
  },
  practiceCameraContainer: {
    flex: 1,
    backgroundColor: "#000",
  },
  practiceCameraPreview: {
    ...StyleSheet.absoluteFillObject,
  },
  practiceCameraLoading: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
  },
  practiceCameraLoadingText: {
    marginTop: 12,
    fontSize: 16,
    color: "#fff",
    fontWeight: "600",
  },
  practiceCameraPermission: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    backgroundColor: "#f7f7f7",
  },
  practicePermissionText: {
    fontSize: 16,
    lineHeight: 22,
    color: "#333",
    textAlign: "center",
    marginBottom: 16,
  },
  practiceCountdownOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.35)",
    paddingHorizontal: 24,
  },
  practiceCountdownNumber: {
    fontSize: 96,
    fontWeight: "800",
    color: "#fff",
    lineHeight: 104,
    marginBottom: 16,
  },
  practiceCountdownMessage: {
    fontSize: 18,
    lineHeight: 26,
    color: "#f5f5f5",
    textAlign: "center",
    fontWeight: "600",
    maxWidth: 320,
  },
  practiceTimerOverlay: {
    position: "absolute",
    top: 56,
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    alignItems: "center",
  },
  practiceTimerLabel: {
    fontSize: 13,
    color: "#e8e8e8",
    fontWeight: "600",
    marginBottom: 4,
  },
  practiceTimerValue: {
    fontSize: 20,
    color: "#fff",
    fontWeight: "700",
  },
  practiceTimerMeta: {
    marginTop: 4,
    fontSize: 13,
    color: "#e8e8e8",
    fontWeight: "600",
    textAlign: "center",
  },
  practiceHiddenVideo: {
    position: "absolute",
    width: 1,
    height: 1,
    opacity: 0,
    bottom: 0,
    left: 0,
  },
  practiceMusicStatus: {
    marginTop: 10,
    fontSize: 15,
    lineHeight: 21,
    color: "#f5f5f5",
    fontWeight: "600",
  },
  practiceMusicStatusOk: {
    color: "#b7f0b7",
  },
  practiceMusicStatusWarn: {
    color: "#ffe08a",
  },
  practiceMusicError: {
    marginTop: 6,
    fontSize: 14,
    lineHeight: 20,
    color: "#ffb4b4",
  },
  practiceCameraOverlay: {
    position: "absolute",
    top: 56,
    left: 16,
    right: 16,
    backgroundColor: "rgba(0,0,0,0.45)",
    borderRadius: 8,
    padding: 14,
  },
  practiceOverlayInstruction: {
    fontSize: 16,
    lineHeight: 22,
    color: "#fff",
    fontWeight: "600",
    marginBottom: 8,
  },
  practiceOverlayDuration: {
    fontSize: 15,
    color: "#f0f0f0",
  },
  practiceFullBodyStatus: {
    marginTop: 10,
    fontSize: 15,
    lineHeight: 21,
    color: "#f5f5f5",
    fontWeight: "600",
  },
  practiceFullBodyStatusOk: {
    color: "#b7f0b7",
  },
  practiceFullBodyStatusWarn: {
    color: "#ffe08a",
  },
  practiceGuidanceBox: {
    marginTop: 8,
  },
  practiceGuidanceText: {
    fontSize: 14,
    lineHeight: 20,
    color: "#ffe08a",
    marginBottom: 4,
  },
  practiceFooterError: {
    fontSize: 13,
    color: "#ffb4b4",
    marginBottom: 8,
    lineHeight: 18,
  },
  practiceCameraFooter: {
    position: "absolute",
    bottom: 32,
    left: 20,
    right: 20,
    gap: 10,
  },
  practiceBackOnCamera: {
    backgroundColor: "rgba(255,255,255,0.92)",
    borderColor: "#fff",
    marginBottom: 0,
    marginTop: 4,
  },
  buttonOutlineOnDarkText: {
    color: "#1a4d8f",
    fontSize: 16,
    fontWeight: "600",
  },
});
