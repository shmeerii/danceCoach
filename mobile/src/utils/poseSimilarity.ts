import {
  getLandmarkByName,
  NormalizedComparisonLandmark,
} from "./poseNormalization";

/** Minimum visibility for a landmark to count in post-practice comparison. */
export const POSE_SIMILARITY_VISIBILITY_THRESHOLD = 0.4;

/** Major landmarks used for single-frame pose similarity. */
export const MAJOR_DANCE_LANDMARKS = [
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
] as const;

export type MajorDanceLandmarkName = (typeof MAJOR_DANCE_LANDMARKS)[number];

type LandmarkGroup = "arms" | "legs" | "torso";

const LANDMARK_GROUP_BY_NAME: Record<MajorDanceLandmarkName, LandmarkGroup> = {
  left_shoulder: "torso",
  right_shoulder: "torso",
  left_elbow: "arms",
  right_elbow: "arms",
  left_wrist: "arms",
  right_wrist: "arms",
  left_hip: "torso",
  right_hip: "torso",
  left_knee: "legs",
  right_knee: "legs",
  left_ankle: "legs",
  right_ankle: "legs",
};

/**
 * Normalized distance at which similarity reaches 0.
 * Uses a generous scale suited to post-practice review, not live coaching.
 */
const SIMILARITY_DISTANCE_SCALE = 2;

export type PoseSimilarityResult = {
  overall_score: number;
  arms_score: number | null;
  legs_score: number | null;
  torso_score: number | null;
  matched_landmarks: number;
  missing_landmarks: string[];
  average_distance: number;
};

function isFiniteNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}

function isLandmarkUsable(
  landmark: NormalizedComparisonLandmark | undefined,
  threshold: number = POSE_SIMILARITY_VISIBILITY_THRESHOLD,
): boolean {
  if (!landmark) {
    return false;
  }
  if (
    !isFiniteNumber(landmark.x) ||
    !isFiniteNumber(landmark.y) ||
    !isFiniteNumber(landmark.z)
  ) {
    return false;
  }
  const visibility = landmark.visibility;
  if (visibility === null || visibility === undefined) {
    return true;
  }
  return visibility >= threshold;
}

function landmarkDistance(
  reference: NormalizedComparisonLandmark,
  live: NormalizedComparisonLandmark,
): number {
  const dx = reference.x - live.x;
  const dy = reference.y - live.y;
  const dz = reference.z - live.z;
  return Math.hypot(dx, dy, dz);
}

function distanceToSimilarityScore(distance: number): number {
  const raw = 100 * (1 - distance / SIMILARITY_DISTANCE_SCALE);
  return Math.round(Math.max(0, Math.min(100, raw)));
}

function averageScore(scores: number[]): number | null {
  if (scores.length === 0) {
    return null;
  }
  return Math.round(
    scores.reduce((sum, score) => sum + score, 0) / scores.length,
  );
}

/**
 * Compare two body-normalized poses for post-practice review.
 * Expects output from normalizePoseForComparison; does not compare raw poses.
 */
export function compareNormalizedPoses(
  referencePose: readonly NormalizedComparisonLandmark[],
  livePose: readonly NormalizedComparisonLandmark[],
): PoseSimilarityResult {
  const missingLandmarks: string[] = [];
  const distances: number[] = [];
  const scores: number[] = [];
  const groupScores: Record<LandmarkGroup, number[]> = {
    arms: [],
    legs: [],
    torso: [],
  };

  for (const name of MAJOR_DANCE_LANDMARKS) {
    const referenceLandmark = getLandmarkByName(referencePose, name);
    const liveLandmark = getLandmarkByName(livePose, name);

    if (!isLandmarkUsable(referenceLandmark) || !isLandmarkUsable(liveLandmark)) {
      missingLandmarks.push(name);
      continue;
    }

    const distance = landmarkDistance(referenceLandmark, liveLandmark);
    const score = distanceToSimilarityScore(distance);

    distances.push(distance);
    scores.push(score);
    groupScores[LANDMARK_GROUP_BY_NAME[name]].push(score);
  }

  const matchedLandmarks = distances.length;
  const averageDistance =
    matchedLandmarks > 0
      ? distances.reduce((sum, distance) => sum + distance, 0) /
        matchedLandmarks
      : 0;

  return {
    overall_score: averageScore(scores) ?? 0,
    arms_score: averageScore(groupScores.arms),
    legs_score: averageScore(groupScores.legs),
    torso_score: averageScore(groupScores.torso),
    matched_landmarks: matchedLandmarks,
    missing_landmarks: missingLandmarks,
    average_distance: averageDistance,
  };
}
