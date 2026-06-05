import {
  BODY_PART_GROUP_LANDMARKS,
  type BodyPartGroupName,
} from "./bodyPartGroups";
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

export type LandmarkErrorIssueLevel = "good" | "minor" | "needs_work" | "poor";

export type LandmarkError = {
  name: MajorDanceLandmarkName;
  distance: number;
  group: BodyPartGroupName[];
  issue_level: LandmarkErrorIssueLevel;
};

export type BiggestLandmarkError = {
  name: MajorDanceLandmarkName;
  distance: number;
  user_friendly_name: string;
};

/**
 * Normalized distance at which similarity reaches 0.
 * Uses a generous scale suited to post-practice review, not live coaching.
 */
/** Generous scale for post-practice review (avoids uniformly harsh demo labels). */
const SIMILARITY_DISTANCE_SCALE = 3;

const USER_FRIENDLY_LANDMARK_NAMES: Partial<
  Record<MajorDanceLandmarkName, string>
> = {
  left_wrist: "left hand",
  right_wrist: "right hand",
  left_ankle: "left foot",
  right_ankle: "right foot",
  left_knee: "left knee",
  right_knee: "right knee",
  left_elbow: "left elbow",
  right_elbow: "right elbow",
};

const LANDMARK_BODY_PART_GROUPS: Record<
  MajorDanceLandmarkName,
  BodyPartGroupName[]
> = MAJOR_DANCE_LANDMARKS.reduce(
  (groupsByLandmark, name) => {
    const groups: BodyPartGroupName[] = [];
    for (const groupName of Object.keys(
      BODY_PART_GROUP_LANDMARKS,
    ) as BodyPartGroupName[]) {
      if (
        BODY_PART_GROUP_LANDMARKS[groupName].some((landmark) => landmark === name)
      ) {
        groups.push(groupName);
      }
    }
    groupsByLandmark[name] = groups;
    return groupsByLandmark;
  },
  {} as Record<MajorDanceLandmarkName, BodyPartGroupName[]>,
);

export type PoseSimilarityResult = {
  overall_score: number;
  arms_score: number | null;
  legs_score: number | null;
  torso_score: number | null;
  left_arm_score: number | null;
  right_arm_score: number | null;
  left_leg_score: number | null;
  right_leg_score: number | null;
  left_side_score: number | null;
  right_side_score: number | null;
  shoulders_score: number | null;
  hips_score: number | null;
  matched_landmarks: number;
  missing_landmarks: string[];
  landmark_errors: LandmarkError[];
  biggest_errors: BiggestLandmarkError[];
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

export function distanceToSimilarityScore(distance: number): number {
  const raw = 100 * (1 - distance / SIMILARITY_DISTANCE_SCALE);
  return Math.round(Math.max(0, Math.min(100, raw)));
}

function distanceToIssueLevel(distance: number): LandmarkErrorIssueLevel {
  const score = distanceToSimilarityScore(distance);
  if (score >= 85) {
    return "good";
  }
  if (score >= 70) {
    return "minor";
  }
  if (score >= 50) {
    return "needs_work";
  }
  return "poor";
}

function scoreFromAverageDistance(distances: readonly number[]): number | null {
  if (distances.length === 0) {
    return null;
  }
  const averageDistance =
    distances.reduce((sum, distance) => sum + distance, 0) / distances.length;
  return distanceToSimilarityScore(averageDistance);
}

function distancesForGroup(
  distancesByLandmark: ReadonlyMap<MajorDanceLandmarkName, number>,
  landmarks: readonly MajorDanceLandmarkName[],
): number[] {
  const distances: number[] = [];
  for (const name of landmarks) {
    const distance = distancesByLandmark.get(name);
    if (distance !== undefined) {
      distances.push(distance);
    }
  }
  return distances;
}

export function getUserFriendlyLandmarkName(name: MajorDanceLandmarkName): string {
  return USER_FRIENDLY_LANDMARK_NAMES[name] ?? name.replace(/_/g, " ");
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
  const distancesByLandmark = new Map<MajorDanceLandmarkName, number>();

  for (const name of MAJOR_DANCE_LANDMARKS) {
    const referenceLandmark = getLandmarkByName(referencePose, name);
    const liveLandmark = getLandmarkByName(livePose, name);

    if (
      !referenceLandmark ||
      !liveLandmark ||
      !isLandmarkUsable(referenceLandmark) ||
      !isLandmarkUsable(liveLandmark)
    ) {
      missingLandmarks.push(name);
      continue;
    }

    distancesByLandmark.set(name, landmarkDistance(referenceLandmark, liveLandmark));
  }

  const matchedLandmarks = distancesByLandmark.size;
  const allDistances = [...distancesByLandmark.values()];

  const landmarkErrors: LandmarkError[] = [...distancesByLandmark.entries()]
    .map(([name, distance]) => ({
      name,
      distance,
      group: LANDMARK_BODY_PART_GROUPS[name],
      issue_level: distanceToIssueLevel(distance),
    }))
    .sort((a, b) => b.distance - a.distance);

  const biggestErrors: BiggestLandmarkError[] = landmarkErrors
    .slice(0, 3)
    .map(({ name, distance }) => ({
      name,
      distance,
      user_friendly_name: getUserFriendlyLandmarkName(name),
    }));

  return {
    overall_score: scoreFromAverageDistance(allDistances) ?? 0,
    arms_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.arms),
    ),
    legs_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.legs),
    ),
    torso_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.torso),
    ),
    left_arm_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.left_arm),
    ),
    right_arm_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.right_arm),
    ),
    left_leg_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.left_leg),
    ),
    right_leg_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.right_leg),
    ),
    left_side_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.left_side),
    ),
    right_side_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.right_side),
    ),
    shoulders_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.shoulders),
    ),
    hips_score: scoreFromAverageDistance(
      distancesForGroup(distancesByLandmark, BODY_PART_GROUP_LANDMARKS.hips),
    ),
    matched_landmarks: matchedLandmarks,
    missing_landmarks: missingLandmarks,
    landmark_errors: landmarkErrors,
    biggest_errors: biggestErrors,
  };
}
