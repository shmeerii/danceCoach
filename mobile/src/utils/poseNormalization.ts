/** Visibility threshold aligned with backend pose extraction. */
export const COMPARISON_VISIBILITY_THRESHOLD = 0.5;

export const REQUIRED_COMPARISON_LANDMARKS = [
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
] as const;

export const FULL_BODY_COMPARISON_LANDMARKS = [
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
] as const;

export type PoseLandmarkInput = {
  id?: number;
  name: string;
  x?: number;
  y?: number;
  z?: number;
  visibility?: number | null;
};

export type NormalizedComparisonLandmark = {
  id: number;
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number | null;
};

type Point3D = {
  x: number;
  y: number;
  z: number;
};

function isFiniteNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}

function hasFiniteCoordinates(landmark: PoseLandmarkInput | undefined): boolean {
  if (!landmark) {
    return false;
  }
  return (
    isFiniteNumber(landmark.x) &&
    isFiniteNumber(landmark.y) &&
    isFiniteNumber(landmark.z)
  );
}

function midpoint(a: Point3D, b: Point3D): Point3D {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
  };
}

function distance2d(a: Point3D, b: Point3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function toPoint3D(landmark: PoseLandmarkInput): Point3D {
  return {
    x: landmark.x ?? 0,
    y: landmark.y ?? 0,
    z: landmark.z ?? 0,
  };
}

export function getLandmarkByName<T extends { name: string }>(
  landmarks: readonly T[],
  name: string,
): T | undefined {
  return landmarks.find((landmark) => landmark.name === name);
}

export function hasRequiredComparisonLandmarks(
  landmarks: readonly PoseLandmarkInput[],
): boolean {
  return REQUIRED_COMPARISON_LANDMARKS.every((name) =>
    hasFiniteCoordinates(getLandmarkByName(landmarks, name)),
  );
}

function isLandmarkVisibleForComparison(
  landmark: PoseLandmarkInput | undefined,
  threshold: number = COMPARISON_VISIBILITY_THRESHOLD,
): boolean {
  if (!landmark) {
    return false;
  }
  const visibility = landmark.visibility;
  if (visibility === null || visibility === undefined) {
    return true;
  }
  return visibility >= threshold;
}

export function calculatePoseVisibilityScore(
  landmarks: readonly PoseLandmarkInput[],
): number | null {
  const values: number[] = [];
  for (const landmark of landmarks) {
    if (isFiniteNumber(landmark.visibility)) {
      values.push(landmark.visibility);
    }
  }
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function isFullBodyUsableForComparison(
  landmarks: readonly PoseLandmarkInput[],
  threshold: number = COMPARISON_VISIBILITY_THRESHOLD,
): boolean {
  return FULL_BODY_COMPARISON_LANDMARKS.every((name) =>
    isLandmarkVisibleForComparison(getLandmarkByName(landmarks, name), threshold),
  );
}

/**
 * Translate pose landmarks to a body-centered coordinate system and scale them
 * so frames from different camera distances are more comparable.
 *
 * Returns a new landmark array, or null when required anchors are missing.
 */
export function normalizePoseForComparison(
  landmarks: readonly PoseLandmarkInput[],
): NormalizedComparisonLandmark[] | null {
  if (!hasRequiredComparisonLandmarks(landmarks)) {
    return null;
  }

  const leftShoulder = getLandmarkByName(landmarks, "left_shoulder")!;
  const rightShoulder = getLandmarkByName(landmarks, "right_shoulder")!;
  const leftHip = getLandmarkByName(landmarks, "left_hip")!;
  const rightHip = getLandmarkByName(landmarks, "right_hip")!;

  const shoulderCenter = midpoint(
    toPoint3D(leftShoulder),
    toPoint3D(rightShoulder),
  );
  const hipCenter = midpoint(toPoint3D(leftHip), toPoint3D(rightHip));
  const bodyCenter = midpoint(shoulderCenter, hipCenter);

  const shoulderWidth = distance2d(
    toPoint3D(leftShoulder),
    toPoint3D(rightShoulder),
  );
  const hipWidth = distance2d(toPoint3D(leftHip), toPoint3D(rightHip));
  const torsoLength = distance2d(shoulderCenter, hipCenter);

  const scale =
    shoulderWidth > 0
      ? shoulderWidth
      : hipWidth > 0
        ? hipWidth
        : torsoLength;

  if (!Number.isFinite(scale) || scale <= 0) {
    return null;
  }

  return landmarks.map((landmark, index) => {
    const x = landmark.x ?? 0;
    const y = landmark.y ?? 0;
    const z = landmark.z ?? 0;

    return {
      id: landmark.id ?? index,
      name: landmark.name,
      x: (x - bodyCenter.x) / scale,
      y: (y - bodyCenter.y) / scale,
      z: (z - bodyCenter.z) / scale,
      visibility:
        landmark.visibility === undefined ? null : landmark.visibility,
    };
  });
}
