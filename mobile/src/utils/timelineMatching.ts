/** Search radius when the closest reference frame has no detected pose. */
export const REFERENCE_FRAME_NEARBY_WINDOW_SECONDS = 0.35;

export type ReferencePoseFrameBase = {
  timestamp_seconds: number;
  pose_detected: boolean;
};

function isFiniteTimestamp(value: number): boolean {
  return Number.isFinite(value);
}

function timestampDistance(
  frame: ReferencePoseFrameBase,
  elapsedSeconds: number,
): number {
  return Math.abs(frame.timestamp_seconds - elapsedSeconds);
}

function findClosestFrameByTime<T extends ReferencePoseFrameBase>(
  frames: readonly T[],
  elapsedSeconds: number,
): T | null {
  let closestFrame: T | null = null;
  let closestDistance = Infinity;

  for (const frame of frames) {
    if (!isFiniteTimestamp(frame.timestamp_seconds)) {
      continue;
    }

    const distance = timestampDistance(frame, elapsedSeconds);
    if (distance < closestDistance) {
      closestFrame = frame;
      closestDistance = distance;
    }
  }

  return closestFrame;
}

function findClosestDetectedFrameWithinWindow<T extends ReferencePoseFrameBase>(
  frames: readonly T[],
  elapsedSeconds: number,
  windowSeconds: number,
): T | null {
  let closestFrame: T | null = null;
  let closestDistance = Infinity;

  for (const frame of frames) {
    if (!frame.pose_detected || !isFiniteTimestamp(frame.timestamp_seconds)) {
      continue;
    }

    const distance = timestampDistance(frame, elapsedSeconds);
    if (distance > windowSeconds || distance >= closestDistance) {
      continue;
    }

    closestFrame = frame;
    closestDistance = distance;
  }

  return closestFrame;
}

/**
 * Pick the reference pose frame that aligns with a live practice timestamp.
 *
 * Matches one-to-one by reference video time: no DTW, no time warping, and no
 * comparison against a distant section of the dance.
 */
export function findBestReferenceFrame<T extends ReferencePoseFrameBase>(
  referencePoseFrames: readonly T[],
  elapsedSeconds: number,
): T | null {
  if (
    !isFiniteTimestamp(elapsedSeconds) ||
    referencePoseFrames.length === 0
  ) {
    return null;
  }

  const closestFrame = findClosestFrameByTime(
    referencePoseFrames,
    elapsedSeconds,
  );
  if (!closestFrame) {
    return null;
  }

  if (closestFrame.pose_detected) {
    return closestFrame;
  }

  return findClosestDetectedFrameWithinWindow(
    referencePoseFrames,
    elapsedSeconds,
    REFERENCE_FRAME_NEARBY_WINDOW_SECONDS,
  );
}
