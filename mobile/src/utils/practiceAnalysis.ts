import {
  normalizePoseForComparison,
  PoseLandmarkInput,
} from "./poseNormalization";
import { compareNormalizedPoses } from "./poseSimilarity";
import {
  findBestReferenceFrame,
  ReferencePoseFrameBase,
} from "./timelineMatching";

const SESSION_COMPLETE_TOLERANCE_SECONDS = 0.75;

export type PracticeLivePoseFrameInput = {
  practice_elapsed_seconds: number;
  pose_detected: boolean;
  full_body_visible: boolean;
  landmarks: readonly unknown[];
};

export type ReferencePoseFrameInput = ReferencePoseFrameBase & {
  landmarks?: readonly unknown[];
};

export type AnalyzePracticeSessionInput = {
  referencePoseFrames: readonly ReferencePoseFrameInput[];
  livePoseFrames: readonly PracticeLivePoseFrameInput[];
  referenceDurationSeconds: number;
};

export type SkippedFrameCounts = {
  no_live_pose: number;
  partial_body: number;
  no_reference_pose: number;
  normalization_failed: number;
};

export type PracticeMomentHighlight = {
  time_seconds: number;
  score: number;
};

export type PerFrameScore = {
  practice_elapsed_seconds: number;
  reference_timestamp_seconds: number;
  overall_score: number;
  arms_score: number | null;
  legs_score: number | null;
  torso_score: number | null;
  matched_landmarks: number;
};

export type PracticeAnalysisResult = {
  session_complete: boolean;
  reference_duration_seconds: number;
  sampled_live_frames: number;
  usable_comparison_frames: number;
  skipped_frames: SkippedFrameCounts;
  overall_score: number | null;
  arms_score: number | null;
  legs_score: number | null;
  torso_score: number | null;
  best_moment: PracticeMomentHighlight | null;
  needs_work_moment: PracticeMomentHighlight | null;
  per_frame_scores: PerFrameScore[];
  plain_language_summary: string[];
  tips: string[];
};

function isPoseLandmarkInput(value: unknown): value is PoseLandmarkInput {
  return (
    typeof value === "object" &&
    value !== null &&
    "name" in value &&
    typeof (value as { name: unknown }).name === "string"
  );
}

function asPoseLandmarks(landmarks: readonly unknown[] | undefined): PoseLandmarkInput[] {
  if (!landmarks) {
    return [];
  }
  return landmarks.filter(isPoseLandmarkInput);
}

function averageRoundedScores(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return Math.round(
    values.reduce((sum, value) => sum + value, 0) / values.length,
  );
}

function averageNullableScores(values: readonly (number | null)[]): number | null {
  const usable = values.filter((value): value is number => value !== null);
  return averageRoundedScores(usable);
}

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function deriveSessionComplete(
  maxElapsedSeconds: number,
  referenceDurationSeconds: number,
): boolean {
  if (!Number.isFinite(referenceDurationSeconds) || referenceDurationSeconds <= 0) {
    return false;
  }

  return (
    maxElapsedSeconds >= referenceDurationSeconds - SESSION_COMPLETE_TOLERANCE_SECONDS
  );
}

function findHighlightMoment(
  perFrameScores: readonly PerFrameScore[],
  mode: "best" | "needs_work",
): PracticeMomentHighlight | null {
  if (perFrameScores.length === 0) {
    return null;
  }

  let selected = perFrameScores[0];
  for (const frameScore of perFrameScores) {
    if (mode === "best") {
      if (frameScore.overall_score > selected.overall_score) {
        selected = frameScore;
      }
    } else if (frameScore.overall_score < selected.overall_score) {
      selected = frameScore;
    }
  }

  return {
    time_seconds: selected.practice_elapsed_seconds,
    score: selected.overall_score,
  };
}

function buildPlainLanguageSummary(
  sessionComplete: boolean,
  referenceDurationSeconds: number,
  maxElapsedSeconds: number,
  usableComparisonFrames: number,
  sampledLiveFrames: number,
  overallScore: number | null,
  armsScore: number | null,
  legsScore: number | null,
  torsoScore: number | null,
  bestMoment: PracticeMomentHighlight | null,
  needsWorkMoment: PracticeMomentHighlight | null,
): string[] {
  const summary: string[] = [];

  if (sessionComplete) {
    summary.push("You made it through the full reference track.");
  } else if (referenceDurationSeconds > 0) {
    summary.push(
      `You practiced for ${formatDuration(maxElapsedSeconds)} of ${formatDuration(referenceDurationSeconds)}.`,
    );
  } else {
    summary.push(`You practiced for ${formatDuration(maxElapsedSeconds)}.`);
  }

  if (usableComparisonFrames === 0) {
    summary.push(
      "There were not enough clear full-body moments to compare against the reference.",
    );
    return summary;
  }

  summary.push(
    `Reviewed ${usableComparisonFrames} of ${sampledLiveFrames} captured practice moments against the reference.`,
  );

  if (overallScore !== null) {
    if (overallScore >= 75) {
      summary.push("Your overall shape stayed close to the reference.");
    } else if (overallScore >= 50) {
      summary.push("Your overall shape matched the reference in several places.");
    } else {
      summary.push(
        "Your overall shape differed from the reference, which gives you clear moments to revisit.",
      );
    }
  }

  const groupScores: Array<{ label: string; score: number | null }> = [
    { label: "arms", score: armsScore },
    { label: "legs", score: legsScore },
    { label: "torso", score: torsoScore },
  ];
  const rankedGroups = groupScores
    .filter((group): group is { label: string; score: number } => group.score !== null)
    .sort((a, b) => b.score - a.score);

  if (rankedGroups.length >= 2) {
    summary.push(
      `${rankedGroups[0].label.charAt(0).toUpperCase()}${rankedGroups[0].label.slice(1)} tracked the reference most closely.`,
    );
    summary.push(
      `${rankedGroups[rankedGroups.length - 1].label.charAt(0).toUpperCase()}${rankedGroups[rankedGroups.length - 1].label.slice(1)} had the most room to revisit.`,
    );
  }

  if (bestMoment) {
    summary.push(
      `A strong match showed up around ${formatDuration(bestMoment.time_seconds)}.`,
    );
  }

  if (
    needsWorkMoment &&
    (!bestMoment || needsWorkMoment.time_seconds !== bestMoment.time_seconds)
  ) {
    summary.push(
      `A useful review point is around ${formatDuration(needsWorkMoment.time_seconds)}.`,
    );
  }

  return summary;
}

function buildTips(
  skippedFrames: SkippedFrameCounts,
  sampledLiveFrames: number,
  usableComparisonFrames: number,
  armsScore: number | null,
  legsScore: number | null,
  torsoScore: number | null,
  needsWorkMoment: PracticeMomentHighlight | null,
): string[] {
  const tips: string[] = [];

  if (sampledLiveFrames > 0) {
    const partialBodyRate = skippedFrames.partial_body / sampledLiveFrames;
    if (partialBodyRate >= 0.25) {
      tips.push(
        "Step back slightly so your shoulders, hips, knees, and ankles stay in frame throughout the dance.",
      );
    }
  }

  if (skippedFrames.no_reference_pose > 0) {
    tips.push(
      "Some practice moments could not be matched to the reference timing, usually where the reference video has missing pose data.",
    );
  }

  const groupScores: Array<{ label: string; score: number }> = [];
  if (armsScore !== null) {
    groupScores.push({ label: "arms", score: armsScore });
  }
  if (legsScore !== null) {
    groupScores.push({ label: "legs", score: legsScore });
  }
  if (torsoScore !== null) {
    groupScores.push({ label: "torso", score: torsoScore });
  }

  if (groupScores.length > 0) {
    groupScores.sort((a, b) => a.score - b.score);
    const focus = groupScores[0];
    if (focus.label === "arms") {
      tips.push(
        "Try finishing each arm line before moving on, especially during faster counts.",
      );
    } else if (focus.label === "legs") {
      tips.push(
        "Give yourself space to complete footwork and knee placement before transitioning.",
      );
    } else {
      tips.push(
        "Check that your shoulders and hips stay level with the reference during shape changes.",
      );
    }
  }

  if (usableComparisonFrames < 3) {
    tips.push(
      "Run the full track once more with steady framing to unlock a richer post-practice review.",
    );
  } else if (needsWorkMoment) {
    tips.push(
      `Rewatch the reference around ${formatDuration(needsWorkMoment.time_seconds)} and practice that section at a comfortable tempo.`,
    );
  }

  return tips.slice(0, 4);
}

/**
 * Compare captured live practice frames to time-aligned reference poses.
 * Intended for post-practice review only; does not provide live coaching.
 */
export function analyzePracticeSession({
  referencePoseFrames,
  livePoseFrames,
  referenceDurationSeconds,
}: AnalyzePracticeSessionInput): PracticeAnalysisResult {
  const skippedFrames: SkippedFrameCounts = {
    no_live_pose: 0,
    partial_body: 0,
    no_reference_pose: 0,
    normalization_failed: 0,
  };

  const perFrameScores: PerFrameScore[] = [];
  let maxElapsedSeconds = 0;

  for (const liveFrame of livePoseFrames) {
    if (Number.isFinite(liveFrame.practice_elapsed_seconds)) {
      maxElapsedSeconds = Math.max(
        maxElapsedSeconds,
        liveFrame.practice_elapsed_seconds,
      );
    }

    const referenceFrame = findBestReferenceFrame(
      referencePoseFrames,
      liveFrame.practice_elapsed_seconds,
    );
    if (!referenceFrame) {
      skippedFrames.no_reference_pose += 1;
      continue;
    }

    if (!liveFrame.pose_detected) {
      skippedFrames.no_live_pose += 1;
      continue;
    }

    if (!liveFrame.full_body_visible) {
      skippedFrames.partial_body += 1;
      continue;
    }

    const referenceLandmarks = asPoseLandmarks(referenceFrame.landmarks);
    const liveLandmarks = asPoseLandmarks(liveFrame.landmarks);

    const normalizedReference = normalizePoseForComparison(referenceLandmarks);
    const normalizedLive = normalizePoseForComparison(liveLandmarks);
    if (!normalizedReference || !normalizedLive) {
      skippedFrames.normalization_failed += 1;
      continue;
    }

    const comparison = compareNormalizedPoses(
      normalizedReference,
      normalizedLive,
    );

    perFrameScores.push({
      practice_elapsed_seconds: liveFrame.practice_elapsed_seconds,
      reference_timestamp_seconds: referenceFrame.timestamp_seconds,
      overall_score: comparison.overall_score,
      arms_score: comparison.arms_score,
      legs_score: comparison.legs_score,
      torso_score: comparison.torso_score,
      matched_landmarks: comparison.matched_landmarks,
    });
  }

  const sessionComplete = deriveSessionComplete(
    maxElapsedSeconds,
    referenceDurationSeconds,
  );
  const usableComparisonFrames = perFrameScores.length;
  const overallScore = averageRoundedScores(
    perFrameScores.map((frame) => frame.overall_score),
  );
  const armsScore = averageNullableScores(
    perFrameScores.map((frame) => frame.arms_score),
  );
  const legsScore = averageNullableScores(
    perFrameScores.map((frame) => frame.legs_score),
  );
  const torsoScore = averageNullableScores(
    perFrameScores.map((frame) => frame.torso_score),
  );
  const bestMoment = findHighlightMoment(perFrameScores, "best");
  const needsWorkMoment = findHighlightMoment(perFrameScores, "needs_work");

  return {
    session_complete: sessionComplete,
    reference_duration_seconds: referenceDurationSeconds,
    sampled_live_frames: livePoseFrames.length,
    usable_comparison_frames: usableComparisonFrames,
    skipped_frames: skippedFrames,
    overall_score: overallScore,
    arms_score: armsScore,
    legs_score: legsScore,
    torso_score: torsoScore,
    best_moment: bestMoment,
    needs_work_moment: needsWorkMoment,
    per_frame_scores: perFrameScores,
    plain_language_summary: buildPlainLanguageSummary(
      sessionComplete,
      referenceDurationSeconds,
      maxElapsedSeconds,
      usableComparisonFrames,
      livePoseFrames.length,
      overallScore,
      armsScore,
      legsScore,
      torsoScore,
      bestMoment,
      needsWorkMoment,
    ),
    tips: buildTips(
      skippedFrames,
      livePoseFrames.length,
      usableComparisonFrames,
      armsScore,
      legsScore,
      torsoScore,
      needsWorkMoment,
    ),
  };
}
