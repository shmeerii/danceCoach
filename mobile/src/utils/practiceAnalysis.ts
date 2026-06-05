import type {
  AnalysisReliability,
  BodyPartBreakdown,
  BodyPartInsight,
  BodyPartIssueLevel,
  DanceSection,
  DetailedFeedback,
  DetailedFeedbackCategory,
  DetailedFeedbackSeverity,
  ImprovementPriority,
  PracticeAnalysisResult,
  ReferenceQualityBreakdown,
  ScoreBreakdown,
  SectionBreakdown,
  SessionSummary,
  SideBreakdown,
  TimeRange,
  TimingBreakdown,
  VisibilityBreakdown,
} from "../types/practiceAnalysis";
import {
  normalizePoseForComparison,
  PoseLandmarkInput,
} from "./poseNormalization";
import {
  compareNormalizedPoses,
  POSE_SIMILARITY_VISIBILITY_THRESHOLD,
  type PoseSimilarityResult,
} from "./poseSimilarity";
import {
  findBestReferenceFrame,
  ReferencePoseFrameBase,
} from "./timelineMatching";

export type { PracticeAnalysisResult } from "../types/practiceAnalysis";

const SESSION_COMPLETE_TOLERANCE_SECONDS = 0.75;
const OUT_OF_FRAME_OFTEN_RATE = 0.25;
const SECTION_TIMING_DROP_THRESHOLD = 15;
const SECTION_BELOW_AVERAGE_GAP = 12;
const SECTION_NEAR_AVERAGE_GAP = 8;
export const MIN_USABLE_FRAMES_FOR_RELIABLE_SCORES = 5;
const SECTION_LENGTH_SECONDS = 5;
const SHORT_REFERENCE_MAX_SECONDS = 20;
const SHORT_REFERENCE_SECTION_COUNT = 4;
export const MIN_SECTION_USABLE_FRAMES = 2;
const SECTION_GOOD_SCORE_THRESHOLD = 70;
const SIMILARITY_DISTANCE_SCALE = 3;
const MAX_IMPROVEMENT_PRIORITIES = 4;
const MAX_AFFECTED_TIME_RANGES_PER_PRIORITY = 3;

const SECTION_MAIN_ISSUE = {
  visibility: "Not enough visible body data",
  arms: "Arms need attention",
  legs: "Legs need attention",
  torso: "Body position needs attention",
} as const;
const REGION_SCORE_SIGNIFICANT_GAP = 5;
const REGION_LOW_SCORE_THRESHOLD = 65;
const ADJACENT_SECTION_SCORE_SWING_THRESHOLD = 15;
const SIDE_IMBALANCE_SCORE_GAP = 8;
const MAX_DETAILED_FEEDBACK = 6;
const FULL_BODY_VISIBILITY_GOOD_THRESHOLD = 85;
const FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD = 60;
const FULL_BODY_VISIBILITY_LIMITED_THRESHOLD = 30;
const MIN_USABLE_FRAME_PERCENT_FOR_HIGH_RELIABILITY = 45;
const MIN_USABLE_FRAME_PERCENT_FOR_LOW_RELIABILITY = 35;
const MIN_AVERAGE_POSE_QUALITY_FOR_HIGH = 55;
const REFERENCE_QUALITY_LIMITS_RELIABILITY = new Set(["poor", "failed"]);
const REFERENCE_MISSING_POSE_DETECTION_THRESHOLD = 75;
const REFERENCE_LOW_CONFIDENCE_SEGMENT_THRESHOLD = 2;
const MIN_NO_REFERENCE_POSE_SKIPS_FOR_MISSING_SECTIONS = 3;
const REFERENCE_QUALITY_SUGGESTION =
  "Try extracting a clearer reference video if results seem inaccurate.";

const LOW_BODY_REGION_SCORE_THRESHOLD = 55;

const USER_FEEDBACK = {
  positiveComplete:
    "Nice work finishing the full practice run with the music.",
  positiveStrongMoments:
    "You stayed close to the reference in several moments.",
  positiveBuilding:
    "Good effort — you are building the routine with each run.",
  positiveDefault:
    "Nice effort — keep dancing with the music and the shapes will click in.",
  allLimbsWeak:
    "The overall movement was different from the reference, so start by practicing the dance slowly with the music.",
  armsLowest: "Focus on hand height and elbow position.",
  legsLowest: "Focus on foot placement and knee bends.",
  torsoLowest: "Focus on facing the same direction as the reference.",
  timingWeak: "Practice entering each move on the beat.",
  partialVisibility:
    "Move farther back so your feet and hands stay in frame.",
  setupFullBody: "Try again with your full body visible.",
  setupPhoneStable: "Keep the phone stable.",
  setupLighting: "Use brighter lighting.",
} as const;

const WEAK_DATA_SETUP_SUGGESTIONS = [
  USER_FEEDBACK.setupFullBody,
  USER_FEEDBACK.setupPhoneStable,
  USER_FEEDBACK.setupLighting,
] as const;

type BodyRegionKey = "arms" | "legs" | "torso";

export const VISIBILITY_PROBLEM_LABELS = {
  feet_missing: "feet missing",
  hands_missing: "hands missing",
  body_too_close: "body too close",
  body_too_far: "body too far",
  low_lighting: "low lighting",
  no_body_detected: "no body detected",
  partial_body_detected: "partial body detected",
} as const;

export type VisibilityProblemLabel =
  (typeof VISIBILITY_PROBLEM_LABELS)[keyof typeof VISIBILITY_PROBLEM_LABELS];

export type PracticeLivePoseFrameInput = {
  practice_elapsed_seconds: number;
  pose_detected: boolean;
  full_body_visible: boolean;
  landmarks: readonly unknown[];
  quality?: Record<string, unknown>;
};

export type ReferencePoseFrameInput = ReferencePoseFrameBase & {
  landmarks?: readonly unknown[];
};

export type ReferenceQualityInput = {
  reference_quality?: string | null;
  pose_detection_percentage?: number | null;
  full_body_visibility_percentage?: number | null;
  low_confidence_segments?: readonly {
    start_time?: number;
    end_time?: number;
  }[];
};

export type AnalyzePracticeSessionInput = {
  referencePoseFrames: readonly ReferencePoseFrameInput[];
  livePoseFrames: readonly PracticeLivePoseFrameInput[];
  referenceDurationSeconds: number;
  referenceQuality?: ReferenceQualityInput;
};

type SkippedFrameCounts = {
  no_live_pose: number;
  partial_body: number;
  no_reference_pose: number;
  normalization_failed: number;
};

type PerFrameScore = {
  practice_elapsed_seconds: number;
  reference_timestamp_seconds: number;
  timing_offset_seconds: number;
  pose_match: PoseSimilarityResult;
};

function perFramePoseMatch(
  frame: PerFrameScore,
): PoseSimilarityResult {
  return frame.pose_match;
}

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

function roundPercentage(value: number): number {
  return Math.round(Math.max(0, Math.min(100, value)));
}

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function scoreToAverageError(score: number | null): number | null {
  if (score === null) {
    return null;
  }
  return Number(
    ((1 - score / 100) * SIMILARITY_DISTANCE_SCALE).toFixed(3),
  );
}

function deriveIssueLevel(score: number | null): BodyPartIssueLevel {
  if (score === null) {
    return "not_enough_data";
  }
  if (score >= 75) {
    return "good";
  }
  if (score >= 60) {
    return "minor";
  }
  if (score >= 45) {
    return "needs_work";
  }
  return "poor";
}

function issueMessage(label: string, score: number | null, issueLevel: BodyPartIssueLevel): string {
  if (issueLevel === "not_enough_data") {
    return `We will have more to say about your ${label} after a clearer practice run.`;
  }
  if (issueLevel === "good") {
    return `Your ${label} tracked the reference closely.`;
  }
  if (issueLevel === "minor") {
    return `Your ${label} were mostly aligned with small differences.`;
  }
  if (issueLevel === "needs_work") {
    return `Your ${label} drifted from the reference in several moments.`;
  }
  return `Your ${label} were often away from the reference shape.`;
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

function buildBodyPartInsight(
  label: string,
  score: number | null,
  averageError: number | null,
): BodyPartInsight {
  const issue_level = deriveIssueLevel(score);
  return {
    score,
    issue_level,
    message: issueMessage(label, score, issue_level),
    average_error: averageError,
  };
}

function buildBodyPartBreakdown(
  perFrameScores: readonly PerFrameScore[],
  torsoScore: number | null,
): BodyPartBreakdown {
  const leftArmScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).left_arm_score),
  );
  const rightArmScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).right_arm_score),
  );
  const leftLegScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).left_leg_score),
  );
  const rightLegScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).right_leg_score),
  );
  const shouldersScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).shoulders_score),
  );
  const hipsScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).hips_score),
  );

  return {
    left_arm: buildBodyPartInsight(
      "left arm",
      leftArmScore,
      scoreToAverageError(leftArmScore),
    ),
    right_arm: buildBodyPartInsight(
      "right arm",
      rightArmScore,
      scoreToAverageError(rightArmScore),
    ),
    left_leg: buildBodyPartInsight(
      "left leg",
      leftLegScore,
      scoreToAverageError(leftLegScore),
    ),
    right_leg: buildBodyPartInsight(
      "right leg",
      rightLegScore,
      scoreToAverageError(rightLegScore),
    ),
    shoulders: buildBodyPartInsight(
      "shoulders",
      shouldersScore,
      scoreToAverageError(shouldersScore),
    ),
    hips: buildBodyPartInsight(
      "hips",
      hipsScore,
      scoreToAverageError(hipsScore),
    ),
    torso: buildBodyPartInsight(
      "torso",
      torsoScore,
      scoreToAverageError(torsoScore),
    ),
  };
}

function buildSessionSummary(
  referenceDurationSeconds: number,
  maxElapsedSeconds: number,
  sessionComplete: boolean,
  sampledLiveFrames: number,
  usableComparisonFrames: number,
  skippedFrames: SkippedFrameCounts,
): SessionSummary {
  const skippedTotal =
    skippedFrames.no_live_pose +
    skippedFrames.partial_body +
    skippedFrames.no_reference_pose +
    skippedFrames.normalization_failed;

  return {
    reference_duration_seconds: referenceDurationSeconds,
    completed_duration_seconds: maxElapsedSeconds,
    session_complete: sessionComplete,
    sampled_live_frames: sampledLiveFrames,
    usable_comparison_frames: usableComparisonFrames,
    skipped_frames_total: skippedTotal,
    usable_frame_percentage:
      sampledLiveFrames > 0
        ? roundPercentage((usableComparisonFrames / sampledLiveFrames) * 100)
        : 0,
  };
}

function scoredSectionsForTiming(
  sectionBreakdown: SectionBreakdown,
): DanceSection[] {
  return sectionBreakdown.sections.filter(
    (section) =>
      section.average_score !== null &&
      section.usable_frames >= MIN_SECTION_USABLE_FRAMES,
  );
}

/**
 * Timing consistency from section scores along the reference timeline (no DTW / no time warp).
 */
function buildTimingBreakdown(
  sectionBreakdown: SectionBreakdown,
  perFrameScores: readonly PerFrameScore[],
): TimingBreakdown {
  const emptyTiming: TimingBreakdown = {
    on_time_percentage: null,
    early_movement_percentage: null,
    late_movement_percentage: null,
    timing_message:
      "Keep practicing with the music to line up your moves with the beat.",
  };

  const sections = scoredSectionsForTiming(sectionBreakdown);
  if (
    sections.length < 2 ||
    perFrameScores.length < MIN_USABLE_FRAMES_FOR_RELIABLE_SCORES
  ) {
    return emptyTiming;
  }

  const sectionScores = sections.map((section) => section.average_score as number);
  const sessionAverage = Math.round(
    sectionScores.reduce((sum, score) => sum + score, 0) / sectionScores.length,
  );

  let adjacentSharpDrops = 0;
  for (let index = 1; index < sections.length; index += 1) {
    const swing = Math.abs(
      (sections[index].average_score as number) -
        (sections[index - 1].average_score as number),
    );
    if (swing >= SECTION_TIMING_DROP_THRESHOLD) {
      adjacentSharpDrops += 1;
    }
  }

  let sectionsWellBelowAverage = 0;
  for (const section of sections) {
    if ((section.average_score as number) <= sessionAverage - SECTION_BELOW_AVERAGE_GAP) {
      sectionsWellBelowAverage += 1;
    }
  }

  let consistentSections = 0;
  for (let index = 0; index < sections.length; index += 1) {
    const score = sections[index].average_score as number;
    const nearSessionAverage =
      Math.abs(score - sessionAverage) <= SECTION_NEAR_AVERAGE_GAP;
    const steadyFromPrevious =
      index === 0 ||
      Math.abs(score - (sections[index - 1].average_score as number)) <
        SECTION_TIMING_DROP_THRESHOLD;
    if (nearSessionAverage && steadyFromPrevious) {
      consistentSections += 1;
    }
  }

  const onTimePercentage = roundPercentage(
    (consistentSections / sections.length) * 100,
  );

  const timingLooksStable =
    adjacentSharpDrops === 0 && sectionsWellBelowAverage === 0;
  const timingMessage = timingLooksStable
    ? "Your timing looked consistent across the dance."
    : "Some transitions may be off-time. Check the weaker time ranges and practice them with the music.";

  return {
    on_time_percentage: onTimePercentage,
    early_movement_percentage: null,
    late_movement_percentage: null,
    timing_message: timingMessage,
  };
}

function buildSideBreakdown(perFrameScores: readonly PerFrameScore[]): SideBreakdown {
  const leftScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).left_side_score),
  );
  const rightScore = averageNullableScores(
    perFrameScores.map((frame) => perFramePoseMatch(frame).right_side_score),
  );

  if (leftScore === null || rightScore === null) {
    return {
      left_side_score: leftScore,
      right_side_score: rightScore,
      side_balance_message:
        "We will compare your left and right sides after a clearer practice run.",
    };
  }

  const difference = Math.abs(leftScore - rightScore);
  let sideBalanceMessage = "Your left and right sides were fairly balanced.";
  if (difference >= SIDE_IMBALANCE_SCORE_GAP) {
    sideBalanceMessage =
      leftScore < rightScore
        ? "Your left side was less consistent than your right side."
        : "Your right side was less consistent than your left side.";
  }

  return {
    left_side_score: leftScore,
    right_side_score: rightScore,
    side_balance_message: sideBalanceMessage,
  };
}

type SectionBoundary = {
  start_time_seconds: number;
  end_time_seconds: number;
};

type SectionFrameCounts = {
  usable_frames: number;
  skipped_frames: number;
};

function deriveSectionBoundaries(durationSeconds: number): SectionBoundary[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return [];
  }

  if (durationSeconds < SHORT_REFERENCE_MAX_SECONDS) {
    const sectionDuration = durationSeconds / SHORT_REFERENCE_SECTION_COUNT;
    return Array.from({ length: SHORT_REFERENCE_SECTION_COUNT }, (_, index) => ({
      start_time_seconds: index * sectionDuration,
      end_time_seconds:
        index === SHORT_REFERENCE_SECTION_COUNT - 1
          ? durationSeconds
          : (index + 1) * sectionDuration,
    }));
  }

  const boundaries: SectionBoundary[] = [];
  let startSeconds = 0;
  while (startSeconds < durationSeconds) {
    const endSeconds = Math.min(
      startSeconds + SECTION_LENGTH_SECONDS,
      durationSeconds,
    );
    boundaries.push({
      start_time_seconds: startSeconds,
      end_time_seconds: endSeconds,
    });
    startSeconds = endSeconds;
  }
  return boundaries;
}

function frameInSection(
  elapsedSeconds: number,
  boundary: SectionBoundary,
  isLastSection: boolean,
): boolean {
  if (!Number.isFinite(elapsedSeconds)) {
    return false;
  }
  if (elapsedSeconds < boundary.start_time_seconds) {
    return false;
  }
  if (isLastSection) {
    return elapsedSeconds <= boundary.end_time_seconds;
  }
  return elapsedSeconds < boundary.end_time_seconds;
}

function sectionIndexForElapsed(
  elapsedSeconds: number,
  boundaries: readonly SectionBoundary[],
): number {
  if (!Number.isFinite(elapsedSeconds) || boundaries.length === 0) {
    return -1;
  }
  for (let index = 0; index < boundaries.length; index += 1) {
    if (
      frameInSection(
        elapsedSeconds,
        boundaries[index],
        index === boundaries.length - 1,
      )
    ) {
      return index;
    }
  }
  return -1;
}

function mainIssueForSection(
  usableFrames: number,
  averageScore: number | null,
  armsScore: number | null,
  legsScore: number | null,
  torsoScore: number | null,
): string {
  if (usableFrames < MIN_SECTION_USABLE_FRAMES) {
    return "Not enough visible body data";
  }

  const regions: Array<{ kind: "arms" | "legs" | "torso"; score: number }> = [];
  if (armsScore !== null) {
    regions.push({ kind: "arms", score: armsScore });
  }
  if (legsScore !== null) {
    regions.push({ kind: "legs", score: legsScore });
  }
  if (torsoScore !== null) {
    regions.push({ kind: "torso", score: torsoScore });
  }

  if (regions.length === 0) {
    return "Not enough visible body data";
  }

  const allRegionsGood = regions.every(
    (region) => region.score >= SECTION_GOOD_SCORE_THRESHOLD,
  );
  if (
    averageScore !== null &&
    averageScore >= SECTION_GOOD_SCORE_THRESHOLD &&
    allRegionsGood
  ) {
    return "Good section";
  }

  regions.sort((a, b) => a.score - b.score);
  const weakest = regions[0];
  if (weakest.kind === "arms") {
    return "Arms need attention";
  }
  if (weakest.kind === "legs") {
    return "Legs need attention";
  }
  return "Body position needs attention";
}

function pickBestAndWeakestSections(
  sections: readonly DanceSection[],
): { best_section: DanceSection | null; weakest_section: DanceSection | null } {
  const scoredSections = sections.filter(
    (section) => section.average_score !== null,
  );
  if (scoredSections.length === 0) {
    return { best_section: null, weakest_section: null };
  }

  let bestSection = scoredSections[0];
  let weakestSection = scoredSections[0];
  for (const section of scoredSections) {
    if (
      section.average_score !== null &&
      bestSection.average_score !== null &&
      section.average_score > bestSection.average_score
    ) {
      bestSection = section;
    }
    if (
      section.average_score !== null &&
      weakestSection.average_score !== null &&
      section.average_score < weakestSection.average_score
    ) {
      weakestSection = section;
    }
  }

  const sameSection =
    bestSection.start_time_seconds === weakestSection.start_time_seconds;
  return {
    best_section: bestSection,
    weakest_section: sameSection ? null : weakestSection,
  };
}

function buildSectionBreakdown(
  perFrameScores: readonly PerFrameScore[],
  durationSeconds: number,
  sectionFrameCounts: readonly SectionFrameCounts[],
): SectionBreakdown {
  const boundaries = deriveSectionBoundaries(durationSeconds);
  if (boundaries.length === 0) {
    return { best_section: null, weakest_section: null, sections: [] };
  }

  const sections: DanceSection[] = boundaries.map((boundary, index) => {
    const isLastSection = index === boundaries.length - 1;
    const framesInSection = perFrameScores.filter((frame) =>
      frameInSection(
        frame.practice_elapsed_seconds,
        boundary,
        isLastSection,
      ),
    );
    const counts = sectionFrameCounts[index] ?? {
      usable_frames: 0,
      skipped_frames: 0,
    };

    const armsScore = averageNullableScores(
      framesInSection.map((frame) => perFramePoseMatch(frame).arms_score),
    );
    const legsScore = averageNullableScores(
      framesInSection.map((frame) => perFramePoseMatch(frame).legs_score),
    );
    const torsoScore = averageNullableScores(
      framesInSection.map((frame) => perFramePoseMatch(frame).torso_score),
    );
    const averageScore = averageRoundedScores(
      framesInSection.map((frame) => perFramePoseMatch(frame).overall_score),
    );

    return {
      start_time_seconds: boundary.start_time_seconds,
      end_time_seconds: boundary.end_time_seconds,
      average_score: averageScore,
      arms_score: armsScore,
      legs_score: legsScore,
      torso_score: torsoScore,
      usable_frames: counts.usable_frames,
      skipped_frames: counts.skipped_frames,
      main_issue: mainIssueForSection(
        counts.usable_frames,
        averageScore,
        armsScore,
        legsScore,
        torsoScore,
      ),
    };
  });

  const { best_section, weakest_section } = pickBestAndWeakestSections(sections);
  return { best_section, weakest_section, sections };
}

type LiveVisibilityStats = {
  sampled_live_frames: number;
  pose_not_detected_frames: number;
  partial_body_frames: number;
  full_body_visible_percentage: number;
};

function isLandmarkVisibleInFrame(landmark: PoseLandmarkInput): boolean {
  const visibility = landmark.visibility;
  if (visibility === null || visibility === undefined) {
    return true;
  }
  return visibility >= POSE_SIMILARITY_VISIBILITY_THRESHOLD;
}

function extractFrameQualityWarnings(
  quality: Record<string, unknown> | undefined,
): string[] {
  if (!quality) {
    return [];
  }

  const messages: string[] = [];
  const seen = new Set<string>();
  const add = (value: unknown) => {
    if (typeof value !== "string" || !value.trim() || seen.has(value)) {
      return;
    }
    seen.add(value);
    messages.push(value);
  };

  if (Array.isArray(quality.warnings)) {
    for (const warning of quality.warnings) {
      add(warning);
    }
  }
  if (Array.isArray(quality.lighting_warnings)) {
    for (const warning of quality.lighting_warnings) {
      add(warning);
    }
  }
  if (Array.isArray(quality.video_warnings)) {
    for (const warning of quality.video_warnings) {
      add(warning);
    }
  }

  add(quality.body_size_warning);
  add(quality.edge_cutoff_warning);
  add(quality.angle_warning);

  return messages;
}

function visibilityProblemsFromWarnings(
  warnings: readonly string[],
): VisibilityProblemLabel[] {
  const problems = new Set<VisibilityProblemLabel>();

  for (const warning of warnings) {
    const lower = warning.toLowerCase();
    if (
      lower.includes("too far") ||
      lower.includes("too small") ||
      lower.includes("appear too far")
    ) {
      problems.add(VISIBILITY_PROBLEM_LABELS.body_too_far);
    }
    if (
      lower.includes("too close") ||
      lower.includes("too large") ||
      lower.includes("out of frame") ||
      lower.includes("partly out")
    ) {
      problems.add(VISIBILITY_PROBLEM_LABELS.body_too_close);
    }
    if (lower.includes("light") || lower.includes("dark")) {
      problems.add(VISIBILITY_PROBLEM_LABELS.low_lighting);
    }
    if (lower.includes("ankle") || lower.includes("foot") || lower.includes("feet")) {
      problems.add(VISIBILITY_PROBLEM_LABELS.feet_missing);
    }
    if (lower.includes("wrist") || lower.includes("hand")) {
      problems.add(VISIBILITY_PROBLEM_LABELS.hands_missing);
    }
  }

  return [...problems];
}

function visibilityProblemsFromLandmarks(
  landmarks: readonly PoseLandmarkInput[],
): VisibilityProblemLabel[] {
  const byName = new Map(landmarks.map((landmark) => [landmark.name, landmark]));
  const problems: VisibilityProblemLabel[] = [];

  const leftAnkle = byName.get("left_ankle");
  const rightAnkle = byName.get("right_ankle");
  if (
    !leftAnkle ||
    !rightAnkle ||
    !isLandmarkVisibleInFrame(leftAnkle) ||
    !isLandmarkVisibleInFrame(rightAnkle)
  ) {
    problems.push(VISIBILITY_PROBLEM_LABELS.feet_missing);
  }

  const leftWrist = byName.get("left_wrist");
  const rightWrist = byName.get("right_wrist");
  if (
    !leftWrist ||
    !rightWrist ||
    !isLandmarkVisibleInFrame(leftWrist) ||
    !isLandmarkVisibleInFrame(rightWrist)
  ) {
    problems.push(VISIBILITY_PROBLEM_LABELS.hands_missing);
  }

  return problems;
}

function collectFrameVisibilityProblems(
  frame: PracticeLivePoseFrameInput,
): VisibilityProblemLabel[] {
  const problems = new Set<VisibilityProblemLabel>();

  if (!frame.pose_detected) {
    problems.add(VISIBILITY_PROBLEM_LABELS.no_body_detected);
    return [...problems];
  }

  if (!frame.full_body_visible) {
    problems.add(VISIBILITY_PROBLEM_LABELS.partial_body_detected);
  }

  for (const problem of visibilityProblemsFromWarnings(
    extractFrameQualityWarnings(frame.quality),
  )) {
    problems.add(problem);
  }

  for (const problem of visibilityProblemsFromLandmarks(
    asPoseLandmarks(frame.landmarks),
  )) {
    problems.add(problem);
  }

  return [...problems];
}

function computeLiveVisibilityStats(
  livePoseFrames: readonly PracticeLivePoseFrameInput[],
): LiveVisibilityStats {
  const sampled = livePoseFrames.length;
  const poseNotDetectedFrames = livePoseFrames.filter(
    (frame) => !frame.pose_detected,
  ).length;
  const partialBodyFrames = livePoseFrames.filter(
    (frame) => frame.pose_detected && !frame.full_body_visible,
  ).length;
  const fullBodyVisibleCount = livePoseFrames.filter(
    (frame) => frame.full_body_visible,
  ).length;

  return {
    sampled_live_frames: sampled,
    pose_not_detected_frames: poseNotDetectedFrames,
    partial_body_frames: partialBodyFrames,
    full_body_visible_percentage:
      sampled > 0 ? roundPercentage((fullBodyVisibleCount / sampled) * 100) : 0,
  };
}

function mostCommonVisibilityProblem(
  livePoseFrames: readonly PracticeLivePoseFrameInput[],
): VisibilityProblemLabel | null {
  const problemCounts = new Map<VisibilityProblemLabel, number>();

  const bump = (problem: VisibilityProblemLabel) => {
    problemCounts.set(problem, (problemCounts.get(problem) ?? 0) + 1);
  };

  for (const frame of livePoseFrames) {
    for (const problem of collectFrameVisibilityProblems(frame)) {
      bump(problem);
    }
  }

  let topProblem: VisibilityProblemLabel | null = null;
  let topCount = 0;
  for (const [problem, count] of problemCounts.entries()) {
    if (count > topCount) {
      topProblem = problem;
      topCount = count;
    }
  }

  return topProblem;
}

function visibilityMessageFromPercentage(
  fullBodyVisiblePercentage: number,
  sampledLiveFrames: number,
): string {
  if (sampledLiveFrames === 0) {
    return "No practice frames were captured to review.";
  }
  if (fullBodyVisiblePercentage >= FULL_BODY_VISIBILITY_GOOD_THRESHOLD) {
    return "Your body stayed clearly visible for most of the practice.";
  }
  if (fullBodyVisiblePercentage >= FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD) {
    return "Your body was visible enough for analysis, but some moments were skipped.";
  }
  if (fullBodyVisiblePercentage >= FULL_BODY_VISIBILITY_LIMITED_THRESHOLD) {
    return "Step back so your full body stays in view for more of the dance.";
  }
  return "Move farther back so your feet and hands stay in frame.";
}

/** True when too little was captured for meaningful user-facing scores. */
export function hasSparsePracticeCapture(
  sessionSummary: SessionSummary,
): boolean {
  return (
    sessionSummary.sampled_live_frames === 0 ||
    sessionSummary.usable_comparison_frames <
      MIN_USABLE_FRAMES_FOR_RELIABLE_SCORES
  );
}

/** Soft intro when some feedback is possible but capture was incomplete. */
export function shouldShowPartialCaptureIntro(
  sessionSummary: SessionSummary,
): boolean {
  if (hasSparsePracticeCapture(sessionSummary)) {
    return false;
  }
  return (
    sessionSummary.usable_frame_percentage < 45 ||
    !sessionSummary.session_complete
  );
}

export function isSectionShowableForUser(section: DanceSection | null): boolean {
  return (
    section !== null &&
    section.usable_frames >= MIN_SECTION_USABLE_FRAMES &&
    section.average_score !== null
  );
}

/** Post-practice technique detail is only appropriate at medium+ reliability. */
export function isReliableEnoughForTechniqueFeedback(
  reliability: AnalysisReliability,
): boolean {
  return reliability.level === "high" || reliability.level === "medium";
}

function hasTrustworthyCameraData(
  visibilityBreakdown: VisibilityBreakdown,
  reliability: AnalysisReliability,
): boolean {
  if (!isReliableEnoughForTechniqueFeedback(reliability)) {
    return false;
  }
  if (reliability.level === "high") {
    return true;
  }
  return (
    visibilityBreakdown.full_body_visible_percentage >=
    FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD
  );
}

function readQualityNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value;
}

function normalizeQualityPercentage(value: number): number {
  if (value <= 1) {
    return roundPercentage(value * 100);
  }
  return roundPercentage(Math.min(100, Math.max(0, value)));
}

function computeAveragePoseDetectionQuality(
  livePoseFrames: readonly PracticeLivePoseFrameInput[],
): number | null {
  const qualityValues: number[] = [];

  for (const frame of livePoseFrames) {
    const quality = frame.quality;
    if (!quality) {
      continue;
    }

    const averageVisibility = readQualityNumber(quality.average_visibility);
    if (averageVisibility !== null) {
      qualityValues.push(normalizeQualityPercentage(averageVisibility));
      continue;
    }

    const fullBodyScore = readQualityNumber(quality.full_body_visibility_score);
    if (fullBodyScore !== null) {
      qualityValues.push(normalizeQualityPercentage(fullBodyScore));
    }
  }

  if (qualityValues.length > 0) {
    return roundPercentage(
      qualityValues.reduce((sum, value) => sum + value, 0) / qualityValues.length,
    );
  }

  if (livePoseFrames.length === 0) {
    return null;
  }

  let total = 0;
  for (const frame of livePoseFrames) {
    if (!frame.pose_detected) {
      continue;
    }
    total += frame.full_body_visible ? 100 : 40;
  }
  return roundPercentage(total / livePoseFrames.length);
}

function framingIssuePhrase(
  problem: VisibilityBreakdown["most_common_visibility_problem"],
): string | null {
  if (!problem) {
    return null;
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.feet_missing) {
    return "your feet were out of frame";
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.hands_missing) {
    return "your hands were out of frame";
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.partial_body_detected) {
    return "your full body was not visible";
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.no_body_detected) {
    return "your body was not detected";
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.body_too_close) {
    return "you were too close to the camera";
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.body_too_far) {
    return "you were too far from the camera";
  }
  if (problem === VISIBILITY_PROBLEM_LABELS.low_lighting) {
    return "lighting was low";
  }
  return problem;
}

function normalizeReferenceQualityValue(
  value: string | null | undefined,
): string | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  return value.trim().toLowerCase();
}

function computePoseDetectionPercentageFromFrames(
  referencePoseFrames: readonly ReferencePoseFrameInput[],
): number | null {
  if (referencePoseFrames.length === 0) {
    return null;
  }
  const detectedCount = referencePoseFrames.filter(
    (frame) => frame.pose_detected,
  ).length;
  return roundPercentage((detectedCount / referencePoseFrames.length) * 100);
}

function deriveReferenceQualityFromDetectionPercentage(
  poseDetectionPercentage: number | null,
): string | null {
  if (poseDetectionPercentage === null) {
    return null;
  }
  if (poseDetectionPercentage <= 20) {
    return "failed";
  }
  if (poseDetectionPercentage < 60) {
    return "poor";
  }
  if (poseDetectionPercentage >= 90) {
    return "good";
  }
  if (poseDetectionPercentage >= 75) {
    return "good";
  }
  return "usable";
}

function buildReferenceQualityBreakdown(
  referenceQualityInput: ReferenceQualityInput | undefined,
  referencePoseFrames: readonly ReferencePoseFrameInput[],
  skippedFrames: SkippedFrameCounts,
): ReferenceQualityBreakdown {
  const poseDetectionFromFrames = computePoseDetectionPercentageFromFrames(
    referencePoseFrames,
  );
  const pose_detection_percentage =
    readQualityNumber(referenceQualityInput?.pose_detection_percentage) ??
    poseDetectionFromFrames;
  const reference_quality =
    normalizeReferenceQualityValue(referenceQualityInput?.reference_quality) ??
    deriveReferenceQualityFromDetectionPercentage(pose_detection_percentage);
  const full_body_visibility_percentage = readQualityNumber(
    referenceQualityInput?.full_body_visibility_percentage,
  );
  const low_confidence_segment_count =
    referenceQualityInput?.low_confidence_segments?.length ?? 0;

  const limits_reliability =
    reference_quality !== null &&
    REFERENCE_QUALITY_LIMITS_RELIABILITY.has(reference_quality);
  const has_missing_pose_sections =
    (pose_detection_percentage !== null &&
      pose_detection_percentage < REFERENCE_MISSING_POSE_DETECTION_THRESHOLD) ||
    low_confidence_segment_count >= REFERENCE_LOW_CONFIDENCE_SEGMENT_THRESHOLD ||
    skippedFrames.no_reference_pose >=
      MIN_NO_REFERENCE_POSE_SKIPS_FOR_MISSING_SECTIONS;

  let summary_message =
    "The reference video looks suitable for comparison.";
  if (reference_quality === "failed") {
    summary_message =
      "The reference video had very limited pose data, which may restrict this analysis.";
  } else if (reference_quality === "poor") {
    summary_message =
      "The reference video had limited pose data, which may affect how much of the dance could be compared.";
  } else if (has_missing_pose_sections) {
    summary_message =
      "Some parts of the reference video had gaps in pose data.";
  }

  return {
    reference_quality,
    pose_detection_percentage,
    full_body_visibility_percentage,
    low_confidence_segment_count,
    limits_reliability,
    has_missing_pose_sections,
    summary_message,
    suggestion: REFERENCE_QUALITY_SUGGESTION,
  };
}

function applyReferenceQualityReliabilityCap(
  level: AnalysisReliability["level"],
  referenceQualityBreakdown: ReferenceQualityBreakdown,
): AnalysisReliability["level"] {
  if (referenceQualityBreakdown.reference_quality === "failed") {
    if (level === "high" || level === "medium") {
      return "low";
    }
  } else if (referenceQualityBreakdown.limits_reliability && level === "high") {
    return "medium";
  }
  return level;
}

function buildReliabilityMessage(
  level: AnalysisReliability["level"],
  sessionSummary: SessionSummary,
  visibilityBreakdown: VisibilityBreakdown,
  referenceQualityBreakdown: ReferenceQualityBreakdown,
): string {
  if (level === "not_enough_data") {
    return USER_FEEDBACK.setupFullBody;
  }

  if (referenceQualityBreakdown.limits_reliability) {
    return USER_FEEDBACK.setupFullBody;
  }

  if (level === "high") {
    return "Your full body stayed visible for most of the session.";
  }

  if (level === "medium") {
    const problem = visibilityBreakdown.most_common_visibility_problem;
    if (
      problem === VISIBILITY_PROBLEM_LABELS.feet_missing ||
      problem === VISIBILITY_PROBLEM_LABELS.hands_missing
    ) {
      return USER_FEEDBACK.partialVisibility;
    }
    return USER_FEEDBACK.partialVisibility;
  }

  if (sessionSummary.usable_frame_percentage < MIN_USABLE_FRAME_PERCENT_FOR_LOW_RELIABILITY) {
    return USER_FEEDBACK.setupFullBody;
  }
  if (!sessionSummary.session_complete) {
    return "Dance through the full song on your next try.";
  }
  return USER_FEEDBACK.partialVisibility;
}

function buildAnalysisReliability(
  sessionSummary: SessionSummary,
  visibilityBreakdown: VisibilityBreakdown,
  livePoseFrames: readonly PracticeLivePoseFrameInput[],
  referenceQualityBreakdown: ReferenceQualityBreakdown,
  skippedFrames: SkippedFrameCounts,
): AnalysisReliability {
  const reasons: string[] = [];
  const {
    usable_comparison_frames,
    sampled_live_frames,
    usable_frame_percentage,
    session_complete,
  } = sessionSummary;
  const fullBodyVisiblePercentage =
    visibilityBreakdown.full_body_visible_percentage;
  const averagePoseDetectionQuality =
    computeAveragePoseDetectionQuality(livePoseFrames);
  const skippedFramesTotal = Math.max(
    0,
    sampled_live_frames - usable_comparison_frames,
  );

  if (referenceQualityBreakdown.reference_quality) {
    reasons.push(
      `Reference video quality: ${referenceQualityBreakdown.reference_quality}.`,
    );
  }
  if (referenceQualityBreakdown.pose_detection_percentage !== null) {
    reasons.push(
      `Pose was detected in about ${referenceQualityBreakdown.pose_detection_percentage}% of reference frames.`,
    );
  }
  if (referenceQualityBreakdown.has_missing_pose_sections) {
    reasons.push(
      "The reference video had missing pose data in some sections.",
    );
  }
  if (referenceQualityBreakdown.limits_reliability) {
    reasons.push(referenceQualityBreakdown.suggestion);
  }
  if (averagePoseDetectionQuality !== null) {
    reasons.push(
      `Average live pose detection quality was about ${averagePoseDetectionQuality}%.`,
    );
  }
  reasons.push(
    `${usable_comparison_frames} of ${sampled_live_frames} captured moments were compared (${usable_frame_percentage}% usable).`,
  );
  reasons.push(
    `Full body was clearly visible in about ${fullBodyVisiblePercentage}% of captured moments.`,
  );
  if (!session_complete) {
    reasons.push("The practice session did not cover the full reference track.");
  }
  if (visibilityBreakdown.most_common_visibility_problem) {
    reasons.push(
      `Most common live visibility issue: ${visibilityBreakdown.most_common_visibility_problem}.`,
    );
  }
  if (skippedFrames.no_reference_pose > 0) {
    reasons.push(
      `${skippedFrames.no_reference_pose} captured moments could not be matched to nearby reference pose data.`,
    );
  }
  if (skippedFramesTotal > 0) {
    reasons.push(`${skippedFramesTotal} captured moments were skipped during analysis.`);
  }

  const buildResult = (
    level: AnalysisReliability["level"],
  ): AnalysisReliability => {
    const cappedLevel = applyReferenceQualityReliabilityCap(
      level,
      referenceQualityBreakdown,
    );
    return {
      level: cappedLevel,
      message: buildReliabilityMessage(
        cappedLevel,
        sessionSummary,
        visibilityBreakdown,
        referenceQualityBreakdown,
      ),
      reasons,
    };
  };

  if (
    sampled_live_frames === 0 ||
    usable_comparison_frames < MIN_USABLE_FRAMES_FOR_RELIABLE_SCORES
  ) {
    return buildResult("not_enough_data");
  }

  const qualifiesForHigh =
    session_complete &&
    usable_frame_percentage >= MIN_USABLE_FRAME_PERCENT_FOR_HIGH_RELIABILITY &&
    fullBodyVisiblePercentage >= FULL_BODY_VISIBILITY_GOOD_THRESHOLD &&
    (averagePoseDetectionQuality === null ||
      averagePoseDetectionQuality >= MIN_AVERAGE_POSE_QUALITY_FOR_HIGH) &&
    !referenceQualityBreakdown.limits_reliability &&
    (referenceQualityBreakdown.pose_detection_percentage === null ||
      referenceQualityBreakdown.pose_detection_percentage >=
        REFERENCE_MISSING_POSE_DETECTION_THRESHOLD);

  if (qualifiesForHigh) {
    return buildResult("high");
  }

  const qualifiesForLow =
    !session_complete ||
    fullBodyVisiblePercentage < FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD ||
    usable_frame_percentage < MIN_USABLE_FRAME_PERCENT_FOR_LOW_RELIABILITY ||
    fullBodyVisiblePercentage < FULL_BODY_VISIBILITY_LIMITED_THRESHOLD ||
    (averagePoseDetectionQuality !== null &&
      averagePoseDetectionQuality < MIN_AVERAGE_POSE_QUALITY_FOR_HIGH - 15) ||
    referenceQualityBreakdown.limits_reliability ||
    (referenceQualityBreakdown.has_missing_pose_sections &&
      (referenceQualityBreakdown.pose_detection_percentage === null ||
        referenceQualityBreakdown.pose_detection_percentage <
          REFERENCE_MISSING_POSE_DETECTION_THRESHOLD));

  if (qualifiesForLow) {
    return buildResult("low");
  }

  return buildResult("medium");
}

function buildVisibilityBreakdown(
  livePoseFrames: readonly PracticeLivePoseFrameInput[],
): VisibilityBreakdown {
  const stats = computeLiveVisibilityStats(livePoseFrames);
  const visibilityMessage = visibilityMessageFromPercentage(
    stats.full_body_visible_percentage,
    stats.sampled_live_frames,
  );

  return {
    full_body_visible_percentage: stats.full_body_visible_percentage,
    most_common_visibility_problem: mostCommonVisibilityProblem(livePoseFrames),
    visibility_message: visibilityMessage,
  };
}

function buildScoreBreakdown(
  perFrameScores: readonly PerFrameScore[],
  timingBreakdown: TimingBreakdown,
  sessionSummary: SessionSummary,
  visibilityBreakdown: VisibilityBreakdown,
): ScoreBreakdown {
  return {
    overall_score: averageRoundedScores(
      perFrameScores.map((frame) => perFramePoseMatch(frame).overall_score),
    ),
    arms_score: averageNullableScores(
      perFrameScores.map((frame) => perFramePoseMatch(frame).arms_score),
    ),
    legs_score: averageNullableScores(
      perFrameScores.map((frame) => perFramePoseMatch(frame).legs_score),
    ),
    torso_score: averageNullableScores(
      perFrameScores.map((frame) => perFramePoseMatch(frame).torso_score),
    ),
    timing_consistency_score: timingBreakdown.on_time_percentage,
    visibility_score: Math.round(
      (sessionSummary.usable_frame_percentage +
        visibilityBreakdown.full_body_visible_percentage) /
        2,
    ),
  };
}

function timeRangeFromSection(section: DanceSection): TimeRange {
  return {
    start_time_seconds: section.start_time_seconds,
    end_time_seconds: section.end_time_seconds,
  };
}

type ImprovementPriorityCandidate = {
  rank_score: number;
  title: string;
  explanation: string;
  suggestion: string;
  affected_time_ranges: TimeRange[];
};

function partialBodySkipRate(
  sessionSummary: SessionSummary,
  skippedFrames: SkippedFrameCounts,
): number {
  if (sessionSummary.sampled_live_frames <= 0) {
    return 0;
  }
  return skippedFrames.partial_body / sessionSummary.sampled_live_frames;
}

function timeRangeDuration(range: TimeRange): number {
  return Math.max(0, range.end_time_seconds - range.start_time_seconds);
}

function groupAdjacentSectionsByMainIssue(
  sections: readonly DanceSection[],
  targetIssue: string,
): TimeRange[] {
  const ranges: TimeRange[] = [];
  let activeStart: number | null = null;
  let activeEnd: number | null = null;

  const flush = (): void => {
    if (activeStart !== null && activeEnd !== null) {
      ranges.push({
        start_time_seconds: activeStart,
        end_time_seconds: activeEnd,
      });
    }
    activeStart = null;
    activeEnd = null;
  };

  for (const section of sections) {
    if (section.main_issue === targetIssue) {
      if (activeStart === null) {
        activeStart = section.start_time_seconds;
      }
      activeEnd = section.end_time_seconds;
    } else {
      flush();
    }
  }
  flush();

  return ranges;
}

function groupAdjacentWeakSectionsByScore(
  sections: readonly DanceSection[],
  scoreThreshold: number = SECTION_GOOD_SCORE_THRESHOLD,
): TimeRange[] {
  const ranges: TimeRange[] = [];
  let activeStart: number | null = null;
  let activeEnd: number | null = null;

  const flush = (): void => {
    if (activeStart !== null && activeEnd !== null) {
      ranges.push({
        start_time_seconds: activeStart,
        end_time_seconds: activeEnd,
      });
    }
    activeStart = null;
    activeEnd = null;
  };

  for (const section of sections) {
    const isWeak =
      section.average_score !== null &&
      section.average_score < scoreThreshold &&
      section.usable_frames >= MIN_SECTION_USABLE_FRAMES;

    if (isWeak) {
      if (activeStart === null) {
        activeStart = section.start_time_seconds;
      }
      activeEnd = section.end_time_seconds;
    } else {
      flush();
    }
  }
  flush();

  return ranges;
}

function limitAffectedTimeRanges(
  ranges: readonly TimeRange[],
  maxRanges: number = MAX_AFFECTED_TIME_RANGES_PER_PRIORITY,
): TimeRange[] {
  if (ranges.length <= maxRanges) {
    return [...ranges];
  }
  return [...ranges]
    .sort((a, b) => timeRangeDuration(b) - timeRangeDuration(a))
    .slice(0, maxRanges)
    .sort((a, b) => a.start_time_seconds - b.start_time_seconds);
}

function affectedRangesForMainIssue(
  sections: readonly DanceSection[],
  targetIssue: string,
  fallbackSection: DanceSection | null = null,
): TimeRange[] {
  const grouped = groupAdjacentSectionsByMainIssue(sections, targetIssue);
  if (grouped.length > 0) {
    return limitAffectedTimeRanges(grouped);
  }
  if (fallbackSection?.main_issue === targetIssue) {
    return [timeRangeFromSection(fallbackSection)];
  }
  return [];
}

function affectedRangesForWeakSections(
  sections: readonly DanceSection[],
  fallbackSection: DanceSection | null = null,
): TimeRange[] {
  const grouped = groupAdjacentWeakSectionsByScore(sections);
  if (grouped.length > 0) {
    return limitAffectedTimeRanges(grouped);
  }
  if (
    fallbackSection &&
    fallbackSection.average_score !== null &&
    fallbackSection.average_score < SECTION_GOOD_SCORE_THRESHOLD
  ) {
    return [timeRangeFromSection(fallbackSection)];
  }
  return [];
}

function largestAdjacentSectionSwing(
  sections: readonly DanceSection[],
): { swing: number; ranges: TimeRange[] } {
  const scored = sections.filter((section) => section.average_score !== null);
  if (scored.length < 2) {
    return { swing: 0, ranges: [] };
  }

  let maxSwing = 0;
  let ranges: TimeRange[] = [];
  for (let index = 1; index < scored.length; index += 1) {
    const previous = scored[index - 1];
    const current = scored[index];
    const swing = Math.abs(
      (current.average_score ?? 0) - (previous.average_score ?? 0),
    );
    if (swing > maxSwing) {
      maxSwing = swing;
      ranges = [
        timeRangeFromSection(previous),
        timeRangeFromSection(current),
      ];
    }
  }

  return { swing: maxSwing, ranges };
}

/** Gap when `target` trails both comparators; otherwise 0 (avoids false positives). */
function regionScoreGapBelowBoth(
  target: number | null,
  otherA: number | null,
  otherB: number | null,
): number {
  if (target === null || otherA === null || otherB === null) {
    return 0;
  }
  if (otherA <= target || otherB <= target) {
    return 0;
  }
  return Math.min(otherA - target, otherB - target);
}

function rankedBodyRegionsFromScores(
  scoreBreakdown: ScoreBreakdown,
): Array<{ region: BodyRegionKey; score: number }> {
  const regions: Array<{ region: BodyRegionKey; score: number }> = [];
  if (scoreBreakdown.arms_score !== null) {
    regions.push({ region: "arms", score: scoreBreakdown.arms_score });
  }
  if (scoreBreakdown.legs_score !== null) {
    regions.push({ region: "legs", score: scoreBreakdown.legs_score });
  }
  if (scoreBreakdown.torso_score !== null) {
    regions.push({ region: "torso", score: scoreBreakdown.torso_score });
  }
  return regions.sort((a, b) => a.score - b.score);
}

function getLowestBodyRegion(
  scoreBreakdown: ScoreBreakdown,
): BodyRegionKey | null {
  const ranked = rankedBodyRegionsFromScores(scoreBreakdown);
  return ranked.length > 0 ? ranked[0].region : null;
}

function allBodyRegionScoresWeakOrMissing(
  scoreBreakdown: ScoreBreakdown,
): boolean {
  const ranked = rankedBodyRegionsFromScores(scoreBreakdown);
  if (ranked.length === 0) {
    return true;
  }
  return ranked.every(
    (region) => region.score < LOW_BODY_REGION_SCORE_THRESHOLD,
  );
}

function regionPracticeSuggestion(region: BodyRegionKey): string {
  if (region === "arms") {
    return USER_FEEDBACK.armsLowest;
  }
  if (region === "legs") {
    return USER_FEEDBACK.legsLowest;
  }
  return USER_FEEDBACK.torsoLowest;
}

function isTimingConsistencyWeak(
  timingBreakdown: TimingBreakdown,
  scoreBreakdown: ScoreBreakdown,
): boolean {
  if (
    timingBreakdown.on_time_percentage !== null &&
    timingBreakdown.on_time_percentage < SECTION_GOOD_SCORE_THRESHOLD
  ) {
    return true;
  }
  if (
    scoreBreakdown.timing_consistency_score !== null &&
    scoreBreakdown.timing_consistency_score < SECTION_GOOD_SCORE_THRESHOLD
  ) {
    return true;
  }
  return timingBreakdown.timing_message.includes("transitions");
}

function hasWeakAnalysisData(
  sessionSummary: SessionSummary,
  reliability: AnalysisReliability,
): boolean {
  return (
    hasSparsePracticeCapture(sessionSummary) ||
    !isReliableEnoughForTechniqueFeedback(reliability)
  );
}

function hasPartialBodyVisibilityIssue(
  sessionSummary: SessionSummary,
  skippedFrames: SkippedFrameCounts,
  visibilityBreakdown: VisibilityBreakdown,
): boolean {
  const bodyNotVisibleRate = partialBodySkipRate(sessionSummary, skippedFrames);
  return (
    bodyNotVisibleRate >= OUT_OF_FRAME_OFTEN_RATE ||
    skippedFrames.partial_body >= 3 ||
    visibilityBreakdown.full_body_visible_percentage <
      FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD
  );
}

function isSetupSuggestion(suggestion: string): boolean {
  return (WEAK_DATA_SETUP_SUGGESTIONS as readonly string[]).includes(
    suggestion.trim(),
  );
}

function buildGuaranteedPositiveComment(
  sessionSummary: SessionSummary,
  scoreBreakdown: ScoreBreakdown,
  sectionBreakdown: SectionBreakdown,
): string {
  if (
    scoreBreakdown.overall_score !== null &&
    scoreBreakdown.overall_score >= SECTION_GOOD_SCORE_THRESHOLD
  ) {
    return USER_FEEDBACK.positiveStrongMoments;
  }
  if (sectionBreakdown.best_section?.main_issue === "Good section") {
    return USER_FEEDBACK.positiveStrongMoments;
  }
  if (sessionSummary.session_complete) {
    return USER_FEEDBACK.positiveComplete;
  }
  if (
    scoreBreakdown.overall_score !== null &&
    scoreBreakdown.overall_score >= 45
  ) {
    return USER_FEEDBACK.positiveBuilding;
  }
  return USER_FEEDBACK.positiveDefault;
}

function ensureMinimumImprovementPriorities(
  priorities: ImprovementPriority[],
  sessionSummary: SessionSummary,
  skippedFrames: SkippedFrameCounts,
  scoreBreakdown: ScoreBreakdown,
  timingBreakdown: TimingBreakdown,
  visibilityBreakdown: VisibilityBreakdown,
  reliability: AnalysisReliability,
): ImprovementPriority[] {
  const merged: ImprovementPriority[] = [...priorities];
  const seen = new Set(merged.map((item) => item.suggestion.trim()));

  const addPriority = (
    title: string,
    explanation: string,
    suggestion: string,
    rankScore = 1,
  ) => {
    const trimmed = suggestion.trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    merged.push({
      priority: 0,
      title,
      explanation,
      suggestion: trimmed,
      affected_time_ranges: [],
    });
    seen.add(trimmed);
  };

  const weakData = hasWeakAnalysisData(sessionSummary, reliability);

  if (hasPartialBodyVisibilityIssue(sessionSummary, skippedFrames, visibilityBreakdown)) {
    addPriority(
      "Stay in frame",
      "Some moments were skipped when your body left the camera view.",
      USER_FEEDBACK.partialVisibility,
      90,
    );
  }

  if (allBodyRegionScoresWeakOrMissing(scoreBreakdown)) {
    addPriority(
      "Overall movement",
      "Start with the big shapes before fine details.",
      USER_FEEDBACK.allLimbsWeak,
      80,
    );
  } else {
    const lowestRegion = getLowestBodyRegion(scoreBreakdown);
    if (lowestRegion) {
      addPriority(
        lowestRegion === "arms"
          ? "Arms"
          : lowestRegion === "legs"
            ? "Legs"
            : "Body direction",
        "This area had the biggest gap from the reference.",
        regionPracticeSuggestion(lowestRegion),
        70,
      );
    }
  }

  if (isTimingConsistencyWeak(timingBreakdown, scoreBreakdown)) {
    addPriority(
      "Timing",
      "Line up your moves with the beat.",
      USER_FEEDBACK.timingWeak,
      60,
    );
  }

  const practiceSuggestionCount = () =>
    merged.filter((item) => !isSetupSuggestion(item.suggestion)).length;

  const practiceDefaults = [
    USER_FEEDBACK.allLimbsWeak,
    USER_FEEDBACK.timingWeak,
    USER_FEEDBACK.armsLowest,
    USER_FEEDBACK.legsLowest,
    USER_FEEDBACK.torsoLowest,
  ];
  for (const suggestion of practiceDefaults) {
    if (practiceSuggestionCount() >= 2) {
      break;
    }
    addPriority("Practice tip", "A simple focus for your next run.", suggestion, 10);
  }

  if (weakData) {
    for (const setupSuggestion of WEAK_DATA_SETUP_SUGGESTIONS) {
      addPriority(
        "Camera setup",
        "A clearer view helps the next review.",
        setupSuggestion,
        5,
      );
    }
  }

  return merged
    .slice(0, MAX_IMPROVEMENT_PRIORITIES)
    .map((item, index) => ({
      ...item,
      priority: index + 1,
    }));
}

function buildGuaranteedPlainLanguageSummary(
  positiveComment: string,
  improvementPriorities: readonly ImprovementPriority[],
  sessionSummary: SessionSummary,
  reliability: AnalysisReliability,
  timingBreakdown: TimingBreakdown,
  scoreBreakdown: ScoreBreakdown,
): string[] {
  const weakData = hasWeakAnalysisData(sessionSummary, reliability);
  const summary: string[] = [positiveComment];

  const practiceSuggestions = improvementPriorities
    .map((item) => item.suggestion.trim())
    .filter((suggestion) => suggestion.length > 0 && !isSetupSuggestion(suggestion));

  for (const suggestion of practiceSuggestions) {
    if (summary.length >= 3) {
      break;
    }
    if (!summary.includes(suggestion)) {
      summary.push(suggestion);
    }
  }

  while (summary.length < 3) {
    const fallback =
      summary.length === 1
        ? USER_FEEDBACK.allLimbsWeak
        : USER_FEEDBACK.timingWeak;
    if (!summary.includes(fallback)) {
      summary.push(fallback);
    } else {
      break;
    }
  }

  if (weakData) {
    for (const setupSuggestion of WEAK_DATA_SETUP_SUGGESTIONS) {
      if (!summary.includes(setupSuggestion)) {
        summary.push(setupSuggestion);
      }
    }
    if (
      isTimingConsistencyWeak(timingBreakdown, scoreBreakdown) &&
      !summary.includes(USER_FEEDBACK.timingWeak)
    ) {
      summary.push(USER_FEEDBACK.timingWeak);
    }
  }

  return summary.slice(0, 5);
}

function ensureMinimumDetailedFeedback(
  feedback: DetailedFeedback[],
  positiveComment: string,
  sessionSummary: SessionSummary,
  scoreBreakdown: ScoreBreakdown,
  timingBreakdown: TimingBreakdown,
  visibilityBreakdown: VisibilityBreakdown,
  improvementPriorities: readonly ImprovementPriority[],
  reliability: AnalysisReliability,
): DetailedFeedback[] {
  const merged = [...feedback];
  const weakData = hasWeakAnalysisData(sessionSummary, reliability);

  const hasCategory = (category: DetailedFeedbackCategory) =>
    merged.some((item) => item.category === category);

  const topPracticeSuggestions = improvementPriorities
    .map((item) => item.suggestion.trim())
    .filter((suggestion) => suggestion.length > 0 && !isSetupSuggestion(suggestion));

  if (!hasCategory("overall")) {
    merged.unshift({
      category: "overall",
      title: "What went well",
      message: positiveComment,
      suggestion:
        topPracticeSuggestions[0] ?? USER_FEEDBACK.allLimbsWeak,
      severity: "info",
    });
  }

  if (
    weakData &&
    !hasCategory("visibility") &&
    !merged.some((item) => isSetupSuggestion(item.suggestion))
  ) {
    merged.push({
      category: "visibility",
      title: "Camera setup",
      message: USER_FEEDBACK.setupFullBody,
      suggestion: USER_FEEDBACK.setupLighting,
      severity: "medium",
    });
  }

  if (
    isTimingConsistencyWeak(timingBreakdown, scoreBreakdown) &&
    !hasCategory("timing")
  ) {
    merged.push({
      category: "timing",
      title: "Timing",
      message: "Some moves started a little early or late.",
      suggestion: USER_FEEDBACK.timingWeak,
      severity: "minor",
    });
  }

  if (
    hasPartialBodyVisibilityIssue(
      sessionSummary,
      {
        no_live_pose: 0,
        partial_body: sessionSummary.skipped_frames_total,
        no_reference_pose: 0,
        normalization_failed: 0,
      },
      visibilityBreakdown,
    ) &&
    !hasCategory("visibility")
  ) {
    merged.push({
      category: "visibility",
      title: "Stay in frame",
      message: "Some moments were skipped when your body left the camera view.",
      suggestion: USER_FEEDBACK.partialVisibility,
      severity: "medium",
    });
  }

  let practiceCount = merged.filter(
    (item) =>
      item.category !== "visibility" &&
      item.category !== "overall" &&
      item.suggestion.trim().length > 0,
  ).length;

  for (const suggestion of topPracticeSuggestions) {
    if (practiceCount >= 2) {
      break;
    }
    if (merged.some((item) => item.suggestion === suggestion)) {
      continue;
    }
    merged.push({
      category: "overall",
      title: "Practice focus",
      message: "A simple next step for your next run.",
      suggestion,
      severity: "minor",
    });
    practiceCount += 1;
  }

  return merged.slice(0, MAX_DETAILED_FEEDBACK);
}

function patchUserFacingFeedback(
  result: PracticeAnalysisResult,
  skippedFrames: SkippedFrameCounts,
): PracticeAnalysisResult {
  const positiveComment = buildGuaranteedPositiveComment(
    result.session_summary,
    result.score_breakdown,
    result.section_breakdown,
  );
  const improvementPriorities = ensureMinimumImprovementPriorities(
    result.improvement_priorities,
    result.session_summary,
    skippedFrames,
    result.score_breakdown,
    result.timing_breakdown,
    result.visibility_breakdown,
    result.reliability,
  );
  const plainLanguageSummary = buildGuaranteedPlainLanguageSummary(
    positiveComment,
    improvementPriorities,
    result.session_summary,
    result.reliability,
    result.timing_breakdown,
    result.score_breakdown,
  );
  const detailedFeedback = ensureMinimumDetailedFeedback(
    result.detailed_feedback,
    positiveComment,
    result.session_summary,
    result.score_breakdown,
    result.timing_breakdown,
    result.visibility_breakdown,
    improvementPriorities,
    result.reliability,
  );

  return {
    ...result,
    improvement_priorities: improvementPriorities,
    plain_language_summary: plainLanguageSummary,
    detailed_feedback: detailedFeedback,
    reliability: {
      ...result.reliability,
      message: hasWeakAnalysisData(result.session_summary, result.reliability)
        ? USER_FEEDBACK.setupFullBody
        : result.reliability.message,
    },
  };
}

function finalizeImprovementPriorities(
  candidates: ImprovementPriorityCandidate[],
): ImprovementPriority[] {
  return candidates
    .sort((a, b) => b.rank_score - a.rank_score)
    .slice(0, MAX_IMPROVEMENT_PRIORITIES)
    .map((candidate, index) => ({
      priority: index + 1,
      title: candidate.title,
      explanation: candidate.explanation,
      suggestion: candidate.suggestion,
      affected_time_ranges: limitAffectedTimeRanges(
        candidate.affected_time_ranges,
      ),
    }));
}

function buildImprovementPriorities(
  sessionSummary: SessionSummary,
  skippedFrames: SkippedFrameCounts,
  scoreBreakdown: ScoreBreakdown,
  timingBreakdown: TimingBreakdown,
  sideBreakdown: SideBreakdown,
  sectionBreakdown: SectionBreakdown,
  visibilityBreakdown: VisibilityBreakdown,
  reliability: AnalysisReliability,
): ImprovementPriority[] {
  const hasReliableShapeScores = hasTrustworthyCameraData(
    visibilityBreakdown,
    reliability,
  );
  const candidates: ImprovementPriorityCandidate[] = [];

  const bodyNotVisibleRate = partialBodySkipRate(sessionSummary, skippedFrames);
  if (hasPartialBodyVisibilityIssue(sessionSummary, skippedFrames, visibilityBreakdown)) {
    candidates.push({
      rank_score:
        100 * bodyNotVisibleRate + skippedFrames.partial_body * 4,
      title: "Stay in frame",
      explanation: visibilityBreakdown.visibility_message,
      suggestion: USER_FEEDBACK.partialVisibility,
      affected_time_ranges: affectedRangesForMainIssue(
        sectionBreakdown.sections,
        SECTION_MAIN_ISSUE.visibility,
        sectionBreakdown.weakest_section,
      ),
    });
  }

  if (allBodyRegionScoresWeakOrMissing(scoreBreakdown)) {
    candidates.push({
      rank_score: 85,
      title: "Overall movement",
      explanation: "Start with the big shapes before fine details.",
      suggestion: USER_FEEDBACK.allLimbsWeak,
      affected_time_ranges: affectedRangesForWeakSections(
        sectionBreakdown.sections,
        sectionBreakdown.weakest_section,
      ),
    });
  } else if (hasReliableShapeScores) {
    const lowestRegion = getLowestBodyRegion(scoreBreakdown);
    if (lowestRegion) {
      const { arms_score, legs_score, torso_score } = scoreBreakdown;
      const armsGap = regionScoreGapBelowBoth(
        arms_score,
        legs_score,
        torso_score,
      );
      const legsGap = regionScoreGapBelowBoth(
        legs_score,
        arms_score,
        torso_score,
      );
      const torsoGap = regionScoreGapBelowBoth(
        torso_score,
        arms_score,
        legs_score,
      );
      const gap =
        lowestRegion === "arms"
          ? armsGap
          : lowestRegion === "legs"
            ? legsGap
            : torsoGap;
      const score =
        lowestRegion === "arms"
          ? arms_score
          : lowestRegion === "legs"
            ? legs_score
            : torso_score;

      candidates.push({
        rank_score:
          (gap > 0 ? gap * 6 : 40) +
          (score !== null ? SECTION_GOOD_SCORE_THRESHOLD - score : 0),
        title:
          lowestRegion === "arms"
            ? "Your arms"
            : lowestRegion === "legs"
              ? "Your legs"
              : "Body position",
        explanation: "This area had the biggest gap from the reference.",
        suggestion: regionPracticeSuggestion(lowestRegion),
        affected_time_ranges: affectedRangesForMainIssue(
          sectionBreakdown.sections,
          lowestRegion === "arms"
            ? SECTION_MAIN_ISSUE.arms
            : lowestRegion === "legs"
              ? SECTION_MAIN_ISSUE.legs
              : SECTION_MAIN_ISSUE.torso,
          sectionBreakdown.weakest_section,
        ),
      });
    }
  }

  if (
    sideBreakdown.left_side_score !== null &&
    sideBreakdown.right_side_score !== null
  ) {
    const sideGap = Math.abs(
      sideBreakdown.left_side_score - sideBreakdown.right_side_score,
    );
    if (sideGap >= SIDE_IMBALANCE_SCORE_GAP) {
      const weakerSide =
        sideBreakdown.left_side_score < sideBreakdown.right_side_score
          ? "left"
          : "right";
      candidates.push({
        rank_score: sideGap * 5,
        title: "Left and right balance",
        explanation:
          weakerSide === "left"
            ? "Your left side was a bit less consistent than your right."
            : "Your right side was a bit less consistent than your left.",
        suggestion:
          weakerSide === "left"
            ? "Run the tricky section once more leading with your left arm and leg."
            : "Run the tricky section once more leading with your right arm and leg.",
        affected_time_ranges: affectedRangesForWeakSections(
          sectionBreakdown.sections,
          sectionBreakdown.weakest_section,
        ),
      });
    }
  }

  const { swing } = largestAdjacentSectionSwing(sectionBreakdown.sections);
  const scoredSectionCount = sectionBreakdown.sections.filter(
    (section) => section.average_score !== null,
  ).length;
  if (
    swing >= ADJACENT_SECTION_SCORE_SWING_THRESHOLD &&
    scoredSectionCount >= 2
  ) {
    candidates.push({
      rank_score: swing * 4,
      title: "Timing with the music",
      explanation:
        "Some parts of the dance matched better than others, which often means the moves started a little early or late.",
      suggestion:
        "Dance with the music and hit the big shapes right when the beat lands.",
      affected_time_ranges: affectedRangesForWeakSections(
        sectionBreakdown.sections,
        sectionBreakdown.weakest_section,
      ),
    });
  } else if (isTimingConsistencyWeak(timingBreakdown, scoreBreakdown)) {
    candidates.push({
      rank_score:
        timingBreakdown.on_time_percentage !== null
          ? 100 - timingBreakdown.on_time_percentage
          : 50,
      title: "Timing consistency",
      explanation: "Line up your moves with the beat.",
      suggestion: USER_FEEDBACK.timingWeak,
      affected_time_ranges: affectedRangesForWeakSections(
        sectionBreakdown.sections,
        sectionBreakdown.weakest_section,
      ),
    });
  }

  return finalizeImprovementPriorities(candidates);
}

function scoreFeedbackSeverity(score: number | null): DetailedFeedbackSeverity {
  if (score === null) {
    return "info";
  }
  if (score >= 75) {
    return "info";
  }
  if (score >= 55) {
    return "minor";
  }
  return "medium";
}

function weakerTimeRangesPhrase(sectionBreakdown: SectionBreakdown): string {
  const weak = sectionBreakdown.weakest_section;
  if (!weak) {
    return "the weaker time ranges";
  }
  return `${formatDuration(weak.start_time_seconds)}–${formatDuration(weak.end_time_seconds)}`;
}

function bodyPartNeedsAttention(insight: BodyPartInsight): boolean {
  return insight.issue_level === "needs_work" || insight.issue_level === "poor";
}

function shouldIncludeArmsFeedback(
  scoreBreakdown: ScoreBreakdown,
  bodyPartBreakdown: BodyPartBreakdown,
): boolean {
  const { arms_score, legs_score, torso_score } = scoreBreakdown;
  if (arms_score === null) {
    return false;
  }
  if (arms_score < SECTION_GOOD_SCORE_THRESHOLD) {
    return true;
  }
  if (
    legs_score !== null &&
    torso_score !== null &&
    arms_score <= legs_score - REGION_SCORE_SIGNIFICANT_GAP &&
    arms_score <= torso_score - REGION_SCORE_SIGNIFICANT_GAP
  ) {
    return true;
  }
  return (
    bodyPartNeedsAttention(bodyPartBreakdown.left_arm) ||
    bodyPartNeedsAttention(bodyPartBreakdown.right_arm)
  );
}

function shouldIncludeLegsFeedback(
  scoreBreakdown: ScoreBreakdown,
  bodyPartBreakdown: BodyPartBreakdown,
): boolean {
  const { arms_score, legs_score, torso_score } = scoreBreakdown;
  if (legs_score === null) {
    return false;
  }
  const upperBody = averageNullableScores([arms_score, torso_score]);
  if (upperBody !== null && legs_score <= upperBody - REGION_SCORE_SIGNIFICANT_GAP) {
    return true;
  }
  if (legs_score < SECTION_GOOD_SCORE_THRESHOLD) {
    return true;
  }
  return (
    bodyPartNeedsAttention(bodyPartBreakdown.left_leg) ||
    bodyPartNeedsAttention(bodyPartBreakdown.right_leg)
  );
}

function shouldIncludeTorsoFeedback(
  scoreBreakdown: ScoreBreakdown,
  bodyPartBreakdown: BodyPartBreakdown,
): boolean {
  const { arms_score, legs_score, torso_score } = scoreBreakdown;
  if (torso_score === null) {
    return false;
  }
  if (torso_score < REGION_LOW_SCORE_THRESHOLD) {
    return true;
  }
  if (
    arms_score !== null &&
    legs_score !== null &&
    torso_score <= arms_score - REGION_SCORE_SIGNIFICANT_GAP &&
    torso_score <= legs_score - REGION_SCORE_SIGNIFICANT_GAP
  ) {
    return true;
  }
  return bodyPartNeedsAttention(bodyPartBreakdown.torso);
}

function shouldIncludeTimingFeedback(timingBreakdown: TimingBreakdown): boolean {
  if (timingBreakdown.on_time_percentage === null) {
    return false;
  }
  return (
    timingBreakdown.timing_message.includes("transitions") ||
    timingBreakdown.on_time_percentage < SECTION_GOOD_SCORE_THRESHOLD
  );
}

function shouldIncludeVisibilityFeedback(
  visibilityBreakdown: VisibilityBreakdown,
  reliability: AnalysisReliability,
): boolean {
  if (!isReliableEnoughForTechniqueFeedback(reliability)) {
    return true;
  }
  return (
    visibilityBreakdown.full_body_visible_percentage <
      FULL_BODY_VISIBILITY_GOOD_THRESHOLD ||
    visibilityBreakdown.most_common_visibility_problem !== null
  );
}

function buildDetailedFeedback(
  scoreBreakdown: ScoreBreakdown,
  bodyPartBreakdown: BodyPartBreakdown,
  timingBreakdown: TimingBreakdown,
  visibilityBreakdown: VisibilityBreakdown,
  sectionBreakdown: SectionBreakdown,
  reliability: AnalysisReliability,
  sessionSummary: SessionSummary,
  skippedFrames: SkippedFrameCounts,
): DetailedFeedback[] {
  const feedback: DetailedFeedback[] = [];
  const push = (item: DetailedFeedback) => {
    if (feedback.length < MAX_DETAILED_FEEDBACK) {
      feedback.push(item);
    }
  };

  const positiveComment = buildGuaranteedPositiveComment(
    sessionSummary,
    scoreBreakdown,
    sectionBreakdown,
  );
  const weakData = hasWeakAnalysisData(sessionSummary, reliability);

  push({
    category: "overall",
    title: "What went well",
    message: positiveComment,
    suggestion:
      allBodyRegionScoresWeakOrMissing(scoreBreakdown)
        ? USER_FEEDBACK.allLimbsWeak
        : regionPracticeSuggestion(
            getLowestBodyRegion(scoreBreakdown) ?? "arms",
          ),
    severity: "info",
  });

  if (
    hasPartialBodyVisibilityIssue(
      sessionSummary,
      skippedFrames,
      visibilityBreakdown,
    )
  ) {
    push({
      category: "visibility",
      title: "Stay in frame",
      message: "Some moments were skipped when your body left the camera view.",
      suggestion: USER_FEEDBACK.partialVisibility,
      severity: "medium",
    });
  }

  if (weakData) {
    push({
      category: "visibility",
      title: "Camera setup",
      message: USER_FEEDBACK.setupFullBody,
      suggestion: USER_FEEDBACK.setupLighting,
      severity: "medium",
    });
  } else if (!allBodyRegionScoresWeakOrMissing(scoreBreakdown)) {
    const lowestRegion = getLowestBodyRegion(scoreBreakdown);
    if (lowestRegion === "arms" && shouldIncludeArmsFeedback(scoreBreakdown, bodyPartBreakdown)) {
      push({
        category: "arms",
        title: "Arms",
        message: "Your arm shapes were the main area to polish this run.",
        suggestion: USER_FEEDBACK.armsLowest,
        severity: scoreFeedbackSeverity(scoreBreakdown.arms_score),
      });
    } else if (
      lowestRegion === "legs" &&
      shouldIncludeLegsFeedback(scoreBreakdown, bodyPartBreakdown)
    ) {
      push({
        category: "legs",
        title: "Legs",
        message: "Your foot and knee placement was the main area to polish.",
        suggestion: USER_FEEDBACK.legsLowest,
        severity: scoreFeedbackSeverity(scoreBreakdown.legs_score),
      });
    } else if (
      lowestRegion === "torso" &&
      shouldIncludeTorsoFeedback(scoreBreakdown, bodyPartBreakdown)
    ) {
      push({
        category: "torso",
        title: "Body direction",
        message: "Your body angle was the main area to polish this run.",
        suggestion: USER_FEEDBACK.torsoLowest,
        severity: scoreFeedbackSeverity(scoreBreakdown.torso_score),
      });
    }
  }

  if (isTimingConsistencyWeak(timingBreakdown, scoreBreakdown)) {
    push({
      category: "timing",
      title: "Timing",
      message: "Some moves started a little early or late.",
      suggestion: USER_FEEDBACK.timingWeak,
      severity: "minor",
    });
  }

  return feedback.slice(0, MAX_DETAILED_FEEDBACK);
}

function sectionTimePhrase(
  section: DanceSection,
  sectionIndex: number,
  sectionCount: number,
): string {
  if (sectionCount <= 1) {
    return "part of the dance";
  }
  if (sectionIndex === 0) {
    return "the beginning";
  }
  if (sectionIndex === sectionCount - 1) {
    return "the end";
  }
  return "the middle";
}

function overallPerformanceSentence(overallScore: number | null): string | null {
  if (overallScore === null) {
    return null;
  }
  if (overallScore >= 75) {
    return "You stayed close to the reference for much of the dance.";
  }
  if (overallScore >= 55) {
    return "You picked up a lot of the routine — a few shapes still need work.";
  }
  return "You are on your way — the tips below show where to focus next.";
}

function strongestAreaSentence(scoreBreakdown: ScoreBreakdown): string | null {
  const regions: Array<{ label: string; score: number }> = [];
  if (scoreBreakdown.arms_score !== null) {
    regions.push({ label: "arm movements", score: scoreBreakdown.arms_score });
  }
  if (scoreBreakdown.legs_score !== null) {
    regions.push({ label: "leg movements", score: scoreBreakdown.legs_score });
  }
  if (scoreBreakdown.torso_score !== null) {
    regions.push({ label: "body position", score: scoreBreakdown.torso_score });
  }
  if (regions.length < 2) {
    return null;
  }

  regions.sort((a, b) => b.score - a.score);
  const strongest = regions[0];
  const weakest = regions[regions.length - 1];
  if (strongest.score - weakest.score < 4) {
    return null;
  }

  if (strongest.label === "arm movements" && weakest.label === "leg movements") {
    return "Your arm movements were stronger than your leg movements.";
  }
  if (strongest.label === "leg movements" && weakest.label === "arm movements") {
    return "Your leg movements were stronger than your arm movements.";
  }

  return `Your ${strongest.label} were stronger than your ${weakest.label}.`;
}

function mainImprovementSentence(
  scoreBreakdown: ScoreBreakdown,
  bodyPartBreakdown: BodyPartBreakdown,
  sectionBreakdown: SectionBreakdown,
): string | null {
  const weakestSection = sectionBreakdown.weakest_section;
  if (weakestSection?.main_issue) {
    const sectionIndex = sectionBreakdown.sections.findIndex(
      (section) =>
        section.start_time_seconds === weakestSection.start_time_seconds,
    );
    const timePhrase = sectionTimePhrase(
      weakestSection,
      sectionIndex >= 0 ? sectionIndex : 0,
      sectionBreakdown.sections.length,
    );

    switch (weakestSection.main_issue) {
      case "Arms need attention":
        return `The main area to improve is arm positioning during ${timePhrase}.`;
      case "Legs need attention":
        return `The main area to improve is foot placement during ${timePhrase}.`;
      case "Body position needs attention":
        return `The main area to improve is body position during ${timePhrase}.`;
      case "Good section":
        return null;
      default:
        break;
    }
  }

  const regionCandidates: { label: string; score: number }[] = [];
  if (scoreBreakdown.arms_score !== null) {
    regionCandidates.push({
      label: "arm positioning",
      score: scoreBreakdown.arms_score,
    });
  }
  if (scoreBreakdown.legs_score !== null) {
    regionCandidates.push({
      label: "foot placement",
      score: scoreBreakdown.legs_score,
    });
  }
  if (scoreBreakdown.torso_score !== null) {
    regionCandidates.push({
      label: "body position",
      score: scoreBreakdown.torso_score,
    });
  }

  if (regionCandidates.length > 0) {
    regionCandidates.sort((a, b) => a.score - b.score);
    return `The main area to improve is ${regionCandidates[0].label}.`;
  }

  const weakBodyPart = (Object.entries(bodyPartBreakdown) as Array<
    [keyof BodyPartBreakdown, BodyPartInsight]
  >)
    .filter(
      ([, insight]) =>
        insight.issue_level === "needs_work" || insight.issue_level === "poor",
    )
    .sort((a, b) => (a[1].score ?? 100) - (b[1].score ?? 100))[0];

  if (weakBodyPart) {
    const label = weakBodyPart[0].replace(/_/g, " ");
    return `The main area to improve is your ${label}.`;
  }

  return null;
}

function visibilitySummarySentence(
  visibilityBreakdown: VisibilityBreakdown,
  sessionSummary: SessionSummary,
): string | null {
  if (sessionSummary.sampled_live_frames === 0) {
    return null;
  }

  if (visibilityBreakdown.full_body_visible_percentage < FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD) {
    return "Some moments were skipped because your full body was not visible.";
  }

  if (
    visibilityBreakdown.full_body_visible_percentage <
      FULL_BODY_VISIBILITY_GOOD_THRESHOLD &&
    sessionSummary.skipped_frames_total > 0
  ) {
    return "A few moments were skipped when your body was partly out of frame.";
  }

  const problem = visibilityBreakdown.most_common_visibility_problem;
  if (
    problem === VISIBILITY_PROBLEM_LABELS.hands_missing ||
    problem === VISIBILITY_PROBLEM_LABELS.feet_missing
  ) {
    return "Some moments were skipped because your hands or feet were out of frame.";
  }

  if (problem === VISIBILITY_PROBLEM_LABELS.low_lighting) {
    return "Lighting made some moments harder to analyze.";
  }

  return null;
}

function encouragingNextStepSentence(
  improvementPriorities: readonly ImprovementPriority[],
  visibilityBreakdown: VisibilityBreakdown,
  timingBreakdown: TimingBreakdown,
  emphasizeSetup: boolean,
): string {
  if (emphasizeSetup) {
    if (
      visibilityBreakdown.most_common_visibility_problem ===
        VISIBILITY_PROBLEM_LABELS.body_too_close ||
      visibilityBreakdown.full_body_visible_percentage <
        FULL_BODY_VISIBILITY_ADEQUATE_THRESHOLD
    ) {
      return "Try again with the phone slightly farther back for a cleaner analysis.";
    }
    return "Run the dance again with your full body in frame before focusing on technique.";
  }

  const topPriority = improvementPriorities[0];
  if (topPriority?.suggestion) {
    return topPriority.suggestion;
  }

  if (timingBreakdown.timing_message.includes("transitions")) {
    return "Practice the weaker sections slowly with the music.";
  }

  return "Run the dance again and focus on the section that felt hardest.";
}

function buildPlainLanguageSummary(
  sessionSummary: SessionSummary,
  scoreBreakdown: ScoreBreakdown,
  _bodyPartBreakdown: BodyPartBreakdown,
  reliability: AnalysisReliability,
  _visibilityBreakdown: VisibilityBreakdown,
  timingBreakdown: TimingBreakdown,
  improvementPriorities: readonly ImprovementPriority[],
  sectionBreakdown: SectionBreakdown,
): string[] {
  const positiveComment = buildGuaranteedPositiveComment(
    sessionSummary,
    scoreBreakdown,
    sectionBreakdown,
  );
  return buildGuaranteedPlainLanguageSummary(
    positiveComment,
    improvementPriorities,
    sessionSummary,
    reliability,
    timingBreakdown,
    scoreBreakdown,
  );
}

/**
 * Compare captured live practice frames to time-aligned reference poses.
 * Intended for post-practice review only; does not provide live coaching.
 */
export function analyzePracticeSession({
  referencePoseFrames,
  livePoseFrames,
  referenceDurationSeconds,
  referenceQuality,
}: AnalyzePracticeSessionInput): PracticeAnalysisResult {
  const skippedFrames: SkippedFrameCounts = {
    no_live_pose: 0,
    partial_body: 0,
    no_reference_pose: 0,
    normalization_failed: 0,
  };

  const perFrameScores: PerFrameScore[] = [];
  let maxElapsedSeconds = 0;

  const sectionDurationSeconds =
    referenceDurationSeconds > 0 ? referenceDurationSeconds : 0;
  const sectionBoundaries = deriveSectionBoundaries(sectionDurationSeconds);
  const sectionFrameCounts: SectionFrameCounts[] = sectionBoundaries.map(
    () => ({ usable_frames: 0, skipped_frames: 0 }),
  );

  const recordSectionSkip = (elapsedSeconds: number): void => {
    const sectionIndex = sectionIndexForElapsed(
      elapsedSeconds,
      sectionBoundaries,
    );
    if (sectionIndex >= 0) {
      sectionFrameCounts[sectionIndex].skipped_frames += 1;
    }
  };

  const recordSectionUsable = (elapsedSeconds: number): void => {
    const sectionIndex = sectionIndexForElapsed(
      elapsedSeconds,
      sectionBoundaries,
    );
    if (sectionIndex >= 0) {
      sectionFrameCounts[sectionIndex].usable_frames += 1;
    }
  };

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
      recordSectionSkip(liveFrame.practice_elapsed_seconds);
      continue;
    }

    if (!liveFrame.pose_detected) {
      skippedFrames.no_live_pose += 1;
      recordSectionSkip(liveFrame.practice_elapsed_seconds);
      continue;
    }

    if (!liveFrame.full_body_visible) {
      skippedFrames.partial_body += 1;
      recordSectionSkip(liveFrame.practice_elapsed_seconds);
      continue;
    }

    const referenceLandmarks = asPoseLandmarks(referenceFrame.landmarks);
    const liveLandmarks = asPoseLandmarks(liveFrame.landmarks);

    const normalizedReference = normalizePoseForComparison(referenceLandmarks);
    const normalizedLive = normalizePoseForComparison(liveLandmarks);
    if (!normalizedReference || !normalizedLive) {
      skippedFrames.normalization_failed += 1;
      recordSectionSkip(liveFrame.practice_elapsed_seconds);
      continue;
    }

    const comparison = compareNormalizedPoses(
      normalizedReference,
      normalizedLive,
    );

    recordSectionUsable(liveFrame.practice_elapsed_seconds);
    perFrameScores.push({
      practice_elapsed_seconds: liveFrame.practice_elapsed_seconds,
      reference_timestamp_seconds: referenceFrame.timestamp_seconds,
      timing_offset_seconds:
        liveFrame.practice_elapsed_seconds - referenceFrame.timestamp_seconds,
      pose_match: comparison,
    });
  }

  const sessionComplete = deriveSessionComplete(
    maxElapsedSeconds,
    referenceDurationSeconds,
  );
  const usableComparisonFrames = perFrameScores.length;
  const sessionSummary = buildSessionSummary(
    referenceDurationSeconds,
    maxElapsedSeconds,
    sessionComplete,
    livePoseFrames.length,
    usableComparisonFrames,
    skippedFrames,
  );
  const visibilityBreakdown = buildVisibilityBreakdown(livePoseFrames);
  const referenceQualityBreakdown = buildReferenceQualityBreakdown(
    referenceQuality,
    referencePoseFrames,
    skippedFrames,
  );
  const reliability = buildAnalysisReliability(
    sessionSummary,
    visibilityBreakdown,
    livePoseFrames,
    referenceQualityBreakdown,
    skippedFrames,
  );
  const sideBreakdown = buildSideBreakdown(perFrameScores);
  const sectionBreakdownDuration =
    referenceDurationSeconds > 0 ? referenceDurationSeconds : maxElapsedSeconds;
  const sectionBreakdown = buildSectionBreakdown(
    perFrameScores,
    sectionBreakdownDuration,
    sectionFrameCounts.length > 0
      ? sectionFrameCounts
      : deriveSectionBoundaries(sectionBreakdownDuration).map(() => ({
          usable_frames: 0,
          skipped_frames: 0,
        })),
  );
  const timingBreakdown = buildTimingBreakdown(sectionBreakdown, perFrameScores);
  const scoreBreakdown = buildScoreBreakdown(
    perFrameScores,
    timingBreakdown,
    sessionSummary,
    visibilityBreakdown,
  );
  const bodyPartBreakdown = buildBodyPartBreakdown(
    perFrameScores,
    scoreBreakdown.torso_score,
  );
  const improvementPriorities = buildImprovementPriorities(
    sessionSummary,
    skippedFrames,
    scoreBreakdown,
    timingBreakdown,
    sideBreakdown,
    sectionBreakdown,
    visibilityBreakdown,
    reliability,
  );
  const detailedFeedback = buildDetailedFeedback(
    scoreBreakdown,
    bodyPartBreakdown,
    timingBreakdown,
    visibilityBreakdown,
    sectionBreakdown,
    reliability,
    sessionSummary,
    skippedFrames,
  );
  const plainLanguageSummary = buildPlainLanguageSummary(
    sessionSummary,
    scoreBreakdown,
    bodyPartBreakdown,
    reliability,
    visibilityBreakdown,
    timingBreakdown,
    improvementPriorities,
    sectionBreakdown,
  );

  return patchUserFacingFeedback(
    {
      session_summary: sessionSummary,
      score_breakdown: scoreBreakdown,
      body_part_breakdown: bodyPartBreakdown,
      side_breakdown: sideBreakdown,
      timing_breakdown: timingBreakdown,
      section_breakdown: sectionBreakdown,
      visibility_breakdown: visibilityBreakdown,
      improvement_priorities: improvementPriorities,
      plain_language_summary: plainLanguageSummary,
      detailed_feedback: detailedFeedback,
      reliability,
      reference_quality_breakdown: referenceQualityBreakdown,
    },
    skippedFrames,
  );
}
