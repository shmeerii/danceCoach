import type {
  BodyPartIssueLevel,
  PracticeAnalysisResult,
  ScoreBreakdown,
  TimingBreakdown,
} from "../types/practiceAnalysis";
import {
  hasSparsePracticeCapture,
  shouldShowPartialCaptureIntro,
} from "./practiceAnalysis";

function isValidNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Friendly label for a 0–100 match score (demo-oriented, not clinical). */
export function scoreToLabel(score: number | null | undefined): string {
  if (!isValidNumber(score)) {
    return "Still gathering";
  }

  const rounded = Math.round(Math.max(0, Math.min(100, score)));
  if (rounded >= 85) {
    return "Very close";
  }
  if (rounded >= 70) {
    return "On track";
  }
  if (rounded >= 55) {
    return "Getting there";
  }
  if (rounded >= 40) {
    return "Keep practicing";
  }
  return "Room to grow";
}

export function issueLevelToLabel(issueLevel: BodyPartIssueLevel): string {
  switch (issueLevel) {
    case "good":
      return "Looking good";
    case "minor":
      return "Small differences";
    case "needs_work":
      return "Worth a focus";
    case "poor":
      return "Worth a focus";
    default:
      return "Still gathering";
  }
}

export function referenceQualityToLabel(
  quality: string | null | undefined,
): string {
  if (typeof quality !== "string" || !quality.trim()) {
    return "Unknown";
  }
  switch (quality.trim().toLowerCase()) {
    case "excellent":
      return "Excellent";
    case "good":
      return "Good";
    case "usable":
      return "Usable";
    case "poor":
      return "Limited";
    case "failed":
      return "Very limited";
    default:
      return quality.trim().charAt(0).toUpperCase() + quality.trim().slice(1);
  }
}

function formatDurationSeconds(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

export function formatTimeRange(
  startSeconds: number | null | undefined,
  endSeconds: number | null | undefined,
): string {
  if (!isValidNumber(startSeconds) || !isValidNumber(endSeconds)) {
    return "Part of the dance";
  }
  return `${formatDurationSeconds(startSeconds)}–${formatDurationSeconds(endSeconds)}`;
}

export function formatScore(score: number | null | undefined): string {
  return scoreToLabel(score);
}

export function formatPercent(value: number | null | undefined): string {
  if (!isValidNumber(value)) {
    return "Still gathering";
  }
  return `${Math.round(Math.max(0, Math.min(100, value)))}%`;
}

export const PARTIAL_CAPTURE_INTRO =
  "Here is friendly feedback from your practice run.";

export const SETUP_PRACTICAL_FIXES = [
  "Try again with your full body visible.",
  "Keep the phone stable.",
  "Use brighter lighting.",
] as const;

export function getPositiveComment(
  analysis: PracticeAnalysisResult,
): string {
  return (
    analysis.plain_language_summary[0] ??
    "Nice effort — keep dancing with the music."
  );
}

export function getPracticeSuggestions(
  analysis: PracticeAnalysisResult,
): string[] {
  const fromPriorities = analysis.improvement_priorities
    .map((item) => item.suggestion.trim())
    .filter(
      (suggestion) =>
        suggestion.length > 0 &&
        !SETUP_PRACTICAL_FIXES.includes(
          suggestion as (typeof SETUP_PRACTICAL_FIXES)[number],
        ),
    );

  const fromSummary = analysis.plain_language_summary
    .slice(1)
    .filter(
      (sentence) =>
        !SETUP_PRACTICAL_FIXES.includes(
          sentence as (typeof SETUP_PRACTICAL_FIXES)[number],
        ),
    );

  const merged = [...fromPriorities, ...fromSummary].filter(
    (tip, index, list) => tip.length > 0 && list.indexOf(tip) === index,
  );

  const defaults = [
    "The overall movement was different from the reference, so start by practicing the dance slowly with the music.",
    "Practice entering each move on the beat.",
  ];

  for (const fallback of defaults) {
    if (merged.length >= 2) {
      break;
    }
    if (!merged.includes(fallback)) {
      merged.push(fallback);
    }
  }

  return merged.slice(0, 4);
}

export function hasDisplayableScore(
  score: number | null | undefined,
): boolean {
  return isValidNumber(score);
}

export function scoreLabelTone(score: number | null | undefined): string {
  const label = scoreToLabel(score);
  switch (label) {
    case "Very close":
    case "On track":
      return "#1f6b34";
    case "Getting there":
      return "#1a4d8f";
    case "Keep practicing":
    case "Room to grow":
      return "#9b5a00";
    default:
      return "#888";
  }
}

export type PerformanceRow = {
  label: string;
  value: string;
};

export function formatTimingForUser(timing: TimingBreakdown): string {
  const message = timing.timing_message.trim();
  if (!message) {
    return "Keep practicing with the music to line up your moves with the beat.";
  }
  const technicalPatterns = [
    /not enough/i,
    /usable pose/i,
    /reliable/i,
    /frame/i,
    /skipped/i,
    /comparison/i,
  ];
  if (technicalPatterns.some((pattern) => pattern.test(message))) {
    return "Keep practicing with the music to line up your moves with the beat.";
  }
  return message;
}

export function getPerformanceRows(
  scoreBreakdown: ScoreBreakdown,
  timing: TimingBreakdown,
): PerformanceRow[] {
  const rows: PerformanceRow[] = [
    {
      label: "Overall performance",
      value: scoreToLabel(scoreBreakdown.overall_score),
    },
    {
      label: "Arms",
      value: scoreToLabel(scoreBreakdown.arms_score),
    },
    {
      label: "Legs",
      value: scoreToLabel(scoreBreakdown.legs_score),
    },
    {
      label: "Body position",
      value: scoreToLabel(scoreBreakdown.torso_score),
    },
    {
      label: "Timing",
      value: formatTimingForUser(timing),
    },
  ];
  return rows;
}

function rankedBodyRegions(
  scoreBreakdown: ScoreBreakdown,
): Array<{ label: string; score: number }> {
  const regions: Array<{ label: string; score: number }> = [];
  if (scoreBreakdown.arms_score !== null) {
    regions.push({ label: "Arms", score: scoreBreakdown.arms_score });
  }
  if (scoreBreakdown.legs_score !== null) {
    regions.push({ label: "Legs", score: scoreBreakdown.legs_score });
  }
  if (scoreBreakdown.torso_score !== null) {
    regions.push({ label: "Body position", score: scoreBreakdown.torso_score });
  }
  return regions.sort((a, b) => b.score - a.score);
}

export function getDemoHeadline(
  analysis: PracticeAnalysisResult,
  stoppedEarly: boolean,
): string {
  if (hasSparsePracticeCapture(analysis.session_summary)) {
    return "Good effort — here are simple next steps";
  }
  if (stoppedEarly) {
    return "Nice effort — here's your review";
  }

  const overall = analysis.score_breakdown.overall_score;
  if (overall !== null && overall >= 75) {
    return "Great session — you stayed close to the dance";
  }
  if (overall !== null && overall >= 55) {
    return "Solid run — a few spots to polish";
  }

  const ranked = rankedBodyRegions(analysis.score_breakdown);
  if (ranked.length >= 2) {
    return `Good work — focus on your ${ranked[ranked.length - 1].label.toLowerCase()} next`;
  }

  return "Good session — here is what to practice next";
}

export function getPartialCaptureIntro(
  analysis: PracticeAnalysisResult,
): string | null {
  if (!shouldShowPartialCaptureIntro(analysis.session_summary)) {
    return null;
  }
  return PARTIAL_CAPTURE_INTRO;
}

export function formatDemoMomentLabel(
  startSeconds: number,
  endSeconds: number,
): string {
  return formatTimeRange(startSeconds, endSeconds);
}
