/**
 * Post-practice analysis model for UI rendering.
 * Aggregated scores and copy only — no raw landmark arrays or per-frame pose data.
 */

// --- Body regions ---

export type BodyPartIssueLevel =
  | "good"
  | "minor"
  | "needs_work"
  | "poor"
  | "not_enough_data";

/** Per-region insight shown in body-part cards and priority lists. */
export type BodyPartInsight = {
  score: number | null;
  issue_level: BodyPartIssueLevel;
  message: string;
  average_error: number | null;
};

export type BodyPartBreakdown = {
  left_arm: BodyPartInsight;
  right_arm: BodyPartInsight;
  left_leg: BodyPartInsight;
  right_leg: BodyPartInsight;
  shoulders: BodyPartInsight;
  hips: BodyPartInsight;
  torso: BodyPartInsight;
};

// --- Session & scores ---

export type SessionSummary = {
  reference_duration_seconds: number;
  completed_duration_seconds: number;
  session_complete: boolean;
  sampled_live_frames: number;
  usable_comparison_frames: number;
  skipped_frames_total: number;
  usable_frame_percentage: number;
};

export type ScoreBreakdown = {
  overall_score: number | null;
  arms_score: number | null;
  legs_score: number | null;
  torso_score: number | null;
  timing_consistency_score: number | null;
  visibility_score: number | null;
};

export type SideBreakdown = {
  left_side_score: number | null;
  right_side_score: number | null;
  side_balance_message: string;
};

export type TimingBreakdown = {
  on_time_percentage: number | null;
  early_movement_percentage: number | null;
  late_movement_percentage: number | null;
  timing_message: string;
};

// --- Time-based sections ---

export type TimeRange = {
  start_time_seconds: number;
  end_time_seconds: number;
};

export type DanceSection = {
  start_time_seconds: number;
  end_time_seconds: number;
  average_score: number | null;
  arms_score: number | null;
  legs_score: number | null;
  torso_score: number | null;
  usable_frames: number;
  skipped_frames: number;
  main_issue: string | null;
};

export type SectionBreakdown = {
  best_section: DanceSection | null;
  weakest_section: DanceSection | null;
  sections: DanceSection[];
};

export type VisibilityBreakdown = {
  full_body_visible_percentage: number;
  most_common_visibility_problem: string | null;
  visibility_message: string;
};

// --- Coaching copy ---

export type ImprovementPriority = {
  priority: number;
  title: string;
  explanation: string;
  suggestion: string;
  affected_time_ranges: TimeRange[];
};

export type DetailedFeedbackCategory =
  | "arms"
  | "legs"
  | "torso"
  | "timing"
  | "visibility"
  | "overall";

export type DetailedFeedbackSeverity = "info" | "minor" | "medium" | "major";

export type DetailedFeedback = {
  category: DetailedFeedbackCategory;
  title: string;
  message: string;
  suggestion: string;
  severity: DetailedFeedbackSeverity;
};

export type AnalysisReliabilityLevel =
  | "high"
  | "medium"
  | "low"
  | "not_enough_data";

export type AnalysisReliability = {
  level: AnalysisReliabilityLevel;
  message: string;
  reasons: string[];
};

export type ReferenceQualityBreakdown = {
  reference_quality: string | null;
  pose_detection_percentage: number | null;
  full_body_visibility_percentage: number | null;
  low_confidence_segment_count: number;
  limits_reliability: boolean;
  has_missing_pose_sections: boolean;
  summary_message: string;
  suggestion: string;
};

// --- Root result ---

/** Full post-practice report returned by `analyzePracticeSession`. */
export type PracticeAnalysisResult = {
  session_summary: SessionSummary;
  score_breakdown: ScoreBreakdown;
  body_part_breakdown: BodyPartBreakdown;
  side_breakdown: SideBreakdown;
  timing_breakdown: TimingBreakdown;
  section_breakdown: SectionBreakdown;
  visibility_breakdown: VisibilityBreakdown;
  improvement_priorities: ImprovementPriority[];
  plain_language_summary: string[];
  detailed_feedback: DetailedFeedback[];
  reliability: AnalysisReliability;
  reference_quality_breakdown: ReferenceQualityBreakdown;
};
