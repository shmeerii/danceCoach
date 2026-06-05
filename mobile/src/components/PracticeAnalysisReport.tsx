import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import {
  formatDemoMomentLabel,
  getDemoHeadline,
  getPartialCaptureIntro,
  getPerformanceRows,
  getPositiveComment,
  getPracticeSuggestions,
  issueLevelToLabel,
  scoreLabelTone,
  SETUP_PRACTICAL_FIXES,
} from "../utils/analysisWording";
import {
  hasSparsePracticeCapture,
  isSectionShowableForUser,
} from "../utils/practiceAnalysis";
import type { DanceSection, PracticeAnalysisResult } from "../types/practiceAnalysis";

function friendlySectionNote(section: DanceSection): string {
  if (!section.main_issue || section.main_issue === "Good section") {
    return "You stayed close to the reference here.";
  }
  if (section.main_issue === "Not enough visible body data") {
    return "Move farther back so your feet and hands stay in frame.";
  }
  if (section.main_issue === "Arms need attention") {
    return "Arm shapes drifted from the reference.";
  }
  if (section.main_issue === "Legs need attention") {
    return "Foot and knee placement drifted from the reference.";
  }
  if (section.main_issue === "Body position needs attention") {
    return "Body angle drifted from the reference.";
  }
  return issueLevelToLabel("needs_work");
}

type PracticeAnalysisReportProps = {
  analysis: PracticeAnalysisResult;
  stoppedEarly: boolean;
};

export function PracticeAnalysisReport({
  analysis,
  stoppedEarly,
}: PracticeAnalysisReportProps) {
  const [showDebug, setShowDebug] = useState(false);
  const { score_breakdown, timing_breakdown, session_summary, reliability } =
    analysis;

  const sparseCapture = hasSparsePracticeCapture(session_summary);
  const partialIntro = getPartialCaptureIntro(analysis);
  const headline = getDemoHeadline(analysis, stoppedEarly);
  const positiveComment = getPositiveComment(analysis);
  const practiceSuggestions = getPracticeSuggestions(analysis);
  const performanceRows = getPerformanceRows(score_breakdown, timing_breakdown);

  const bestSection = isSectionShowableForUser(
    analysis.section_breakdown.best_section,
  )
    ? analysis.section_breakdown.best_section
    : null;
  const practiceSection = isSectionShowableForUser(
    analysis.section_breakdown.weakest_section,
  )
    ? analysis.section_breakdown.weakest_section
    : null;

  const simpleTips = analysis.improvement_priorities
    .slice(0, 4)
    .map((item) => item.suggestion.trim())
    .filter((tip, index, list) => tip.length > 0 && list.indexOf(tip) === index);

  return (
    <>
      <View style={styles.heroCard}>
        <Text style={styles.heroTitle}>{headline}</Text>
        <Text style={styles.heroSubtitle}>{positiveComment}</Text>
        {stoppedEarly && (
          <Text style={styles.heroSubtitle}>
            You stopped before the full track ended — this review covers what you
            completed.
          </Text>
        )}
        {partialIntro && (
          <Text style={styles.heroSubtitle}>{partialIntro}</Text>
        )}
      </View>

      {sparseCapture && (
        <View style={styles.setupCard}>
          <Text style={styles.cardTitle}>Before your next try</Text>
          {SETUP_PRACTICAL_FIXES.map((fix) => (
            <Text key={fix} style={styles.bulletLine}>
              • {fix}
            </Text>
          ))}
        </View>
      )}

      {!sparseCapture && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Your performance</Text>
          {performanceRows.map((row, index) => (
            <View
              key={row.label}
              style={[styles.performanceRow, index > 0 && styles.performanceRowBorder]}
            >
              <Text style={styles.performanceLabel}>{row.label}</Text>
              <Text
                style={[
                  styles.performanceValue,
                  row.label !== "Timing" && {
                    color: scoreLabelTone(
                      row.label === "Overall performance"
                        ? score_breakdown.overall_score
                        : row.label === "Arms"
                          ? score_breakdown.arms_score
                          : row.label === "Legs"
                            ? score_breakdown.legs_score
                            : score_breakdown.torso_score,
                    ),
                  },
                ]}
              >
                {row.value}
              </Text>
            </View>
          ))}
        </View>
      )}

      {bestSection && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Best moment</Text>
          <Text style={styles.momentTime}>
            {formatDemoMomentLabel(
              bestSection.start_time_seconds,
              bestSection.end_time_seconds,
            )}
          </Text>
          <Text style={styles.bodyText}>{friendlySectionNote(bestSection)}</Text>
        </View>
      )}

      {practiceSection && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Area to practice</Text>
          <Text style={styles.momentTime}>
            {formatDemoMomentLabel(
              practiceSection.start_time_seconds,
              practiceSection.end_time_seconds,
            )}
          </Text>
          <Text style={styles.bodyText}>
            {friendlySectionNote(practiceSection)}
          </Text>
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Practice suggestions</Text>
        {(simpleTips.length > 0 ? simpleTips : practiceSuggestions).map((tip) => (
          <Text key={tip} style={styles.bulletLine}>
            • {tip}
          </Text>
        ))}
      </View>

      {__DEV__ && (
        <>
          <Pressable
            style={styles.debugToggle}
            onPress={() => setShowDebug((open) => !open)}
            accessibilityRole="button"
          >
            <Text style={styles.debugToggleText}>
              {showDebug ? "Hide debug details" : "Show debug details"}
            </Text>
          </Pressable>
          {showDebug && (
            <View style={styles.debugCard}>
              <Text style={styles.debugLine}>
                Reliability level (internal): {reliability.level}
              </Text>
              <Text style={styles.debugLine}>
                Compared moments: {session_summary.usable_comparison_frames} /{" "}
                {session_summary.sampled_live_frames}
              </Text>
              <Text style={styles.debugLine}>
                Skipped moments: {session_summary.skipped_frames_total}
              </Text>
              {reliability.reasons.map((reason) => (
                <Text key={reason} style={styles.debugLine}>
                  • {reason}
                </Text>
              ))}
            </View>
          )}
        </>
      )}
    </>
  );
}

const styles = StyleSheet.create({
  heroCard: {
    backgroundColor: "#e8f2fc",
    borderRadius: 12,
    padding: 18,
    marginTop: 8,
    borderWidth: 1,
    borderColor: "#c5daf0",
  },
  heroTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: "#1a4d8f",
    lineHeight: 28,
  },
  heroSubtitle: {
    fontSize: 15,
    lineHeight: 22,
    color: "#444",
    marginTop: 10,
  },
  card: {
    backgroundColor: "#f5f8fc",
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#d8e4f2",
  },
  setupCard: {
    backgroundColor: "#fff8e6",
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#e8d48a",
  },
  cardTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: "#1a4d8f",
    marginBottom: 10,
  },
  bodyText: {
    fontSize: 15,
    lineHeight: 22,
    color: "#333",
  },
  bulletLine: {
    fontSize: 15,
    lineHeight: 23,
    color: "#333",
    marginTop: 6,
  },
  performanceRow: {
    marginTop: 4,
  },
  performanceRowBorder: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "#e2ebf5",
  },
  performanceLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: "#333",
  },
  performanceValue: {
    fontSize: 15,
    lineHeight: 22,
    color: "#555",
    marginTop: 4,
  },
  momentTime: {
    fontSize: 15,
    fontWeight: "600",
    color: "#1a4d8f",
    marginBottom: 6,
  },
  debugToggle: {
    marginTop: 16,
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 4,
  },
  debugToggleText: {
    fontSize: 12,
    color: "#888",
    textDecorationLine: "underline",
  },
  debugCard: {
    backgroundColor: "#f3f3f3",
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  debugLine: {
    fontSize: 12,
    lineHeight: 18,
    color: "#555",
    marginTop: 4,
  },
});
