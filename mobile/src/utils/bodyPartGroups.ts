import type { MajorDanceLandmarkName } from "./poseSimilarity";

/** Landmark names match `POSE_LANDMARK_NAMES` / serialized pose JSON (major body points only). */

export const ARMS = [
  "left_shoulder",
  "left_elbow",
  "left_wrist",
  "right_shoulder",
  "right_elbow",
  "right_wrist",
] as const satisfies readonly MajorDanceLandmarkName[];

export const LEFT_ARM = [
  "left_shoulder",
  "left_elbow",
  "left_wrist",
] as const satisfies readonly MajorDanceLandmarkName[];

export const RIGHT_ARM = [
  "right_shoulder",
  "right_elbow",
  "right_wrist",
] as const satisfies readonly MajorDanceLandmarkName[];

export const LEGS = [
  "left_hip",
  "left_knee",
  "left_ankle",
  "right_hip",
  "right_knee",
  "right_ankle",
] as const satisfies readonly MajorDanceLandmarkName[];

export const LEFT_LEG = [
  "left_hip",
  "left_knee",
  "left_ankle",
] as const satisfies readonly MajorDanceLandmarkName[];

export const RIGHT_LEG = [
  "right_hip",
  "right_knee",
  "right_ankle",
] as const satisfies readonly MajorDanceLandmarkName[];

export const TORSO = [
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
] as const satisfies readonly MajorDanceLandmarkName[];

export const SHOULDERS = [
  "left_shoulder",
  "right_shoulder",
] as const satisfies readonly MajorDanceLandmarkName[];

export const HIPS = [
  "left_hip",
  "right_hip",
] as const satisfies readonly MajorDanceLandmarkName[];

export const LEFT_SIDE = [
  "left_shoulder",
  "left_elbow",
  "left_wrist",
  "left_hip",
  "left_knee",
  "left_ankle",
] as const satisfies readonly MajorDanceLandmarkName[];

export const RIGHT_SIDE = [
  "right_shoulder",
  "right_elbow",
  "right_wrist",
  "right_hip",
  "right_knee",
  "right_ankle",
] as const satisfies readonly MajorDanceLandmarkName[];

export const BODY_PART_GROUP_LANDMARKS = {
  arms: ARMS,
  left_arm: LEFT_ARM,
  right_arm: RIGHT_ARM,
  legs: LEGS,
  left_leg: LEFT_LEG,
  right_leg: RIGHT_LEG,
  torso: TORSO,
  shoulders: SHOULDERS,
  hips: HIPS,
  left_side: LEFT_SIDE,
  right_side: RIGHT_SIDE,
} as const;

export type BodyPartGroupName = keyof typeof BODY_PART_GROUP_LANDMARKS;

export type MainDanceGroupName = "arms" | "legs" | "torso";

export type SideGroupName = "left_arm" | "right_arm" | "left_leg" | "right_leg";

const MAIN_DANCE_GROUPS: readonly MainDanceGroupName[] = ["arms", "legs", "torso"];

const SIDE_GROUPS: readonly SideGroupName[] = [
  "left_arm",
  "right_arm",
  "left_leg",
  "right_leg",
];

const BODY_PART_DISPLAY_NAMES: Record<BodyPartGroupName, string> = {
  arms: "Arms",
  left_arm: "Left arm",
  right_arm: "Right arm",
  legs: "Legs",
  left_leg: "Left leg",
  right_leg: "Right leg",
  torso: "Torso",
  shoulders: "Shoulders",
  hips: "Hips",
  left_side: "Left side",
  right_side: "Right side",
};

export function getLandmarksForGroup(
  groupName: BodyPartGroupName,
): readonly MajorDanceLandmarkName[] {
  return BODY_PART_GROUP_LANDMARKS[groupName];
}

export function getBodyPartDisplayName(groupName: BodyPartGroupName): string {
  return BODY_PART_DISPLAY_NAMES[groupName];
}

export function getMainDanceGroups(): readonly MainDanceGroupName[] {
  return MAIN_DANCE_GROUPS;
}

export function getSideGroups(): readonly SideGroupName[] {
  return SIDE_GROUPS;
}
