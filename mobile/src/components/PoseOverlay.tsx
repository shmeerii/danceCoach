import { useCallback, useState } from "react";
import { LayoutChangeEvent, StyleSheet, View } from "react-native";
import Svg, { Circle, Line } from "react-native-svg";
import type { PoseLandmarkInput } from "../utils/poseNormalization";

/** Front camera preview is mirrored horizontally relative to landmark x coordinates. */
export const FRONT_CAMERA_PREVIEW_MIRRORED = true;

const LANDMARK_VISIBILITY_THRESHOLD = 0.5;
const LANDMARK_RADIUS = 5;
const SKELETON_STROKE_WIDTH = 3;

const POSE_SKELETON_CONNECTIONS: [string, string][] = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["left_ankle", "left_heel"],
  ["left_ankle", "left_foot_index"],
  ["left_heel", "left_foot_index"],
  ["right_ankle", "right_heel"],
  ["right_ankle", "right_foot_index"],
  ["right_heel", "right_foot_index"],
];

type PoseOverlayProps = {
  poseDetected: boolean;
  landmarks: PoseLandmarkInput[];
};

function isFiniteNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}

function isLandmarkVisible(landmark: PoseLandmarkInput | undefined): boolean {
  if (!landmark || !isFiniteNumber(landmark.x) || !isFiniteNumber(landmark.y)) {
    return false;
  }

  const visibility = landmark.visibility;
  if (visibility === null || visibility === undefined) {
    return true;
  }

  return visibility >= LANDMARK_VISIBILITY_THRESHOLD;
}

function mapNormalizedX(x: number, width: number): number {
  const normalizedX = FRONT_CAMERA_PREVIEW_MIRRORED ? 1 - x : x;
  return normalizedX * width;
}

function mapNormalizedY(y: number, height: number): number {
  return y * height;
}

export function PoseOverlay({ poseDetected, landmarks }: PoseOverlayProps) {
  const [layout, setLayout] = useState({ width: 0, height: 0 });

  const handleLayout = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setLayout({ width, height });
  }, []);

  if (!poseDetected || layout.width <= 0 || layout.height <= 0) {
    return (
      <View
        style={styles.container}
        pointerEvents="none"
        onLayout={handleLayout}
      />
    );
  }

  const landmarkByName = new Map(
    landmarks.map((landmark) => [landmark.name, landmark]),
  );

  const visibleLandmarks = landmarks.filter(isLandmarkVisible);

  const skeletonLines = POSE_SKELETON_CONNECTIONS.flatMap(([startName, endName]) => {
    const start = landmarkByName.get(startName);
    const end = landmarkByName.get(endName);
    if (!isLandmarkVisible(start) || !isLandmarkVisible(end)) {
      return [];
    }

    return [
      {
        key: `${startName}-${endName}`,
        x1: mapNormalizedX(start!.x!, layout.width),
        y1: mapNormalizedY(start!.y!, layout.height),
        x2: mapNormalizedX(end!.x!, layout.width),
        y2: mapNormalizedY(end!.y!, layout.height),
      },
    ];
  });

  return (
    <View
      style={styles.container}
      pointerEvents="none"
      onLayout={handleLayout}
    >
      <Svg width={layout.width} height={layout.height}>
        {skeletonLines.map((line) => (
          <Line
            key={line.key}
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            stroke="rgba(255, 255, 255, 0.9)"
            strokeWidth={SKELETON_STROKE_WIDTH}
            strokeLinecap="round"
          />
        ))}
        {visibleLandmarks.map((landmark) => (
          <Circle
            key={landmark.name}
            cx={mapNormalizedX(landmark.x!, layout.width)}
            cy={mapNormalizedY(landmark.y!, layout.height)}
            r={LANDMARK_RADIUS}
            fill="#4ade80"
            stroke="#14532d"
            strokeWidth={1.5}
          />
        ))}
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
  },
});
