/** @type {import('expo/config').ExpoConfig} */
module.exports = ({ config }) => ({
  ...config,
  extra: {
    ...config.extra,
    // Baked into the APK at build time. Override via mobile/.env (EXPO_PUBLIC_BACKEND_URL).
    backendUrl:
      process.env.EXPO_PUBLIC_BACKEND_URL ?? "http://192.168.1.199:8000",
  },
});
