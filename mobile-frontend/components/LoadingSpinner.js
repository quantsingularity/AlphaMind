import { StyleSheet, View } from "react-native";
import { ActivityIndicator, Text, useTheme } from "react-native-paper";

export default function LoadingSpinner({
  message = "Loading...",
  size = "large",
  fullScreen = true,
}) {
  const theme = useTheme();

  return (
    <View
      style={[
        fullScreen ? styles.container : styles.inline,
        {
          backgroundColor: fullScreen ? theme.colors.background : "transparent",
        },
      ]}
      accessible
      accessibilityRole="progressbar"
      accessibilityLabel={message}
    >
      <ActivityIndicator size={size} color={theme.colors.primary} />
      {!!message && (
        <Text
          variant="bodyMedium"
          style={[styles.message, { color: theme.colors.onSurfaceVariant }]}
        >
          {message}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
  },
  inline: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 20,
  },
  message: {
    marginTop: 16,
  },
});
