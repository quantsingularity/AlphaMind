import { StyleSheet, View } from "react-native";
import { Button, Text, useTheme } from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

export default function ErrorMessage({
  message,
  onRetry,
  title = "Something went wrong",
}) {
  const theme = useTheme();

  return (
    <View style={styles.container} accessible accessibilityRole="alert">
      <Icon name="alert-circle-outline" size={64} color={theme.colors.error} />
      <Text
        variant="titleMedium"
        style={[styles.title, { color: theme.colors.error }]}
      >
        {title}
      </Text>
      {!!message && (
        <Text
          variant="bodyMedium"
          style={[styles.message, { color: theme.colors.onSurfaceVariant }]}
        >
          {message}
        </Text>
      )}
      {onRetry && (
        <Button mode="contained" onPress={onRetry} style={styles.button}>
          Try Again
        </Button>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  button: {
    marginTop: 16,
  },
  container: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    padding: 20,
  },
  message: {
    marginTop: 8,
    textAlign: "center",
  },
  title: {
    marginTop: 16,
    textAlign: "center",
  },
});
