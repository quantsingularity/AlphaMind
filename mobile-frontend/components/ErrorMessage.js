import { StyleSheet, View } from "react-native";
import { Button, Icon, Text, useTheme } from "react-native-paper";

export default function ErrorMessage({
  title = "Something went wrong",
  message,
  onRetry,
}) {
  const theme = useTheme();

  const styles = StyleSheet.create({
    container: {
      alignItems: "center",
      backgroundColor: theme.colors.background,
      flex: 1,
      justifyContent: "center",
      padding: 24,
    },
    errorBox: {
      alignItems: "center",
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.error,
      borderRadius: 8,
      borderWidth: 1,
      elevation: 2,
      maxWidth: 360,
      padding: 20,
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 3,
      width: "100%",
    },
    icon: {
      marginBottom: 12,
    },
    message: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 13,
      lineHeight: 20,
      marginBottom: 16,
      textAlign: "center",
    },
    retryButton: {
      borderRadius: 6,
    },
    retryButtonContent: {
      paddingHorizontal: 8,
      paddingVertical: 2,
    },
    title: {
      color: theme.colors.error,
      fontSize: 16,
      fontWeight: "700",
      marginBottom: 8,
      textAlign: "center",
    },
  });

  return (
    <View style={styles.container} accessible={true} accessibilityRole="alert">
      <View style={styles.errorBox}>
        <View style={styles.icon}>
          <Icon
            source="alert-circle-outline"
            size={30}
            color={theme.colors.error}
          />
        </View>
        <Text style={styles.title}>{title}</Text>
        {message && <Text style={styles.message}>{message}</Text>}
        {onRetry && (
          <Button
            mode="contained"
            onPress={onRetry}
            style={styles.retryButton}
            contentStyle={styles.retryButtonContent}
            buttonColor={theme.colors.primary}
          >
            Try Again
          </Button>
        )}
      </View>
    </View>
  );
}
