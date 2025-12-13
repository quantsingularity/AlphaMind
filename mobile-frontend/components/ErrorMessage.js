import React from "react";
import { View, StyleSheet } from "react-native";
import { Text, Button, useTheme } from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

export default function ErrorMessage({ message, onRetry }) {
  const theme = useTheme();

  return (
    <View style={styles.container}>
      <Icon name="alert-circle-outline" size={64} color={theme.colors.error} />
      <Text style={[styles.message, { color: theme.colors.error }]}>{message}</Text>
      {onRetry && (
        <Button mode="contained" onPress={onRetry} style={styles.button}>
          Retry
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
    marginTop: 16,
    textAlign: "center",
  },
});
