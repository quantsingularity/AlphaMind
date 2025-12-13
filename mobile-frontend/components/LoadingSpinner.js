import React from "react";
import { View, StyleSheet } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";

export default function LoadingSpinner({ message = "Loading...", size = "large" }) {
  return (
    <View style={styles.container}>
      <ActivityIndicator size={size} />
      {message && <Text style={styles.message}>{message}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
  },
  message: {
    marginTop: 16,
  },
});
