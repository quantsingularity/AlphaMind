import React from "react";
import { View, StyleSheet } from "react-native";
import { Card, Title, Text, useTheme } from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

export default function KPICard({
  title,
  value,
  change,
  changeColor,
  icon,
  isLoading,
}) {
  const theme = useTheme();

  return (
    <Card style={styles.card}>
      <Card.Content style={styles.cardContent}>
        {icon && (
          <Icon
            name={icon}
            size={32}
            color={theme.colors.primary}
            style={styles.icon}
          />
        )}
        <View style={styles.textContainer}>
          <Text
            style={[styles.title, { color: theme.colors.onSurfaceVariant }]}
          >
            {title}
          </Text>
          <Title style={styles.value}>{isLoading ? "..." : value}</Title>
          {change && (
            <Text
              style={[
                styles.change,
                { color: changeColor || theme.colors.onSurface },
              ]}
            >
              {change}
            </Text>
          )}
        </View>
      </Card.Content>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    marginBottom: 16,
    width: "48%",
  },
  cardContent: {
    alignItems: "center",
    paddingHorizontal: 8,
    paddingVertical: 12,
  },
  change: {
    fontSize: 12,
    marginTop: 2,
  },
  icon: {
    marginBottom: 8,
  },
  textContainer: {
    alignItems: "center",
  },
  title: {
    fontSize: 12,
    marginBottom: 2,
    textAlign: "center",
  },
  value: {
    fontSize: 16,
    lineHeight: 20,
    textAlign: "center",
  },
});
