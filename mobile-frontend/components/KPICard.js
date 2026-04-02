import { StyleSheet, View } from "react-native";
import { Card, Text, useTheme } from "react-native-paper";
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
    <Card
      style={styles.card}
      accessible
      accessibilityLabel={`${title}: ${isLoading ? "loading" : value}`}
    >
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
            variant="labelSmall"
            style={[styles.title, { color: theme.colors.onSurfaceVariant }]}
          >
            {title}
          </Text>
          <Text
            variant="titleMedium"
            style={[styles.value, { color: theme.colors.onSurface }]}
          >
            {isLoading ? "—" : value}
          </Text>
          {!!change && (
            <Text
              variant="labelSmall"
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
    marginTop: 2,
    textAlign: "center",
  },
  icon: {
    marginBottom: 8,
  },
  textContainer: {
    alignItems: "center",
  },
  title: {
    marginBottom: 2,
    textAlign: "center",
  },
  value: {
    textAlign: "center",
  },
});
