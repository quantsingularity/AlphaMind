import React from "react";
import { StyleSheet, ScrollView, Text } from "react-native"; // Added Text import
import {
  // Surface, // Removed unused import
  // Text, // Removed unused import
  Headline,
  Paragraph,
  Card,
  Title,
  useTheme,
} from "react-native-paper";

export default function FeaturesScreen() {
  const theme = useTheme();

  return (
    <ScrollView
      contentContainerStyle={[
        styles.container,
        { backgroundColor: theme.colors.background },
      ]}
    >
      <Headline style={styles.title}>
        <Text>Key Features</Text>
      </Headline>
      <Paragraph style={styles.paragraph}>
        <Text>Discover the core capabilities of the AlphaMind platform.</Text>
      </Paragraph>

      <Card style={styles.card}>
        <Card.Content>
          <Title>
            <Text>AI/ML Core</Text>
          </Title>
          <Paragraph>
            <Text>
              Leverage advanced machine learning models for predictive analytics
              and strategy generation.
            </Text>
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>
            <Text>Quantitative Research</Text>
          </Title>
          <Paragraph>
            <Text>
              Access powerful tools for backtesting, factor analysis, and
              portfolio optimization.
            </Text>
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>
            <Text>Alternative Data Integration</Text>
          </Title>
          <Paragraph>
            <Text>
              Incorporate diverse datasets like satellite imagery, social media
              sentiment, and more.
            </Text>
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>
            <Text>Risk Management</Text>
          </Title>
          <Paragraph>
            <Text>
              Utilize sophisticated risk models and real-time monitoring to
              protect capital.
            </Text>
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>
            <Text>Execution Infrastructure</Text>
          </Title>
          <Paragraph>
            <Text>
              Connect seamlessly with brokers for low-latency order execution
              and management.
            </Text>
          </Paragraph>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  card: {
    marginBottom: 16,
    width: "100%",
  },
  container: {
    alignItems: "center",
    flexGrow: 1,
    padding: 20,
  },
  paragraph: {
    marginBottom: 24,
    textAlign: "center",
  },
  title: {
    marginBottom: 16,
    textAlign: "center",
  },
});
