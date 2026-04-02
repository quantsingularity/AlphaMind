import { ScrollView, StyleSheet } from "react-native";
import { Card, Headline, Paragraph, Title, useTheme } from "react-native-paper";

export default function FeaturesScreen() {
  const theme = useTheme();

  const features = [
    {
      key: "ai",
      icon: "brain",
      title: "AI/ML Core",
      description:
        "Leverage advanced machine learning models for predictive analytics and strategy generation.",
    },
    {
      key: "quant",
      icon: "chart-bar",
      title: "Quantitative Research",
      description:
        "Access powerful tools for backtesting, factor analysis, and portfolio optimization.",
    },
    {
      key: "altdata",
      icon: "satellite-uplink",
      title: "Alternative Data Integration",
      description:
        "Incorporate diverse datasets like satellite imagery, social media sentiment, and more.",
    },
    {
      key: "risk",
      icon: "shield-check",
      title: "Risk Management",
      description:
        "Utilize sophisticated risk models and real-time monitoring to protect capital.",
    },
    {
      key: "exec",
      icon: "lightning-bolt",
      title: "Execution Infrastructure",
      description:
        "Connect seamlessly with brokers for low-latency order execution and management.",
    },
  ];

  return (
    <ScrollView
      contentContainerStyle={[
        styles.container,
        { backgroundColor: theme.colors.background },
      ]}
    >
      <Headline style={styles.title}>Key Features</Headline>
      <Paragraph style={styles.paragraph}>
        Discover the core capabilities of the AlphaMind platform.
      </Paragraph>

      {features.map((feature) => (
        <Card key={feature.key} style={styles.card}>
          <Card.Content>
            <Title>{feature.title}</Title>
            <Paragraph>{feature.description}</Paragraph>
          </Card.Content>
        </Card>
      ))}
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
