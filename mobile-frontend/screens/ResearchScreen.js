import React, { useEffect, useState } from "react";
import {
  StyleSheet,
  ScrollView,
  Alert,
  Text,
  RefreshControl,
} from "react-native";
import {
  Headline,
  Paragraph,
  Card,
  Title,
  Button,
  useTheme,
} from "react-native-paper";
import { researchService } from "../services/researchService";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorMessage from "../components/ErrorMessage";

export default function ResearchScreen() {
  const theme = useTheme();
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchPapers = async () => {
    try {
      setError(null);
      const data = await researchService.getPapers();
      setPapers(data);
    } catch (err) {
      setError(err.message || "Failed to load research papers");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPapers();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchPapers();
    setRefreshing(false);
  };

  const handlePress = (paper) => {
    Alert.alert("Research Paper", `Title: ${paper.title}\n\nOpening paper...`, [
      {
        text: "Cancel",
        style: "cancel",
      },
      {
        text: "Open",
        onPress: () => {
          // In production, open paper URL or navigate to detail screen
          console.log("Opening paper:", paper.url);
        },
      },
    ]);
  };

  if (loading) {
    return <LoadingSpinner message="Loading research papers..." />;
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={fetchPapers} />;
  }

  return (
    <ScrollView
      contentContainerStyle={[
        styles.container,
        { backgroundColor: theme.colors.background },
      ]}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <Headline style={styles.title}>
        <Text>Research Insights</Text>
      </Headline>
      <Paragraph style={styles.paragraph}>
        <Text>
          Explore the latest publications and findings from the AlphaMind
          research team.
        </Text>
      </Paragraph>

      {papers.map((item) => (
        <Card key={item.id} style={styles.card}>
          <Card.Content>
            <Title>
              <Text>{item.title}</Text>
            </Title>
            <Paragraph>
              <Text>{item.summary}</Text>
            </Paragraph>
            {item.authors && (
              <Text style={styles.authors}>By: {item.authors.join(", ")}</Text>
            )}
            {item.date && (
              <Text style={styles.date}>Published: {item.date}</Text>
            )}
          </Card.Content>
          <Card.Actions>
            <Button onPress={() => handlePress(item)}>
              <Text>Read More</Text>
            </Button>
          </Card.Actions>
        </Card>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  authors: {
    fontSize: 12,
    fontStyle: "italic",
    marginTop: 8,
  },
  card: {
    marginBottom: 16,
    width: "100%",
  },
  container: {
    alignItems: "center",
    flexGrow: 1,
    padding: 20,
  },
  date: {
    fontSize: 12,
    marginTop: 4,
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
