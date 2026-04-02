import { useCallback, useEffect, useState } from "react";
import {
  Alert,
  RefreshControl,
  ScrollView,
  StyleSheet,
  View,
} from "react-native";
import {
  Button,
  Card,
  Headline,
  Paragraph,
  Text,
  Title,
  useTheme,
} from "react-native-paper";
import ErrorMessage from "../components/ErrorMessage";
import LoadingSpinner from "../components/LoadingSpinner";
import { researchService } from "../services/researchService";

export default function ResearchScreen() {
  const theme = useTheme();
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchPapers = useCallback(async () => {
    try {
      setError(null);
      const data = await researchService.getPapers();
      setPapers(data);
    } catch (err) {
      setError(err.message || "Failed to load research papers");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPapers();
  }, [fetchPapers]);

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
      <Headline style={styles.title}>Research Insights</Headline>
      <Paragraph style={styles.paragraph}>
        Explore the latest publications and findings from the AlphaMind research
        team.
      </Paragraph>

      {papers.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={[styles.emptyText, { color: theme.colors.outline }]}>
            No research papers available at this time.
          </Text>
        </View>
      ) : (
        papers.map((item) => (
          <Card key={item.id} style={styles.card}>
            <Card.Content>
              <Title>{item.title}</Title>
              <Paragraph>{item.summary}</Paragraph>
              {item.authors && (
                <Text style={styles.authors}>
                  By: {item.authors.join(", ")}
                </Text>
              )}
              {item.date && (
                <Text style={styles.date}>Published: {item.date}</Text>
              )}
              {item.category && (
                <Text
                  style={[styles.category, { color: theme.colors.primary }]}
                >
                  {item.category}
                </Text>
              )}
            </Card.Content>
            <Card.Actions>
              <Button onPress={() => handlePress(item)}>Read More</Button>
            </Card.Actions>
          </Card>
        ))
      )}
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
  category: {
    fontSize: 12,
    fontWeight: "bold",
    marginTop: 4,
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
  emptyContainer: {
    alignItems: "center",
    marginTop: 40,
    padding: 20,
  },
  emptyText: {
    fontSize: 16,
    textAlign: "center",
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
