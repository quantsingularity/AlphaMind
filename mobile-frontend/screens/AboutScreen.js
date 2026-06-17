import { Linking, ScrollView, StyleSheet, View } from "react-native";
import { Button, Icon, Text, useTheme } from "react-native-paper";

const TEAM = [
  {
    role: "Quantitative Research Lead",
    focus: "ML architecture and alpha generation",
  },
  {
    role: "Execution Engineer",
    focus: "Low-latency systems and order routing",
  },
  { role: "Risk Analyst", focus: "Portfolio risk, stress testing, VaR" },
  { role: "Data Engineer", focus: "Alternative data pipelines and feeds" },
];

const STACK = [
  {
    group: "AI / ML",
    items: ["PyTorch", "TensorFlow", "Scikit-learn", "Hugging Face"],
  },
  { group: "Backend", items: ["Python 3.10+", "Flask", "Celery", "Redis"] },
  {
    group: "Frontend",
    items: ["React 19", "React Native", "TypeScript", "Recharts"],
  },
  {
    group: "Infrastructure",
    items: ["Docker", "PostgreSQL", "Kafka", "Prometheus"],
  },
];

export default function AboutScreen() {
  const theme = useTheme();
  const styles = createStyles(theme);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.hero}>
        <Text style={styles.eyebrow}>About</Text>
        <Text style={styles.title}>AlphaMind</Text>
        <Text style={styles.subtitle}>
          Institutional-grade quantitative trading powered by AI
        </Text>
      </View>

      <View style={styles.section}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Our mission</Text>
          <Text style={styles.body}>
            AlphaMind combines alternative data, advanced machine learning, and
            high-frequency execution to pursue superior risk-adjusted returns.
            By leveraging Temporal Fusion Transformers, reinforcement learning,
            and real-time data processing, it adapts to market regimes and
            discovers alpha where traditional methods fall short.
          </Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Core team</Text>
        <View style={styles.list}>
          {TEAM.map((m) => (
            <View key={m.role} style={styles.card}>
              <Text style={styles.roleTitle}>{m.role}</Text>
              <Text style={styles.roleFocus}>{m.focus}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Technology stack</Text>
        <View style={styles.card}>
          {STACK.map((s, idx) => (
            <View
              key={s.group}
              style={[styles.stackGroup, idx > 0 && styles.stackGroupBorder]}
            >
              <Text style={styles.stackLabel}>{s.group}</Text>
              <View style={styles.chipRow}>
                {s.items.map((it) => (
                  <View key={it} style={styles.chip}>
                    <Text style={styles.chipText}>{it}</Text>
                  </View>
                ))}
              </View>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.section}>
        <View style={styles.card}>
          <View style={styles.osRow}>
            <Icon source="github" size={26} color={theme.colors.onSurface} />
            <View style={styles.osTextWrap}>
              <Text style={styles.cardTitle}>Open source</Text>
              <Text style={styles.body}>
                Contributions, issues, and feature requests are welcome.
              </Text>
            </View>
          </View>
          <Button
            mode="contained"
            style={styles.osButton}
            contentStyle={styles.osButtonContent}
            onPress={() =>
              Linking.openURL("https://github.com/quantsingularity/AlphaMind")
            }
          >
            View on GitHub
          </Button>
        </View>
      </View>
    </ScrollView>
  );
}

const createStyles = (theme) =>
  StyleSheet.create({
    body: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 14,
      lineHeight: 21,
    },
    card: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      padding: 16,
    },
    cardTitle: {
      color: theme.colors.onSurface,
      fontSize: 16,
      fontWeight: "700",
      marginBottom: 6,
    },
    chip: {
      backgroundColor: theme.colors.surfaceVariant,
      borderRadius: 999,
      paddingHorizontal: 12,
      paddingVertical: 6,
    },
    chipRow: {
      flexDirection: "row",
      flexWrap: "wrap",
      gap: 8,
    },
    chipText: {
      color: theme.colors.onSurface,
      fontSize: 12,
    },
    container: {
      backgroundColor: theme.colors.background,
      flexGrow: 1,
    },
    eyebrow: {
      color: theme.colors.primary,
      fontSize: 12,
      fontWeight: "700",
      letterSpacing: 1,
      marginBottom: 6,
      textTransform: "uppercase",
    },
    hero: {
      backgroundColor: theme.colors.surface,
      borderBottomColor: theme.colors.outlineVariant,
      borderBottomWidth: 1,
      paddingBottom: 18,
      paddingHorizontal: 16,
      paddingTop: 24,
    },
    list: {
      gap: 12,
    },
    osButton: {
      borderRadius: 8,
    },
    osButtonContent: {
      paddingVertical: 6,
    },
    osRow: {
      alignItems: "center",
      flexDirection: "row",
      gap: 12,
      marginBottom: 14,
    },
    osTextWrap: {
      flex: 1,
    },
    roleFocus: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 13,
    },
    roleTitle: {
      color: theme.colors.onSurface,
      fontSize: 14,
      fontWeight: "700",
      marginBottom: 3,
    },
    section: {
      paddingHorizontal: 16,
      paddingVertical: 14,
    },
    sectionTitle: {
      color: theme.colors.onBackground,
      fontSize: 18,
      fontWeight: "700",
      marginBottom: 12,
    },
    stackGroup: {
      paddingVertical: 12,
    },
    stackGroupBorder: {
      borderTopColor: theme.colors.outlineVariant,
      borderTopWidth: 1,
    },
    stackLabel: {
      color: theme.colors.primary,
      fontSize: 12,
      fontWeight: "700",
      letterSpacing: 0.8,
      marginBottom: 8,
      textTransform: "uppercase",
    },
    subtitle: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 14,
    },
    title: {
      color: theme.colors.onBackground,
      fontSize: 28,
      fontWeight: "800",
      letterSpacing: -0.5,
      marginBottom: 4,
    },
  });
