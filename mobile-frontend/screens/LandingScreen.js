import { ScrollView, StyleSheet, View } from "react-native";
import { Button, Text, useTheme } from "react-native-paper";
import { SafeAreaView } from "react-native-safe-area-context";
import ThemeToggle from "../components/ThemeToggle";

const FEATURES = [
  {
    title: "Alternative Data",
    body: "Satellite, filings, and social signals fused into tradable features.",
  },
  {
    title: "Quant Research",
    body: "Temporal Fusion Transformers and reinforcement learning at the core.",
  },
  {
    title: "Execution Engine",
    body: "Low-latency, cost-aware order routing across venues.",
  },
  {
    title: "Risk Management",
    body: "Regime-aware VaR, stress testing, and live exposure limits.",
  },
];

const STATS = [
  { label: "Sharpe", value: "2.4" },
  { label: "Max DD", value: "9%" },
  { label: "Profit Factor", value: "4.1" },
];

export default function LandingScreen({ navigation }) {
  const theme = useTheme();
  const styles = createStyles(theme);

  return (
    <SafeAreaView style={styles.safe} edges={["top", "bottom"]}>
      <View style={styles.topBar}>
        <ThemeToggle color={theme.colors.onBackground} />
      </View>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.hero}>
          <View style={styles.logoCircle}>
            <Text style={styles.logoLetter}>α</Text>
          </View>
          <Text style={styles.brand}>AlphaMind</Text>
          <Text style={styles.badge}>Institutional-grade · Quant AI</Text>
          <Text style={styles.headline}>
            Where alternative data becomes durable alpha
          </Text>
          <Text style={styles.sub}>
            A quantitative trading system pairing machine learning, alternative
            data, and high-frequency execution for desks that measure edge in
            basis points.
          </Text>

          <View style={styles.ctaGroup}>
            <Button
              mode="contained"
              style={styles.primaryBtn}
              contentStyle={styles.btnContent}
              onPress={() => navigation.navigate("Register")}
            >
              Create an Account
            </Button>
            <Button
              mode="outlined"
              style={styles.outlineBtn}
              contentStyle={styles.btnContent}
              textColor={theme.colors.primary}
              onPress={() => navigation.navigate("Login")}
            >
              Sign In
            </Button>
          </View>

          <View style={styles.statsRow}>
            {STATS.map((s) => (
              <View key={s.label} style={styles.statItem}>
                <Text style={styles.statValue}>{s.value}</Text>
                <Text style={styles.statLabel}>{s.label}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.eyebrow}>Capabilities</Text>
          <Text style={styles.sectionTitle}>Four engines, one edge</Text>
          <View style={styles.featureGrid}>
            {FEATURES.map((f) => (
              <View key={f.title} style={styles.featureCard}>
                <View style={styles.featureDot} />
                <Text style={styles.featureTitle}>{f.title}</Text>
                <Text style={styles.featureBody}>{f.body}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            Automating systematic, risk-aware trading.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const createStyles = (theme) =>
  StyleSheet.create({
    badge: {
      backgroundColor: theme.colors.primaryLight,
      borderRadius: 999,
      color: theme.colors.primary,
      fontSize: 12,
      fontWeight: "700",
      marginBottom: 18,
      overflow: "hidden",
      paddingHorizontal: 12,
      paddingVertical: 5,
    },
    brand: {
      color: theme.colors.onBackground,
      fontSize: 24,
      fontWeight: "800",
      letterSpacing: -0.3,
      marginBottom: 12,
    },
    btnContent: {
      paddingVertical: 8,
    },
    ctaGroup: {
      alignSelf: "stretch",
      gap: 12,
    },
    eyebrow: {
      color: theme.colors.primary,
      fontSize: 12,
      fontWeight: "700",
      letterSpacing: 1,
      marginBottom: 6,
      textTransform: "uppercase",
    },
    featureBody: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 14,
      lineHeight: 20,
    },
    featureCard: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 14,
      borderWidth: 1,
      padding: 18,
    },
    featureDot: {
      backgroundColor: theme.colors.accent,
      borderRadius: 5,
      height: 10,
      marginBottom: 10,
      width: 10,
    },
    featureGrid: {
      gap: 12,
    },
    featureTitle: {
      color: theme.colors.onSurface,
      fontSize: 16,
      fontWeight: "700",
      marginBottom: 4,
    },
    footer: {
      alignItems: "center",
      marginTop: 28,
    },
    footerText: {
      color: theme.colors.onSurfaceFaint,
      fontSize: 13,
    },
    headline: {
      color: theme.colors.onBackground,
      fontSize: 30,
      fontWeight: "800",
      letterSpacing: -0.5,
      lineHeight: 36,
      marginBottom: 14,
      textAlign: "center",
    },
    hero: {
      alignItems: "center",
      paddingBottom: 24,
      paddingTop: 16,
    },
    logoCircle: {
      alignItems: "center",
      backgroundColor: theme.colors.primary,
      borderRadius: 18,
      elevation: 8,
      height: 64,
      justifyContent: "center",
      marginBottom: 16,
      shadowColor: theme.colors.primary,
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: 0.35,
      shadowRadius: 12,
      width: 64,
    },
    logoLetter: {
      color: "#ffffff",
      fontSize: 30,
      fontWeight: "900",
    },
    outlineBtn: {
      borderColor: theme.colors.primary,
      borderRadius: 8,
    },
    primaryBtn: {
      borderRadius: 8,
    },
    safe: {
      backgroundColor: theme.colors.background,
      flex: 1,
    },
    scroll: {
      padding: 24,
      paddingBottom: 40,
    },
    section: {
      marginTop: 16,
    },
    sectionTitle: {
      color: theme.colors.onBackground,
      fontSize: 22,
      fontWeight: "800",
      letterSpacing: -0.3,
      marginBottom: 16,
    },
    statItem: {
      alignItems: "center",
      flex: 1,
    },
    statLabel: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 12,
      marginTop: 2,
    },
    statValue: {
      color: theme.colors.onBackground,
      fontSize: 20,
      fontVariant: ["tabular-nums"],
      fontWeight: "800",
    },
    statsRow: {
      alignSelf: "stretch",
      borderTopColor: theme.colors.outlineVariant,
      borderTopWidth: 1,
      flexDirection: "row",
      justifyContent: "space-between",
      marginTop: 28,
      paddingTop: 18,
    },
    sub: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 15,
      lineHeight: 22,
      marginBottom: 24,
      textAlign: "center",
    },
    topBar: {
      alignItems: "flex-end",
      paddingHorizontal: 8,
      paddingTop: 4,
    },
  });
