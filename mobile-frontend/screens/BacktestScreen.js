import { useEffect, useMemo, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import {
  ActivityIndicator,
  Button,
  Chip,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import { researchService } from "../services/researchService";

const formatCurrency = (n) =>
  `$${Number(n ?? 0).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

const pct = (n) =>
  `${Number(n ?? 0) >= 0 ? "+" : ""}${Number(n ?? 0).toFixed(2)}%`;

export default function BacktestScreen() {
  const theme = useTheme();
  const styles = useMemo(() => createStyles(theme), [theme]);

  const [strategies, setStrategies] = useState([]);
  const [strategyId, setStrategyId] = useState(null);
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [capital, setCapital] = useState("100000");

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    researchService
      .getStrategies()
      .then((list) => {
        if (!active) return;
        const arr = Array.isArray(list) ? list : [];
        setStrategies(arr);
        if (arr.length > 0) setStrategyId(arr[0].id);
      })
      .catch(() => active && setStrategies([]));
    return () => {
      active = false;
    };
  }, []);

  const runBacktest = async () => {
    if (!strategyId) {
      setError("Select a strategy to backtest.");
      return;
    }
    setError(null);
    setRunning(true);
    setResult(null);
    try {
      const res = await researchService.runBacktest(
        strategyId,
        startDate,
        endDate,
        Number(capital) || 0,
      );
      setResult(res);
    } catch {
      setError(
        "The backtest could not be run. Check the API connection and inputs.",
      );
    } finally {
      setRunning(false);
    }
  };

  const metrics = result
    ? [
        {
          label: "Total Return",
          value: pct(result.totalReturn),
          tone: result.totalReturn >= 0 ? "pos" : "neg",
        },
        {
          label: "Annualised",
          value: pct(result.annualisedReturn),
          tone: result.annualisedReturn >= 0 ? "pos" : "neg",
        },
        {
          label: "Sharpe",
          value: Number(result.sharpeRatio ?? 0).toFixed(2),
          tone: "neutral",
        },
        {
          label: "Sortino",
          value: Number(result.sortinoRatio ?? 0).toFixed(2),
          tone: "neutral",
        },
        {
          label: "Max Drawdown",
          value: `${(Number(result.maxDrawdown ?? 0) * 100).toFixed(1)}%`,
          tone: "neg",
        },
        {
          label: "Win Rate",
          value: `${(Number(result.winRate ?? 0) * 100).toFixed(1)}%`,
          tone: "neutral",
        },
        {
          label: "Profit Factor",
          value: Number(result.profitFactor ?? 0).toFixed(2),
          tone: "neutral",
        },
        {
          label: "Final Capital",
          value: formatCurrency(result.finalCapital),
          tone: "neutral",
        },
      ]
    : [];

  const toneColor = (tone) =>
    tone === "pos"
      ? theme.colors.success
      : tone === "neg"
        ? theme.colors.error
        : theme.colors.onSurface;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.hero}>
        <Text style={styles.title}>Backtest</Text>
        <Text style={styles.subtitle}>
          Simulate a strategy over a historical window
        </Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.label}>Strategy</Text>
        {strategies.length === 0 ? (
          <Text style={styles.hint}>
            No strategies available. Connect the API to load strategies.
          </Text>
        ) : (
          <View style={styles.chipRow}>
            {strategies.map((s) => (
              <Chip
                key={s.id}
                selected={s.id === strategyId}
                showSelectedOverlay
                onPress={() => setStrategyId(s.id)}
                style={styles.chip}
              >
                {s.name ?? s.id}
              </Chip>
            ))}
          </View>
        )}
      </View>

      <View style={styles.section}>
        <View style={styles.row}>
          <TextInput
            mode="outlined"
            label="Start date"
            value={startDate}
            onChangeText={setStartDate}
            placeholder="YYYY-MM-DD"
            style={styles.flexInput}
            autoCapitalize="none"
          />
          <TextInput
            mode="outlined"
            label="End date"
            value={endDate}
            onChangeText={setEndDate}
            placeholder="YYYY-MM-DD"
            style={styles.flexInput}
            autoCapitalize="none"
          />
        </View>
        <TextInput
          mode="outlined"
          label="Initial capital"
          value={capital}
          onChangeText={setCapital}
          keyboardType="numeric"
          left={<TextInput.Affix text="$" />}
          style={styles.input}
        />
        <Button
          mode="contained"
          onPress={runBacktest}
          disabled={running}
          style={styles.runButton}
          contentStyle={styles.runButtonContent}
        >
          {running ? "Running..." : "Run backtest"}
        </Button>
        {!!error && <Text style={styles.error}>{error}</Text>}
      </View>

      {running && (
        <View style={styles.loading}>
          <ActivityIndicator color={theme.colors.primary} />
          <Text style={styles.hint}>Simulating trades...</Text>
        </View>
      )}

      {result && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Results</Text>
          <View style={styles.metricGrid}>
            {metrics.map((m) => (
              <View key={m.label} style={styles.metricCard}>
                <Text style={styles.metricLabel}>{m.label}</Text>
                <Text
                  style={[styles.metricValue, { color: toneColor(m.tone) }]}
                >
                  {m.value}
                </Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const createStyles = (theme) =>
  StyleSheet.create({
    chip: {
      marginBottom: 4,
    },
    chipRow: {
      flexDirection: "row",
      flexWrap: "wrap",
      gap: 8,
    },
    container: {
      backgroundColor: theme.colors.background,
      flexGrow: 1,
      paddingBottom: 32,
    },
    error: {
      color: theme.colors.error,
      fontSize: 13,
      marginTop: 12,
    },
    flexInput: {
      backgroundColor: theme.colors.surface,
      flex: 1,
      marginBottom: 12,
    },
    hero: {
      backgroundColor: theme.colors.surface,
      borderBottomColor: theme.colors.outlineVariant,
      borderBottomWidth: 1,
      paddingBottom: 18,
      paddingHorizontal: 16,
      paddingTop: 24,
    },
    hint: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 13,
    },
    input: {
      backgroundColor: theme.colors.surface,
      marginBottom: 16,
    },
    label: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 12,
      fontWeight: "700",
      letterSpacing: 0.5,
      marginBottom: 10,
      textTransform: "uppercase",
    },
    loading: {
      alignItems: "center",
      gap: 10,
      paddingVertical: 24,
    },
    metricCard: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      marginBottom: 12,
      padding: 16,
      width: "48.5%",
    },
    metricGrid: {
      flexDirection: "row",
      flexWrap: "wrap",
      justifyContent: "space-between",
    },
    metricLabel: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 11,
      fontWeight: "600",
      letterSpacing: 0.5,
      marginBottom: 8,
      textTransform: "uppercase",
    },
    metricValue: {
      fontSize: 19,
      fontVariant: ["tabular-nums"],
      fontWeight: "800",
    },
    row: {
      flexDirection: "row",
      gap: 12,
    },
    runButton: {
      borderRadius: 8,
    },
    runButtonContent: {
      paddingVertical: 6,
    },
    section: {
      paddingHorizontal: 16,
      paddingVertical: 16,
    },
    sectionTitle: {
      color: theme.colors.onBackground,
      fontSize: 18,
      fontWeight: "700",
      marginBottom: 12,
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
