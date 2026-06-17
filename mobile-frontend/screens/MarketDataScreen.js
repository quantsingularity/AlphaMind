import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshControl, ScrollView, StyleSheet, View } from "react-native";
import {
  ActivityIndicator,
  Text,
  TouchableRipple,
  useTheme,
} from "react-native-paper";
import { marketService } from "../services/marketService";

const formatCurrency = (n) =>
  `$${Number(n ?? 0).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

const formatNumber = (n) => Number(n ?? 0).toLocaleString("en-US");

export default function MarketDataScreen() {
  const theme = useTheme();
  const styles = useMemo(() => createStyles(theme), [theme]);

  const [quotes, setQuotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await marketService.getQuotes();
      setQuotes(Array.isArray(data) ? data : []);
      setError(null);
    } catch {
      setError("Could not load market data. Check the API connection.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  return (
    <ScrollView
      contentContainerStyle={styles.container}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          colors={[theme.colors.primary]}
          tintColor={theme.colors.primary}
        />
      }
    >
      <View style={styles.hero}>
        <Text style={styles.title}>Market Data</Text>
        <Text style={styles.subtitle}>
          Live quotes across the tracked universe
        </Text>
      </View>

      <View style={styles.section}>
        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color={theme.colors.primary} />
          </View>
        ) : error ? (
          <View style={styles.card}>
            <Text style={styles.error}>{error}</Text>
          </View>
        ) : quotes.length === 0 ? (
          <View style={styles.card}>
            <Text style={styles.hint}>No quotes available.</Text>
          </View>
        ) : (
          <View style={styles.table}>
            <View style={styles.tableHeader}>
              <Text style={[styles.th, styles.cTicker]}>Ticker</Text>
              <Text style={[styles.th, styles.cNum]}>Last</Text>
              <Text style={[styles.th, styles.cNum]}>Change</Text>
              <Text style={[styles.th, styles.cNum]}>Volume</Text>
            </View>
            {quotes.map((q, i) => {
              const change = q.open ? ((q.last - q.open) / q.open) * 100 : 0;
              const changeColor =
                change >= 0 ? theme.colors.success : theme.colors.error;
              return (
                <TouchableRipple
                  key={q.ticker ?? i}
                  onPress={() => {}}
                  rippleColor={theme.colors.primaryLight}
                >
                  <View
                    style={[
                      styles.tableRow,
                      i === quotes.length - 1 && styles.tableRowLast,
                    ]}
                  >
                    <Text style={[styles.tickerText, styles.cTicker]}>
                      {q.ticker}
                    </Text>
                    <Text style={[styles.td, styles.cNum]}>
                      {formatCurrency(q.last)}
                    </Text>
                    <Text
                      style={[styles.td, styles.cNum, { color: changeColor }]}
                    >
                      {change >= 0 ? "+" : ""}
                      {change.toFixed(2)}%
                    </Text>
                    <Text style={[styles.td, styles.cNum]}>
                      {formatNumber(q.volume)}
                    </Text>
                  </View>
                </TouchableRipple>
              );
            })}
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const createStyles = (theme) =>
  StyleSheet.create({
    cNum: { flex: 1, textAlign: "right" },
    cTicker: { flex: 1.4 },
    card: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      padding: 20,
    },
    container: {
      backgroundColor: theme.colors.background,
      flexGrow: 1,
      paddingBottom: 32,
    },
    error: {
      color: theme.colors.error,
      fontSize: 14,
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
      fontSize: 14,
    },
    loading: {
      paddingVertical: 32,
    },
    section: {
      padding: 16,
    },
    subtitle: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 14,
    },
    table: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      overflow: "hidden",
    },
    tableHeader: {
      backgroundColor: theme.colors.surfaceVariant,
      borderBottomColor: theme.colors.outlineVariant,
      borderBottomWidth: 1,
      flexDirection: "row",
      paddingHorizontal: 12,
      paddingVertical: 10,
    },
    tableRow: {
      alignItems: "center",
      borderBottomColor: theme.colors.outlineVariant,
      borderBottomWidth: 1,
      flexDirection: "row",
      paddingHorizontal: 12,
      paddingVertical: 13,
    },
    tableRowLast: {
      borderBottomWidth: 0,
    },
    td: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 13,
      fontVariant: ["tabular-nums"],
    },
    th: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 10,
      fontWeight: "700",
      letterSpacing: 0.5,
      textTransform: "uppercase",
    },
    tickerText: {
      color: theme.colors.onSurface,
      fontSize: 13,
      fontWeight: "700",
    },
    title: {
      color: theme.colors.onBackground,
      fontSize: 28,
      fontWeight: "800",
      letterSpacing: -0.5,
      marginBottom: 4,
    },
  });
