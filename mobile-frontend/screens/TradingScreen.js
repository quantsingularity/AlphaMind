import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshControl, ScrollView, StyleSheet, View } from "react-native";
import {
  ActivityIndicator,
  Button,
  Chip,
  SegmentedButtons,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import { tradingService } from "../services/tradingService";

const formatCurrency = (n) =>
  `$${Number(n ?? 0).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

const ORDER_TYPES = ["MARKET", "LIMIT", "STOP"];

export default function TradingScreen() {
  const theme = useTheme();
  const styles = useMemo(() => createStyles(theme), [theme]);

  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const [ticker, setTicker] = useState("");
  const [side, setSide] = useState("BUY");
  const [quantity, setQuantity] = useState("");
  const [orderType, setOrderType] = useState("MARKET");
  const [price, setPrice] = useState("");

  const needsPrice = orderType !== "MARKET";

  const loadOrders = useCallback(async () => {
    try {
      const data = await tradingService.getOrders();
      setOrders(Array.isArray(data) ? data : []);
    } catch {
      setOrders([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOrders();
  }, [loadOrders]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadOrders();
    setRefreshing(false);
  }, [loadOrders]);

  const submit = async () => {
    setError(null);
    const qty = Number(quantity);
    if (!ticker.trim()) return setError("Enter a ticker.");
    if (!qty || qty <= 0)
      return setError("Enter a quantity greater than zero.");
    if (needsPrice && (!Number(price) || Number(price) <= 0)) {
      return setError("Enter a price for limit and stop orders.");
    }
    setSubmitting(true);
    try {
      await tradingService.createOrder({
        ticker: ticker.trim().toUpperCase(),
        side,
        quantity: qty,
        orderType,
        price: needsPrice ? Number(price) : undefined,
      });
      setTicker("");
      setQuantity("");
      setPrice("");
      await loadOrders();
    } catch {
      setError("The order could not be submitted. Check the API connection.");
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async (id) => {
    try {
      await tradingService.cancelOrder(id);
      await loadOrders();
    } catch {
      setError("The order could not be cancelled.");
    }
  };

  const statusColor = (status) => {
    if (status === "filled") return theme.colors.success;
    if (status === "cancelled" || status === "rejected")
      return theme.colors.error;
    if (status === "pending") return theme.colors.warning;
    return theme.colors.onSurfaceVariant;
  };

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
        <Text style={styles.title}>Trading</Text>
        <Text style={styles.subtitle}>
          Submit orders and monitor the blotter
        </Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Order ticket</Text>
        <View style={styles.card}>
          <TextInput
            mode="outlined"
            label="Ticker"
            value={ticker}
            onChangeText={setTicker}
            autoCapitalize="characters"
            placeholder="AAPL"
            style={styles.input}
          />
          <SegmentedButtons
            value={side}
            onValueChange={setSide}
            style={styles.segment}
            buttons={[
              { value: "BUY", label: "Buy" },
              { value: "SELL", label: "Sell" },
            ]}
          />
          <TextInput
            mode="outlined"
            label="Quantity"
            value={quantity}
            onChangeText={setQuantity}
            keyboardType="numeric"
            placeholder="100"
            style={styles.input}
          />
          <View style={styles.typeRow}>
            {ORDER_TYPES.map((t) => (
              <Chip
                key={t}
                selected={orderType === t}
                showSelectedOverlay
                onPress={() => setOrderType(t)}
                style={styles.chip}
              >
                {t}
              </Chip>
            ))}
          </View>
          {needsPrice && (
            <TextInput
              mode="outlined"
              label={orderType === "LIMIT" ? "Limit price" : "Stop price"}
              value={price}
              onChangeText={setPrice}
              keyboardType="numeric"
              left={<TextInput.Affix text="$" />}
              style={styles.input}
            />
          )}
          {!!error && <Text style={styles.error}>{error}</Text>}
          <Button
            mode="contained"
            onPress={submit}
            disabled={submitting}
            style={styles.submit}
            contentStyle={styles.submitContent}
            buttonColor={
              side === "SELL" ? theme.colors.error : theme.colors.primary
            }
          >
            {submitting
              ? "Submitting..."
              : `Submit ${side.toLowerCase()} order`}
          </Button>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Order blotter</Text>
        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color={theme.colors.primary} />
          </View>
        ) : orders.length === 0 ? (
          <View style={styles.card}>
            <Text style={styles.hint}>
              No orders yet. Submitted orders appear here.
            </Text>
          </View>
        ) : (
          <View style={styles.list}>
            {orders.map((o) => (
              <View key={o.id} style={styles.orderCard}>
                <View style={styles.orderTop}>
                  <View style={styles.orderLeft}>
                    <Text
                      style={[
                        styles.orderSide,
                        {
                          color:
                            o.side === "BUY"
                              ? theme.colors.success
                              : theme.colors.error,
                        },
                      ]}
                    >
                      {o.side}
                    </Text>
                    <Text style={styles.orderTicker}>{o.ticker}</Text>
                  </View>
                  <Text
                    style={[
                      styles.orderStatus,
                      { color: statusColor(o.status) },
                    ]}
                  >
                    {o.status}
                  </Text>
                </View>
                <View style={styles.orderMeta}>
                  <Text style={styles.metaText}>
                    {o.quantity} @ {o.orderType}
                    {o.price ? ` ${formatCurrency(o.price)}` : ""}
                  </Text>
                  {o.status === "pending" && (
                    <Text style={styles.cancel} onPress={() => cancel(o.id)}>
                      Cancel
                    </Text>
                  )}
                </View>
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const createStyles = (theme) =>
  StyleSheet.create({
    cancel: {
      color: theme.colors.error,
      fontSize: 13,
      fontWeight: "600",
    },
    card: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      padding: 16,
    },
    chip: {
      marginBottom: 4,
    },
    container: {
      backgroundColor: theme.colors.background,
      flexGrow: 1,
      paddingBottom: 32,
    },
    error: {
      color: theme.colors.error,
      fontSize: 13,
      marginBottom: 8,
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
    input: {
      backgroundColor: theme.colors.surface,
      marginBottom: 12,
    },
    list: {
      gap: 10,
    },
    loading: {
      paddingVertical: 24,
    },
    metaText: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 13,
      fontVariant: ["tabular-nums"],
    },
    orderCard: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      padding: 14,
    },
    orderLeft: {
      alignItems: "center",
      flexDirection: "row",
      gap: 8,
    },
    orderMeta: {
      alignItems: "center",
      flexDirection: "row",
      justifyContent: "space-between",
    },
    orderSide: {
      fontSize: 13,
      fontWeight: "800",
    },
    orderStatus: {
      fontSize: 12,
      fontWeight: "700",
      textTransform: "uppercase",
    },
    orderTicker: {
      color: theme.colors.onSurface,
      fontSize: 15,
      fontWeight: "700",
    },
    orderTop: {
      alignItems: "center",
      flexDirection: "row",
      justifyContent: "space-between",
      marginBottom: 6,
    },
    section: {
      paddingHorizontal: 16,
      paddingVertical: 12,
    },
    sectionTitle: {
      color: theme.colors.onBackground,
      fontSize: 18,
      fontWeight: "700",
      marginBottom: 12,
    },
    segment: {
      marginBottom: 12,
    },
    submit: {
      borderRadius: 8,
      marginTop: 4,
    },
    submitContent: {
      paddingVertical: 6,
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
    typeRow: {
      flexDirection: "row",
      gap: 8,
      marginBottom: 12,
    },
  });
