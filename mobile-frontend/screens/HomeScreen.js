import React, { useCallback, useEffect, useMemo } from "react";
import {
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Headline, Paragraph, useTheme } from "react-native-paper";
import { useDispatch, useSelector } from "react-redux";
import ErrorMessage from "../components/ErrorMessage";
import KPICard from "../components/KPICard";
import LoadingSpinner from "../components/LoadingSpinner";
import { fetchPortfolio } from "../store/slices/portfolioSlice";

export default function HomeScreen() {
  const theme = useTheme();
  const dispatch = useDispatch();

  const { data, loading, error } = useSelector((state) => state.portfolio);
  const { user } = useSelector((state) => state.auth);
  const [refreshing, setRefreshing] = React.useState(false);

  useEffect(() => {
    dispatch(fetchPortfolio());
  }, [dispatch]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await dispatch(fetchPortfolio());
    setRefreshing(false);
  }, [dispatch]);

  const kpiData = useMemo(() => {
    if (!data) {
      return [
        {
          title: "Portfolio Value",
          value: "$0.00",
          change: "0.0%",
          changeColor: "gray",
          icon: "chart-line",
        },
        {
          title: "Daily P&L",
          value: "$0.00",
          change: "0.0%",
          changeColor: "gray",
          icon: "trending-up",
        },
        {
          title: "Sharpe Ratio",
          value: "0.00",
          change: "",
          changeColor: "gray",
          icon: "chart-bell-curve-cumulative",
        },
        {
          title: "Active Strategies",
          value: "0",
          change: "",
          changeColor: "gray",
          icon: "robot",
        },
      ];
    }

    const portfolioValue = `$${(data.value ?? 0).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
    const dailyPnLNum = data.dailyPnL ?? 0;
    const dailyPnL = `${dailyPnLNum >= 0 ? "+" : ""}$${Math.abs(
      dailyPnLNum,
    ).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
    const dailyPnLPercent = data.dailyPnLPercent ?? 0;
    const dailyPnLChange = `${dailyPnLPercent >= 0 ? "+" : ""}${dailyPnLPercent.toFixed(2)}%`;
    const changeColor = dailyPnLPercent >= 0 ? "green" : "red";

    return [
      {
        title: "Portfolio Value",
        value: portfolioValue,
        change: dailyPnLChange,
        changeColor,
        icon: "chart-line",
      },
      {
        title: "Daily P&L",
        value: dailyPnL,
        change: dailyPnLChange,
        changeColor,
        icon: "trending-up",
      },
      {
        title: "Sharpe Ratio",
        value: (data.sharpeRatio ?? 0).toFixed(2),
        change: "",
        changeColor: theme.colors.primary,
        icon: "chart-bell-curve-cumulative",
      },
      {
        title: "Active Strategies",
        value: String(data.activeStrategies ?? 0),
        change: "",
        changeColor: theme.colors.primary,
        icon: "robot",
      },
    ];
  }, [data, theme.colors.primary]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: {
          backgroundColor: theme.colors.background,
          flexGrow: 1,
          padding: 16,
        },
        infoText: {
          color: theme.colors.outline,
          marginTop: 16,
          textAlign: "center",
        },
        kpiContainer: {
          flexDirection: "row",
          flexWrap: "wrap",
          justifyContent: "space-between",
          marginBottom: 24,
        },
        paragraph: {
          fontSize: 16,
          marginBottom: 24,
          textAlign: "center",
        },
        title: {
          marginBottom: 8,
          textAlign: "center",
        },
        welcome: {
          color: theme.colors.onBackground,
          fontSize: 14,
          marginBottom: 16,
          textAlign: "center",
        },
      }),
    [theme],
  );

  if (loading && !data) {
    return <LoadingSpinner message="Loading portfolio..." />;
  }

  if (error && !data) {
    return (
      <ErrorMessage
        message={error}
        onRetry={() => dispatch(fetchPortfolio())}
      />
    );
  }

  return (
    <ScrollView
      contentContainerStyle={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {user && (
        <Text style={styles.welcome}>
          Welcome back, {user.name || user.email}!
        </Text>
      )}

      <Headline style={styles.title}>AlphaMind Dashboard</Headline>
      <Paragraph style={styles.paragraph}>
        Real-time overview of your quantitative trading performance.
      </Paragraph>

      <View style={styles.kpiContainer}>
        {kpiData.map((kpi, index) => (
          <KPICard key={index} {...kpi} isLoading={loading} />
        ))}
      </View>

      <Paragraph style={styles.infoText}>
        Navigate using the bottom tabs to explore features, documentation, and
        research.
      </Paragraph>
    </ScrollView>
  );
}
