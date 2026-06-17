import { ScrollView, StyleSheet, View } from "react-native";
import { Divider, Icon, Text, useTheme } from "react-native-paper";
import { TouchableRipple } from "react-native-paper";

const ITEMS = [
  {
    label: "Trading",
    description: "Place orders and view the blotter",
    icon: "swap-horizontal",
    route: "Trading",
  },
  {
    label: "Market Data",
    description: "Live quotes and price history",
    icon: "chart-line",
    route: "MarketData",
  },
  {
    label: "Backtest",
    description: "Run a strategy backtest",
    icon: "chart-timeline-variant",
    route: "Backtest",
  },
  {
    label: "Research",
    description: "Latest papers and applied research",
    icon: "flask-outline",
    route: "Research",
  },
  {
    label: "Documentation",
    description: "Platform reference and guides",
    icon: "file-document-outline",
    route: "Documentation",
  },
  {
    label: "About",
    description: "Mission, team, and technology",
    icon: "information-outline",
    route: "About",
  },
  {
    label: "Settings",
    description: "Appearance, notifications, and account",
    icon: "cog-outline",
    route: "Settings",
  },
];

export default function MoreScreen({ navigation }) {
  const theme = useTheme();
  const styles = createStyles(theme);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.hero}>
        <Text style={styles.title}>More</Text>
        <Text style={styles.subtitle}>Research, docs, and workspace</Text>
      </View>

      <View style={styles.section}>
        <View style={styles.card}>
          {ITEMS.map((item, idx) => (
            <View key={item.route}>
              {idx > 0 && <Divider style={styles.divider} />}
              <TouchableRipple
                onPress={() => navigation.navigate(item.route)}
                rippleColor={theme.colors.primaryLight}
              >
                <View style={styles.row}>
                  <View style={styles.iconWrap}>
                    <Icon
                      source={item.icon}
                      size={20}
                      color={theme.colors.primary}
                    />
                  </View>
                  <View style={styles.rowText}>
                    <Text style={styles.rowLabel}>{item.label}</Text>
                    <Text style={styles.rowDesc}>{item.description}</Text>
                  </View>
                  <Icon
                    source="chevron-right"
                    size={22}
                    color={theme.colors.onSurfaceFaint}
                  />
                </View>
              </TouchableRipple>
            </View>
          ))}
        </View>
      </View>
    </ScrollView>
  );
}

const createStyles = (theme) =>
  StyleSheet.create({
    card: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.outlineVariant,
      borderRadius: 12,
      borderWidth: 1,
      overflow: "hidden",
    },
    container: {
      backgroundColor: theme.colors.background,
      flexGrow: 1,
    },
    divider: {
      backgroundColor: theme.colors.outlineVariant,
    },
    hero: {
      backgroundColor: theme.colors.surface,
      borderBottomColor: theme.colors.outlineVariant,
      borderBottomWidth: 1,
      paddingBottom: 18,
      paddingHorizontal: 16,
      paddingTop: 24,
    },
    iconWrap: {
      alignItems: "center",
      backgroundColor: theme.colors.primaryLight,
      borderRadius: 10,
      height: 38,
      justifyContent: "center",
      width: 38,
    },
    row: {
      alignItems: "center",
      flexDirection: "row",
      gap: 14,
      paddingHorizontal: 16,
      paddingVertical: 16,
    },
    rowDesc: {
      color: theme.colors.onSurfaceVariant,
      fontSize: 12,
      marginTop: 2,
    },
    rowLabel: {
      color: theme.colors.onSurface,
      fontSize: 15,
      fontWeight: "700",
    },
    rowText: {
      flex: 1,
    },
    section: {
      padding: 16,
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
