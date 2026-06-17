import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Icon, useTheme } from "react-native-paper";
import ThemeToggle from "../components/ThemeToggle";
import AboutScreen from "../screens/AboutScreen";
import BacktestScreen from "../screens/BacktestScreen";
import MarketDataScreen from "../screens/MarketDataScreen";
import DocumentationScreen from "../screens/DocumentationScreen";
import FeaturesScreen from "../screens/FeaturesScreen";
import HomeScreen from "../screens/HomeScreen";
import MoreScreen from "../screens/MoreScreen";
import PortfolioScreen from "../screens/PortfolioScreen";
import ResearchScreen from "../screens/ResearchScreen";
import TradingScreen from "../screens/TradingScreen";
import RiskScreen from "../screens/RiskScreen";
import SettingsScreen from "../screens/SettingsScreen";

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function TabBarIcon({ name, color, size }) {
  return <Icon source={name} color={color} size={size} />;
}

function MainTabs() {
  const theme = useTheme();

  return (
    <Tab.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.onSurfaceVariant,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.outlineVariant,
        },
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarLabel: "Home",
          tabBarIcon: (props) => (
            <TabBarIcon name="view-dashboard-outline" {...props} />
          ),
        }}
      />
      <Tab.Screen
        name="Strategies"
        component={FeaturesScreen}
        options={{
          tabBarIcon: (props) => <TabBarIcon name="trending-up" {...props} />,
        }}
      />
      <Tab.Screen
        name="Portfolio"
        component={PortfolioScreen}
        options={{
          tabBarIcon: (props) => (
            <TabBarIcon name="briefcase-outline" {...props} />
          ),
        }}
      />
      <Tab.Screen
        name="Risk"
        component={RiskScreen}
        options={{
          tabBarIcon: (props) => (
            <TabBarIcon name="shield-alert-outline" {...props} />
          ),
        }}
      />
      <Tab.Screen
        name="More"
        component={MoreScreen}
        options={{
          tabBarIcon: (props) => (
            <TabBarIcon name="dots-horizontal" {...props} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  const theme = useTheme();

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: theme.colors.surface },
        headerTintColor: theme.colors.onSurface,
        headerTitleStyle: { fontWeight: "700" },
        contentStyle: { backgroundColor: theme.colors.background },
        headerRight: () => <ThemeToggle color={theme.colors.onSurface} />,
      }}
    >
      <Stack.Screen
        name="Tabs"
        component={MainTabs}
        options={{ headerShown: false }}
      />
      <Stack.Screen name="Trading" component={TradingScreen} />
      <Stack.Screen
        name="MarketData"
        component={MarketDataScreen}
        options={{ title: "Market Data" }}
      />
      <Stack.Screen name="Backtest" component={BacktestScreen} />
      <Stack.Screen name="Research" component={ResearchScreen} />
      <Stack.Screen name="Documentation" component={DocumentationScreen} />
      <Stack.Screen name="About" component={AboutScreen} />
      <Stack.Screen name="Settings" component={SettingsScreen} />
    </Stack.Navigator>
  );
}
