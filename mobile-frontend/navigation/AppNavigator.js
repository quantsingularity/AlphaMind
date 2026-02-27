import React, { useState } from "react";
import { BottomNavigation } from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

import HomeScreen from "../screens/HomeScreen";
import FeaturesScreen from "../screens/FeaturesScreen";
import DocumentationScreen from "../screens/DocumentationScreen";
import ResearchScreen from "../screens/ResearchScreen";
import SettingsScreen from "../screens/SettingsScreen"; // Import SettingsScreen

const HomeRoute = () => <HomeScreen />;
const FeaturesRoute = () => <FeaturesScreen />;
const DocsRoute = () => <DocumentationScreen />;
const ResearchRoute = () => <ResearchScreen />;
const SettingsRoute = () => <SettingsScreen />; // Define SettingsRoute

export default function AppNavigator() {
  const [index, setIndex] = useState(0);
  const [routes] = useState([
    {
      key: "home",
      title: "Home",
      focusedIcon: "home",
      unfocusedIcon: "home-outline",
    },
    {
      key: "features",
      title: "Features",
      focusedIcon: "star",
      unfocusedIcon: "star-outline",
    },
    {
      key: "docs",
      title: "Docs",
      focusedIcon: "file-document",
      unfocusedIcon: "file-document-outline",
    },
    {
      key: "research",
      title: "Research",
      focusedIcon: "flask",
      unfocusedIcon: "flask-outline",
    },
    {
      key: "settings",
      title: "Settings",
      focusedIcon: "cog",
      unfocusedIcon: "cog-outline",
    }, // Add settings route
  ]);

  const renderScene = BottomNavigation.SceneMap({
    home: HomeRoute,
    features: FeaturesRoute,
    docs: DocsRoute,
    research: ResearchRoute,
    settings: SettingsRoute, // Add settings scene
  });

  const renderIcon = ({ route, focused, color }) => {
    return (
      <Icon
        name={focused ? route.focusedIcon : route.unfocusedIcon}
        size={24}
        color={color}
      />
    );
  };

  return (
    <BottomNavigation
      navigationState={{ index, routes }}
      onIndexChange={setIndex}
      renderScene={renderScene}
      renderIcon={renderIcon}
      // Optional: Customize appearance
      // activeColor="#6200ee"
      // inactiveColor="#9e9e9e"
      // barStyle={{ backgroundColor: '#ffffff' }}
    />
  );
}
