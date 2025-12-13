import React from "react";
import { StyleSheet, ScrollView, View, Alert } from "react-native";
import {
  Headline,
  Paragraph,
  List,
  Switch,
  RadioButton,
  Text,
  Button,
  useTheme,
} from "react-native-paper";
import { useDispatch, useSelector } from "react-redux";
import { setTheme, setNotifications } from "../store/slices/settingsSlice";
import { logoutUser } from "../store/slices/authSlice";

export default function SettingsScreen() {
  const theme = useTheme();
  const dispatch = useDispatch();

  const settings = useSelector((state) => state.settings);
  const { user } = useSelector((state) => state.auth);

  const handleThemeChange = (newTheme) => {
    dispatch(setTheme(newTheme));
  };

  const handleNotificationToggle = (key) => {
    dispatch(setNotifications({ [key]: !settings.notifications[key] }));
  };

  const handleLogout = () => {
    Alert.alert("Logout", "Are you sure you want to logout?", [
      {
        text: "Cancel",
        style: "cancel",
      },
      {
        text: "Logout",
        onPress: () => dispatch(logoutUser()),
        style: "destructive",
      },
    ]);
  };

  return (
    <ScrollView
      contentContainerStyle={[styles.container, { backgroundColor: theme.colors.background }]}
    >
      <Headline style={styles.title}>
        <Text>Settings</Text>
      </Headline>
      <Paragraph style={styles.paragraph}>
        <Text>Customize your app experience.</Text>
      </Paragraph>

      {user && (
        <List.Section>
          <List.Subheader>
            <Text>Account</Text>
          </List.Subheader>
          <List.Item
            title={user.name || "User"}
            description={user.email}
            left={(props) => <List.Icon {...props} icon="account" />}
          />
        </List.Section>
      )}

      <List.Section>
        <List.Subheader>
          <Text>Appearance</Text>
        </List.Subheader>
        <View style={styles.radioGroup}>
          <Text style={styles.radioLabel}>Theme</Text>
          <RadioButton.Group onValueChange={handleThemeChange} value={settings.theme}>
            <View style={styles.radioButtonItem}>
              <RadioButton value="light" />
              <Text>Light</Text>
            </View>
            <View style={styles.radioButtonItem}>
              <RadioButton value="dark" />
              <Text>Dark</Text>
            </View>
            <View style={styles.radioButtonItem}>
              <RadioButton value="system" />
              <Text>System Default</Text>
            </View>
          </RadioButton.Group>
        </View>
      </List.Section>

      <List.Section>
        <List.Subheader>
          <Text>Notifications</Text>
        </List.Subheader>
        <List.Item
          title="Enable Trade Alerts"
          right={() => (
            <Switch
              value={settings.notifications.tradeAlerts}
              onValueChange={() => handleNotificationToggle("tradeAlerts")}
            />
          )}
        />
        <List.Item
          title="Enable Research Updates"
          right={() => (
            <Switch
              value={settings.notifications.researchUpdates}
              onValueChange={() => handleNotificationToggle("researchUpdates")}
            />
          )}
        />
        <List.Item
          title="Enable Price Alerts"
          right={() => (
            <Switch
              value={settings.notifications.priceAlerts}
              onValueChange={() => handleNotificationToggle("priceAlerts")}
            />
          )}
        />
      </List.Section>

      <Button
        mode="contained"
        onPress={handleLogout}
        style={styles.logoutButton}
        buttonColor={theme.colors.error}
      >
        <Text style={{ color: "white" }}>Logout</Text>
      </Button>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
  },
  logoutButton: {
    marginTop: 32,
  },
  paragraph: {
    marginBottom: 24,
    textAlign: "center",
  },
  radioButtonItem: {
    alignItems: "center",
    flexDirection: "row",
    marginBottom: 4,
  },
  radioGroup: {
    marginBottom: 16,
    paddingHorizontal: 16,
  },
  radioLabel: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 8,
  },
  title: {
    marginBottom: 16,
    textAlign: "center",
  },
});
