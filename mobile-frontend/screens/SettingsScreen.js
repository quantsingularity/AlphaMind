import { Alert, ScrollView, StyleSheet, View } from "react-native";
import {
  Button,
  Divider,
  Headline,
  List,
  Paragraph,
  RadioButton,
  Switch,
  Text,
  useTheme,
} from "react-native-paper";
import { useDispatch, useSelector } from "react-redux";
import { logoutUser } from "../store/slices/authSlice";
import {
  resetSettings,
  setNotifications,
  setTheme,
} from "../store/slices/settingsSlice";

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

  const handleResetSettings = () => {
    Alert.alert(
      "Reset Settings",
      "Are you sure you want to reset all settings to defaults?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          onPress: () => dispatch(resetSettings()),
          style: "destructive",
        },
      ],
    );
  };

  return (
    <ScrollView
      contentContainerStyle={[
        styles.container,
        { backgroundColor: theme.colors.background },
      ]}
    >
      <Headline style={styles.title}>Settings</Headline>
      <Paragraph style={styles.paragraph}>
        Customize your app experience.
      </Paragraph>

      {user && (
        <>
          <List.Section>
            <List.Subheader>Account</List.Subheader>
            <List.Item
              title={user.name || "User"}
              description={user.email}
              left={(props) => <List.Icon {...props} icon="account-circle" />}
            />
          </List.Section>
          <Divider />
        </>
      )}

      <List.Section>
        <List.Subheader>Appearance</List.Subheader>
        <View style={styles.radioGroup}>
          <Text style={styles.radioLabel}>Theme</Text>
          <RadioButton.Group
            onValueChange={handleThemeChange}
            value={settings.theme}
          >
            <RadioButton.Item label="Light" value="light" />
            <RadioButton.Item label="Dark" value="dark" />
            <RadioButton.Item label="System Default" value="system" />
          </RadioButton.Group>
        </View>
      </List.Section>

      <Divider />

      <List.Section>
        <List.Subheader>Notifications</List.Subheader>
        <List.Item
          title="Trade Alerts"
          description="Get notified when trades are executed"
          right={() => (
            <Switch
              value={settings.notifications.tradeAlerts}
              onValueChange={() => handleNotificationToggle("tradeAlerts")}
            />
          )}
        />
        <List.Item
          title="Research Updates"
          description="Receive new research paper notifications"
          right={() => (
            <Switch
              value={settings.notifications.researchUpdates}
              onValueChange={() => handleNotificationToggle("researchUpdates")}
            />
          )}
        />
        <List.Item
          title="Price Alerts"
          description="Alert on significant price movements"
          right={() => (
            <Switch
              value={settings.notifications.priceAlerts}
              onValueChange={() => handleNotificationToggle("priceAlerts")}
            />
          )}
        />
      </List.Section>

      <Divider />

      <View style={styles.buttonGroup}>
        <Button
          mode="outlined"
          onPress={handleResetSettings}
          style={styles.resetButton}
        >
          Reset to Defaults
        </Button>

        <Button
          mode="contained"
          onPress={handleLogout}
          style={styles.logoutButton}
          buttonColor={theme.colors.error}
          textColor={theme.colors.onError}
        >
          Logout
        </Button>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  buttonGroup: {
    gap: 12,
    marginTop: 32,
    paddingHorizontal: 4,
  },
  container: {
    flexGrow: 1,
    padding: 20,
  },
  logoutButton: {
    marginBottom: 16,
  },
  paragraph: {
    marginBottom: 24,
    textAlign: "center",
  },
  radioGroup: {
    marginBottom: 8,
    paddingHorizontal: 16,
  },
  radioLabel: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 4,
    marginTop: 8,
  },
  resetButton: {},
  title: {
    marginBottom: 16,
    textAlign: "center",
  },
});
