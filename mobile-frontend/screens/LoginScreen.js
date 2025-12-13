import React, { useState } from "react";
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from "react-native";
import { TextInput, Button, Text, Headline, useTheme, Snackbar } from "react-native-paper";
import { useDispatch, useSelector } from "react-redux";
import { loginUser, clearError } from "../store/slices/authSlice";

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const dispatch = useDispatch();
  const { loading, error } = useSelector((state) => state.auth);
  const theme = useTheme();

  const handleLogin = async () => {
    if (!email || !password) {
      return;
    }
    dispatch(loginUser({ email, password }));
  };

  const handleDismissError = () => {
    dispatch(clearError());
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.container}
    >
      <ScrollView
        contentContainerStyle={[styles.scrollContent, { backgroundColor: theme.colors.background }]}
      >
        <Headline style={styles.title}>Welcome to AlphaMind</Headline>
        <Text style={styles.subtitle}>Login to access your trading dashboard</Text>

        <TextInput
          label="Email"
          value={email}
          onChangeText={setEmail}
          mode="outlined"
          keyboardType="email-address"
          autoCapitalize="none"
          autoComplete="email"
          style={styles.input}
          left={<TextInput.Icon icon="email" />}
        />

        <TextInput
          label="Password"
          value={password}
          onChangeText={setPassword}
          mode="outlined"
          secureTextEntry={!showPassword}
          autoCapitalize="none"
          autoComplete="password"
          style={styles.input}
          left={<TextInput.Icon icon="lock" />}
          right={
            <TextInput.Icon
              icon={showPassword ? "eye-off" : "eye"}
              onPress={() => setShowPassword(!showPassword)}
            />
          }
        />

        <Button
          mode="contained"
          onPress={handleLogin}
          loading={loading}
          disabled={loading || !email || !password}
          style={styles.button}
        >
          Login
        </Button>

        <Button
          mode="text"
          onPress={() => navigation.navigate("Register")}
          style={styles.linkButton}
        >
          Don't have an account? Register
        </Button>

        <Snackbar
          visible={!!error}
          onDismiss={handleDismissError}
          duration={3000}
          action={{
            label: "Dismiss",
            onPress: handleDismissError,
          }}
        >
          {error}
        </Snackbar>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  button: {
    marginTop: 16,
  },
  container: {
    flex: 1,
  },
  input: {
    marginBottom: 16,
  },
  linkButton: {
    marginTop: 8,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    padding: 20,
  },
  subtitle: {
    marginBottom: 32,
    textAlign: "center",
  },
  title: {
    marginBottom: 8,
    textAlign: "center",
  },
});
