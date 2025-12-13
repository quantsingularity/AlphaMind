import React, { useState } from "react";
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from "react-native";
import { TextInput, Button, Text, Headline, useTheme, Snackbar } from "react-native-paper";
import { useDispatch, useSelector } from "react-redux";
import { registerUser, clearError } from "../store/slices/authSlice";

export default function RegisterScreen({ navigation }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [localError, setLocalError] = useState("");

  const dispatch = useDispatch();
  const { loading, error } = useSelector((state) => state.auth);
  const theme = useTheme();

  const handleRegister = async () => {
    // Validation
    if (!name || !email || !password || !confirmPassword) {
      setLocalError("All fields are required");
      return;
    }

    if (password !== confirmPassword) {
      setLocalError("Passwords do not match");
      return;
    }

    if (password.length < 8) {
      setLocalError("Password must be at least 8 characters");
      return;
    }

    setLocalError("");
    dispatch(registerUser({ name, email, password }));
  };

  const handleDismissError = () => {
    dispatch(clearError());
    setLocalError("");
  };

  const displayError = error || localError;

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.container}
    >
      <ScrollView
        contentContainerStyle={[styles.scrollContent, { backgroundColor: theme.colors.background }]}
      >
        <Headline style={styles.title}>Create Account</Headline>
        <Text style={styles.subtitle}>Join AlphaMind to start trading</Text>

        <TextInput
          label="Full Name"
          value={name}
          onChangeText={setName}
          mode="outlined"
          autoCapitalize="words"
          autoComplete="name"
          style={styles.input}
          left={<TextInput.Icon icon="account" />}
        />

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

        <TextInput
          label="Confirm Password"
          value={confirmPassword}
          onChangeText={setConfirmPassword}
          mode="outlined"
          secureTextEntry={!showPassword}
          autoCapitalize="none"
          style={styles.input}
          left={<TextInput.Icon icon="lock-check" />}
        />

        <Button
          mode="contained"
          onPress={handleRegister}
          loading={loading}
          disabled={loading || !name || !email || !password || !confirmPassword}
          style={styles.button}
        >
          Register
        </Button>

        <Button mode="text" onPress={() => navigation.navigate("Login")} style={styles.linkButton}>
          Already have an account? Login
        </Button>

        <Snackbar
          visible={!!displayError}
          onDismiss={handleDismissError}
          duration={3000}
          action={{
            label: "Dismiss",
            onPress: handleDismissError,
          }}
        >
          {displayError}
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
