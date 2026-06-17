import { useColorScheme } from "react-native";
import { IconButton } from "react-native-paper";
import { useDispatch, useSelector } from "react-redux";
import { setTheme } from "../store/slices/settingsSlice";

// A quick light/dark toggle. It reads the current effective theme (resolving
// "system" against the OS scheme) and flips to the opposite explicit theme.
export default function ThemeToggle({ size = 22, color }) {
  const dispatch = useDispatch();
  const preference = useSelector((state) => state.settings.theme);
  const systemScheme = useColorScheme();

  const effective =
    preference === "system"
      ? systemScheme === "dark"
        ? "dark"
        : "light"
      : preference;

  const isDark = effective === "dark";

  return (
    <IconButton
      icon={isDark ? "weather-sunny" : "weather-night"}
      size={size}
      iconColor={color}
      onPress={() => dispatch(setTheme(isDark ? "light" : "dark"))}
      accessibilityLabel={
        isDark ? "Switch to light theme" : "Switch to dark theme"
      }
    />
  );
}
