// FilmUpload — pick a film clip to run through the pipeline.
//
// A minimal upload affordance for the viewer: opens the system document picker
// filtered to video files and hands the chosen file back to the caller. Wiring
// the selected clip to a backend that runs `ml.cli` is app-specific and left to
// the host application.
import React from "react";
import { Pressable, Text, StyleSheet } from "react-native";
import * as DocumentPicker from "expo-document-picker";

import { palette } from "../theme/colors";

export type PickedFilm = {
  uri: string;
  name: string;
  size?: number | null;
  mimeType?: string | null;
};

type Props = {
  onPick: (film: PickedFilm) => void;
  label?: string;
};

export function FilmUpload({ onPick, label = "Upload film" }: Props) {
  const pick = React.useCallback(async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (result.canceled) return;
    const asset = result.assets[0];
    onPick({
      uri: asset.uri,
      name: asset.name,
      size: asset.size,
      mimeType: asset.mimeType,
    });
  }, [onPick]);

  return (
    <Pressable style={styles.button} onPress={pick} accessibilityRole="button">
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: palette.primary,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 10,
    alignItems: "center",
  },
  label: {
    color: palette.white,
    fontSize: 16,
    fontWeight: "600",
  },
});
