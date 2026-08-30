import { useEffect, useState } from "react";
import { Platform } from "react-native";

// On web, Skia needs CanvasKit (wasm) loaded before a <Canvas> can render.
// Native is always ready. If the wasm fetch fails (no signal), the field
// renderer falls back to text — text always works (spec §4.2).

let loaded = Platform.OS !== "web";
let failed = false;
let loading: Promise<void> | null = null;

export function useSkiaReady(): { ready: boolean; failed: boolean } {
  const [, force] = useState(0);

  useEffect(() => {
    if (loaded || failed) return;
    if (!loading) {
      loading = import("@shopify/react-native-skia/lib/module/web")
        .then(({ LoadSkiaWeb }) =>
          // canvaskit.wasm is copied into public/ (postinstall) so it ships
          // with the site instead of hitting a CDN at runtime.
          LoadSkiaWeb({ locateFile: (file: string) => `/${file}` })
        )
        .then(() => {
          loaded = true;
        })
        .catch(() => {
          failed = true;
        });
    }
    let mounted = true;
    loading.finally(() => {
      if (mounted) force((n) => n + 1);
    });
    return () => {
      mounted = false;
    };
  }, []);

  return { ready: loaded, failed };
}
