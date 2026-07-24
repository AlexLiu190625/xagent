import { defineConfig, mergeConfig } from "vitest/config"
import baseConfig from "./vitest.config"

const widgetConfig = mergeConfig(baseConfig, defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: [
        "public/widget.js",
        "src/components/widget/public-agent-chat-page.tsx",
      ],
      reporter: ["text", "json-summary"],
      reportsDirectory: "coverage/widget",
      thresholds: {
        perFile: true,
        statements: 1,
        branches: 1,
        functions: 1,
        lines: 1,
      },
    },
  },
}))

export default defineConfig({
  ...widgetConfig,
  test: {
    ...widgetConfig.test,
    // Vite concatenates arrays during merge, so replace the base glob after
    // merging to keep this command strictly targeted.
    include: [
      "src/components/widget/widget-bootstrap.test.ts",
      "src/components/widget/public-agent-chat-page.test.tsx",
    ],
  },
})
