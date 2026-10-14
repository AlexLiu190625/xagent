import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it, vi } from "vitest"

import { DEAD_AFTER_MS } from "./use-websocket"

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({}),
}))

const widgetScript = readFileSync(resolve(process.cwd(), "public/widget.js"), "utf8")

const readWidgetConstant = (name: string): number => {
  const match = new RegExp(`var ${name} = (\\d+);`).exec(widgetScript)
  if (!match) throw new Error(`widget.js does not define ${name} as a number literal`)
  return Number(match[1])
}

describe("widget dead threshold mirrors the hook", () => {
  it("uses the hook's dead threshold and a 60 second stable window", () => {
    const widgetDeadAfter = readWidgetConstant("CLIENT_DEAD_AFTER_MS")
    const trafficSpan = readWidgetConstant("STABLE_TRAFFIC_SPAN_MS")

    expect(widgetDeadAfter).toBe(DEAD_AFTER_MS)
    expect(widgetDeadAfter + trafficSpan).toBe(60_000)
    expect(widgetScript).toContain(
      "var SESSION_STABLE_CONNECTION_MS = CLIENT_DEAD_AFTER_MS + STABLE_TRAFFIC_SPAN_MS;",
    )
  })
})
