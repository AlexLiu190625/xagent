import { describe, expect, it } from "vitest"

import { translations } from "./translations"

const runtimeKeys = [
  "title",
  "delegatedAuthorization",
  "addInput",
  "addBinding",
  "noInputs",
  "noBindings",
  "boundHeaders",
  "boundBodyFields",
] as const

describe("translations", () => {
  it("exposes MCP runtime labels at the component namespace", () => {
    for (const locale of [translations.en, translations.zh]) {
      for (const key of runtimeKeys) {
        expect(locale.tools.mcp.runtime[key]).toBeTruthy()
        expect(locale.tools.mcp.runtime[key]).not.toBe(`tools.mcp.runtime.${key}`)
      }
    }
  })
})
