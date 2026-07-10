import { describe, expect, it } from "vitest"

import { translations } from "./translations"

const runtimeLabelKeys = [
  "title",
  "delegatedAuthorization",
  "authorizationRequiresDelegated",
  "inputs",
  "bindings",
  "addInput",
  "addBinding",
  "noInputs",
  "noBindings",
  "noSourceKeys",
  "boundHeaders",
  "boundBodyFields",
  "toolArgumentsHidden",
  "key",
  "path",
  "required",
] as const

const runtimeInputTypes = [
  "context",
  "secrets",
  "auth_selector",
] as const

const runtimeValueTypes = [
  "string",
  "object",
] as const

const runtimeTargetTypes = [
  "mcp_meta",
  "tool_arguments",
  "transport_headers",
  "headers",
  "body_field",
] as const

const runtimeErrorKeys = [
  "sourceMissing",
  "sourceUnsupported",
  "targetUnsupported",
  "authSelectorBinding",
  "secretTarget",
  "contextTarget",
  "objectHeader",
  "duplicateInput",
  "targetMissing",
  "authorizationRequiresDelegated",
] as const

describe("translations", () => {
  it("exposes MCP runtime labels at the component namespace", () => {
    for (const locale of [translations.en, translations.zh]) {
      const runtime = locale.tools.mcp.runtime
      for (const key of runtimeLabelKeys) {
        expect(runtime[key]).toBeTruthy()
        expect(runtime[key]).not.toBe(`tools.mcp.runtime.${key}`)
      }
      for (const key of runtimeInputTypes) {
        expect(runtime.inputTypes[key]).toBeTruthy()
        expect(runtime.inputTypes[key]).not.toBe(`tools.mcp.runtime.inputTypes.${key}`)
      }
      for (const key of runtimeValueTypes) {
        expect(runtime.valueTypes[key]).toBeTruthy()
        expect(runtime.valueTypes[key]).not.toBe(`tools.mcp.runtime.valueTypes.${key}`)
      }
      for (const key of runtimeTargetTypes) {
        expect(runtime.targetTypes[key]).toBeTruthy()
        expect(runtime.targetTypes[key]).not.toBe(`tools.mcp.runtime.targetTypes.${key}`)
      }
      for (const key of runtimeErrorKeys) {
        expect(runtime.errors[key]).toBeTruthy()
        expect(runtime.errors[key]).not.toBe(`tools.mcp.runtime.errors.${key}`)
      }
    }
  })
})
