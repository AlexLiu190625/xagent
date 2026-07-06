import React, { useState } from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CustomApiForm } from "./custom-api-form"
import type { MCPServerFormData } from "./custom-api-form"
import { RuntimeInputsForm } from "./runtime-inputs-form"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

afterEach(() => {
  cleanup()
})

function RuntimeFormHarness({
  connectorType = "mcp",
}: {
  connectorType?: "mcp" | "custom_api"
}) {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "records",
    transport: connectorType === "mcp" ? "streamable_http" : "custom_api",
    description: "",
    config: {},
  })

  return (
    <>
      <RuntimeInputsForm
        connectorType={connectorType}
        formData={formData}
        setFormData={setFormData}
      />
      <pre data-testid="form-state">{JSON.stringify(formData)}</pre>
    </>
  )
}

function formState(): MCPServerFormData {
  return JSON.parse(screen.getByTestId("form-state").textContent || "{}")
}

function CustomApiHarness() {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "records",
    transport: "custom_api",
    description: "",
    method: "POST",
    headers: {
      account_id: "static",
      "X-Static": "ok",
    },
    body: "{}",
    config: {},
    runtime_input_schema: {
      context: {
        account_id: { type: "string", required: false },
      },
    },
    runtime_bindings: [
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "headers", key: "account_id" },
      },
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "body_field", path: "account.id" },
      },
    ],
  })
  const [env, setEnv] = useState<{ key: string; value: string }[]>([])

  return (
    <CustomApiForm
      mcpFormData={formData}
      setMcpFormData={setFormData}
      customApiEnv={env}
      setCustomApiEnv={setEnv}
    />
  )
}

describe("RuntimeInputsForm", () => {
  it("writes MCP runtime input declarations and bindings to top-level form data", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    expect(formState().runtime_input_schema).toEqual({
      context: {
        account_id: { type: "string", required: false },
      },
    })

    fireEvent.click(screen.getByText("tools.mcp.runtime.addBinding"))
    expect(formState().runtime_bindings).toEqual([
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "mcp_meta", key: "account_id" },
      },
    ])
  })

  it("writes Custom API bindings and delegated authorization to top-level form data", () => {
    render(<RuntimeFormHarness connectorType="custom_api" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    fireEvent.click(screen.getByText("tools.mcp.runtime.addBinding"))
    fireEvent.click(
      screen.getByRole("switch", {
        name: "tools.mcp.runtime.delegatedAuthorization",
      }),
    )

    expect(formState().runtime_bindings).toEqual([
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "headers", key: "account_id" },
      },
    ])
    expect(formState().allow_delegated_authorization).toBe(true)
  })

  it("shows validation errors for authorization bindings without delegated authorization", () => {
    render(
      <RuntimeInputsForm
        connectorType="custom_api"
        formData={{
          name: "records",
          transport: "custom_api",
          description: "",
          config: {},
          runtime_input_schema: {
            secrets: {
              authorization: { type: "string", required: false },
            },
          },
          runtime_bindings: [
            {
              source: { input_type: "secrets", key: "authorization" },
              target: { target_type: "headers", key: "Authorization" },
            },
          ],
          allow_delegated_authorization: false,
        }}
        setFormData={vi.fn()}
      />,
    )

    expect(
      screen.getByText("tools.mcp.runtime.errors.authorizationRequiresDelegated"),
    ).toBeInTheDocument()
  })

  it("renders Custom API runtime-bound headers and body fields as read-only references", () => {
    render(<CustomApiHarness />)

    fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))

    expect(screen.getByText("tools.mcp.runtime.boundHeaders")).toBeInTheDocument()
    expect(screen.getByText("tools.mcp.runtime.boundBodyFields")).toBeInTheDocument()
    expect(
      screen.getAllByDisplayValue("account_id").some((item) => item.hasAttribute("disabled")),
    ).toBe(true)
    expect(screen.getAllByDisplayValue("$account_id").length).toBeGreaterThanOrEqual(2)
    expect(
      screen.getAllByDisplayValue("account.id").some((item) => item.hasAttribute("disabled")),
    ).toBe(true)
  })
})
