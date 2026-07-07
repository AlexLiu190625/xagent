import React, { useState } from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CustomApiForm } from "./custom-api-form"
import type { MCPServerFormData } from "./custom-api-form"
import { RuntimeInputsForm, getRuntimeConfigError } from "./runtime-inputs-form"

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
  onValidationErrorChange,
}: {
  connectorType?: "mcp" | "custom_api"
  onValidationErrorChange?: (error: string | null) => void
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
        onValidationErrorChange={onValidationErrorChange}
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

  it("keeps input focus while editing runtime keys", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInput = screen.getByPlaceholderText("tools.mcp.runtime.key")
    keyInput.focus()

    fireEvent.change(keyInput, { target: { value: "tenant_id" } })

    expect(document.activeElement).toBe(keyInput)
    expect(formState().runtime_input_schema).toEqual({
      context: {
        tenant_id: { type: "string", required: false },
      },
    })
  })

  it("keeps runtime input rows visible while a key is temporarily empty", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInput = screen.getByPlaceholderText("tools.mcp.runtime.key")
    keyInput.focus()

    fireEvent.change(keyInput, { target: { value: "" } })

    expect(document.activeElement).toBe(keyInput)
    expect(screen.getByPlaceholderText("tools.mcp.runtime.key")).toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.runtime.noInputs")).not.toBeInTheDocument()

    fireEvent.change(keyInput, { target: { value: "tenant_id" } })
    expect(formState().runtime_input_schema).toEqual({
      context: {
        tenant_id: { type: "string", required: false },
      },
    })
  })

  it("warns when runtime input keys are duplicated within an input type", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInputs = screen.getAllByPlaceholderText("tools.mcp.runtime.key")
    fireEvent.change(keyInputs[1], { target: { value: "account_id" } })

    expect(
      screen.getByText("tools.mcp.runtime.errors.duplicateInput"),
    ).toBeInTheDocument()
  })

  it("reports duplicate runtime input errors before schema folding hides them", async () => {
    const onValidationErrorChange = vi.fn()
    render(
      <RuntimeFormHarness
        connectorType="mcp"
        onValidationErrorChange={onValidationErrorChange}
      />,
    )

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInputs = screen.getAllByPlaceholderText("tools.mcp.runtime.key")
    fireEvent.change(keyInputs[1], { target: { value: "account_id" } })

    expect(formState().runtime_input_schema).toEqual({
      context: {
        account_id: { type: "string", required: false },
      },
    })
    await waitFor(() => {
      expect(onValidationErrorChange).toHaveBeenLastCalledWith(
        "tools.mcp.runtime.errors.duplicateInput",
      )
    })
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

  it("returns a save-blocking validation error for invalid runtime bindings", () => {
    expect(
      getRuntimeConfigError(
        {
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
        },
        "custom_api",
      ),
    ).toBe("tools.mcp.runtime.errors.authorizationRequiresDelegated")
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
