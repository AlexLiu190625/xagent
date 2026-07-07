/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const copyToClipboardMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/lib/browser-location", () => ({
  getBrowserLocationOrigin: () => "http://app.local",
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: copyToClipboardMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
  },
}))

import { AgentWidgetSettingsDialog } from "./agent-widget-settings-dialog"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

function renderDialog(
  props?: Partial<React.ComponentProps<typeof AgentWidgetSettingsDialog>>,
) {
  const onWidgetConfigUpdated = vi.fn()
  render(
    <AgentWidgetSettingsDialog
      agentId={42}
      agentName="Widget Agent"
      open
      onOpenChange={vi.fn()}
      widgetConfig={{
        widget_enabled: false,
        allowed_domains: ["example.com"],
      }}
      onWidgetConfigUpdated={onWidgetConfigUpdated}
      {...props}
    />,
  )
  return { onWidgetConfigUpdated }
}

describe("AgentWidgetSettingsDialog", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    copyToClipboardMock.mockReset()
    copyToClipboardMock.mockResolvedValue(true)
    apiRequestMock.mockImplementation((url: string, options?: { body?: string }) => {
      if (url.endsWith("/widget-key/rotate")) {
        return Promise.resolve(jsonResponse({
          agent_id: 42,
          widget_enabled: true,
          widget_key: "wk-rotated-key",
        }))
      }
      if (url.endsWith("/widget-key")) {
        return Promise.resolve(jsonResponse({
          agent_id: 42,
          widget_enabled: true,
          widget_key: "wk-test-key",
        }))
      }
      const updates = options?.body ? JSON.parse(options.body) : {}
      return Promise.resolve(jsonResponse({
        id: 42,
        widget_enabled: false,
        allowed_domains: ["example.com"],
        ...updates,
      }))
    })
  })

  function agentUpdateCalls() {
    return apiRequestMock.mock.calls.filter(
      ([, options]) => (options as { method?: string } | undefined)?.method === "PUT",
    )
  }

  afterEach(() => {
    cleanup()
  })

  it("updates widget_enabled through the agent update endpoint", async () => {
    const { onWidgetConfigUpdated } = renderDialog()

    expect(screen.getByRole("button", { name: "common.back" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("switch", { name: "appWidget.dialog.enabledLabel" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ widget_enabled: true }),
      })
    })
    expect(onWidgetConfigUpdated).toHaveBeenCalledWith(expect.objectContaining({
      widget_enabled: true,
    }))
    expect(toastSuccessMock).toHaveBeenCalledWith("appWidget.messages.updated")
  })

  it("adds a normalized allowed domain through the agent update endpoint", async () => {
    const { onWidgetConfigUpdated } = renderDialog()

    fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
      target: { value: "Docs.Example.com" },
    })
    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.addDomain" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allowed_domains: ["example.com", "docs.example.com"] }),
      })
    })
    expect(onWidgetConfigUpdated).toHaveBeenCalledWith(expect.objectContaining({
      allowed_domains: ["example.com", "docs.example.com"],
    }))
    await waitFor(() => {
      expect(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder")).toHaveValue("")
    })
  })

  it("rejects domains with a scheme, path, or wildcard and shows an inline error", () => {
    renderDialog()

    for (const invalidValue of ["https://example.com", "example.com/path", "*.example.com"]) {
      fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
        target: { value: invalidValue },
      })
      fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.addDomain" }))

      expect(agentUpdateCalls()).toHaveLength(0)
      expect(screen.getByText("appWidget.dialog.invalidDomain")).toBeInTheDocument()
      expect(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder")).toHaveValue(invalidValue)
    }
  })

  it("clears the inline domain error once the input changes", () => {
    renderDialog()

    fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
      target: { value: "https://example.com" },
    })
    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.addDomain" }))
    expect(screen.getByText("appWidget.dialog.invalidDomain")).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
      target: { value: "docs.example.com" },
    })
    expect(screen.queryByText("appWidget.dialog.invalidDomain")).not.toBeInTheDocument()
  })

  it("accepts the * wildcard entry", async () => {
    renderDialog()

    fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
      target: { value: "*" },
    })
    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.addDomain" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allowed_domains: ["example.com", "*"] }),
      })
    })
  })

  it("removes an allowed domain through the agent update endpoint", async () => {
    renderDialog()

    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.removeDomain" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allowed_domains: [] }),
      })
    })
  })

  it("ignores duplicate domain additions", () => {
    renderDialog()

    fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
      target: { value: "Example.com" },
    })
    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.addDomain" }))

    expect(agentUpdateCalls()).toHaveLength(0)
  })

  it("shows an error and keeps local state unchanged when updates fail", async () => {
    const { onWidgetConfigUpdated } = renderDialog()
    // Let the on-open widget-key fetch settle first so the failure below
    // lands on the domain update, not the key fetch.
    await waitFor(() => {
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    })
    apiRequestMock.mockImplementationOnce(() =>
      Promise.resolve(jsonResponse({ detail: "Nope" }, { status: 500 })),
    )

    fireEvent.change(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder"), {
      target: { value: "docs.example.com" },
    })
    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.addDomain" }))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalled()
    })
    expect(onWidgetConfigUpdated).not.toHaveBeenCalled()
    expect(screen.queryByText("docs.example.com")).not.toBeInTheDocument()
    // The typed value is kept so the user can retry without retyping.
    expect(screen.getByPlaceholderText("appWidget.dialog.domainPlaceholder")).toHaveValue("docs.example.com")
  })

  it("fetches and shows the widget key when opened", async () => {
    renderDialog()

    await waitFor(() => {
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    })
    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/widget-key")
  })

  it("copies the widget key", async () => {
    renderDialog()
    await waitFor(() => {
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.copyWidgetKey" }))

    await waitFor(() => {
      expect(copyToClipboardMock).toHaveBeenCalledWith("wk-test-key")
    })
    expect(toastSuccessMock).toHaveBeenCalledWith("common.copied")
  })

  it("rotates the widget key after the operator confirms", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    try {
      renderDialog()
      await waitFor(() => {
        expect(screen.getByText("wk-test-key")).toBeInTheDocument()
      })

      fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.rotateWidgetKey" }))

      await waitFor(() => {
        expect(apiRequestMock).toHaveBeenCalledWith(
          "http://api.local/api/agents/42/widget-key/rotate",
          { method: "POST" },
        )
      })
      expect(confirmSpy).toHaveBeenCalledWith("appWidget.dialog.rotateWidgetKeyConfirm")
      await waitFor(() => {
        expect(screen.getByText("wk-rotated-key")).toBeInTheDocument()
      })
      expect(toastSuccessMock).toHaveBeenCalledWith("appWidget.messages.widgetKeyRotated")
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("does not rotate the widget key when the operator declines", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false)
    try {
      renderDialog()
      await waitFor(() => {
        expect(screen.getByText("wk-test-key")).toBeInTheDocument()
      })

      fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.rotateWidgetKey" }))

      expect(apiRequestMock).not.toHaveBeenCalledWith(
        "http://api.local/api/agents/42/widget-key/rotate",
        { method: "POST" },
      )
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("copies the embed snippet with the widget key for the selected agent", async () => {
    renderDialog()
    await waitFor(() => {
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "appWidget.dialog.copySnippet" }))

    await waitFor(() => {
      expect(copyToClipboardMock).toHaveBeenCalledWith(expect.stringContaining("widget.js"))
    })
    expect(copyToClipboardMock).toHaveBeenCalledWith(expect.stringContaining('src="http://app.local/widget.js"'))
    expect(copyToClipboardMock).not.toHaveBeenCalledWith(expect.stringContaining("http://api.local/widget.js"))
    expect(copyToClipboardMock).toHaveBeenCalledWith(expect.stringContaining('data-widget-key="wk-test-key"'))
    expect(copyToClipboardMock).not.toHaveBeenCalledWith(expect.stringContaining("data-agent-id"))
    expect(toastSuccessMock).toHaveBeenCalledWith("common.copied")
  })
})
