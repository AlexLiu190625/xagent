import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CustomMcpForm } from "./custom-mcp-form"
import { MCPServerFormData } from "./custom-api-form"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => vi.fn((key: string) => key))

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

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: translateMock,
  }),
}))

function okJson(data: unknown): Response {
  return {
    ok: true,
    json: vi.fn().mockResolvedValue(data),
  } as unknown as Response
}

function renderMcpOAuthForm(overrides: Partial<MCPServerFormData> = {}) {
  const formData: MCPServerFormData = {
    name: "records",
    transport: "streamable_http",
    description: "",
    config: {
      url: "https://mcp.example.com/mcp",
      auth: {
        type: "mcp_oauth",
        resource: "https://mcp.example.com/mcp",
        issuer: "https://auth.example.com",
        scope: "records.read",
        client_id: "client-123",
      },
    },
    ...overrides,
  }

  return render(
    <CustomMcpForm
      mcpFormData={formData}
      setMcpFormData={vi.fn()}
      transports={[]}
      serverId={42}
    />
  )
}

describe("CustomMcpForm MCP OAuth", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    cleanup()
  })

  it("starts connect through the JSON authorization URL response", async () => {
    const popup = {
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    const openMock = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/connect",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Accept: "application/json",
            "Content-Type": "application/json",
          }),
        })
      )
    })

    const connectCall = apiRequestMock.mock.calls.find(([url]) =>
      String(url).endsWith("/oauth/connect")
    )
    expect(connectCall).toBeTruthy()
    const [, connectOptions] = connectCall as [string, RequestInit]
    const body = JSON.parse(String(connectOptions.body))
    expect(body).toMatchObject({
      resource: "https://mcp.example.com/mcp",
      issuer: "https://auth.example.com",
      scope: "records.read",
    })
    expect(body).not.toHaveProperty("access_token")
    expect(body).not.toHaveProperty("refresh_token")
    expect(body).not.toHaveProperty("resource_owner_key")
    expect(openMock).toHaveBeenCalledWith("about:blank", "_blank")
    expect(popup.opener).toBeNull()
    expect(popup.location.href).toBe("https://auth.example.com/authorize")
  })
})
