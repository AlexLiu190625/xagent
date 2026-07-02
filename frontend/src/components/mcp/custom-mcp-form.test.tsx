import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CustomMcpForm, isHttpMcpOAuthTransport } from "./custom-mcp-form"
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

function deferredResponse() {
  let resolve!: (response: Response) => void
  const promise = new Promise<Response>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

async function flushPromises() {
  await Promise.resolve()
  await Promise.resolve()
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
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
    cleanup()
  })

  it("starts connect through the JSON authorization URL response", async () => {
    const popup = {
      closed: false,
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

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })

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

    const connectCall = apiRequestMock.mock.calls.find(([url]) =>
      String(url).endsWith("/oauth/connect")
    )
    expect(connectCall).toBeTruthy()
    const [, connectOptions] = connectCall as [string, RequestInit]
    const body = JSON.parse(String(connectOptions.body))
    expect(body).toHaveProperty("redirect_after")
    expect(body).not.toHaveProperty("resource")
    expect(body).not.toHaveProperty("issuer")
    expect(body).not.toHaveProperty("scope")
    expect(body).not.toHaveProperty("resource_metadata_url")
    expect(body).not.toHaveProperty("access_token")
    expect(body).not.toHaveProperty("refresh_token")
    expect(body).not.toHaveProperty("resource_owner_key")
    expect(openMock).toHaveBeenCalledWith("about:blank", "_blank")
    expect(popup.opener).toBeNull()
    expect(popup.location.href).toBe("https://auth.example.com/authorize")
  })

  it("treats websocket as an MCP OAuth-capable transport", () => {
    expect(isHttpMcpOAuthTransport("streamable_http")).toBe(true)
    expect(isHttpMcpOAuthTransport("sse")).toBe(true)
    expect(isHttpMcpOAuthTransport("websocket")).toBe(true)
    expect(isHttpMcpOAuthTransport("stdio")).toBe(false)
  })

  it("does not start OAuth connect when the authorization popup is blocked", async () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })
    apiRequestMock.mockClear()

    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))

    expect(toastErrorMock).toHaveBeenCalledWith("tools.mcp.dialog.oauthConnectFailed")
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("polls OAuth status until the authorization popup is closed", async () => {
    const onOAuthStatusChange = vi.fn()
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

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
    }

    render(
      <CustomMcpForm
        mcpFormData={formData}
        setMcpFormData={vi.fn()}
        serverId={42}
        onOAuthStatusChange={onOAuthStatusChange}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })
    expect(popup.location.href).toBe("https://auth.example.com/authorize")

    const statusCallsBeforePolling = apiRequestMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/oauth/status")
    ).length

    act(() => {
      vi.advanceTimersByTime(3000)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).endsWith("/oauth/status"))
        .length
    ).toBe(statusCallsBeforePolling + 1)
    expect(onOAuthStatusChange).not.toHaveBeenCalled()

    popup.closed = true
    act(() => {
      vi.advanceTimersByTime(3000)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(onOAuthStatusChange).toHaveBeenCalledTimes(1)
  })

  it("clears OAuth polling when the form unmounts", async () => {
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

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

    const { unmount } = renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })

    const statusCallsBeforeUnmount = apiRequestMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/oauth/status")
    ).length

    unmount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })

    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).endsWith("/oauth/status"))
        .length
    ).toBe(statusCallsBeforeUnmount)
  })

  it("ignores pending OAuth status responses after unmount", async () => {
    const pendingStatus = deferredResponse()
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return pendingStatus.promise
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { unmount } = renderMcpOAuthForm()
    unmount()

    await act(async () => {
      pendingStatus.resolve(okJson({ server_id: 42, grants: [] }))
      await flushPromises()
    })

    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(toastSuccessMock).not.toHaveBeenCalled()
  })

  it("does not reschedule polling when unmounted while a poll is awaiting status", async () => {
    const onOAuthStatusChange = vi.fn()
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const pendingPolledStatus = deferredResponse()
    let statusCalls = 0

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        statusCalls += 1
        if (statusCalls === 1) {
          return Promise.resolve(okJson({ server_id: 42, grants: [] }))
        }
        return pendingPolledStatus.promise
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

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
    }

    const { unmount } = render(
      <CustomMcpForm
        mcpFormData={formData}
        setMcpFormData={vi.fn()}
        serverId={42}
        onOAuthStatusChange={onOAuthStatusChange}
      />
    )

    await waitFor(() => {
      expect(statusCalls).toBe(1)
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })
    act(() => {
      vi.advanceTimersByTime(3000)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(statusCalls).toBe(2)

    unmount()
    await act(async () => {
      pendingPolledStatus.resolve(okJson({ server_id: 42, grants: [] }))
      await flushPromises()
      await vi.advanceTimersByTimeAsync(3000)
    })

    expect(statusCalls).toBe(2)
    expect(onOAuthStatusChange).not.toHaveBeenCalled()
  })

  it("restores asynchronously loaded masked OAuth client secrets on blur", async () => {
    const setMcpFormData = vi.fn()
    apiRequestMock.mockResolvedValue(okJson({ server_id: 42, grants: [] }))
    const baseFormData: MCPServerFormData = {
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
    }

    const { rerender } = render(
      <CustomMcpForm
        mcpFormData={baseFormData}
        setMcpFormData={setMcpFormData}
        serverId={42}
      />
    )

    const loadedFormData: MCPServerFormData = {
      ...baseFormData,
      config: {
        ...baseFormData.config!,
        auth: {
          ...baseFormData.config!.auth,
          client_secret: "********",
        },
      },
    }
    rerender(
      <CustomMcpForm
        mcpFormData={loadedFormData}
        setMcpFormData={setMcpFormData}
        serverId={42}
      />
    )

    await waitFor(() => {
      expect(screen.getByLabelText("tools.mcp.dialog.clientSecret")).toHaveValue(
        "********"
      )
    })

    const clearedFormData: MCPServerFormData = {
      ...loadedFormData,
      config: {
        ...loadedFormData.config!,
        auth: {
          ...loadedFormData.config!.auth,
          client_secret: "",
        },
      },
    }
    rerender(
      <CustomMcpForm
        mcpFormData={clearedFormData}
        setMcpFormData={setMcpFormData}
        serverId={42}
      />
    )

    fireEvent.blur(screen.getByLabelText("tools.mcp.dialog.clientSecret"))

    const updater = setMcpFormData.mock.calls.at(-1)?.[0]
    expect(typeof updater).toBe("function")
    const nextState = updater(clearedFormData)
    expect(nextState.config.auth.client_secret).toBe("********")
  })

  it("keeps an explicitly cleared masked OAuth client secret empty on blur", async () => {
    apiRequestMock.mockResolvedValue(okJson({ server_id: 42, grants: [] }))
    const loadedFormData: MCPServerFormData = {
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
          client_secret: "********",
        },
      },
    }

    function Harness() {
      const [formData, setFormData] = React.useState(loadedFormData)
      return (
        <CustomMcpForm
          mcpFormData={formData}
          setMcpFormData={setFormData}
          serverId={42}
        />
      )
    }

    render(<Harness />)
    const clientSecret = screen.getByLabelText("tools.mcp.dialog.clientSecret")

    fireEvent.focus(clientSecret)
    expect(clientSecret).toHaveValue("")
    fireEvent.change(clientSecret, {
      target: { value: "temporary-secret" },
    })
    expect(clientSecret).toHaveValue("temporary-secret")
    fireEvent.change(clientSecret, {
      target: { value: "" },
    })
    fireEvent.blur(clientSecret)

    expect(clientSecret).toHaveValue("")
  })
})
