import { act, renderHook } from "@testing-library/react"
import { StrictMode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  buildWidgetSessionWebSocketUrl,
  useWidgetSession,
} from "./use-widget-session"

const PARENT_ORIGIN = "https://embed.example"

function updateMessage(overrides: Record<string, unknown> = {}) {
  const now = Date.now()
  return {
    xagent: true,
    v: 1,
    type: "session_update",
    session_token: "st_session_token",
    session_token_expires_at: new Date(now + 15 * 60_000).toISOString(),
    absolute_expires_at: new Date(now + 30 * 60_000).toISOString(),
    agent: {
      id: 42,
      name: "Support Agent",
      description: "Helps with schedules",
      logo_url: "https://cdn.example/logo.png",
      suggested_prompts: ["Show my schedule"],
    },
    ...overrides,
  }
}

function dispatchFromParent(data: Record<string, unknown>, origin = PARENT_ORIGIN, source: MessageEventSource | null = window) {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", { data, origin, source }))
  })
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe("useWidgetSession", () => {
  it("announces credential-free readiness with the bootstrap target origin", () => {
    const postMessage = vi.spyOn(window, "postMessage")

    renderHook(() => useWidgetSession())

    expect(postMessage).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "ready" },
      "*",
    )
  })

  it("pins the first valid parent origin and rejects messages from another origin", () => {
    const { result } = renderHook(() => useWidgetSession())

    dispatchFromParent(updateMessage(), PARENT_ORIGIN, null)
    dispatchFromParent({ xagent: true, v: 2, type: "session_terminal", code: "ignored" })
    dispatchFromParent(updateMessage())
    dispatchFromParent(
      { xagent: true, v: 1, type: "session_terminal", code: "session_expired" },
      "https://other.example",
    )

    expect(result.current.status).toBe("active")
    expect(result.current.parentOrigin).toBe(PARENT_ORIGIN)
  })

  it("sends an exact-origin reconnect request once and removes the usable token", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => {
      result.current.requestReconnect("ws_closed")
      result.current.requestReconnect("ws_closed")
    })

    expect(result.current.status).toBe("refreshing")
    expect(result.current.session).toBeNull()
    expect(postMessage).toHaveBeenLastCalledWith(
      { xagent: true, v: 1, type: "reconnect_request", reason: "ws_closed" },
      PARENT_ORIGIN,
    )
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)
  })

  it("does nothing when a retained reconnect callback runs after unmount", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result, unmount } = renderHook(() => useWidgetSession(), { wrapper: StrictMode })
    dispatchFromParent(updateMessage())
    const requestReconnect = result.current.requestReconnect
    const snapshot = result.current

    unmount()
    postMessage.mockClear()
    act(() => requestReconnect("ws_closed"))

    expect(postMessage).not.toHaveBeenCalled()
    expect(result.current).toBe(snapshot)
  })

  it("rejects a whitespace-only token without normalizing a nonblank raw token", () => {
    const first = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_token: "   " }))

    expect(first.result.current.status).toBe("terminal")
    expect(first.result.current.session).toBeNull()
    first.unmount()

    const second = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_token: "  st.raw.value  " }))

    expect(second.result.current.status).toBe("active")
    expect(second.result.current.session?.token).toBe("  st.raw.value  ")
  })

  it.each([
    ["missing", undefined],
    ["non-array", "not-an-array"],
  ])("rejects %s suggested prompts", (_label, suggestedPrompts) => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      agent: {
        id: 42,
        name: "Support Agent",
        suggested_prompts: suggestedPrompts,
      },
    }))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("unexpected_error")
    expect(result.current.session).toBeNull()
  })

  it("requests one refresh instead of exposing a token with less than one minute remaining", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_token_expires_at: new Date(Date.now() + 59_000).toISOString() }))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.session).toBeNull()
    expect(postMessage).toHaveBeenLastCalledWith(
      { xagent: true, v: 1, type: "reconnect_request", reason: "token_expired" },
      PARENT_ORIGIN,
    )
  })

  it("fails terminal when absolute expiry is expired", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ absolute_expires_at: new Date(Date.now() - 1).toISOString() }))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("session_expired")
    expect(result.current.session).toBeNull()
  })

  it("fails terminal when absolute expiry precedes token expiry in a fresh hook", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 20 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
    }))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("session_expired")
    expect(result.current.session).toBeNull()
  })

  it("accepts a terminal first response and ignores later updates", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent({ xagent: true, v: 1, type: "session_terminal", code: "reconnect_invalid" })
    dispatchFromParent(updateMessage())

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("reconnect_invalid")
    expect(result.current.session).toBeNull()
  })

  it("clears active token, agent, and warning timer on terminal", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-26T00:00:00.000Z"))
    const clearTimeout = vi.spyOn(globalThis, "clearTimeout")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 11 * 60_000).toISOString(),
    }))

    expect(result.current.session?.agent.name).toBe("Support Agent")
    dispatchFromParent({ xagent: true, v: 1, type: "session_terminal", code: "session_expired" })
    act(() => vi.advanceTimersByTime(2 * 60_000))

    expect(result.current.status).toBe("terminal")
    expect(result.current.session).toBeNull()
    expect(result.current.isAbsoluteExpiryWarningVisible).toBe(false)
    expect(clearTimeout).toHaveBeenCalled()
  })

  it("allowlists the complete update and drops unknown fields", () => {
    const tokenExpiresAt = new Date(Date.now() + 15 * 60_000).toISOString()
    const absoluteExpiresAt = new Date(Date.now() + 30 * 60_000).toISOString()
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token: "st_exact",
      session_token_expires_at: tokenExpiresAt,
      absolute_expires_at: absoluteExpiresAt,
      ignored_root: "discard",
      agent: {
        id: 42,
        name: "Support Agent",
        description: "Helps with schedules",
        logo_url: "https://cdn.example/logo.png",
        suggested_prompts: ["Show my schedule"],
        ignored_agent: "discard",
      },
    }))

    expect(result.current.session).toEqual({
      token: "st_exact",
      tokenExpiresAt,
      absoluteExpiresAt,
      generation: 1,
      agent: {
        id: 42,
        name: "Support Agent",
        description: "Helps with schedules",
        logoUrl: "https://cdn.example/logo.png",
        suggestedPrompts: ["Show my schedule"],
      },
    })
  })

  it("accepts a token with exactly sixty seconds remaining", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-26T00:00:00.000Z"))
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 20 * 60_000).toISOString(),
    }))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.token).toBe("st_session_token")
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(0)
  })

  it("shows the expiry warning at the ten-minute boundary", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-26T00:00:00.000Z"))
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 11 * 60_000).toISOString(),
    }))

    expect(result.current.isAbsoluteExpiryWarningVisible).toBe(false)
    act(() => vi.advanceTimersByTime(60_000))
    expect(result.current.isAbsoluteExpiryWarningVisible).toBe(true)
  })

  it("removes its listener and expiry timer on unmount", () => {
    vi.useFakeTimers()
    const removeEventListener = vi.spyOn(window, "removeEventListener")
    const clearTimeout = vi.spyOn(globalThis, "clearTimeout")
    const { unmount } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 11 * 60_000).toISOString(),
    }))

    unmount()

    expect(removeEventListener).toHaveBeenCalledWith("message", expect.any(Function))
    expect(clearTimeout).toHaveBeenCalled()
  })

  it("derives the session endpoint from the iframe origin", () => {
    expect(buildWidgetSessionWebSocketUrl("https://chat.example")).toBe(
      "wss://chat.example/v1/external/chat/sessions/ws",
    )
    expect(buildWidgetSessionWebSocketUrl("http://localhost:3000")).toBe(
      "ws://localhost:3000/v1/external/chat/sessions/ws",
    )
  })
})
