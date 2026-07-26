import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AuthProvider, useAuth } from "@/contexts/auth-context"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, clearStoredAuth, writeAuthCache } from "@/lib/auth-cache"

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn(async () => new Response(null, { status: 404 })),
  refreshStoredAccessToken: vi.fn(),
}))

afterEach(cleanup)

function AuthProbe() {
  const { checkAuth, token } = useAuth()
  const [checkResult, setCheckResult] = React.useState("pending")
  return (
    <>
      <span data-testid="access-token">{token || "none"}</span>
      <span data-testid="check-result">{checkResult}</span>
      <button onClick={() => {
        void checkAuth().then(result => setCheckResult(String(result)))
      }}>
        Check auth
      </button>
    </>
  )
}

function AuthRefreshProbe() {
  const { refreshAccessToken, token } = useAuth()
  const [refreshResult, setRefreshResult] = React.useState("pending")
  return (
    <>
      <span data-testid="refresh-access-token">{token || "none"}</span>
      <span data-testid="refresh-result">{refreshResult}</span>
      <button onClick={() => {
        void refreshAccessToken().then(result => setRefreshResult(String(result)))
      }}>
        Refresh access token
      </button>
    </>
  )
}

describe("AuthProvider storage synchronization", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    vi.mocked(apiRequest).mockImplementation(
      async () => new Response(null, { status: 404 })
    )
  })

  it("ignores non-object auth cache payloads without a runtime error", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "access-token",
      "refresh-token",
      120,
      240
    )
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("access-token")
    })

    act(() => {
      window.dispatchEvent(new StorageEvent("storage", {
        key: AUTH_CACHE_KEY,
        newValue: "null",
      }))
    })

    expect(screen.getByTestId("access-token")).toHaveTextContent("access-token")
    expect(consoleError).not.toHaveBeenCalled()
  })

  it("updates auth state from a valid cross-tab cache payload", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
    })

    writeAuthCache(
      { id: "2", username: "bob", email: null, is_admin: false },
      "bob-access",
      "bob-refresh",
      120,
      240
    )
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", {
        key: AUTH_CACHE_KEY,
        newValue: localStorage.getItem(AUTH_CACHE_KEY),
      }))
    })

    expect(screen.getByTestId("access-token")).toHaveTextContent("bob-access")
  })

  it("keeps auth state when a 401 leaves the refresh cache intact", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    vi.mocked(apiRequest).mockImplementation(async (url) =>
      new Response(null, {
        status: String(url).endsWith("/api/auth/verify") ? 401 : 404,
        headers: { "Error-Type": "TokenExpired" },
      })
    )

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Check auth" }))

    await waitFor(() => {
      expect(screen.getByTestId("check-result")).toHaveTextContent("true")
    })
    expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
  })

  it("clears auth state when a 401 follows a rejected refresh", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    vi.mocked(apiRequest).mockImplementation(async (url) => {
      if (String(url).endsWith("/api/auth/verify")) {
        clearStoredAuth()
        return new Response(null, {
          status: 401,
          headers: { "Error-Type": "TokenExpired" },
        })
      }
      return new Response(null, { status: 404 })
    })

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Check auth" }))

    await waitFor(() => {
      expect(screen.getByTestId("check-result")).toHaveTextContent("false")
      expect(screen.getByTestId("access-token")).toHaveTextContent("none")
    })
  })

  it("does not log out a replacement session after an older refresh is rejected", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    let resolveRefresh!: (value: { accessToken: null; rejected: true }) => void
    const refreshPromise = new Promise<{ accessToken: null; rejected: true }>((resolve) => {
      resolveRefresh = resolve
    })
    vi.mocked(refreshStoredAccessToken).mockReturnValue(refreshPromise)

    render(
      <AuthProvider>
        <AuthRefreshProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))
    await waitFor(() => {
      expect(refreshStoredAccessToken).toHaveBeenCalledWith("alice-access", "1")
    })

    writeAuthCache(
      { id: "2", username: "bob", email: null, is_admin: false },
      "bob-access",
      "bob-refresh",
      120,
      240
    )
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", {
        key: AUTH_CACHE_KEY,
        newValue: localStorage.getItem(AUTH_CACHE_KEY),
      }))
    })
    expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("bob-access")

    await act(async () => {
      resolveRefresh({ accessToken: null, rejected: true })
      await refreshPromise
    })

    await waitFor(() => {
      expect(screen.getByTestId("refresh-result")).toHaveTextContent("false")
    })
    expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("bob-access")
    expect(JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "null")).toMatchObject({
      token: "bob-access",
      refreshToken: "bob-refresh",
      user: { id: "2" },
    })
  })

  it("logs out the current session after its own refresh is rejected", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    vi.mocked(refreshStoredAccessToken).mockResolvedValue({
      accessToken: null,
      rejected: true,
    })
    vi.spyOn(console, "error").mockImplementation(() => {})

    render(
      <AuthProvider>
        <AuthRefreshProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))

    await waitFor(() => {
      expect(screen.getByTestId("refresh-result")).toHaveTextContent("false")
      expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("none")
    })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })
})
