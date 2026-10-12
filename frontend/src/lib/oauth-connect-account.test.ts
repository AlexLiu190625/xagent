import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))

import { openBuiltinOAuthPopup } from "./oauth-connect"

function fakePopup(): Window {
  return { closed: false, close: vi.fn() } as unknown as Window
}

async function settle(popup: Window, pending: Promise<unknown>) {
  ;(popup as { closed: boolean }).closed = true
  await vi.advanceTimersByTimeAsync(500)
  await pending
}

describe("openBuiltinOAuthPopup account parameter", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it("passes the account label to the login URL", async () => {
    const popup = fakePopup()
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup)

    const pending = openBuiltinOAuthPopup({ provider: "zendesk", appId: "zendesk", token: "t", account: "acme" })

    const url = new URL(windowOpenSpy.mock.calls[0][0] as string)
    expect(url.pathname).toBe("/api/auth/zendesk/login")
    expect(url.searchParams.get("account")).toBe("acme")
    expect(url.searchParams.get("app_id")).toBe("zendesk")
    expect(url.searchParams.get("redirect")).toBe(window.location.href)
    await settle(popup, pending)
  })

  it("passes the account on the bare batch login too", async () => {
    const popup = fakePopup()
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup)

    const pending = openBuiltinOAuthPopup({ provider: "zendesk", token: "t", account: "acme" })

    const url = new URL(windowOpenSpy.mock.calls[0][0] as string)
    expect(url.searchParams.get("account")).toBe("acme")
    expect(url.searchParams.has("app_id")).toBe(false)
    await settle(popup, pending)
  })

  it("leaves the login URL unchanged when no account is given", async () => {
    const popup = fakePopup()
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup)

    const pending = openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "t" })

    expect(windowOpenSpy.mock.calls[0][0]).toBe(
      `http://api.local/api/auth/google/login?token=t&app_id=gmail&redirect=${encodeURIComponent(window.location.href)}`,
    )
    await settle(popup, pending)
  })
})
