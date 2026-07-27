import { describe, expect, it } from "vitest"
import { isAuthPublicPath, isExternalRoutePath } from "./auth-pages"

describe("auth public paths", () => {
  it("allows the OIDC callback route through the auth guard", () => {
    expect(isAuthPublicPath("/auth/oidc/callback")).toBe(true)
  })

  it("classifies widget and share routes as external provider boundaries", () => {
    expect(isExternalRoutePath("/widget/chat/session")).toBe(true)
    expect(isExternalRoutePath("/share/public-token")).toBe(true)
    expect(isExternalRoutePath("/settings")).toBe(false)
    expect(isExternalRoutePath("/widgets")).toBe(false)
    expect(isExternalRoutePath("/widget-admin")).toBe(false)
    expect(isExternalRoutePath("/share-settings")).toBe(false)
  })
})
