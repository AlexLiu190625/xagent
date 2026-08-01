import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const homeExtensionRenderMock = vi.hoisted(() => vi.fn())

async function createHomeExtensionMock() {
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  const HomePageExtension = ReactModule.memo(() => {
    ReactModule.useState(null)
    homeExtensionRenderMock()
    return ReactModule.createElement("div", { "data-testid": "home-extension" })
  })
  return { HomePageExtension }
}

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    locale: "en",
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    setTaskId: vi.fn(),
    setPendingMessage: vi.fn(),
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({
    appName: "Xagent",
    whiteLogoPath: "/logo-white.png",
  }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    status: "idle",
    hasAsrModel: true,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/components/welcome-modal", () => ({
  WelcomeModal: () => null,
}))

vi.mock("@/lib/home-page-extension", createHomeExtensionMock)

import Home from "./page"

function jsonResponse(data: unknown) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

describe("Home", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    homeExtensionRenderMock.mockClear()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") {
        return Promise.resolve(jsonResponse({ tasks: [] }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  })

  afterEach(() => cleanup())

  it("renders a hook-bearing configured extension as one component", async () => {
    render(<Home />)

    expect(screen.getByRole("button", { name: "voiceInput.start" })).toBeInTheDocument()
    const extension = await screen.findByTestId("home-extension")
    expect(extension).toBeInTheDocument()
    expect(extension.parentElement).toHaveAttribute(
      "data-slot",
      "home-page-extension",
    )
    expect(extension.parentElement).toHaveClass("shrink-0")
    expect(screen.getAllByTestId("home-extension")).toHaveLength(1)
    expect(homeExtensionRenderMock).toHaveBeenCalled()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(2))
  })

  it("renders the shipped default extension in an inert canonical slot", async () => {
    vi.doUnmock("@/lib/home-page-extension")
    vi.resetModules()
    try {
      const { default: DefaultHome } = await import("./page")

      const { container } = render(<DefaultHome />)
      const slot = container.querySelector('[data-slot="home-page-extension"]')

      expect(slot).toBeInTheDocument()
      expect(slot).toHaveClass("shrink-0")
      expect(slot).toBeEmptyDOMElement()
      expect(
        Array.from(slot?.classList ?? []).some((className) =>
          /(?:^|:)(?:p[trblxy]?|m[trblxy]?|min-h)-/.test(className),
        ),
      ).toBe(false)
    } finally {
      vi.doMock("@/lib/home-page-extension", createHomeExtensionMock)
      vi.resetModules()
    }
  })
})
