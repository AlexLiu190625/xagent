import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const cardRenderMock = vi.hoisted(() => vi.fn())
const providerLifetime = vi.hoisted(() => ({ mounts: 0, unmounts: 0 }))

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
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => ({ get: () => null }),
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
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    dispatch: vi.fn(),
    setTaskId: vi.fn(),
    setPendingMessage: vi.fn(),
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    status: "idle",
    hasAsrModel: true,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/lib/build-page-extension", async () => {
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  const BuildPageExtensionProvider = ReactModule.memo(
    ({ children }: { children: React.ReactNode }) => {
      ReactModule.useEffect(() => {
        providerLifetime.mounts += 1
        return () => {
          providerLifetime.unmounts += 1
        }
      }, [])
      return ReactModule.createElement(
        "div",
        { "data-testid": "build-extension-provider" },
        children,
      )
    },
  )
  const BuildAgentCardExtension = ReactModule.memo(
    ({ agentId }: { agentId: number }) => {
      ReactModule.useState(null)
      cardRenderMock(agentId)
      return ReactModule.createElement("div", {
        "data-testid": `agent-card-supplement-${agentId}`,
      })
    },
  )
  return {
    BuildPageExtensionProvider,
    BuildAgentCardExtension,
  }
})

vi.mock("@/components/build/deploy-agent-dialog", () => ({
  DeployAgentDialog: () => null,
}))

vi.mock("@/components/build/agent-triggers-dialog", () => ({
  AgentTriggersDialog: () => null,
}))

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

import BuildsPage from "./page"

const agent = {
  id: 42,
  name: "Research Agent",
  description: "Researches launch topics",
  logo_url: null,
  status: "draft",
  created_at: "2026-07-01T00:00:00Z",
  updated_at: "2026-07-02T00:00:00Z",
  widget_enabled: false,
  allowed_domains: [],
  can_edit: true,
  can_publish: true,
  can_delete: true,
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe("BuildsPage extension boundaries", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    routerReplaceMock.mockReset()
    cardRenderMock.mockClear()
    providerLifetime.mounts = 0
    providerLifetime.unmounts = 0
  })

  afterEach(() => cleanup())

  it("renders a hook-bearing supplement for a real Agent card", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([agent]))

    render(<BuildsPage />)

    await screen.findByText("Research Agent")
    expect(screen.getByTestId("agent-card-supplement-42")).toBeInTheDocument()
    expect(cardRenderMock).toHaveBeenCalledWith(42)
  })

  it("keeps the page provider mounted across loading and a real publish refresh", async () => {
    const initialAgents = deferred<Response>()
    const refreshedAgents = deferred<Response>()
    let agentListRequests = 0

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        agentListRequests += 1
        return agentListRequests === 1
          ? initialAgents.promise
          : refreshedAgents.promise
      }
      if (
        url === "http://api.local/api/agents/42/publish" &&
        options?.method === "POST"
      ) {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const view = render(<BuildsPage />)
    expect(screen.getByTestId("build-extension-provider")).toBeInTheDocument()
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 0 })

    await act(async () => {
      initialAgents.resolve(jsonResponse([agent]))
      await initialAgents.promise
    })
    await screen.findByText("Research Agent")

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.actions.publish",
    }))
    await waitFor(() => expect(agentListRequests).toBe(2))
    expect(screen.getByText("common.loading")).toBeInTheDocument()
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 0 })

    await act(async () => {
      refreshedAgents.resolve(jsonResponse([]))
      await refreshedAgents.promise
    })
    await screen.findByText("builds.emptyState.title")
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 0 })

    view.unmount()
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 1 })
  })

  it("renders voice input in the create dialog", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([]))

    render(<BuildsPage />)
    await waitFor(() => {
      expect(screen.queryByText("common.loading")).not.toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.header.create",
    }))

    expect(screen.getByRole("button", {
      name: "voiceInput.start",
    })).toBeInTheDocument()
  })

  it("renders privileged actions for an editable Agent", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([agent]))

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    for (const name of [
      "builds.list.actions.apiKey",
      "builds.list.actions.triggers",
      "builds.list.actions.publish",
      "builds.list.actions.delete",
      "builds.list.actions.edit",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument()
    }
    expect(screen.queryByRole("button", {
      name: "builds.list.actions.viewConfig",
    })).not.toBeInTheDocument()
  })

  it("limits a published read-only Agent to run and view actions", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([{
      ...agent,
      status: "published",
      can_edit: false,
      can_publish: false,
      can_delete: false,
    }]))

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    expect(screen.getByRole("button", {
      name: "builds.list.actions.chat",
    })).toBeInTheDocument()
    expect(screen.getByRole("button", {
      name: "builds.list.actions.viewConfig",
    })).toBeInTheDocument()
    for (const name of [
      "builds.list.actions.apiKey",
      "builds.list.actions.triggers",
      "builds.list.actions.publish",
      "builds.list.actions.delete",
      "builds.list.actions.edit",
    ]) {
      expect(screen.queryByRole("button", { name })).not.toBeInTheDocument()
    }
  })
})
