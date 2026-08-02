import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const resolveTaskLlmSelectionMock = vi.hoisted(() => vi.fn())
const homeExtensionRenderMock = vi.hoisted(() => vi.fn())
const setPendingMessageMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

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

vi.mock("@/lib/models", () => ({
  resolveTaskLlmSelection: resolveTaskLlmSelectionMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock },
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock }),
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
    setTaskId: setTaskIdMock,
    setPendingMessage: setPendingMessageMock,
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

const successfulSelection = {
  kind: "success" as const,
  llmIds: ["general", null, null, null] as [string, null, null, null],
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
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

function input(): HTMLTextAreaElement {
  return screen.getByPlaceholderText("home.hero.searchPlaceholder")
}

function submitButton(): HTMLButtonElement {
  const button = input().parentElement?.querySelector("button:not([aria-label])")
  if (!(button instanceof HTMLButtonElement)) throw new Error("Home submit button not found")
  return button
}

function typePrompt(value: string) {
  fireEvent.input(input(), { target: { value } })
}

function submitWithEnter() {
  fireEvent.keyDown(input(), { key: "Enter" })
}

function taskCore(taskId = 7) {
  return {
    task_id: taskId,
    title: "created task",
    status: "running",
    created_at: "2026-01-01T00:00:00Z",
  }
}

describe("Home", () => {
  let consoleErrorMock: ReturnType<typeof vi.spyOn>

  function expectNoTaskCreatePublication() {
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(routerPushMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  }

  beforeEach(() => {
    apiRequestMock.mockReset()
    resolveTaskLlmSelectionMock.mockReset()
    homeExtensionRenderMock.mockClear()
    setPendingMessageMock.mockReset()
    setTaskIdMock.mockReset()
    routerPushMock.mockReset()
    toastErrorMock.mockReset()
    consoleErrorMock = vi.spyOn(console, "error").mockImplementation(() => undefined)
    resolveTaskLlmSelectionMock.mockResolvedValue(successfulSelection)
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

  afterEach(() => {
    consoleErrorMock.mockRestore()
    cleanup()
  })

  it("renders a hook-bearing configured extension as one component", async () => {
    render(<Home />)

    expect(screen.getByRole("button", { name: "voiceInput.start" })).toBeInTheDocument()
    const extension = await screen.findByTestId("home-extension")
    expect(extension).toBeInTheDocument()
    expect(extension.parentElement).toHaveAttribute("data-slot", "home-page-extension")
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
      expect(slot).toHaveClass("shrink-0", { exact: true })
      expect(slot).not.toHaveAttribute("style")
      expect(slot).toBeEmptyDOMElement()
    } finally {
      vi.doMock("@/lib/home-page-extension", createHomeExtensionMock)
      vi.resetModules()
    }
  })

  it("uses the shared resolver, real task body parser, and ordered successful commit", async () => {
    const events: string[] = []
    setPendingMessageMock.mockImplementation(() => events.push("pending"))
    setTaskIdMock.mockImplementation(() => {
      events.push("taskId")
      expect(input()).toHaveValue("  hello\n  world  ")
      expect(input().style.height).toBe("56px")
    })
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") {
        expect(options?.method).toBe("POST")
        expect(JSON.parse(String(options?.body))).toEqual({
          title: "hello world",
          description: "hello\n  world",
          llm_ids: ["general", null, null, null],
        })
        return Promise.resolve(jsonResponse(taskCore(9)))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<Home />)
    typePrompt("  hello\n  world  ")
    input().style.height = "56px"
    submitWithEnter()

    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(9))
    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)
    expect(events).toEqual(["pending", "taskId"])
    expect(setPendingMessageMock).toHaveBeenCalledWith({
      message: "hello\n  world",
      files: [],
      targetTaskId: 9,
    })
    expect(input().value).toBe("")
    expect(input().style.height).toBe("auto")
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("keeps no_model distinct from an operational resolver failure", async () => {
    resolveTaskLlmSelectionMock.mockResolvedValueOnce({ kind: "no_model" })
    render(<Home />)
    const noModelPrompt = "  no model draft  "
    typePrompt(noModelPrompt)
    input().style.height = "72px"
    const noModelStyle = input().getAttribute("style")
    submitWithEnter()
    expect(await screen.findByText("chatPage.input.noModelAlert")).toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(input().value).toBe(noModelPrompt)
    expect(input()).toHaveAttribute("style", noModelStyle)

    cleanup()
    resolveTaskLlmSelectionMock.mockResolvedValueOnce({
      kind: "operational_error",
      error: new Error("resolver failure"),
    })
    render(<Home />)
    const operationalErrorPrompt = "  operational error draft  "
    typePrompt(operationalErrorPrompt)
    input().style.height = "64px"
    const operationalErrorStyle = input().getAttribute("style")
    submitWithEnter()
    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
    expect(input().value).toBe(operationalErrorPrompt)
    expect(input()).toHaveAttribute("style", operationalErrorStyle)
  })

  it("uses the ref token as a same-act Enter/click latch", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    render(<Home />)
    typePrompt("prompt")

    await act(async () => {
      submitWithEnter()
      fireEvent.click(submitButton())
    })

    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
  })

  it("does not POST or publish effects after unmount during model resolution", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()

    view.unmount()
    await act(async () => selection.resolve(successfulSelection))

    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("silences a rejected resolver after Home unmount", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)

    view.unmount()
    await act(async () => selection.reject(new Error("resolver unavailable")))

    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expectNoTaskCreatePublication()
  })

  it("silences a rejected create transport after Home unmount", async () => {
    const response = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return response.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    view.unmount()
    await act(async () => response.reject(new Error("transport unavailable")))

    expectNoTaskCreatePublication()
  })

  it("silences parseApiResponse's empty-data fallback after Response.text() rejects post-unmount", async () => {
    const body = deferred<string>()
    const taskResponse = new Response("unreadable")
    const text = vi.fn(() => body.promise)
    Object.defineProperty(taskResponse, "text", { value: text })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(text).toHaveBeenCalledTimes(1))

    view.unmount()
    await act(async () => {
      body.reject(new Error("body unavailable"))
      await new Promise((resolve) => setTimeout(resolve, 20))
    })

    expectNoTaskCreatePublication()
  })

  it("does not parse a response that arrives after unmount", async () => {
    const response = deferred<Response>()
    const taskResponse = new Response(JSON.stringify(taskCore()))
    const text = vi.fn(taskResponse.text.bind(taskResponse))
    Object.defineProperty(taskResponse, "text", { value: text })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return response.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    view.unmount()
    await act(async () => response.resolve(taskResponse))

    expect(text).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("allows parsing to finish after unmount but publishes no effects", async () => {
    const body = deferred<string>()
    const taskResponse = new Response("")
    const text = vi.fn(() => body.promise)
    Object.defineProperty(taskResponse, "text", { value: text })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(text).toHaveBeenCalledTimes(1))

    view.unmount()
    await act(async () => body.resolve(JSON.stringify(taskCore())))

    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it.each([
    ["non-OK", jsonResponse(taskCore(), { status: 500 })],
    ["empty", new Response("")],
    ["malformed", new Response("{")],
    ["invalid core", jsonResponse({ id: 7 })],
  ])("keeps task actions at zero and reports current %s create failure", async (_name, taskResponse) => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("prompt")
    submitWithEnter()

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
  })

  it("reports a current unreadable task body as one operational failure", async () => {
    const taskResponse = new Response("unreadable")
    Object.defineProperty(taskResponse, "text", {
      value: vi.fn().mockRejectedValue(new Error("body unavailable")),
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("prompt")
    submitWithEnter()

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
  })

  it("never overwrites a newer B draft or its height after A succeeds or fails", async () => {
    const result = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return result.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typePrompt("B")
    input().style.height = "72px"

    await act(async () => result.resolve(jsonResponse(taskCore())))
    expect(input().value).toBe("B")
    expect(input().style.height).toBe("72px")

    cleanup()
    const failed = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return failed.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typePrompt("B")
    input().style.height = "72px"
    await act(async () => failed.resolve(jsonResponse(taskCore(), { status: 500 })))

    expect(input().value).toBe("B")
    expect(input().style.height).toBe("72px")
  })

  it("preserves A after an ABA edit even when the old attempt succeeds", async () => {
    const result = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return result.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typePrompt("B")
    typePrompt("A")
    input().style.height = "64px"

    await act(async () => result.resolve(jsonResponse(taskCore())))

    expect(input().value).toBe("A")
    expect(input().style.height).toBe("64px")
  })

  it("does not classify a synchronous commit collaborator throw as an operational create failure", async () => {
    setTaskIdMock.mockImplementation(() => { throw new Error("commit failed") })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(jsonResponse(taskCore()))
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("prompt")
    submitWithEnter()

    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(7))
    expect(setPendingMessageMock).toHaveBeenCalledTimes(1)
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
    expect(input().value).toBe("prompt")

    const resolverCallsBeforeRetry = resolveTaskLlmSelectionMock.mock.calls.length
    setTaskIdMock.mockReset()
    await waitFor(() => expect(submitButton()).toBeEnabled())
    fireEvent.click(submitButton())
    await waitFor(() => expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(resolverCallsBeforeRetry + 1))
    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(7))
    expect(toastErrorMock).not.toHaveBeenCalled()
  })
})
