import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Issue #969: the non-update builder actions (publish, unpublish, publish from
// the creation success dialog, optimize instructions) must never pass a raw
// `detail` payload to toast.error. Each action renders only displayable
// strings, keeps its own localized fallback for unreadable or malformed
// responses, and leaves the builder mounted after the failure.

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://api.local",
    getWsUrl: () => "ws://api.local",
  }
})

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: {
      messages: [],
      traceEvents: [],
      currentTask: null,
      isProcessing: false,
      isHistoryLoading: false,
      taskId: null,
      filePreview: { isOpen: false },
      dagExecution: null,
      steps: [],
    },
    setTaskId: vi.fn(),
    sendMessage: vi.fn(),
    dispatch: vi.fn(),
    closeFilePreview: vi.fn(),
    pauseTask: vi.fn(),
    resumeTask: vi.fn(),
    openFilePreview: vi.fn(),
    requestStatus: vi.fn(),
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", user: { id: "1", is_admin: false } }),
}))

// The i18n return value must be referentially stable: AgentSshBindings keys a
// fetch effect on `t`, so a per-render `t` identity turns that effect into an
// unbounded fetch/render loop under jsdom.
vi.mock("@/contexts/i18n-context", () => {
  const i18n = {
    locale: "en",
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }
  return { useI18n: () => i18n }
})

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: toastErrorMock, success: vi.fn() } }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ middlePanel }: { middlePanel: React.ReactNode }) => (
    <div>{middlePanel}</div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({ AgentBuilderChat: () => null }))
vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
}))
vi.mock("@/components/chat/FileMentionDropdown", () => ({ FileMentionDropdown: () => null }))
vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    isOpen: false,
    items: [],
    selectedIndex: 0,
    selectItem: vi.fn(),
    close: vi.fn(),
  }),
}))
vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: () => <div data-testid="multi-select" />,
}))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"

function agentResponse(status: "draft" | "published") {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Existing Agent",
    description: "",
    instructions: "You are an existing agent.",
    execution_mode: "balanced",
    models: { general: "10" },
    knowledge_bases: [],
    skills: [],
    tool_categories: ["basic"],
    suggested_prompts: [],
    logo_url: null,
    status,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    can_edit: true,
  }
}

// Error payload matrix shared by every action: each case must surface
// `expected` (with `FALLBACK` replaced by the action-specific i18n key)
// instead of handing the raw payload to the toaster.
const FALLBACK = "__ACTION_FALLBACK__"

const ERROR_CASES = [
  {
    name: "a plain string detail",
    body: JSON.stringify({ detail: " Action failed with string detail " }),
    expected: "Action failed with string detail",
  },
  {
    name: "a structured detail message",
    body: JSON.stringify({ detail: { message: "Action failed", context: [] } }),
    expected: "Action failed",
  },
  {
    name: "a detail object without a readable message",
    body: JSON.stringify({ detail: { code: 123 } }),
    expected: FALLBACK,
  },
  {
    name: "FastAPI validation detail messages",
    body: JSON.stringify({
      detail: [
        { msg: " Field is required " },
        " Invalid value ",
        { message: " Unsupported option " },
        { msg: " " },
      ],
    }),
    expected: "Field is required; Invalid value; Unsupported option",
  },
  {
    name: "a detail array without readable entries",
    body: JSON.stringify({ detail: [1, true, null, { msg: " " }] }),
    expected: FALLBACK,
  },
  {
    name: "an empty response body",
    body: null,
    expected: FALLBACK,
  },
  {
    name: "a non-JSON response body",
    body: "<html>Bad Gateway</html>",
    expected: FALLBACK,
  },
] as const

function errorResponse(body: string | null) {
  return new Response(body, {
    status: 422,
    headers: { "Content-Type": "application/json" },
  })
}

function installEditModeApi(
  status: "draft" | "published",
  failingPath: string,
  failingBody: string | null
) {
  apiRequestMock.mockImplementation((url: string, opts?: { method?: string }) => {
    if (opts?.method === "POST" && url.endsWith(failingPath))
      return Promise.resolve(errorResponse(failingBody))
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available"))
      return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(status)), { status: 200 })
      )
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

// Create-mode mock for the success-dialog publish path: agent creation
// succeeds (which opens the dialog), publish fails with the payload under test.
function installCreateModeApi(failingBody: string | null) {
  apiRequestMock.mockImplementation((url: string, opts?: { method?: string }) => {
    if (opts?.method === "POST" && url.endsWith(`/api/agents/${AGENT_ID}/publish`))
      return Promise.resolve(errorResponse(failingBody))
    if (opts?.method === "POST" && url.endsWith("/api/agents"))
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse("draft")), { status: 200 })
      )
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available"))
      return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(
        new Response(
          JSON.stringify([{ config_type: "general", model: { id: 10 } }]),
          { status: 200 }
        )
      )
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

async function waitForLoadedBuilder() {
  await waitFor(() =>
    expect(
      screen.getByPlaceholderText("builds.configForm.name.placeholder")
    ).toHaveValue("Existing Agent")
  )
}

async function expectToast(expected: string) {
  await waitFor(() => {
    expect(toastErrorMock).toHaveBeenCalled()
    expect(toastErrorMock.mock.calls.at(-1)?.[0]).toBe(expected)
  })
}

beforeEach(() => {
  apiRequestMock.mockReset()
  toastErrorMock.mockReset()
  globalThis.WebSocket = vi.fn() as unknown as typeof WebSocket
})

afterEach(() => cleanup())

describe("AgentBuilder publish error handling (issue #969)", () => {
  it.each(ERROR_CASES)(
    "handles $name without unmounting the builder",
    async ({ body, expected }) => {
      installEditModeApi("draft", `/api/agents/${AGENT_ID}/publish`, body)
      render(<AgentBuilder agentId={AGENT_ID} />)
      await waitForLoadedBuilder()

      fireEvent.click(screen.getByText("builds.editor.header.publish"))

      await expectToast(
        expected === FALLBACK ? "builds.publication.publishFailed" : expected
      )
      expect(screen.getByDisplayValue("Existing Agent")).toBeInTheDocument()
    }
  )
})

describe("AgentBuilder network failure handling (issue #969)", () => {
  it("uses the generic fallback when the publish request itself rejects", async () => {
    // A transport-level rejection must not be swallowed by the response-body
    // parsing path: it hits the outer catch and shows the generic fallback.
    installEditModeApi("draft", "__no_failing_path__", null)
    const base = apiRequestMock.getMockImplementation()!
    apiRequestMock.mockImplementation((url: string, opts?: { method?: string }) => {
      if (opts?.method === "POST" && url.endsWith(`/api/agents/${AGENT_ID}/publish`))
        return Promise.reject(new TypeError("network down"))
      return base(url, opts)
    })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await waitForLoadedBuilder()

    fireEvent.click(screen.getByText("builds.editor.header.publish"))

    await expectToast("builds.editor.error.unknown")
    expect(screen.getByDisplayValue("Existing Agent")).toBeInTheDocument()
  })
})

describe("AgentBuilder unpublish error handling (issue #969)", () => {
  it.each(ERROR_CASES)(
    "handles $name without unmounting the builder",
    async ({ body, expected }) => {
      installEditModeApi("published", `/api/agents/${AGENT_ID}/unpublish`, body)
      render(<AgentBuilder agentId={AGENT_ID} />)
      await waitForLoadedBuilder()

      fireEvent.click(screen.getByText("builds.editor.header.unpublish"))

      await expectToast(
        expected === FALLBACK ? "builds.publication.unpublishFailed" : expected
      )
      expect(screen.getByDisplayValue("Existing Agent")).toBeInTheDocument()
    }
  )
})

describe("AgentBuilder optimize instructions error handling (issue #969)", () => {
  it.each(ERROR_CASES)(
    "handles $name without unmounting the builder",
    async ({ body, expected }) => {
      installEditModeApi("draft", "/api/agents/optimize-instructions", body)
      render(<AgentBuilder agentId={AGENT_ID} />)
      await waitForLoadedBuilder()

      fireEvent.click(screen.getByText("builds.configForm.instructions.optimize"))

      await expectToast(
        expected === FALLBACK
          ? "builds.configForm.instructions.optimizeError"
          : expected
      )
      expect(screen.getByDisplayValue("Existing Agent")).toBeInTheDocument()
    }
  )
})

describe("AgentBuilder success-dialog publish error handling (issue #969)", () => {
  it.each(ERROR_CASES)(
    "handles $name without unmounting the builder",
    async ({ body, expected }) => {
      installCreateModeApi(body)
      render(<AgentBuilder />)

      const nameInput = await screen.findByPlaceholderText(
        "builds.configForm.name.placeholder"
      )
      fireEvent.change(nameInput, { target: { value: "New Agent" } })

      // Instructions live in a contentEditable div, not a form control.
      const editor = document.querySelector("[contenteditable]") as HTMLElement
      expect(editor).toBeTruthy()
      editor.textContent = "You are a new agent."
      fireEvent.input(editor)

      // The default model arrives asynchronously from /api/models/user-default;
      // retry the create click until validation passes and the dialog opens.
      // After creation the header shows its own (disabled) publish button, so
      // scope the click to the success dialog.
      await waitFor(() => {
        if (!screen.queryByRole("dialog")) {
          fireEvent.click(screen.getByText("builds.editor.header.create"))
        }
        expect(screen.getByRole("dialog")).toBeInTheDocument()
      })
      fireEvent.click(
        within(screen.getByRole("dialog")).getByText("builds.editor.header.publish")
      )

      await expectToast(
        expected === FALLBACK ? "builds.publication.publishFailed" : expected
      )
      // The background form is aria-hidden behind the dialog overlay, so
      // assert survival via the dialog staying mounted after the failure.
      expect(screen.getByRole("dialog")).toBeInTheDocument()
    }
  )
})
