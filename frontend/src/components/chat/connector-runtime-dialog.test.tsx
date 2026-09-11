import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  READ_FAILURE_STATUSES,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
  type ConnectorRuntimeSection,
  type ConnectorRuntimeType,
} from "@/lib/connector-runtime-api"

const pathnameRef = vi.hoisted(() => ({ current: "/task/1" as string | null }))
const appStateRef = vi.hoisted(() => ({ taskId: 1 as number | null }))
const sendMessageMock = vi.hoisted(() =>
  vi.fn<(message: string, config?: { clientMessageId?: string }, files?: File[]) => Promise<void>>(
    async () => {},
  ),
)
const authUserRef = vi.hoisted(() => ({ current: { id: "u1" } as { id: string } | null }))
const fetchMock = vi.hoisted(() => vi.fn())
const submitMock = vi.hoisted(() => vi.fn())
const toastMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({ usePathname: () => pathnameRef.current }))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ state: { taskId: appStateRef.taskId }, sendMessage: sendMessageMock }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUserRef.current }),
}))

// Wrapped in a stable function so a spy installed on `toastMock` after the
// first render still intercepts every call, regardless of when the dialog
// component's own module-level import binding was evaluated.
vi.mock("@/components/ui/sonner", () => ({ toast: (...args: unknown[]) => toastMock(...args) }))

// A referentially stable value: the xagent frontend test convention this
// mirrors (i18n stubs must not return a new object every render) exists
// because a fresh object here would make any effect depending on `t`
// re-fire every render.
const i18nValue = {
  t: (key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}:${JSON.stringify(vars)}` : key,
}
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => i18nValue }))

vi.mock("@/lib/connector-runtime-api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/connector-runtime-api")>()
  return {
    ...actual,
    fetchTaskConnectorRuntimeRequirements: (...args: unknown[]) => fetchMock(...args),
    submitTaskConnectorRuntimeValues: (...args: unknown[]) => submitMock(...args),
  }
})

import { ConnectorRuntimeDialog } from "./connector-runtime-dialog"
import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialog,
  useConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogActions,
} from "@/contexts/connector-runtime-dialog-context"

const REF_A = { connector_type: "custom_api", connector_id: 1 }
const REF_B = { connector_type: "mcp", connector_id: 2 }

// Shared by the two source-scanning tests below instead of each re-reading.
const libSource = readFileSync(path.resolve(__dirname, "../../lib/connector-runtime-api.ts"), "utf8")
const dialogSource = readFileSync(path.resolve(__dirname, "./connector-runtime-dialog.tsx"), "utf8")
const providerSource = readFileSync(path.resolve(__dirname, "../../contexts/connector-runtime-dialog-context.tsx"), "utf8")

function input(
  overrides: Partial<ConnectorRuntimeInput> & { section: ConnectorRuntimeSection; key: string; type: ConnectorRuntimeType },
): ConnectorRuntimeInput {
  return { required: false, satisfied: false, expired: false, ...overrides }
}
function connector(ref: typeof REF_A, name: string, inputs: ConnectorRuntimeInput[]): ConnectorRuntimeConnector {
  return { connector_ref: ref, name, inputs }
}
function report(satisfied: boolean, connectors: ConnectorRuntimeConnector[]): ConnectorRuntimeReport {
  return { satisfied, secrets_expires_at: null, connectors }
}
function ok(r: ConnectorRuntimeReport) { return { ok: true as const, report: r } }

let latestActions: ConnectorRuntimeDialogActions
let latestState: { request: unknown; payload: unknown }

function Probe({ mounted = true }: { mounted?: boolean }) {
  latestActions = useConnectorRuntimeDialogActions()
  const { request, payload } = useConnectorRuntimeDialog()
  latestState = { request, payload }
  return mounted ? <ConnectorRuntimeDialog /> : null
}

function renderHarness() {
  return render(
    <ConnectorRuntimeDialogProvider>
      <Probe />
    </ConnectorRuntimeDialogProvider>,
  )
}

async function openForTask(taskId = 1) {
  await act(async () => {
    latestActions.openForTask(taskId)
  })
}

// Delivery and the later open are two separate updates in production (the
// delivered-turn stash is written long before any failure frame retargets
// it), so this mirrors that with two acts rather than one -- batching both
// into a single act would not exercise the same "read the already-committed
// stash" path openForTask's handoff depends on.
async function recordThenOpen(
  delivery: { taskId: number; clientMessageId: string; text: string; files?: File[] },
  taskId = delivery.taskId,
) {
  await act(async () => {
    latestActions.recordDelivery(delivery)
  })
  await act(async () => {
    latestActions.openForTask(taskId)
  })
}

beforeEach(() => {
  pathnameRef.current = "/task/1"
  appStateRef.taskId = 1
  authUserRef.current = { id: "u1" }
  sendMessageMock.mockReset()
  sendMessageMock.mockResolvedValue(undefined)
  fetchMock.mockReset()
  // A harmless default for any refresh this test does not explicitly stub
  // (a failed save whose disposition sets `refresh: true`): resolving to a
  // read failure just means the dialog keeps showing the report it already
  // had, and avoids an unhandled rejection from an un-mocked call.
  fetchMock.mockResolvedValue({ ok: false, kind: "malformed" })
  submitMock.mockReset()
  toastMock.mockReset()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("keeps read outcomes in three distinct tiers", () => {
  it("keeps read outcomes in three distinct tiers", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    // The full read-failure warn matrix: every status READ_FAILURE_STATUSES
    // lists, 418 standing in for "any other non-200", and the two non-HTTP
    // failure kinds. `.every` over the spy's calls would pass on an empty
    // array, so this pins down the exact call list instead.
    const expectedWarnCalls: unknown[][] = []

    for (const status of READ_FAILURE_STATUSES) {
      fetchMock.mockReset()
      fetchMock.mockResolvedValueOnce({ ok: false, kind: "http", status })
      renderHarness()
      await openForTask()
      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expectedWarnCalls.push(["[connector-runtime] requirements read failed", status])
      cleanup()
    }
    fetchMock.mockReset()
    fetchMock.mockResolvedValueOnce({ ok: false, kind: "http", status: 418 })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expectedWarnCalls.push(["[connector-runtime] requirements read failed", 418])
    cleanup()

    fetchMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expectedWarnCalls.push(["[connector-runtime] requirements read failed", "transport"])
    cleanup()

    fetchMock.mockResolvedValueOnce({ ok: false, kind: "malformed" })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expectedWarnCalls.push(["[connector-runtime] requirements read failed", "malformed"])
    cleanup()

    expect(warnSpy.mock.calls).toEqual(expectedWarnCalls)

    // Met: nothing to fill at all, and met with an unfilled optional key.
    fetchMock.mockResolvedValueOnce(ok(report(true, [])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    cleanup()

    fetchMock.mockResolvedValueOnce(ok(report(true, [
      connector(REF_A, "A", [input({ section: "context", key: "optional", type: "string" })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    cleanup()

    // Needs a fill: the dialog becomes visible.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
  })
})

describe("renders each row by section, type, satisfaction and outcome", () => {
  it("renders each row by section, type, satisfaction and outcome", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "textKey", type: "string", required: true }),
        input({ section: "context", key: "objectKey", type: "object", required: true }),
        input({ section: "context", key: "doneKey", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "secretKey", type: "string", required: false }),
        input({ section: "auth_selector", key: "authKey", type: "string", required: false }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    expect(screen.getByLabelText("textKey")).toBeInstanceOf(HTMLInputElement)
    expect(screen.getByLabelText("objectKey")).toBeInstanceOf(HTMLTextAreaElement)
    expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument()
    expect(screen.getByText("secretKey")).toBeInTheDocument()
    expect(screen.getByText("authKey")).toBeInTheDocument()
    expect(screen.getAllByText("connectorRuntime.unsupportedNote")).toHaveLength(2)
    expect(screen.queryByRole("textbox", { name: "secretKey" })).not.toBeInTheDocument()
    expect(screen.queryByRole("textbox", { name: "authKey" })).not.toBeInTheDocument()
    expect(document.querySelector('input[type="password"]')).not.toBeInTheDocument()
    cleanup()

    // unsupported_only: an unfilled context row (bad key name included)
    // renders no control and no "saved, cannot be changed" hint, but the
    // key-name warning still shows.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "bad key", type: "string", required: false }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.contextNote")).not.toBeInTheDocument()
    expect(screen.getByText("connectorRuntime.keyNameWarning")).toBeInTheDocument()
  })
})

describe("keeps drafts and the snapshot when re-requested while open", () => {
  it("keeps drafts and the snapshot when re-requested while open", async () => {
    // A second request for an already-open task keeps a draft for a key
    // still unsatisfied, but drops one the refreshed report now reports
    // satisfied.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "stillMissing", type: "string", required: true }),
        input({ section: "context", key: "getsFilled", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("stillMissing"), { target: { value: "draft-a" } })
    fireEvent.change(screen.getByLabelText("getsFilled"), { target: { value: "draft-b" } })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "stillMissing", type: "string", required: true }),
        input({ section: "context", key: "getsFilled", type: "string", required: true, satisfied: true }),
      ]),
    ])))
    await openForTask() // same task: a second request, not a remount
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())
    expect(screen.getByLabelText("stillMissing")).toHaveValue("draft-a")
    expect(screen.queryByLabelText("getsFilled")).not.toBeInTheDocument()
    cleanup()
    fetchMock.mockClear()

    // A read started earlier must not overwrite a fresher read's result if
    // it resolves later (dropped via the request's own seq).
    let resolveFirst: (v: unknown) => void = () => {}
    let resolveSecond: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveFirst = res }))
    renderHarness()
    await openForTask()
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveSecond = res }))
    await openForTask()
    await act(async () => { resolveSecond(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "fromSecondRequest", type: "string", required: true })]),
    ]))) })
    await waitFor(() => expect(screen.getByLabelText("fromSecondRequest")).toBeInTheDocument())
    await act(async () => { resolveFirst(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "fromStaleFirstRequest", type: "string", required: true })]),
    ]))) })
    expect(screen.queryByLabelText("fromStaleFirstRequest")).not.toBeInTheDocument()
    expect(screen.getByLabelText("fromSecondRequest")).toBeInTheDocument()
    cleanup()
    fetchMock.mockClear()

    // A same-task re-request with no fresh stash in between keeps the resend
    // snapshot the first request already carried.
    const tokenReport = ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ]))
    fetchMock.mockResolvedValueOnce(tokenReport)
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-4", text: "keep me" })
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.saveAndResend")).toBeInTheDocument())
    fetchMock.mockResolvedValueOnce(tokenReport)
    await openForTask() // second request, no recordDelivery first
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    expect(sendMessageMock.mock.calls[0][0]).toBe("keep me")
    cleanup()
    fetchMock.mockClear()

    // A save in flight when the task is re-requested must not leave the
    // dialog stuck once the stale save settles.
    fetchMock.mockResolvedValueOnce(tokenReport)
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    let resolveSave: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSave = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    fetchMock.mockResolvedValueOnce(tokenReport)
    await openForTask() // retargets this same dialog instance mid-save
    await act(async () => { resolveSave(ok(report(true, []))) })
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    cleanup()
    fetchMock.mockClear()

    // The save half of a save-and-resend settles, then the resend it
    // triggers is still in flight when the task is re-requested: submitting
    // must still reset once that stale resend settles, or the dialog stays
    // stuck exactly like the save-only case above.
    fetchMock.mockResolvedValueOnce(tokenReport)
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-5", text: "keep me" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    let resolveSend: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveSend = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    fetchMock.mockResolvedValueOnce(tokenReport)
    await openForTask() // retargets this same dialog instance mid-resend
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await act(async () => { resolveSend() })
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })
})

describe("flags a non-object JSON draft on blur and does not submit it", () => {
  it("flags a non-object JSON draft on blur and does not submit it", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "config", type: "object", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    const field = screen.getByLabelText("config")

    for (const badValue of ["[]", "1", '"x"', "{"]) {
      fireEvent.change(field, { target: { value: badValue } })
      fireEvent.blur(field)
      expect(screen.getByText("connectorRuntime.objectInvalid")).toBeInTheDocument()
      expect(submitMock).not.toHaveBeenCalled()
    }

    fireEvent.change(field, { target: { value: '{"a":1}' } })
    fireEvent.blur(field)
    expect(screen.queryByText("connectorRuntime.objectInvalid")).not.toBeInTheDocument()
  })
})

describe("refreshes after a conflict and drops newly satisfied keys", () => {
  it("refreshes after a conflict and drops newly satisfied keys", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "a", type: "string", required: true }),
        input({ section: "context", key: "b", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    fireEvent.change(screen.getByLabelText("a"), { target: { value: "1" } })
    fireEvent.change(screen.getByLabelText("b"), { target: { value: "2" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 409, code: "runtime_context_immutable",
      reason: "conflict.context.a", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "a", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "b", type: "string", required: true }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())

    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2))
    expect(submitMock.mock.calls[1][1]).toEqual([{ connector_ref: REF_A, context: { b: "2" } }])
  })
})

describe("locates a field error by connector and key", () => {
  it("locates a field error by connector and key", async () => {
    const twoConnectors = report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      connector(REF_B, "B", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])
    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    fireEvent.change(screen.getAllByLabelText("token")[1], { target: { value: "y" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.token", connectorRef: REF_B,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.typeString:{\"key\":\"token\"}")).toBeInTheDocument())
    expect(screen.getAllByText(/connectorRuntime.errors.typeString/)).toHaveLength(1)
    cleanup()

    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 409, code: "runtime_context_immutable",
      reason: "conflict.context.token", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getAllByText(/connectorRuntime.errors.conflict/)).toHaveLength(1))
    cleanup()

    // No connector_ref at all: whole-dialog scope.
    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({ ok: false, kind: "http", status: 500 })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.contactAdmin")).toBeInTheDocument())
    cleanup()

    // A located key the current report no longer has: falls back to dialog scope.
    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.missing", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    // Falls back to whole-dialog scope, which renders without the {key} var.
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.typeString")).toBeInTheDocument())
  })
})

describe("resends the snapshot under a fresh id", () => {
  it("resends the snapshot under a fresh id", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-1", text: "hello" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    expect(sendMessageMock.mock.calls[0][0]).toBe("hello")
    expect(sendMessageMock.mock.calls[0][1]?.clientMessageId).not.toBe("orig-1")
    expect(sendMessageMock.mock.calls[0][2]).toEqual([])
    expect(submitMock).toHaveBeenCalledTimes(1)
    cleanup()
    sendMessageMock.mockClear()
    submitMock.mockClear()

    // With attachments: the same File objects travel to sendMessage.
    const file = new File(["x"], "a.txt")
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-2", text: "hi", files: [file] })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    expect(sendMessageMock.mock.calls[0][2]).toEqual([file])
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())
    expect(submitMock).toHaveBeenCalledTimes(1)

    sendMessageMock.mockResolvedValueOnce(undefined)
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
    expect(submitMock).toHaveBeenCalledTimes(1)
    const firstId = sendMessageMock.mock.calls[0][1]?.clientMessageId
    const secondId = sendMessageMock.mock.calls[1][1]?.clientMessageId
    expect(secondId).not.toBe(firstId)
    expect(secondId).not.toBe("orig-2")
    cleanup()
    sendMessageMock.mockClear()
    submitMock.mockClear()

    // Only a save-only path is offered without a snapshot; resolving 200
    // with a still-blocking-only-secrets outcome sends first, then closes.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    // Spy installed right after the first render, before anything reads
    // `close` off the shared actions object: a component that already
    // destructured the pre-spy function from an earlier render would call
    // that captured reference straight through the spy.
    const closeSpy = vi.spyOn(latestActions, "close")
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-3", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
      ]),
    ])))
    sendMessageMock.mockClear()
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    const sendOrder = sendMessageMock.mock.invocationCallOrder[0]
    await waitFor(() => expect(closeSpy).toHaveBeenCalled())
    const closeOrder = closeSpy.mock.invocationCallOrder[closeSpy.mock.invocationCallOrder.length - 1]
    expect(sendOrder).toBeLessThan(closeOrder)
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    cleanup()
    sendMessageMock.mockClear()
    submitMock.mockClear()

    // A hung resend must not fire twice from a double click on the retry
    // button.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-4", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())
    sendMessageMock.mockClear()
    let resolveRetry: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveRetry = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    await act(async () => { resolveRetry() })
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(sendMessageMock).toHaveBeenCalledTimes(1)
  })
})

async function openSimpleDialog(taskId = 1, withStash = true) {
  fetchMock.mockResolvedValueOnce(ok(report(false, [
    connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
  ])))
  renderHarness()
  if (withStash) await recordThenOpen({ taskId, clientMessageId: `orig-${taskId}`, text: "hi" })
  else await openForTask(taskId)
  await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
}

describe("clears the stash when the dialog closes", () => {
  it("clears the stash when the dialog closes", async () => {
    await openSimpleDialog()
    // A fresh delivery lands for this task while the dialog is already open
    // (its own stash was already claimed into the request on open, so this
    // is the only way to put a non-null payload in front of the close click
    // below -- without it this assertion would pass even if closing stopped
    // clearing the stash).
    await act(async () => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "late-clean", text: "hi" })
    })
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(latestState).toEqual({ request: null, payload: null }))
    cleanup()

    // Read failure while a delivery lands mid-read: the stash it wrote survives.
    fetchMock.mockImplementationOnce(async () => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "late", text: "hi" })
      return { ok: false, kind: "http", status: 500 }
    })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(latestState.request).toBeNull())
    expect((latestState.payload as { clientMessageId: string } | null)?.clientMessageId).toBe("late")
    cleanup()

    // Resent: the stash now holds the just-resent turn, not null.
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect((latestState.payload as { clientMessageId: string } | null)?.clientMessageId).not.toBeNull()
    cleanup()

    // Submitting in progress: X is ignored, nothing changes.
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    let resolveSubmit: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    resolveSubmit(ok(report(true, [])))
  })
})

describe("clears the stash at the task-switch and reset cleanup points", () => {
  it.each([
    ["drops the stash and request of a task switched away from", async () => {
      await openSimpleDialog(1)
      // A fresh delivery for the same task after the original stash was
      // already claimed into the request on open -- without this, `payload`
      // would already be null before the switch below, and this assertion
      // would pass even if switching tasks stopped narrowing the stash.
      await act(async () => {
        latestActions.recordDelivery({ taskId: 1, clientMessageId: "still-here", text: "hi" })
      })
      await act(async () => { latestActions.retainOnlyTask(2) })
      expect(latestState).toEqual({ request: null, payload: null })
    }],
    ["clears the stash on conversation reset", async () => {
      await openSimpleDialog(1)
      await act(async () => { latestActions.retainOnlyTask(null) })
      expect(latestState).toEqual({ request: null, payload: null })
      cleanup()

      // The dialog unmounts (AppProvider going away), a late delivery still
      // lands, then it remounts with no viewed task: the mount-time task
      // effect (not the unmount cleanup, which already ran) wipes it.
      appStateRef.taskId = null
      const toggle = render(providerTree())
      toggle.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
      await act(async () => { latestActions.recordDelivery({ taskId: 1, clientMessageId: "x", text: "hi" }) })
      toggle.rerender(providerTree())
      await waitFor(() => expect(latestState.payload).toBeNull())
    }],
  ])("%s", async (_name, run) => { await run() })
})

function providerTree() {
  return <ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>
}

describe("clears the stash and request when the signed-in user changes", () => {
  it("clears the stash and request when the signed-in user changes", async () => {
    fetchMock.mockResolvedValue(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    const { rerender } = render(providerTree())
    await recordThenOpen({ taskId: 1, clientMessageId: "x", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    // Same identity, re-rendered: a control row proving the effect only
    // fires on an actual id change, not on every render.
    rerender(providerTree())
    expect(screen.getByRole("dialog")).toBeInTheDocument()

    authUserRef.current = { id: "u2" }
    rerender(providerTree())
    await waitFor(() => expect(latestState).toEqual({ request: null, payload: null }))
    cleanup()

    // Logout: id -> null clears the same way.
    authUserRef.current = { id: "u1" }
    const second = render(providerTree())
    await recordThenOpen({ taskId: 1, clientMessageId: "y", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    authUserRef.current = null
    second.rerender(providerTree())
    await waitFor(() => expect(latestState).toEqual({ request: null, payload: null }))
  })
})

describe("closes after save by the shared outcome", () => {
  it("closes after save by the shared outcome", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
        input({ section: "secrets", key: "s2", type: "string", required: false }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    // Opening-time annotation names only the required secret, not the optional one.
    expect(screen.getByText('connectorRuntime.stillMissingAfterSave:{"keys":"s1"}')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("k1"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
        input({ section: "secrets", key: "s2", type: "string", required: false }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(toastMock).toHaveBeenCalledTimes(1)
    expect(toastMock.mock.calls[0][0]).toBe('connectorRuntime.onlyUnsupportedRemaining:{"keys":"s1"}')
    cleanup()
    toastMock.mockClear()

    // Two required context keys, only one filled: stays open, no toast.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true }),
        input({ section: "context", key: "k2", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("k1"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "k2", type: "string", required: true }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(toastMock).not.toHaveBeenCalled()
  })
})

describe("offers resend only with this request's snapshot", () => {
  it("offers resend only with this request's snapshot", async () => {
    await openSimpleDialog(1, false)
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.actions.saveAndResend")).not.toBeInTheDocument()
    cleanup()

    await openSimpleDialog(1, true)
    expect(screen.getByText("connectorRuntime.actions.saveAndResend")).toBeInTheDocument()
  })
})

describe("keeps the key-name hint out of the submit rule", () => {
  it("keeps the key-name hint out of the submit rule", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "bad key", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    expect(screen.getByText("connectorRuntime.keyNameWarning")).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("bad key"), { target: { value: "x" } })
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
  })
})

describe("keeps the dialog open and every draft on a failed save", () => {
  it("keeps the dialog open and every draft on a failed save", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "kept-draft" } })

    for (const outcome of [
      { ok: false as const, kind: "transport" as const },
      { ok: false as const, kind: "coded" as const, status: 503, code: "connector_runtime_unavailable" },
      { ok: false as const, kind: "http" as const, status: 500 },
      { ok: false as const, kind: "coded" as const, status: 400, code: "invalid_runtime_context", reason: "runtime input key must match [A-Za-z0-9_-]+", connectorRef: REF_A },
    ]) {
      submitMock.mockResolvedValueOnce(outcome)
      fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
      await waitFor(() => expect(submitMock).toHaveBeenCalled())
      expect(screen.getByRole("dialog")).toBeInTheDocument()
      expect(screen.getByLabelText("token")).toHaveValue("kept-draft")
    }

    // The two retryable dispositions offer a retry button that re-POSTs the
    // same body.
    submitMock.mockClear()
    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.retry")).toBeInTheDocument())
    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.retry"))
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2))
    expect(submitMock.mock.calls[0][1]).toEqual(submitMock.mock.calls[1][1])
  })
})

describe("logs only a fixed prefix and a status", () => {
  it("logs only a fixed prefix and a status", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {})
    const infoSpy = vi.spyOn(console, "info").mockImplementation(() => {})

    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "s3cr3t-draft" } })
    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.network")).toBeInTheDocument())
    cleanup()

    fetchMock.mockReset()
    fetchMock.mockResolvedValueOnce({ ok: false, kind: "http", status: 500 })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(warnSpy).toHaveBeenCalled())

    expect(warnSpy.mock.calls).toEqual([["[connector-runtime] requirements read failed", 500]])
    const leaked = JSON.stringify([errorSpy.mock.calls, logSpy.mock.calls, infoSpy.mock.calls])
    expect(leaked).not.toContain("s3cr3t-draft")
  })
})

describe("never renders server text as HTML", () => {
  it("never renders server text as HTML", () => {
    const sources = [
      { source: libSource, marker: "export function readConnectorRuntimeReport" },
      { source: providerSource, marker: "export function ConnectorRuntimeDialogProvider" },
      { source: dialogSource, marker: "export function ConnectorRuntimeDialog" },
    ]
    expect(sources).toHaveLength(3)
    for (const { source, marker } of sources) {
      expect(source).toContain(marker)
      expect(source).not.toContain("dangerouslySetInnerHTML")
      expect(source).not.toContain("innerHTML")
    }
  })
})

function leafTranslationPaths(obj: Record<string, unknown>, prefix: string): string[] {
  return Object.entries(obj).flatMap(([k, v]) =>
    typeof v === "string" ? [`${prefix}${k}`] : leafTranslationPaths(v as Record<string, unknown>, `${prefix}${k}.`),
  )
}

describe("resolves every connectorRuntime key the dialog uses", () => {
  it("resolves every connectorRuntime key the dialog uses", async () => {
    const { translations, resolveTranslation } = await import("@/i18n/translations")
    expect(libSource).toContain("export function readConnectorRuntimeReport")
    expect(dialogSource).toContain("export function ConnectorRuntimeDialog")
    expect(dialogSource).not.toMatch(/connectorRuntime\.\$\{/)
    expect(libSource).not.toMatch(/connectorRuntime\.\$\{/)

    const usedKeys = new Set<string>()
    for (const source of [libSource, dialogSource]) {
      for (const match of source.matchAll(/"(connectorRuntime\.[a-zA-Z0-9_.]+)"/g)) usedKeys.add(match[1])
    }
    const definedKeys = new Set(
      leafTranslationPaths(translations.en.connectorRuntime as Record<string, unknown>, "connectorRuntime."),
    )
    expect(Array.from(usedKeys).sort()).toEqual(Array.from(definedKeys).sort())

    for (const locale of ["en", "zh"] as const) {
      for (const key of usedKeys) {
        expect(resolveTranslation(locale, key as never)).not.toBe(key)
      }
    }
    expect(translations.zh.connectorRuntime.keyNameWarning).toContain("在改名之前这个连接器每次都会失败")
  })
})

describe("does not open off the host routes", () => {
  it("does not open off the host routes", async () => {
    // Not a host route when the request arrives: no read, request cleared.
    pathnameRef.current = "/settings"
    renderHarness()
    await openForTask()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(latestState.request).toBeNull()
    cleanup()

    // A host route when requested, but the path changes before the read
    // resolves: still not shown.
    pathnameRef.current = "/task/1"
    let resolveFetch: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveFetch = res }))
    renderHarness()
    await openForTask()
    pathnameRef.current = "/settings"
    await act(async () => {
      resolveFetch(ok(report(false, [
        connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      ])))
    })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(latestState.request).toBeNull()
    cleanup()

    // Already visible, then the path leaves the host routes: closes with
    // "left-host"; a mid-read delivery's stash survives.
    pathnameRef.current = "/task/1"
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    const first = renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    await act(async () => { latestActions.recordDelivery({ taskId: 1, clientMessageId: "z", text: "hi" }) })
    // "not-shown"/"resent" also keep the stash, so only a spy on the actual
    // argument proves this passed "left-host".
    const closeSpy = vi.spyOn(latestActions, "close")
    pathnameRef.current = "/settings"
    first.rerender(providerTree())
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(closeSpy).toHaveBeenCalledWith("left-host")
    expect(latestState.request).toBeNull()
    expect((latestState.payload as { clientMessageId: string } | null)?.clientMessageId).toBe("z")
    cleanup()

    // Reverse control: moving between two host routes (a trailing slash on
    // the same workforce run page) does not close it.
    pathnameRef.current = "/workforces/7/run"
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    const second = renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    pathnameRef.current = "/workforces/7/run/"
    second.rerender(providerTree())
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(latestState.request).not.toBeNull()
  })
})

describe("does nothing after unmount when a request settles", () => {
  it("does nothing after unmount when a request settles", async () => {
    // POST in flight when the tree unmounts: resolving it afterward does
    // nothing. Uses a resend snapshot and "save and resend" (not "save
    // only") so a broken alive-after-unmount guard would show up as a
    // spurious sendMessage call, not just a same-outcome no-op.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let resolveSubmit: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    const first = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "unmount-guard", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    first.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    sendMessageMock.mockClear()
    await act(async () => { resolveSubmit(ok(report(true, []))) })
    expect(sendMessageMock).not.toHaveBeenCalled()
    expect(toastMock).not.toHaveBeenCalled()
    cleanup()

    // GET in flight when the task is switched away from: resolving it does nothing.
    let resolveFetch: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveFetch = res }))
    renderHarness()
    await openForTask()
    await act(async () => { latestActions.retainOnlyTask(2) })
    await act(async () => {
      resolveFetch(ok(report(false, [
        connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      ])))
    })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    cleanup()

    // Resend's sendMessage in flight when the tree unmounts: rejecting it
    // afterward does nothing.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let rejectSend: (e: Error) => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((_res, rej) => { rejectSend = rej }))
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    const third = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "x", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    third.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    await act(async () => { rejectSend(new Error("closed")) })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })
})
