import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const authUserRef = { current: { id: "u1" } as { id: string } | null }
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUserRef.current }),
}))

import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialog,
  useConnectorRuntimeDialogActions,
  useConnectorRuntimeDialogActionsIfMounted,
  transitionRequest,
  type ConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogCloseOutcome,
  type ConnectorRuntimeDialogRequest,
  type ConnectorRuntimeDialogState,
  type ConnectorRuntimeDialogValue,
  type ConnectorRuntimeResendPayload,
  type RequestInput,
  type SessionCheckCause,
} from "./connector-runtime-dialog-context"

afterEach(() => {
  cleanup()
  authUserRef.current = { id: "u1" }
})

describe("useConnectorRuntimeDialogActionsIfMounted", () => {
  it("does not warn when called with no provider above it", () => {
    // A fresh module instance for this test file (vitest isolates modules
    // per file by default): warnCalledOutsideProvider's own "already warned
    // about this action" set starts empty here, so this is a real, not
    // vacuous, check that no warning fires -- unlike calling the same
    // action name from within the large app-context-chat.test.tsx suite,
    // where an unrelated earlier test can already have exhausted that
    // action name's one-time warning.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let actions: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActionsIfMounted()
      return null
    }
    render(<Probe />)
    act(() => {
      actions?.openForTask(1)
      actions?.close("dismissed", 1)
      actions?.recordDelivery({ taskId: 1, clientMessageId: "x", text: "hi" })
      actions?.retainOnlyTask(1)
      actions?.forgetDelivery(1)
    })
    expect(warnSpy).not.toHaveBeenCalled()
    warnSpy.mockRestore()
  })

  it("still warns through the plain hook with no provider above it (the wiring-mistake case this dev warning exists for)", () => {
    // Same fresh-module guarantee as above, in the other direction: proves
    // the silent behavior above is specific to the "if mounted" entry point
    // and not a change to warnCalledOutsideProvider itself, which must keep
    // catching an actual wiring mistake (a consumer mounted as a sibling of
    // the provider instead of inside it).
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let actions: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActions()
      return null
    }
    render(<Probe />)
    act(() => { actions?.openForTask(1) })
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("called outside ConnectorRuntimeDialogProvider"),
    )
    warnSpy.mockRestore()
  })

  it("returns the same actions a real provider hands to the plain hook", () => {
    // Inside a real provider there is only one actions object (the
    // provider's own useMemo), so this must read as that same object, not a
    // second one -- otherwise a consumer holding onto this hook's result in
    // a ref (as app-context-chat.tsx does) would diverge from one reading
    // useConnectorRuntimeDialogActions() directly.
    let ifMounted: ConnectorRuntimeDialogActions | undefined
    let plain: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      ifMounted = useConnectorRuntimeDialogActionsIfMounted()
      plain = useConnectorRuntimeDialogActions()
      return null
    }
    render(
      <ConnectorRuntimeDialogProvider>
        <Probe />
      </ConnectorRuntimeDialogProvider>,
    )
    expect(ifMounted).toBe(plain)
  })
})

describe("pending candidate state machine", () => {
  let latestActions: ConnectorRuntimeDialogActions
  let latestState: ConnectorRuntimeDialogValue

  function Probe() {
    latestActions = useConnectorRuntimeDialog()
    latestState = useConnectorRuntimeDialog()
    return null
  }

  function renderProbe() {
    return render(
      <ConnectorRuntimeDialogProvider>
        <Probe />
      </ConnectorRuntimeDialogProvider>,
    )
  }

  it("promotes a staged candidate to the stash once its delivery is recorded (transition)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toBeNull()
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toEqual({ taskId: 1, clientMessageId: "cm-1", text: "hi", files: [] })
  })

  it("withdraws a staged candidate on discard, so its late delivery is not recorded (cancellation)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    act(() => {
      latestActions.discardPendingDelivery("cm-1")
    })
    // The send that staged this candidate threw or returned early; its
    // delivery acknowledgement arriving late must not resurrect the stash --
    // recordDelivery only redeems a ticket that is still outstanding.
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toBeNull()
  })

  it("drops a staged candidate when its task settles, so a late delivery is not recorded (settlement discard)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    // A non-triggering settlement frame (task_completed, or a task_error not
    // in the trigger set) reaches forgetDelivery, not openForTask.
    act(() => {
      latestActions.forgetDelivery(1)
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toBeNull()
  })

  it("does not claim either of two in-flight candidates for the same task (T5)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "first" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-2", text: "second" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })

  it("replaces a repeat stage under the same clientMessageId in place", () => {
    // A retry that reuses the caller-supplied id stages a second time under
    // that same id. Appending instead of replacing would leave two entries
    // for one turn, which openForTask reads as two in-flight turns and
    // withholds the whole resend chain for -- so the retry-id reuse this
    // dialog depends on would cost the user the resend button.
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "first" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "second" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-1", text: "second", files: [],
    })
  })

  it("does not fall through to the confirmed stash on ambiguity (Major A)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    // A second and third turn are staged after "AAA" was already delivered
    // and confirmed -- the ambiguous pair must not make openForTask fall
    // through the rest of the chain and hand out that older, already-sent
    // turn instead of degrading to save-only, same as T5 above (which has
    // no confirmed stash to fall through to in the first place).
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-b", text: "BBB" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-c", text: "CCC" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })

  it("keeps an already-open dialog's own snapshot through a later ambiguous frame", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-a", text: "AAA", files: [],
    })
    // A newer turn is sent and confirmed while this dialog is still open...
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-x", text: "XXX" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-x", text: "XXX" })
    })
    // ...and then two more turns are staged before a second terminal frame
    // retargets this same open dialog.
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-b", text: "BBB" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-c", text: "CCC" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    // Ambiguity discards the confirmed stash ("XXX") same as always, but
    // must not make the button the user is already looking at disappear or
    // switch to a different turn: it keeps carrying "AAA".
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-a", text: "AAA", files: [],
    })
  })

  it("hands the confirmed stash to a plain single-turn resend with nothing staged", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-a", text: "AAA", files: [],
    })
  })

  it("prefers a staged candidate over an already-confirmed stash (T6)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-old", text: "old" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-old", text: "old" })
    })
    // A second turn sent afterward is staged but not yet acknowledged when
    // the triggering terminal frame arrives.
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-new", text: "new" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-new", text: "new", files: [],
    })
  })

  it("never claims a candidate staged for a different task (T7)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    act(() => {
      latestActions.openForTask(2)
    })
    expect(latestState.request?.resendPayload).toBeNull()
    // Task 1's own candidate is untouched by task 2's settlement.
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-1", text: "hi", files: [],
    })
  })

  it("clears a pending candidate when the task it belongs to is switched away from (T8)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    act(() => {
      latestActions.retainOnlyTask(2)
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })

  it("clears a pending candidate when the signed-in identity changes (T8)", () => {
    const { rerender } = renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    authUserRef.current = { id: "u2" }
    rerender(
      <ConnectorRuntimeDialogProvider>
        <Probe />
      </ConnectorRuntimeDialogProvider>,
    )
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })
})

// Shared by the two transitionRequest tables below.
const delivery = (taskId: number, clientMessageId: string) => ({
  taskId, clientMessageId, text: clientMessageId, files: [] as File[],
})
const stash1 = delivery(1, "stash-1")
const stash2 = delivery(2, "stash-2")
const staged1 = delivery(1, "staged-1")
const staged1b = delivery(1, "staged-1b")
const staged2 = delivery(2, "staged-2")
const keptSnap = delivery(1, "kept-1")
const request = (
  taskId: number,
  seq: number,
  resendPayload: ConnectorRuntimeResendPayload | null = null,
  trigger: ConnectorRuntimeDialogRequest["trigger"] = "turn_failure",
): ConnectorRuntimeDialogRequest => ({ taskId, seq, resendPayload, trigger })
const state = (overrides: Partial<ConnectorRuntimeDialogState> = {}): ConnectorRuntimeDialogState => (
  { seq: 4, request: null, payload: null, pending: [], dismissedCheck: null, ...overrides }
)
const open = (taskId: number): RequestInput => ({ type: "open", taskId, trigger: "turn_failure" })
const closeAs = (outcome: ConnectorRuntimeDialogCloseOutcome, taskId = 1): RequestInput => (
  { type: "close", taskId, outcome }
)
const check = (taskId: number, cause: SessionCheckCause): RequestInput => (
  { type: "open", taskId, trigger: "session_open", cause }
)
const sessionCheck = (taskId: number, seq = 4) => request(taskId, seq, null, "session_open")


// Every request transition the provider's actions made before they were
// moved into transitionRequest, pinned as data: each row is one previous
// state, one input, and the state that must come out. A row whose expected
// state is `"same"` must return the very object it was given -- the
// provider relies on that to skip a re-render when nothing changed.
describe("transitionRequest keeps today's transitions", () => {
  const rows: Array<[string, ConnectorRuntimeDialogState, RequestInput, ConnectorRuntimeDialogState | "same"]> = [
    ["open with nothing to claim", state(), open(1),
      state({ seq: 5, request: request(1, 5) })],
    ["open for a task whose request already carries a snapshot keeps it", state({ request: request(1, 4, keptSnap) }), open(1),
      state({ seq: 5, request: request(1, 5, keptSnap) })],
    ["open claims exactly one staged turn over the stash, and clears both", state({ payload: stash1, pending: [staged1, staged2] }), open(1),
      state({ seq: 5, request: request(1, 5, staged1), payload: null, pending: [staged2] })],
    ["open claims the stash when nothing is staged", state({ payload: stash1 }), open(1),
      state({ seq: 5, request: request(1, 5, stash1), payload: null })],
    ["open leaves another task's stash alone", state({ payload: stash2 }), open(1),
      state({ seq: 5, request: request(1, 5), payload: stash2 })],
    ["open claims nothing when two turns are staged", state({ payload: stash1, pending: [staged1, staged1b] }), open(1),
      state({ seq: 5, request: request(1, 5), payload: null, pending: [] })],
    ["open keeps an open request's snapshot when two turns are staged", state({ request: request(1, 4, keptSnap), pending: [staged1, staged1b] }), open(1),
      state({ seq: 5, request: request(1, 5, keptSnap), pending: [] })],
    ["open for another task replaces the request and drops its snapshot", state({ request: request(2, 4, stash2) }), open(1),
      state({ seq: 5, request: request(1, 5) })],
    ["close with no request", state({ payload: stash1 }), closeAs("dismissed"), "same"],
    ["close dismissed drops the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("dismissed"),
      state({ payload: null })],
    ["close resent keeps the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("resent"),
      state({ payload: stash1 })],
    ["close not-shown keeps the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("not-shown"),
      state({ payload: stash1 })],
    ["close left-host keeps the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("left-host"),
      state({ payload: stash1 })],
    ["retain the request's own task", state({ request: request(1, 4), payload: stash2, pending: [staged1, staged2] }), { type: "retain", taskId: 1 },
      state({ request: request(1, 4), payload: null, pending: [staged1] })],
    ["retain another task", state({ request: request(1, 4), payload: stash1, pending: [staged1] }), { type: "retain", taskId: 2 },
      state()],
    ["retain no task", state({ request: request(1, 4), payload: stash1, pending: [staged2] }), { type: "retain", taskId: null },
      state()],
    ["retain with nothing to drop", state({ request: request(1, 4), payload: stash1, pending: [staged1] }), { type: "retain", taskId: 1 }, "same"],
    ["forget takes the snapshot off an open request, seq unchanged", state({ request: request(1, 4, keptSnap), payload: stash1, pending: [staged1, staged2] }), { type: "forget", taskId: 1 },
      state({ request: request(1, 4), payload: null, pending: [staged2] })],
    ["forget leaves a request with no snapshot as is", state({ request: request(1, 4), payload: stash1 }), { type: "forget", taskId: 1 },
      state({ request: request(1, 4), payload: null })],
    ["forget with nothing for that task", state({ request: request(2, 4, stash2), payload: stash2, pending: [staged2] }), { type: "forget", taskId: 1 }, "same"],
    ["identity change with nothing held", state(), { type: "identity-changed" }, "same"],
    ["identity change drops the request, the stash and every staged turn", state({ request: request(1, 4, keptSnap), payload: stash2, pending: [staged1] }), { type: "identity-changed" },
      state()],
  ]

  it.each(rows)("%s", (_name, prev, input, expected) => {
    const next = transitionRequest(prev, input)
    if (expected === "same") expect(next).toBe(prev)
    else expect(next).toEqual(expected)
  })

  it("keeps an unchanged request object when retaining its own task", () => {
    const prev = state({ request: request(1, 4, keptSnap), payload: stash2 })
    expect(transitionRequest(prev, { type: "retain", taskId: 1 }).request).toBe(prev.request)
  })
})

// Session checks, the per-task close and the dismissed-check mark, as data.
// Columns follow the current request: none, a turn failure (TF) or a
// session check (SO); the first-gate column does not exist yet.
describe("transitionRequest opens and closes requests by trigger", () => {
  const tf = (taskId: number, snap: ConnectorRuntimeResendPayload | null = null) => request(taskId, 4, snap)
  const rows: Array<[string, ConnectorRuntimeDialogState, RequestInput, ConnectorRuntimeDialogState | "same"]> = [
    // B0: a close from another task's dialog leaves every request alone.
    ["B0 none: close from task 2", state({ payload: stash1 }), closeAs("dismissed", 2), "same"],
    ["B0 TF: close from task 2 keeps task 1's turn failure", state({ request: tf(1, keptSnap), payload: stash1 }), closeAs("dismissed", 2), "same"],
    ["B0 SO: close from task 2 keeps task 1's session check", state({ request: sessionCheck(1) }), closeAs("not-shown", 2), "same"],
    // B1: a turn failure for the same task upgrades a session check.
    ["B1a SO: upgrade claims the stash", state({ request: sessionCheck(1), payload: stash1 }), open(1),
      state({ seq: 5, request: request(1, 5, stash1), payload: null })],
    ["B1a SO: upgrade claims the one staged turn", state({ request: sessionCheck(1), payload: stash1, pending: [staged1] }), open(1),
      state({ seq: 5, request: request(1, 5, staged1), payload: null, pending: [] })],
    ["B1b SO: another task's turn failure replaces it", state({ request: sessionCheck(2) }), open(1),
      state({ seq: 5, request: request(1, 5) })],
    // B2a: nothing new for a task that already has a request.
    ["B2a TF: opened", state({ request: tf(1, keptSnap) }), check(1, "opened"), "same"],
    ["B2a TF: reconnected", state({ request: tf(1, keptSnap) }), check(1, "reconnected"), "same"],
    ["B2a SO: opened", state({ request: sessionCheck(1) }), check(1, "opened"), "same"],
    // B2b: a session check touches neither the stash nor the tickets.
    ["B2b none: opened", state({ payload: stash2, pending: [staged2] }), check(1, "opened"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), payload: stash2, pending: [staged2] })],
    ["B2b TF: replaces another task's turn failure", state({ request: tf(2, stash2), payload: stash2 }), check(1, "opened"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), payload: stash2 })],
    ["B2b SO: replaces another task's session check", state({ request: sessionCheck(2) }), check(1, "reconnected"),
      state({ seq: 5, request: request(1, 5, null, "session_open") })],
    ["B2b none: a reconnect does not yield to this task's stash", state({ payload: stash1 }), check(1, "reconnected"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), payload: stash1 })],
    // B2c: a ticket for this task, for either cause.
    ["B2c none: opened yields to a ticket", state({ pending: [staged1] }), check(1, "opened"), "same"],
    ["B2c none: reconnected yields to a ticket", state({ pending: [staged1] }), check(1, "reconnected"), "same"],
    ["B2c TF: yields without replacing another task's request", state({ request: tf(2), pending: [staged1] }), check(1, "opened"), "same"],
    ["B2c SO: yields without replacing another task's request", state({ request: sessionCheck(2), pending: [staged1] }), check(1, "reconnected"), "same"],
    // B2d: a fresh view yields to this task's stash.
    ["B2d none: opened yields to the stash", state({ payload: stash1 }), check(1, "opened"), "same"],
    ["B2d SO: opened yields to the stash", state({ request: sessionCheck(2), payload: stash1 }), check(1, "opened"), "same"],
    // B2e: a reconnect stays quiet for a check the user closed this view.
    ["B2e none: reconnected after the user closed it", state({ dismissedCheck: 1 }), check(1, "reconnected"), "same"],
    ["B2e TF: reconnected after the user closed it", state({ request: tf(2), dismissedCheck: 1 }), check(1, "reconnected"), "same"],
    // B4-B7: closing a session check never drops the stash.
    ["B4 SO: dismissed keeps the stash and marks the task", state({ request: sessionCheck(1), payload: stash1 }), closeAs("dismissed"),
      state({ payload: stash1, dismissedCheck: 1 })],
    ["B5 SO: resent keeps the stash", state({ request: sessionCheck(1), payload: stash1 }), closeAs("resent"),
      state({ payload: stash1 })],
    ["B6 SO: not-shown keeps the stash, no mark", state({ request: sessionCheck(1), payload: stash1 }), closeAs("not-shown"),
      state({ payload: stash1 })],
    ["B7 SO: left-host keeps the stash, no mark", state({ request: sessionCheck(1), payload: stash1 }), closeAs("left-host"),
      state({ payload: stash1 })],
    ["B4 TF: dismissed drops the stash and marks nothing", state({ request: tf(1), payload: stash1 }), closeAs("dismissed"),
      state()],
    // B8-B10 for a session check.
    ["B8a SO: retain its own task", state({ request: sessionCheck(1) }), { type: "retain", taskId: 1 }, "same"],
    ["B8b SO: retain another task", state({ request: sessionCheck(1) }), { type: "retain", taskId: 2 }, state()],
    ["B9a SO: forget keeps the request object", state({ request: sessionCheck(1), payload: stash1 }), { type: "forget", taskId: 1 },
      state({ request: sessionCheck(1) })],
    ["B9b SO: forget another task", state({ request: sessionCheck(1) }), { type: "forget", taskId: 2 }, "same"],
    ["B10 SO: identity change", state({ request: sessionCheck(1) }), { type: "identity-changed" }, state()],
    // The dismissed-check mark: what clears it.
    ["mark: a fresh view clears it and checks again", state({ dismissedCheck: 1 }), check(1, "opened"),
      state({ seq: 5, request: request(1, 5, null, "session_open") })],
    ["mark: a fresh view of another task clears it", state({ dismissedCheck: 1, pending: [staged2] }), check(2, "opened"),
      state({ pending: [staged2] })],
    ["mark: a turn failure for that task clears it", state({ dismissedCheck: 1 }), open(1),
      state({ seq: 5, request: request(1, 5) })],
    ["mark: a turn failure for another task keeps it", state({ dismissedCheck: 1 }), open(2),
      state({ seq: 5, request: request(2, 5), dismissedCheck: 1 })],
    ["mark: switching away clears it", state({ dismissedCheck: 1 }), { type: "retain", taskId: 2 }, state()],
    ["mark: staying on that task keeps it", state({ dismissedCheck: 1 }), { type: "retain", taskId: 1 }, "same"],
    ["mark: an identity change clears it even with nothing else held", state({ dismissedCheck: 1 }), { type: "identity-changed" }, state()],
  ]

  it.each(rows)("%s", (_name, prev, input, expected) => {
    const next = transitionRequest(prev, input)
    if (expected === "same") expect(next).toBe(prev)
    else expect(next).toEqual(expected)
  })

  it("stays quiet on a reconnect only until the task is viewed afresh", () => {
    const steps: RequestInput[] = [check(1, "opened"), closeAs("dismissed"), check(1, "reconnected")]
    const afterReconnect = steps.reduce(transitionRequest, state())
    expect(afterReconnect.request).toBeNull()
    const reopened = transitionRequest(afterReconnect, check(1, "opened"))
    expect(reopened.request).toEqual(request(1, 6, null, "session_open"))
    const closedUnseen = transitionRequest(reopened, closeAs("not-shown"))
    expect(transitionRequest(closedUnseen, check(1, "reconnected")).request).toEqual(request(1, 7, null, "session_open"))
  })

  it("keeps a session check when another task's dialog closes in the same batch", () => {
    let actions: ConnectorRuntimeDialogActions | undefined
    let value: ConnectorRuntimeDialogValue | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActions()
      value = useConnectorRuntimeDialog()
      return null
    }
    render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    act(() => { actions?.openSessionCheck(1, "opened") })
    act(() => {
      actions?.openSessionCheck(2, "opened")
      actions?.close("not-shown", 1)
    })
    expect(value?.request).toMatchObject({ taskId: 2, trigger: "session_open" })
    act(() => {
      actions?.openSessionCheck(0, "opened")
      actions?.openSessionCheck(-3, "reconnected")
      actions?.openSessionCheck(1.5, "opened")
    })
    expect(value?.request).toMatchObject({ taskId: 2 })
  })

  it("gives openSessionCheck the same two no-provider defaults as every other action", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let silent: ConnectorRuntimeDialogActions | undefined
    let plain: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      silent = useConnectorRuntimeDialogActionsIfMounted()
      plain = useConnectorRuntimeDialogActions()
      return null
    }
    render(<Probe />)
    silent?.openSessionCheck(1, "opened")
    expect(warnSpy).not.toHaveBeenCalled()
    plain?.openSessionCheck(1, "opened")
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("openSessionCheck() was called outside"))
    warnSpy.mockRestore()
  })
})
