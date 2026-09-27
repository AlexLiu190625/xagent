import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { I18nProvider } from "@/contexts/i18n-context"
import type { AppState } from "@/contexts/app-context-chat"

// The task page's session-check trigger. Kept apart from page-client.test.tsx
// because that file's app-context mock does not expose `isConnected`, which
// every reconnect case here depends on.

const navigation = vi.hoisted(() => ({ params: { id: "1" } as { id: string } }))
vi.mock("next/navigation", () => ({
  useParams: () => navigation.params,
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="conversation-panel" />,
}))

vi.mock("@/contexts/auth-context", () => ({ useAuth: () => ({ user: { id: "u1" } }) }))

const app = vi.hoisted(() => ({
  taskId: 1 as number | null,
  isConnected: false,
  setTaskId: vi.fn(),
  closeFilePreview: vi.fn(),
}))
vi.mock("@/contexts/app-context-chat", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/contexts/app-context-chat")>()
  return {
    ...actual,
    useApp: () => ({
      state: { taskId: app.taskId, currentTask: null, dagExecution: null, steps: [] } as Partial<AppState>,
      isConnected: app.isConnected,
      setTaskId: app.setTaskId,
      closeFilePreview: app.closeFilePreview,
    }),
  }
})

import TaskDetailPage, {
  INITIAL_SESSION_CHECK_WATCH,
  nextSessionCheck,
  type SessionCheckWatch,
} from "./page-client"
import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialog,
  useConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogValue,
  type SessionCheckCause,
} from "@/contexts/connector-runtime-dialog-context"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  navigation.params = { id: "1" }
  app.taskId = 1
  app.isConnected = false
})

// One observation per effect run: [URL-and-state task (null when they
// disagree), the state's task, isConnected] and the check it must ask for.
type Step = [number | null, number | null, boolean, SessionCheckCause | null]

function replay(steps: Step[]): Array<SessionCheckCause | null> {
  let watch: SessionCheckWatch = INITIAL_SESSION_CHECK_WATCH
  return steps.map(([viewed, stateTaskId, isConnected]) => {
    const { next, check } = nextSessionCheck(watch, { viewed, stateTaskId, isConnected })
    watch = next
    return check
  })
}

describe("nextSessionCheck", () => {
  it.each<[string, Step[]]>([
    ["E1 mount before connecting: the first connect is skipped, a later one checks", [
      [5, 5, false, "opened"], [5, 5, true, null], [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
    ["E2 mount already connected, then a reconnect", [
      [5, 5, true, "opened"], [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
    ["E3 arrive from another task: its stale connected value is not this view's first connect", [
      [null, 3, true, null], [5, 5, true, "opened"], [5, 5, false, null], [5, 5, true, null],
      [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
    ["E4 soft navigation from one task to another", [
      [3, 3, true, "opened"], [3, 3, false, null], [3, 3, true, "reconnected"],
      [null, 3, true, null], [5, 5, true, "opened"], [5, 5, false, null], [5, 5, true, null],
      [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
    ["E5 back from the new-conversation page, which cleared the task", [
      [null, null, false, null], [3, 3, false, "opened"], [3, 3, true, null],
    ]],
    ["E6 a URL id that is not a number", [[null, 5, true, null], [null, 5, false, null], [null, 5, true, null]]],
    ["E7 no task in the app state", [[null, null, true, null], [null, null, false, null]]],
    ["E8 the same observation twice (strict mode)", [[5, 5, true, "opened"], [5, 5, true, null]]],
    ["E9 several failed reconnects, then one that works", [
      [5, 5, true, "opened"], [5, 5, false, null], [5, 5, false, null], [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
    ["E16 a switch whose disconnect and reconnect land in one render skips the next reconnect", [
      [3, 3, true, "opened"], [null, 3, true, null], [5, 5, true, "opened"], [5, 5, true, null],
      [5, 5, false, null], [5, 5, true, null], [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
    ["E19 the URL briefly disagrees, then shows the same task again: its connection is this view's own", [
      [5, 5, true, "opened"], [null, 5, true, null], [5, 5, true, "opened"], [5, 5, false, null], [5, 5, true, "reconnected"],
    ]],
  ])("%s", (_name, steps) => {
    expect(replay(steps)).toEqual(steps.map(step => step[3]))
  })
})

describe("the task page asks for a session check", () => {
  let actions: ConnectorRuntimeDialogActions | undefined
  let value: ConnectorRuntimeDialogValue | undefined
  function Probe() {
    actions = useConnectorRuntimeDialogActions()
    value = useConnectorRuntimeDialog()
    return null
  }
  function tree(page: boolean, strict = false) {
    const body = (
      <I18nProvider initialLocale="en">
        <ConnectorRuntimeDialogProvider>
          <Probe />
          {page && <TaskDetailPage />}
        </ConnectorRuntimeDialogProvider>
      </I18nProvider>
    )
    return strict ? <React.StrictMode>{body}</React.StrictMode> : body
  }
  // Mounts the provider first so the page reads the spied action.
  function mountPage(strict = false) {
    const view = render(tree(false, strict))
    const spy = vi.spyOn(actions as ConnectorRuntimeDialogActions, "openSessionCheck")
    view.rerender(tree(true, strict))
    return { view, spy, update: () => view.rerender(tree(true, strict)) }
  }

  it("checks once, as opened, when it mounts on the viewed task", () => {
    const { spy } = mountPage()
    expect(spy.mock.calls).toEqual([[1, "opened"]])
    expect(value?.request).toMatchObject({ taskId: 1, trigger: "session_open", resendPayload: null })
  })

  it("does not check while the URL and the app state disagree", () => {
    navigation.params = { id: "2" }
    const { spy, update } = mountPage()
    expect(spy).not.toHaveBeenCalled()
    app.taskId = 2
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[2, "opened"]])
  })

  it("checks once more after a soft navigation to another task", () => {
    const { spy, update } = mountPage()
    navigation.params = { id: "2" }
    act(() => { update() })
    app.taskId = 2
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[1, "opened"], [2, "opened"]])
  })

  it("checks once when the app state fills in the task after mount", () => {
    navigation.params = { id: "3" }
    app.taskId = null
    const { spy, update } = mountPage()
    app.taskId = 3
    act(() => { update() })
    app.isConnected = true
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[3, "opened"]])
  })

  it("checks again, as reconnected, when the same view's connection comes back", () => {
    app.isConnected = true
    const { spy, update } = mountPage()
    app.isConnected = false
    act(() => { update() })
    app.isConnected = true
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[1, "opened"], [1, "reconnected"]])
  })

  it("checks once under StrictMode", () => {
    const { spy } = mountPage(true)
    expect(spy.mock.calls).toEqual([[1, "opened"]])
  })
})
