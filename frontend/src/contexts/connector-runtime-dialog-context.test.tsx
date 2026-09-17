import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: { id: "u1" } }),
}))

import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialogActions,
  useConnectorRuntimeDialogActionsIfMounted,
  type ConnectorRuntimeDialogActions,
} from "./connector-runtime-dialog-context"

afterEach(() => {
  cleanup()
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
      actions?.close("dismissed")
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
