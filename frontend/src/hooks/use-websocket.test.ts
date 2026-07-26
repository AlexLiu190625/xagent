import { useLayoutEffect } from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  type WebSocketConnection,
  useWebSocket,
} from "./use-websocket"
import { refreshStoredAccessToken } from "@/lib/api-wrapper"
import { readAuthCache, writeAuthCache } from "@/lib/auth-cache"

const authState = vi.hoisted(() => ({
  user: { id: "user-1" } as { id: string } | null,
  token: "token" as string | null,
  refreshToken: "refresh-token" as string | null,
  refreshAccessToken: vi.fn<
    (
      expectedAccessToken?: string | null,
      expectedUserId?: string | null,
    ) => Promise<boolean>
  >(),
}))
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authState,
}))

class MockWebSocket {
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []
  static constructorError: Error | null = null

  readyState = 0
  protocol = ""
  onopen: (() => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  send = vi.fn()
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED
  })

  constructor(
    public url: string,
    public protocols?: string | string[],
  ) {
    if (MockWebSocket.constructorError) throw MockWebSocket.constructorError
    MockWebSocket.instances.push(this)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  receive(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
  }

  triggerError() {
    this.onerror?.(new Event("error"))
  }

  triggerClose(code = 1006, reason = "network lost") {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.({ code, reason } as CloseEvent)
  }
}

const sessionConnection = (
  overrides: Partial<WebSocketConnection> = {},
): WebSocketConnection => ({
  identity: "widget-session:1",
  url: "wss://embed.example/v1/external/chat/sessions/ws",
  protocols: ["xagent-session-v1", "xagent-session-token.st_secret"],
  expectedProtocol: "xagent-session-v1",
  chatTaskIdMode: "omit",
  credentialOwner: { kind: "external" },
  ...overrides,
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

describe("useWebSocket message delivery", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.constructorError = null
    localStorage.clear()
    authState.user = { id: "user-1" }
    authState.token = "token"
    authState.refreshToken = "refresh-token"
    authState.refreshAccessToken.mockReset()
    vi.stubGlobal("WebSocket", MockWebSocket)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("rejects without clearing the caller when the socket is not open", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      autoConnect: false,
    }))

    await expect(result.current.sendChatMessage("keep this draft")).rejects.toThrow(
      "connection is not ready",
    )
  })

  it("resolves only after the server accepts the durable message", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    let delivery!: Promise<{ client_message_id: string; turn_id: string }>
    act(() => {
      delivery = result.current.sendChatMessage("durable guidance")
    })
    expect(socket.send).toHaveBeenCalledOnce()
    const sent = JSON.parse(socket.send.mock.calls[0][0])
    expect(sent.client_message_id).toBeTruthy()

    let settled = false
    void delivery.finally(() => {
      settled = true
    })
    await Promise.resolve()
    expect(settled).toBe(false)

    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: sent.client_message_id,
        turn_id: sent.client_message_id,
      })
    })

    await expect(delivery).resolves.toEqual({
      client_message_id: sent.client_message_id,
      turn_id: sent.client_message_id,
    })
  })

  it("assigns idempotency keys to pause and resume commands", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    act(() => {
      result.current.pauseTask()
      result.current.resumeTask()
    })

    const pause = JSON.parse(socket.send.mock.calls[0][0])
    const resume = JSON.parse(socket.send.mock.calls[1][0])
    expect(pause.command_id).toBeTruthy()
    expect(resume.command_id).toBeTruthy()
    expect(resume.command_id).not.toBe(pause.command_id)
  })

  it("allows an unacknowledged draft to retry with the same id", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const first = result.current.sendChatMessage(
      "retry me",
      undefined,
      false,
      "stable-turn-1",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "stable-turn-1",
        message: "temporary failure",
      })
    })
    await expect(first).rejects.toThrow("temporary failure")

    const retry = result.current.sendChatMessage(
      "retry me",
      undefined,
      false,
      "stable-turn-1",
    )
    expect(socket.send).toHaveBeenCalledTimes(2)
    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: "stable-turn-1",
        turn_id: "stable-turn-1",
      })
    })
    await expect(retry).resolves.toEqual({
      client_message_id: "stable-turn-1",
      turn_id: "stable-turn-1",
    })
  })

  it("rejects concurrent reuse of a pending client message id without replacing its owner", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const first = result.current.sendChatMessage(
      "first owner",
      undefined,
      false,
      "shared-pending-id",
    )
    const second = result.current.sendChatMessage(
      "second owner",
      undefined,
      false,
      "shared-pending-id",
    )
    await expect(second).rejects.toThrow("already pending")
    expect(socket.send).toHaveBeenCalledOnce()

    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "shared-pending-id",
    }))
    await expect(first).resolves.toEqual({
      client_message_id: "shared-pending-id",
      turn_id: "shared-pending-id",
    })
  })

  it("marks definitive rejections so the composer can use a fresh id", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "retry with a new id",
      undefined,
      false,
      "failed-turn-1",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "failed-turn-1",
        message: "previous delivery failed",
        retry_with_new_id: true,
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "previous delivery failed",
      retryWithNewId: true,
    })
  })

  it("allows the same text to be sent again after the first ack", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const first = result.current.sendChatMessage("ok")
    const firstPayload = JSON.parse(socket.send.mock.calls[0][0])
    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: firstPayload.client_message_id,
      })
    })
    await first

    const second = result.current.sendChatMessage("ok")
    expect(socket.send).toHaveBeenCalledTimes(2)
    const secondPayload = JSON.parse(socket.send.mock.calls[1][0])
    expect(secondPayload.client_message_id).not.toBe(firstPayload.client_message_id)
    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: secondPayload.client_message_id,
      })
    })
    await second
  })

  it("rejects a pending delivery when the socket closes", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage("keep after disconnect")
    act(() => socket.triggerClose())

    await expect(delivery).rejects.toThrow("Connection closed")
  })

  it("rejects an unacknowledged delivery after 30 seconds", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    vi.useFakeTimers()

    try {
      const delivery = result.current.sendChatMessage("timeout draft")
      const rejection = expect(delivery).rejects.toThrow("not acknowledged")
      await act(async () => {
        vi.advanceTimersByTime(30000)
      })
      await rejection
    } finally {
      vi.useRealTimers()
    }
  })
})

describe("useWebSocket normalized connections", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.constructorError = null
    localStorage.clear()
    authState.user = { id: "user-1" }
    authState.token = "token"
    authState.refreshToken = "refresh-token"
    authState.refreshAccessToken.mockReset()
    vi.stubGlobal("WebSocket", MockWebSocket)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("normalizes an undefined connection through the unchanged legacy URL", async () => {
    renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
      connection: undefined,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    expect(MockWebSocket.instances[0].url).toBe("ws://localhost/ws/chat/7?token=token")
    expect(MockWebSocket.instances[0].protocols).toBeUndefined()
  })

  it("keeps the task id in the legacy chat frame", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage("legacy", undefined, false, "legacy-turn")
    expect(JSON.parse(socket.send.mock.calls[0][0])).toEqual({
      type: "chat",
      message: "legacy",
      task_id: 7,
      client_message_id: "legacy-turn",
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "legacy-turn",
    }))
    await delivery
  })

  it("treats an explicit null connection as disabled even when legacy inputs exist", async () => {
    renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
      connection: null,
    }))

    await Promise.resolve()
    expect(MockWebSocket.instances).toHaveLength(0)
  })

  it("constructs the exact Session URL and subprotocols and requires the server echo", async () => {
    const onConnect = vi.fn()
    const connection = sessionConnection()
    const { result } = renderHook(() => useWebSocket({ connection, onConnect }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    expect(socket.url).toBe("wss://embed.example/v1/external/chat/sessions/ws")
    expect(socket.url).not.toContain("st_secret")
    expect(socket.protocols).toEqual([
      "xagent-session-v1",
      "xagent-session-token.st_secret",
    ])

    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    expect(result.current.isConnected).toBe(true)
    expect(onConnect).toHaveBeenCalledOnce()
  })

  it("sanitizes a WebSocket constructor failure before logging or exposing it", async () => {
    const secret = "xagent-session-token.st_constructor_secret"
    MockWebSocket.constructorError = new Error(`constructor rejected ${secret}`)
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    const onError = vi.fn()

    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection({
        protocols: ["xagent-session-v1", secret],
      }),
      onError,
    }))

    await waitFor(() => expect(result.current.connectionError).not.toBeNull())
    expect(onError).toHaveBeenCalledOnce()
    const exposed = [
      result.current.connectionError?.message,
      ...onError.mock.calls.map(([error]) => (error as Error).message),
      ...consoleError.mock.calls.flat().map(String),
    ].join(" ")
    expect(exposed).not.toContain(secret)
    expect(result.current.connectionError?.message).toBe(
      "Failed to create WebSocket connection.",
    )
  })

  it("fails closed when the server omits the required Session subprotocol echo", async () => {
    const onConnect = vi.fn()
    const onError = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onConnect,
      onError,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    expect(result.current.isConnected).toBe(false)
    expect(onConnect).not.toHaveBeenCalled()
    expect(onError).toHaveBeenCalledOnce()
    expect(socket.close).toHaveBeenCalled()
  })

  it("sends and acknowledges taskless Session chat without a task id", async () => {
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "create lazily",
      undefined,
      false,
      "session-turn-1",
    )
    const sent = JSON.parse(socket.send.mock.calls[0][0])
    expect(sent).toEqual({
      type: "chat",
      message: "create lazily",
      client_message_id: "session-turn-1",
    })

    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "session-turn-1",
      turn_id: "server-turn-1",
    }))
    await expect(delivery).resolves.toEqual({
      client_message_id: "session-turn-1",
      turn_id: "server-turn-1",
    })
  })

  it.each([4001, 4003, 1011])(
    "runs the Session close delegate first and suppresses legacy handling for %s",
    async (code) => {
      vi.useFakeTimers()
      authState.refreshAccessToken.mockResolvedValue(true)
      const order: string[] = []
      const onError = vi.fn()
      const { result } = renderHook(() => useWebSocket({
        connection: sessionConnection(),
        onConnectionClose: () => {
          order.push("delegate")
          return "handled"
        },
        onDisconnect: () => {
          order.push("disconnect")
        },
        onError,
      }))
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      const socket = MockWebSocket.instances[0]
      socket.protocol = "xagent-session-v1"
      act(() => socket.open())
      const delivery = result.current.sendChatMessage("pending")
      const rejection = expect(delivery).rejects.toThrow("Connection closed")

      act(() => socket.triggerClose(code))
      await rejection
      await act(async () => {
        await vi.runAllTimersAsync()
      })

      expect(order).toEqual(["delegate"])
      expect(result.current.isConnected).toBe(false)
      expect(authState.refreshAccessToken).not.toHaveBeenCalled()
      expect(onError).not.toHaveBeenCalled()
      expect(MockWebSocket.instances).toHaveLength(1)
    },
  )

  it("fails closed and sanitizes when the Session close delegate throws", async () => {
    const secret = "xagent-session-token.st_close_delegate_secret"
    authState.refreshAccessToken.mockResolvedValue(true)
    const onError = vi.fn()
    const onDisconnect = vi.fn()
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onConnectionClose: () => {
        throw new Error(`close delegate leaked ${secret}`)
      },
      onDisconnect,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    vi.useFakeTimers()
    const delivery = result.current.sendChatMessage(
      "pending",
      undefined,
      false,
      "close-handler-pending",
    )
    const deliveryOutcome = delivery.then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    let escaped: unknown
    try {
      act(() => socket.triggerClose(4001))
    } catch (error) {
      escaped = error
    }

    expect(escaped).toBeUndefined()
    expect(await deliveryOutcome).toContain("Connection closed")
    expect(result.current.isConnected).toBe(false)
    expect(result.current.connectionError).not.toBeNull()
    expect(onError).toHaveBeenCalledOnce()
    expect(onDisconnect).not.toHaveBeenCalled()
    expect(authState.refreshAccessToken).not.toHaveBeenCalled()
    await act(async () => {
      await vi.runAllTimersAsync()
    })
    expect(MockWebSocket.instances).toHaveLength(1)
    const exposed = [
      result.current.connectionError?.message,
      ...onError.mock.calls.map(([error]) => (error as Error).message),
      ...consoleError.mock.calls.flat().map(String),
    ].join(" ")
    expect(exposed).not.toContain(secret)
  })

  it("keeps legacy auth refresh, retry, pause, resume, and status behavior", async () => {
    authState.refreshAccessToken.mockResolvedValue(false)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 9,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const first = MockWebSocket.instances[0]
    act(() => first.open())

    act(() => {
      result.current.pauseTask()
      result.current.resumeTask()
      result.current.requestStatus()
    })
    expect(first.send.mock.calls.map(([frame]) => JSON.parse(frame))).toEqual([
      expect.objectContaining({ type: "pause_task", task_id: 9 }),
      expect.objectContaining({ type: "resume_task", task_id: 9 }),
      { type: "status_request", task_id: 9 },
    ])

    act(() => first.triggerClose(4001))
    expect(authState.refreshAccessToken).toHaveBeenCalledOnce()
    expect(authState.refreshAccessToken).toHaveBeenCalledWith("token", "user-1")

    const retryHook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 10,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const retrySocket = MockWebSocket.instances[1]
    act(() => retrySocket.open())
    vi.useFakeTimers()
    act(() => retrySocket.triggerClose(1011))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(MockWebSocket.instances).toHaveLength(3)
    retryHook.unmount()
  })

  it.each([
    ["unmount", "resolve"],
    ["unmount", "reject"],
    ["disconnect", "resolve"],
    ["disconnect", "reject"],
    ["replacement", "resolve"],
    ["replacement", "reject"],
  ] as const)(
    "ignores late auth work after %s when refresh will %s",
    async (lifecycle, settlement) => {
      const refresh = deferred<boolean>()
      authState.refreshAccessToken.mockReturnValue(refresh.promise)
      const onError = vi.fn()
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
      const hook = renderHook(
        ({ taskId }) => useWebSocket({
          url: "ws://localhost",
          taskId,
          onError,
        }),
        { initialProps: { taskId: 1 } },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const oldSocket = MockWebSocket.instances[0]
      act(() => oldSocket.open())
      vi.useFakeTimers()

      act(() => oldSocket.triggerClose(4001))
      expect(authState.refreshAccessToken).toHaveBeenCalledOnce()
      expect(authState.refreshAccessToken).toHaveBeenCalledWith("token", "user-1")

      let expectedSocketCount = 1
      if (lifecycle === "unmount") {
        hook.unmount()
      } else if (lifecycle === "disconnect") {
        act(() => hook.result.current.disconnect())
      } else {
        hook.rerender({ taskId: 2 })
        expectedSocketCount = 2
        const replacementSocket = MockWebSocket.instances[1]
        act(() => replacementSocket.open())
      }

      await act(async () => {
        if (settlement === "resolve") {
          refresh.resolve(true)
        } else {
          refresh.reject(
            new Error("refresh failed xagent-session-token.st_refresh_secret"),
          )
        }
        await Promise.resolve()
      })
      await act(async () => {
        await vi.runAllTimersAsync()
      })

      expect(MockWebSocket.instances).toHaveLength(expectedSocketCount)
      expect(onError).not.toHaveBeenCalled()
      expect(consoleError.mock.calls.flat().join(" ")).not.toContain(
        "st_refresh_secret",
      )
      if (lifecycle !== "unmount") hook.unmount()
    },
  )

  it.each([
    ["an explicit legacy token", {
      url: "ws://localhost",
      taskId: 1,
      token: "token",
    }, "ws://localhost/ws/chat/1?token=token"],
    ["an explicit empty legacy token", {
      url: "ws://localhost",
      taskId: 1,
      token: "",
    }, "ws://localhost/ws/chat/1"],
    ["an explicit Session descriptor", {
      connection: sessionConnection(),
    }, "wss://embed.example/v1/external/chat/sessions/ws"],
  ] as const)(
    "never refreshes AuthContext credentials for %s",
    async (_name, options, expectedUrl) => {
      vi.useFakeTimers()
      authState.refreshAccessToken.mockResolvedValue(true)
      const onError = vi.fn()
      const hook = renderHook(() => useWebSocket({ ...options, onError }))
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      const socket = MockWebSocket.instances[0]
      expect(socket.url).toBe(expectedUrl)
      if ("connection" in options) socket.protocol = "xagent-session-v1"
      act(() => socket.open())

      act(() => socket.triggerClose(4001))
      await act(async () => {
        await vi.runAllTimersAsync()
      })

      expect(authState.refreshAccessToken).not.toHaveBeenCalled()
      expect(onError).toHaveBeenCalledOnce()
      expect((onError.mock.calls[0][0] as Error).message).toBe("Authentication failed")
      expect(MockWebSocket.instances).toHaveLength(1)
      hook.unmount()
    },
  )

  it("reconnects an auth-owned descriptor with the refreshed token only once", async () => {
    const refresh = deferred<boolean>()
    authState.token = "old-auth-token"
    authState.refreshAccessToken.mockReturnValue(refresh.promise)
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    expect(oldSocket.url).toBe("ws://localhost/ws/chat/1?token=old-auth-token")
    act(() => oldSocket.open())
    vi.useFakeTimers()

    act(() => oldSocket.triggerClose(4001))
    expect(authState.refreshAccessToken).toHaveBeenCalledWith(
      "old-auth-token",
      "user-1",
    )
    authState.token = "new-auth-token"
    hook.rerender()
    expect(MockWebSocket.instances).toHaveLength(2)
    const refreshedSocket = MockWebSocket.instances[1]
    expect(refreshedSocket.url).toBe("ws://localhost/ws/chat/1?token=new-auth-token")
    act(() => refreshedSocket.open())

    await act(async () => {
      refresh.resolve(true)
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(MockWebSocket.instances).toHaveLength(2)
    hook.unmount()
  })

  it("does not reuse a replacement user's token for an uncommitted old descriptor", async () => {
    const aliceUser = {
      id: "user-a",
      username: "alice",
      email: null,
      is_admin: false,
    }
    const bobUser = {
      id: "user-b",
      username: "bob",
      email: null,
      is_admin: false,
    }
    writeAuthCache(aliceUser, "user-a-token", "user-a-refresh", 120, 240)
    authState.user = { id: aliceUser.id }
    authState.token = "user-a-token"
    authState.refreshAccessToken.mockImplementation(
      async (expectedAccessToken, expectedUserId) => {
        const result = await refreshStoredAccessToken(
          expectedAccessToken,
          expectedUserId,
        )
        return result.accessToken !== null
      },
    )
    const onError = vi.fn()
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    vi.useFakeTimers()

    writeAuthCache(bobUser, "user-b-token", "user-b-refresh", 120, 240)
    act(() => oldSocket.triggerClose(4001))
    await act(async () => {
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(authState.refreshAccessToken).toHaveBeenCalledWith(
      "user-a-token",
      "user-a",
    )
    expect(readAuthCache()).toMatchObject({
      token: "user-b-token",
      user: { id: "user-b" },
    })
    expect(onError).toHaveBeenCalledOnce()
    expect(MockWebSocket.instances).toHaveLength(1)
    hook.unmount()
  })

  it("retires a closing owner before a same-descriptor replacement claims its id", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn(() => oldUpload.promise)
    const onDisconnect = vi.fn()
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
      onDisconnect,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    let oldSettled = false
    const oldOutcome = hook.result.current.sendChatMessage(
      "old upload",
      [new File(["old"], "old.txt")],
      false,
      "same-attempt-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    oldSocket.readyState = MockWebSocket.CLOSING
    act(() => hook.result.current.connect())
    expect(MockWebSocket.instances).toHaveLength(2)
    const replacementSocket = MockWebSocket.instances[1]
    act(() => replacementSocket.open())
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled
    const replacementDelivery = hook.result.current.sendChatMessage(
      "replacement",
      undefined,
      false,
      "same-attempt-id",
    )
    const replacementOutcome = replacementDelivery.then(
      ack => ack,
      error => `rejected:${(error as Error).message}`,
    )

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-upload" }])
      await Promise.resolve()
    })
    act(() => {
      oldSocket.triggerClose(1000, "late old close")
      replacementSocket.receive({
        type: "message_accepted",
        client_message_id: "same-attempt-id",
      })
    })

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("replaced")
    await expect(replacementOutcome).resolves.toMatchObject({
      client_message_id: "same-attempt-id",
    })
    expect(oldSocket.close).toHaveBeenCalledOnce()
    expect(onDisconnect).toHaveBeenCalledOnce()
    hook.unmount()
  })

  it("does not preserve a closing owner when its replacement constructor fails", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn(() => oldUpload.promise)
    const onDisconnect = vi.fn()
    const onError = vi.fn()
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
      onDisconnect,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    let oldSettled = false
    const oldOutcome = hook.result.current.sendChatMessage(
      "old upload",
      [new File(["old"], "old.txt")],
      false,
      "constructor-failure-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    oldSocket.readyState = MockWebSocket.CLOSING
    MockWebSocket.constructorError = new Error("constructor secret")
    act(() => hook.result.current.connect())
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled
    MockWebSocket.constructorError = null
    act(() => hook.result.current.connect())
    expect(MockWebSocket.instances).toHaveLength(2)
    const replacementSocket = MockWebSocket.instances[1]
    act(() => replacementSocket.open())

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-upload" }])
      await Promise.resolve()
    })
    act(() => oldSocket.triggerClose(1000, "late old close"))

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("replaced")
    expect(oldSocket.close).toHaveBeenCalledOnce()
    expect(onDisconnect).toHaveBeenCalledOnce()
    expect(onError).toHaveBeenCalledOnce()
    expect(consoleError.mock.calls.flat().join(" ")).not.toContain("constructor secret")
    expect(hook.result.current.isConnected).toBe(true)
    hook.unmount()
  })

  it.each([
    ["failed", false],
    ["successful", true],
  ] as const)(
    "fences a stale %s auth refresh after a same-descriptor manual connect",
    async (_settlement, refreshSucceeded) => {
      const refresh = deferred<boolean>()
      authState.refreshAccessToken.mockReturnValue(refresh.promise)
      const onError = vi.fn()
      const hook = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        onError,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const oldSocket = MockWebSocket.instances[0]
      act(() => oldSocket.open())
      vi.useFakeTimers()

      act(() => oldSocket.triggerClose(4001))
      expect(authState.refreshAccessToken).toHaveBeenCalledOnce()
      expect(authState.refreshAccessToken).toHaveBeenCalledWith("token", "user-1")
      act(() => hook.result.current.connect())
      expect(MockWebSocket.instances).toHaveLength(2)
      const currentSocket = MockWebSocket.instances[1]
      act(() => currentSocket.open())
      if (refreshSucceeded) {
        act(() => currentSocket.triggerClose(1000, "current socket closed cleanly"))
      }

      await act(async () => {
        refresh.resolve(refreshSucceeded)
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1000)
      })

      expect(onError).not.toHaveBeenCalled()
      expect(MockWebSocket.instances).toHaveLength(2)
      hook.unmount()
    },
  )

  it.each([
    "default close",
    "explicit disconnect",
    "descriptor replacement",
    "unmount",
  ] as const)(
    "notifies the socket-owning disconnect callback exactly once on %s",
    async (operation) => {
      const owningDisconnect = vi.fn()
      const latestDisconnect = vi.fn()
      const initialConnection = sessionConnection()
      const hook = renderHook(
        ({ connection, onDisconnect }) => useWebSocket({
          connection,
          onDisconnect,
        }),
        {
          initialProps: {
            connection: initialConnection,
            onDisconnect: owningDisconnect,
          },
        },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const ownedSocket = MockWebSocket.instances[0]
      ownedSocket.protocol = "xagent-session-v1"
      act(() => ownedSocket.open())

      if (operation === "descriptor replacement") {
        hook.rerender({
          connection: sessionConnection({
            url: "wss://embed.example/v1/external/chat/sessions/replacement/ws",
          }),
          onDisconnect: latestDisconnect,
        })
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      } else {
        hook.rerender({
          connection: sessionConnection(),
          onDisconnect: latestDisconnect,
        })
        expect(MockWebSocket.instances).toHaveLength(1)
        if (operation === "default close") {
          act(() => ownedSocket.triggerClose(1006))
        } else if (operation === "explicit disconnect") {
          act(() => hook.result.current.disconnect())
        } else {
          hook.unmount()
        }
      }

      act(() => ownedSocket.triggerClose(1011))
      expect(owningDisconnect).toHaveBeenCalledOnce()
      expect(latestDisconnect).not.toHaveBeenCalled()
      if (operation !== "unmount") hook.unmount()
    },
  )

  it.each([
    "default close",
    "explicit disconnect",
    "descriptor replacement",
    "unmount",
  ] as const)(
    "finishes owner retirement when the disconnect callback throws on %s",
    async (operation) => {
      const secret = "disconnect callback raw secret"
      const oldUpload = deferred<Array<{ file_id: string }>>()
      const uploadFiles = vi.fn(() => oldUpload.promise)
      const throwingDisconnect = vi.fn(() => {
        throw new Error(secret)
      })
      const safeDisconnect = vi.fn()
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
      const hook = renderHook(
        ({ taskId, onDisconnect }) => useWebSocket({
          url: "ws://localhost",
          taskId,
          uploadFiles,
          onDisconnect,
        }),
        {
          initialProps: {
            taskId: 1,
            onDisconnect: throwingDisconnect,
          },
        },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const oldSocket = MockWebSocket.instances[0]
      act(() => oldSocket.open())
      const pendingOutcome = hook.result.current.sendChatMessage(
        "pending",
        undefined,
        false,
        "throwing-disconnect-pending",
      ).then(
        () => "resolved",
        error => `rejected:${(error as Error).message}`,
      )
      const preparationOutcome = hook.result.current.sendChatMessage(
        "preparing",
        [new File(["data"], "data.txt")],
        false,
        "throwing-disconnect-preparation",
      ).then(
        () => "resolved",
        error => `rejected:${(error as Error).message}`,
      )

      let escaped: unknown
      try {
        if (operation === "default close") {
          act(() => oldSocket.triggerClose(1006))
        } else if (operation === "explicit disconnect") {
          act(() => hook.result.current.disconnect())
        } else if (operation === "descriptor replacement") {
          hook.rerender({
            taskId: 2,
            onDisconnect: safeDisconnect,
          })
        } else {
          hook.unmount()
        }
      } catch (error) {
        escaped = error
      }
      await act(async () => {
        await Promise.resolve()
      })
      const stateWasDisconnected = operation === "unmount"
        ? true
        : hook.result.current.isConnected === false

      await act(async () => {
        oldUpload.resolve([{ file_id: "late-file" }])
        await Promise.resolve()
      })

      let replacementContinued = operation === "unmount"
      if (operation !== "unmount" && escaped === undefined) {
        if (operation !== "descriptor replacement") {
          hook.rerender({
            taskId: 1,
            onDisconnect: safeDisconnect,
          })
          act(() => hook.result.current.connect())
        }
        replacementContinued = MockWebSocket.instances.length === 2
        if (replacementContinued) {
          act(() => MockWebSocket.instances[1].open())
        }
        hook.unmount()
      } else if (operation !== "unmount") {
        hook.unmount()
      }

      expect(escaped).toBeUndefined()
      expect(stateWasDisconnected).toBe(true)
      expect(oldSocket.readyState).toBe(MockWebSocket.CLOSED)
      expect(await pendingOutcome).toContain("rejected:")
      expect(await preparationOutcome).toContain("rejected:")
      expect(throwingDisconnect).toHaveBeenCalledOnce()
      expect(replacementContinued).toBe(true)
      expect(consoleError).toHaveBeenCalledWith(
        "WebSocket disconnect handler failed",
      )
      const exposed = [
        String(escaped),
        ...consoleError.mock.calls.flat().map(String),
      ].join(" ")
      expect(exposed).not.toContain(secret)
    },
  )

  it("does not reconnect for a new descriptor object with identical values", async () => {
    const hook = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    hook.rerender({ connection: sessionConnection() })
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it("makes stale open, message, error, and close callbacks inert after replacement", async () => {
    const onConnect = vi.fn()
    const onMessage = vi.fn()
    const onError = vi.fn()
    const onDisconnect = vi.fn()
    const onConnectionClose = vi.fn(() => "handled" as const)
    const { result, rerender } = renderHook(
      ({ connection }) => useWebSocket({
        connection,
        onConnect,
        onMessage,
        onError,
        onDisconnect,
        onConnectionClose,
      }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]

    rerender({ connection: sessionConnection({ identity: "widget-session:2" }) })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const currentSocket = MockWebSocket.instances[1]
    currentSocket.protocol = "xagent-session-v1"
    act(() => currentSocket.open())
    onConnect.mockClear()
    onMessage.mockClear()
    onError.mockClear()
    onDisconnect.mockClear()
    onConnectionClose.mockClear()

    oldSocket.protocol = "xagent-session-v1"
    act(() => {
      oldSocket.open()
      oldSocket.receive({ type: "task_info", task_id: 999 })
      oldSocket.triggerError()
      oldSocket.triggerClose(1011)
    })

    expect(result.current.isConnected).toBe(true)
    expect(result.current.lastMessage).toBeNull()
    expect(onConnect).not.toHaveBeenCalled()
    expect(onMessage).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()
    expect(onDisconnect).not.toHaveBeenCalled()
    expect(onConnectionClose).not.toHaveBeenCalled()
    expect(MockWebSocket.instances).toHaveLength(2)
  })

  it("rejects only old pending delivery on replacement and ignores its late ack", async () => {
    const { result, rerender } = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    oldSocket.protocol = "xagent-session-v1"
    act(() => oldSocket.open())
    const oldDelivery = result.current.sendChatMessage(
      "old",
      undefined,
      false,
      "shared-id",
    )
    const oldRejection = expect(oldDelivery).rejects.toThrow("replaced")

    rerender({ connection: sessionConnection({ identity: "widget-session:2" }) })
    await oldRejection
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const currentSocket = MockWebSocket.instances[1]
    currentSocket.protocol = "xagent-session-v1"
    act(() => currentSocket.open())
    const currentDelivery = result.current.sendChatMessage(
      "new",
      undefined,
      false,
      "shared-id",
    )
    let currentSettled = false
    void currentDelivery.finally(() => {
      currentSettled = true
    })

    act(() => {
      oldSocket.receive({
        type: "message_accepted",
        client_message_id: "shared-id",
      })
      oldSocket.triggerClose(1011)
    })
    await Promise.resolve()
    expect(currentSettled).toBe(false)

    act(() => currentSocket.receive({
      type: "message_accepted",
      client_message_id: "shared-id",
    }))
    await expect(currentDelivery).resolves.toMatchObject({
      client_message_id: "shared-id",
    })
  })

  it("invalidates pending and dedupe state on delivery generation without reconnecting", async () => {
    const connection = sessionConnection()
    const { result, rerender } = renderHook(
      ({ deliveryGeneration }) => useWebSocket({
        connection,
        deliveryGeneration,
      }),
      { initialProps: { deliveryGeneration: 0 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    const oldDelivery = result.current.sendChatMessage("same text")
    const oldRejection = expect(oldDelivery).rejects.toThrow("generation")

    rerender({ deliveryGeneration: 1 })
    await oldRejection
    expect(MockWebSocket.instances).toHaveLength(1)

    const replacementDelivery = result.current.sendChatMessage("same text")
    expect(socket.send).toHaveBeenCalledTimes(2)
    const replacementFrame = JSON.parse(socket.send.mock.calls[1][0])
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: replacementFrame.client_message_id,
    }))
    await expect(replacementDelivery).resolves.toMatchObject({
      client_message_id: replacementFrame.client_message_id,
    })
  })

  it("commits the generation fence before a consumer layout effect can acknowledge", async () => {
    const connection = sessionConnection()
    const { result, rerender } = renderHook(
      ({ acknowledgeOnCommit, deliveryGeneration }) => {
        const webSocket = useWebSocket({
          connection,
          deliveryGeneration,
        })
        useLayoutEffect(() => {
          if (!acknowledgeOnCommit) return
          MockWebSocket.instances[0]?.receive({
            type: "message_accepted",
            client_message_id: "layout-fence-id",
          })
        }, [acknowledgeOnCommit, deliveryGeneration])
        return webSocket
      },
      {
        initialProps: {
          acknowledgeOnCommit: false,
          deliveryGeneration: 0,
        },
      },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    const oldDelivery = result.current.sendChatMessage(
      "old generation",
      undefined,
      false,
      "layout-fence-id",
    )
    const outcome = oldDelivery.then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    rerender({
      acknowledgeOnCommit: true,
      deliveryGeneration: 1,
    })

    expect(await outcome).toContain("generation")
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it("keeps duplicate detection local to each hook", async () => {
    const first = renderHook(() => useWebSocket({
      connection: sessionConnection({ identity: "shared-identity" }),
    }))
    const second = renderHook(() => useWebSocket({
      connection: sessionConnection({ identity: "shared-identity" }),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const firstSocket = MockWebSocket.instances[0]
    const secondSocket = MockWebSocket.instances[1]
    firstSocket.protocol = "xagent-session-v1"
    secondSocket.protocol = "xagent-session-v1"
    act(() => {
      firstSocket.open()
      secondSocket.open()
    })

    const firstDelivery = first.result.current.sendChatMessage("same")
    const secondDelivery = second.result.current.sendChatMessage("same")
    expect(firstSocket.send).toHaveBeenCalledOnce()
    expect(secondSocket.send).toHaveBeenCalledOnce()

    const firstFrame = JSON.parse(firstSocket.send.mock.calls[0][0])
    const secondFrame = JSON.parse(secondSocket.send.mock.calls[0][0])
    act(() => {
      firstSocket.receive({
        type: "message_accepted",
        client_message_id: firstFrame.client_message_id,
      })
      secondSocket.receive({
        type: "message_accepted",
        client_message_id: secondFrame.client_message_id,
      })
    })
    await Promise.all([firstDelivery, secondDelivery])
  })

  it("uses the current socket for retained raw protocol sends", async () => {
    const { result, rerender } = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    oldSocket.protocol = "xagent-session-v1"
    act(() => oldSocket.open())
    const sendRaw = result.current.sendMessage

    rerender({ connection: sessionConnection({ identity: "widget-session:2" }) })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const currentSocket = MockWebSocket.instances[1]
    currentSocket.protocol = "xagent-session-v1"
    act(() => currentSocket.open())
    act(() => sendRaw({ type: "new_conversation" }))

    expect(oldSocket.send).not.toHaveBeenCalled()
    expect(currentSocket.send).toHaveBeenCalledWith(
      JSON.stringify({ type: "new_conversation" }),
    )
  })

  it("fails taskless file delivery before any upload", async () => {
    const uploadFiles = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      uploadFiles,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())

    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    await expect(result.current.sendChatMessage("with file", [file])).rejects.toThrow(
      "not supported",
    )
    expect(uploadFiles).not.toHaveBeenCalled()
    expect(socket.send).not.toHaveBeenCalled()
  })

  it("claims a client message id before awaiting its upload", async () => {
    const upload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn(() => upload.promise)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    const file = new File(["data"], "data.txt")

    const first = result.current.sendChatMessage(
      "first owner",
      [file],
      false,
      "shared-upload-id",
    )
    const second = result.current.sendChatMessage(
      "second owner",
      [file],
      false,
      "shared-upload-id",
    )
    const secondOutcome = second.then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    await act(async () => {
      upload.resolve([{ file_id: "uploaded-file" }])
      await Promise.resolve()
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "shared-upload-id",
    }))

    await expect(first).resolves.toMatchObject({
      client_message_id: "shared-upload-id",
    })
    expect(await secondOutcome).toContain("already pending")
    expect(uploadFiles).toHaveBeenCalledOnce()
    expect(socket.send).toHaveBeenCalledOnce()
  })

  it("releases an upload claim after preparation fails", async () => {
    const uploadFiles = vi.fn()
      .mockRejectedValueOnce(new Error("upload failed"))
      .mockResolvedValueOnce([{ file_id: "uploaded-file" }])
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    const file = new File(["data"], "data.txt")

    await expect(result.current.sendChatMessage(
      "first attempt",
      [file],
      false,
      "retry-upload-id",
    )).rejects.toThrow("upload failed")

    const retry = result.current.sendChatMessage(
      "retry",
      [file],
      false,
      "retry-upload-id",
    )
    await waitFor(() => expect(socket.send).toHaveBeenCalledOnce())
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "retry-upload-id",
    }))

    await expect(retry).resolves.toMatchObject({
      client_message_id: "retry-upload-id",
    })
    expect(uploadFiles).toHaveBeenCalledTimes(2)
  })

  it("rejects replaced upload preparation promptly and preserves the replacement claim", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const replacementUpload = deferred<Array<{ file_id: string }>>()
    const unexpectedThirdUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn()
      .mockReturnValueOnce(oldUpload.promise)
      .mockReturnValueOnce(replacementUpload.promise)
      .mockReturnValueOnce(unexpectedThirdUpload.promise)
    const { result, rerender } = renderHook(
      ({ taskId }) => useWebSocket({
        url: "ws://localhost",
        taskId,
        uploadFiles,
      }),
      { initialProps: { taskId: 1 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    const file = new File(["data"], "data.txt")
    let oldSettled = false
    const oldOutcome = result.current.sendChatMessage(
      "old owner",
      [file],
      false,
      "reclaimed-upload-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    rerender({ taskId: 2 })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const replacementSocket = MockWebSocket.instances[1]
    act(() => replacementSocket.open())
    const replacementDelivery = result.current.sendChatMessage(
      "replacement owner",
      [file],
      false,
      "reclaimed-upload-id",
    )
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-file" }])
      await Promise.resolve()
    })
    const duplicateOutcome = result.current.sendChatMessage(
      "must not steal replacement",
      [file],
      false,
      "reclaimed-upload-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    await act(async () => {
      replacementUpload.resolve([{ file_id: "replacement-file" }])
      unexpectedThirdUpload.resolve([{ file_id: "unexpected-file" }])
      await Promise.resolve()
    })
    act(() => replacementSocket.receive({
      type: "message_accepted",
      client_message_id: "reclaimed-upload-id",
    }))

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("connection changed")
    expect(await duplicateOutcome).toContain("already pending")
    expect(uploadFiles).toHaveBeenCalledTimes(2)
    expect(oldSocket.send).not.toHaveBeenCalled()
    expect(replacementSocket.send).toHaveBeenCalledOnce()
    await expect(replacementDelivery).resolves.toMatchObject({
      client_message_id: "reclaimed-upload-id",
    })
  })

  it("releases upload preparation on a delivery generation change", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const currentUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn()
      .mockReturnValueOnce(oldUpload.promise)
      .mockReturnValueOnce(currentUpload.promise)
    const { result, rerender } = renderHook(
      ({ deliveryGeneration }) => useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        deliveryGeneration,
        uploadFiles,
      }),
      { initialProps: { deliveryGeneration: 0 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    const file = new File(["data"], "data.txt")
    let oldSettled = false
    const oldOutcome = result.current.sendChatMessage(
      "old generation",
      [file],
      false,
      "generation-upload-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    rerender({ deliveryGeneration: 1 })
    const currentDelivery = result.current.sendChatMessage(
      "current generation",
      [file],
      false,
      "generation-upload-id",
    )
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-file" }])
      currentUpload.resolve([{ file_id: "current-file" }])
      await Promise.resolve()
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "generation-upload-id",
    }))

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("generation")
    expect(uploadFiles).toHaveBeenCalledTimes(2)
    await expect(currentDelivery).resolves.toMatchObject({
      client_message_id: "generation-upload-id",
    })
  })

  it("fails closed when an upload completes after its connection was replaced", async () => {
    let finishUpload!: (files: Array<{ file_id: string }>) => void
    const uploadFiles = vi.fn(() => new Promise<Array<{ file_id: string }>>((resolve) => {
      finishUpload = resolve
    }))
    const { result, rerender } = renderHook(
      ({ taskId }) => useWebSocket({
        url: "ws://localhost",
        taskId,
        uploadFiles,
      }),
      { initialProps: { taskId: 1 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    const delivery = result.current.sendChatMessage(
      "upload",
      [new File(["data"], "data.txt")],
    )
    const rejection = expect(delivery).rejects.toThrow("connection changed")
    await waitFor(() => expect(uploadFiles).toHaveBeenCalledOnce())

    rerender({ taskId: 2 })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    await act(async () => {
      finishUpload([{ file_id: "uploaded-file" }])
      await Promise.resolve()
    })

    expect(oldSocket.send).not.toHaveBeenCalled()
    await rejection
  })
})
