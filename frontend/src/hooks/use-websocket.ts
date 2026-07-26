"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { generateClientMessageId, getWsUrl, getUploadApiUrl } from "@/lib/utils"
import { isFinalAnswerStreamEventType } from "@/lib/streaming-final-answer"

interface RecentMessage {
  message: string
  timestamp: number
  connectionIdentity: string
  descriptorKey: string
  lifecycleEpoch: number
  attemptEpoch: number
  deliveryGeneration: number
  clientMessageId: string
}

const MESSAGE_DUPLICATE_THRESHOLD = 2000 // Same message within 2 seconds is considered duplicate

interface WebSocketMessage {
  type: string
  data: unknown
  timestamp: string
  task_id?: number
  step_id?: string
  event_id?: string
  event_type?: string
  message_id?: string
  delta?: string
  content?: string
  run_id?: string | null
  state_version?: number
  control_state?: "idle" | "running" | "pause_requested" | "paused" | "resume_requested" | "waiting_for_user" | "completed" | "failed"
  status?: unknown
  task?: Record<string, unknown>
}

interface MessageDeliveryAck {
  client_message_id: string
  turn_id: string
}

export type WebSocketCredentialOwner =
  | {
    kind: "auth-context"
    accessToken: string
    userId: string | null
  }
  | { kind: "external" }

export interface WebSocketConnection {
  identity: string
  url: string
  protocols?: string[]
  expectedProtocol?: string
  taskId?: number
  chatTaskIdMode: "required" | "omit"
  credentialOwner: WebSocketCredentialOwner
}

interface PendingDelivery {
  resolve: (ack: MessageDeliveryAck) => void
  reject: (error: Error) => void
  timeout: ReturnType<typeof setTimeout>
  connectionIdentity: string
  descriptorKey: string
  lifecycleEpoch: number
  attemptEpoch: number
  deliveryGeneration: number
  socket: WebSocket
}

interface MessagePreparationClaim {
  cancellation: Promise<never>
  cancel: (error: Error) => void
  cancelled: boolean
  connectionIdentity: string
  descriptorKey: string
  lifecycleEpoch: number
  attemptEpoch: number
  deliveryGeneration: number
  socket: WebSocket
}

interface WebSocketCallbacks {
  onConnectionClose?: (event: CloseEvent) => "handled" | "default"
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
  onMessage?: (message: WebSocketMessage) => void
}

interface SocketOwner {
  socket: WebSocket
  connection: WebSocketConnection
  descriptorKey: string
  lifecycleEpoch: number
  attemptEpoch: number
  callbacks: WebSocketCallbacks
  refreshAccessToken: (
    expectedAccessToken?: string | null,
    expectedUserId?: string | null,
  ) => Promise<boolean>
  disconnectNotified: boolean
}

interface OwnerRetirementOptions {
  pendingError: Error
  preparationError: Error
  close?: { code?: number; reason?: string }
  notifyDisconnect: boolean
}

export interface UseWebSocketOptions {
  url?: string
  taskId?: number
  token?: string
  buildWebSocketUrl?: (params: { baseUrl: string; taskId: number; token?: string }) => string
  uploadFiles?: (files: File[], params: { taskId?: number | null; taskType: string }) => Promise<Array<{ file_id: string; name?: string; size?: number; type?: string }>>
  connection?: WebSocketConnection | null
  deliveryGeneration?: number
  onConnectionClose?: (event: CloseEvent) => "handled" | "default"
  autoConnect?: boolean
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
}

const useIsomorphicLayoutEffect = typeof window === "undefined"
  ? useEffect
  : useLayoutEffect

const getConnectionDescriptorKey = (
  connection: WebSocketConnection | null,
): string | null => connection
  ? JSON.stringify({
    chatTaskIdMode: connection.chatTaskIdMode,
    credentialOwnerKind: connection.credentialOwner.kind,
    credentialOwnerUserId: connection.credentialOwner.kind === "auth-context"
      ? connection.credentialOwner.userId
      : null,
    expectedProtocol: connection.expectedProtocol ?? null,
    identity: connection.identity,
    protocols: connection.protocols ?? null,
    taskId: connection.taskId ?? null,
    url: connection.url,
  })
  : null

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url,
    taskId,
    token,
    buildWebSocketUrl,
    uploadFiles,
    connection: connectionOption,
    deliveryGeneration = 0,
    onConnectionClose,
    autoConnect = true,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options

  const {
    token: authToken,
    user: authUser,
    refreshAccessToken,
  } = useAuth()

  const normalizedConnection = useMemo<WebSocketConnection | null>(() => {
    if (connectionOption !== undefined) return connectionOption
    if (!taskId) return null

    const baseUrl = url ?? getWsUrl()
    const hasExplicitToken = token !== undefined
    const effectiveToken = hasExplicitToken ? token : authToken || undefined
    return {
      identity: `legacy-task:${taskId}`,
      url: buildWebSocketUrl
        ? buildWebSocketUrl({
          baseUrl,
          taskId,
          token: effectiveToken,
        })
        : `${baseUrl}/ws/chat/${taskId}${effectiveToken ? `?token=${effectiveToken}` : ""}`,
      taskId,
      chatTaskIdMode: "required",
      credentialOwner: !hasExplicitToken && authToken
        ? {
          kind: "auth-context",
          accessToken: authToken,
          userId: authUser?.id ? String(authUser.id) : null,
        }
        : { kind: "external" },
    }
  }, [
    authToken,
    authUser?.id,
    buildWebSocketUrl,
    connectionOption,
    taskId,
    token,
    url,
  ])
  const connectionDescriptorKey = getConnectionDescriptorKey(normalizedConnection)

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionError, setConnectionError] = useState<Error | null>(null)
  const isConnectingRef = useRef(false)

  const socketRef = useRef<WebSocket | null>(null)
  const socketOwnerRef = useRef<SocketOwner | null>(null)
  const connectionRef = useRef<WebSocketConnection | null>(normalizedConnection)
  const descriptorKeyRef = useRef<string | null>(connectionDescriptorKey)
  const retryTimersRef = useRef(new Set<ReturnType<typeof setTimeout>>())
  const reconnectAttemptsRef = useRef(0)
  const deliveryGenerationRef = useRef(deliveryGeneration)
  const deliveryIdentityRef = useRef(normalizedConnection?.identity ?? null)
  const lifecycleEpochRef = useRef(0)
  const attemptEpochRef = useRef(0)
  const mountedRef = useRef(false)
  const tokenRef = useRef(token !== undefined ? token : authToken)
  const pendingDeliveriesRef = useRef(new Map<string, PendingDelivery>())
  const preparationsRef = useRef(new Map<string, MessagePreparationClaim>())
  const recentMessagesRef = useRef<RecentMessage[]>([])
  const callbacksRef = useRef<WebSocketCallbacks>({
    onConnectionClose,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  })
  const refreshAccessTokenRef = useRef(refreshAccessToken)
  const maxReconnectAttempts = 3

  const rejectPendingDeliveries = useCallback((
    error: Error,
    matches: (pending: PendingDelivery) => boolean = () => true,
  ) => {
    for (const [clientMessageId, pending] of pendingDeliveriesRef.current) {
      if (!matches(pending)) continue
      clearTimeout(pending.timeout)
      pending.reject(error)
      pendingDeliveriesRef.current.delete(clientMessageId)
    }
  }, [])

  const rejectPreparations = useCallback((
    error: Error,
    matches: (claim: MessagePreparationClaim) => boolean = () => true,
  ) => {
    for (const [clientMessageId, claim] of preparationsRef.current) {
      if (!matches(claim)) continue
      if (preparationsRef.current.get(clientMessageId) !== claim) continue
      preparationsRef.current.delete(clientMessageId)
      claim.cancel(error)
    }
  }, [])

  const clearRecentMessages = useCallback((
    matches: (recent: RecentMessage) => boolean = () => true,
  ) => {
    recentMessagesRef.current = recentMessagesRef.current.filter(
      recent => !matches(recent),
    )
  }, [])

  const clearRetryTimers = useCallback(() => {
    for (const timer of retryTimersRef.current) clearTimeout(timer)
    retryTimersRef.current.clear()
  }, [])

  const invalidateLifecycle = useCallback(() => {
    lifecycleEpochRef.current++
    clearRetryTimers()
  }, [clearRetryTimers])

  const isCurrentLifecycle = useCallback((
    lifecycleEpoch: number,
    descriptorKey: string,
  ) => (
    mountedRef.current
    && lifecycleEpochRef.current === lifecycleEpoch
    && descriptorKeyRef.current === descriptorKey
  ), [])

  const isCurrentOwner = useCallback((owner: SocketOwner) => (
    isCurrentLifecycle(owner.lifecycleEpoch, owner.descriptorKey)
    && socketOwnerRef.current === owner
    && socketRef.current === owner.socket
  ), [isCurrentLifecycle])

  const isCurrentAttempt = useCallback((owner: SocketOwner) => (
    isCurrentLifecycle(owner.lifecycleEpoch, owner.descriptorKey)
    && attemptEpochRef.current === owner.attemptEpoch
  ), [isCurrentLifecycle])

  const isCurrentSocket = useCallback((socket: WebSocket, identity: string) => {
    const owner = socketOwnerRef.current
    return Boolean(
      owner
      && owner.socket === socket
      && owner.connection.identity === identity
      && isCurrentOwner(owner),
    )
  }, [isCurrentOwner])

  const notifyDisconnect = useCallback((owner: SocketOwner) => {
    if (owner.disconnectNotified) return
    owner.disconnectNotified = true
    try {
      owner.callbacks.onDisconnect?.()
    } catch {
      console.error("WebSocket disconnect handler failed")
    }
  }, [])

  const retireOwner = useCallback((
    owner: SocketOwner,
    options: OwnerRetirementOptions,
  ) => {
    const wasCurrent = (
      socketOwnerRef.current === owner
      && socketRef.current === owner.socket
    )
    if (wasCurrent) {
      clearRetryTimers()
      socketRef.current = null
      socketOwnerRef.current = null
    }
    rejectPendingDeliveries(
      options.pendingError,
      pending => (
        pending.socket === owner.socket
        && pending.descriptorKey === owner.descriptorKey
        && pending.lifecycleEpoch === owner.lifecycleEpoch
        && pending.attemptEpoch === owner.attemptEpoch
      ),
    )
    rejectPreparations(
      options.preparationError,
      claim => (
        claim.socket === owner.socket
        && claim.descriptorKey === owner.descriptorKey
        && claim.lifecycleEpoch === owner.lifecycleEpoch
        && claim.attemptEpoch === owner.attemptEpoch
      ),
    )
    clearRecentMessages(
      recent => (
        recent.descriptorKey === owner.descriptorKey
        && recent.lifecycleEpoch === owner.lifecycleEpoch
        && recent.attemptEpoch === owner.attemptEpoch
      ),
    )
    if (!wasCurrent) return false

    if (
      options.close
      && owner.socket.readyState !== WebSocket.CLOSED
    ) {
      try {
        owner.socket.close(options.close.code, options.close.reason)
      } catch {
        console.error("WebSocket close failed")
      }
    }
    if (mountedRef.current) setIsConnected(false)
    isConnectingRef.current = false
    if (options.notifyDisconnect) notifyDisconnect(owner)
    return true
  }, [
    clearRecentMessages,
    clearRetryTimers,
    notifyDisconnect,
    rejectPendingDeliveries,
    rejectPreparations,
  ])

  const scheduleRetry = useCallback((
    callback: () => void,
    delay: number,
    lifecycleEpoch: number,
    descriptorKey: string,
    attemptEpoch: number,
  ) => {
    const timer = setTimeout(() => {
      retryTimersRef.current.delete(timer)
      if (
        !isCurrentLifecycle(lifecycleEpoch, descriptorKey)
        || attemptEpochRef.current !== attemptEpoch
      ) return
      callback()
    }, delay)
    retryTimersRef.current.add(timer)
  }, [isCurrentLifecycle])

  useIsomorphicLayoutEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      invalidateLifecycle()
    }
  }, [invalidateLifecycle])

  useIsomorphicLayoutEffect(() => {
    const previousDescriptorKey = descriptorKeyRef.current
    const previousIdentity = deliveryIdentityRef.current
    const previousGeneration = deliveryGenerationRef.current
    const nextIdentity = normalizedConnection?.identity ?? null

    if (previousDescriptorKey !== connectionDescriptorKey) {
      invalidateLifecycle()
      if (previousDescriptorKey !== null) {
        rejectPreparations(
          new Error("Message not sent: the connection changed before delivery."),
          claim => claim.descriptorKey === previousDescriptorKey,
        )
      }
    }

    if (
      previousIdentity !== null
      && previousIdentity === nextIdentity
      && previousGeneration !== deliveryGeneration
    ) {
      rejectPendingDeliveries(
        new Error("Message delivery generation changed before acknowledgement."),
        pending => (
          pending.connectionIdentity === previousIdentity
          && pending.deliveryGeneration === previousGeneration
        ),
      )
      rejectPreparations(
        new Error("Message delivery generation changed before preparation completed."),
        claim => (
          claim.connectionIdentity === previousIdentity
          && claim.deliveryGeneration === previousGeneration
        ),
      )
      clearRecentMessages(
        recent => (
          recent.connectionIdentity === previousIdentity
          && recent.deliveryGeneration === previousGeneration
        ),
      )
    }

    descriptorKeyRef.current = connectionDescriptorKey
    connectionRef.current = normalizedConnection
    deliveryIdentityRef.current = nextIdentity
    deliveryGenerationRef.current = deliveryGeneration
    callbacksRef.current = {
      onConnectionClose,
      onConnect,
      onDisconnect,
      onError,
      onMessage,
    }
    refreshAccessTokenRef.current = refreshAccessToken
  })

  // Update token ref when token changes
  useEffect(() => {
    tokenRef.current = token !== undefined ? token : authToken
  }, [token, authToken])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (
      isConnectingRef.current
      || (
        socketRef.current
        && socketRef.current.readyState < WebSocket.CLOSING
      )
    ) return

    const connection = connectionRef.current
    const descriptorKey = descriptorKeyRef.current
    if (!connection) return
    if (!descriptorKey) return
    if (connection.chatTaskIdMode === "required" && !connection.taskId) return

    const previousOwner = socketOwnerRef.current
    if (previousOwner) {
      retireOwner(previousOwner, {
        pendingError: new Error("Connection replaced before the message was accepted."),
        preparationError: new Error("Connection replaced before the message was prepared."),
        close: {
          code: 1000,
          reason: "Connection replaced",
        },
        notifyDisconnect: true,
      })
    } else if (socketRef.current) {
      socketRef.current = null
    }

    isConnectingRef.current = true
    try {
      // Test if the URL is valid before creating WebSocket
      if (!connection.url.startsWith('ws://') && !connection.url.startsWith('wss://')) {
        throw new Error("Invalid WebSocket URL configuration")
      }

      const attemptEpoch = ++attemptEpochRef.current
      const socket = connection.protocols
        ? new WebSocket(connection.url, connection.protocols)
        : new WebSocket(connection.url)
      const owner: SocketOwner = {
        callbacks: callbacksRef.current,
        connection,
        descriptorKey,
        disconnectNotified: false,
        lifecycleEpoch: lifecycleEpochRef.current,
        attemptEpoch,
        refreshAccessToken: refreshAccessTokenRef.current,
        socket,
      }
      socketRef.current = socket
      socketOwnerRef.current = owner

      socket.onopen = () => {
        if (!isCurrentOwner(owner)) return
        if (connection.expectedProtocol && socket.protocol !== connection.expectedProtocol) {
          const protocolError = new Error("WebSocket subprotocol negotiation failed.")
          socketRef.current = null
          socketOwnerRef.current = null
          isConnectingRef.current = false
          setIsConnected(false)
          setConnectionError(protocolError)
          owner.callbacks.onError?.(protocolError)
          socket.close(1002, "WebSocket subprotocol mismatch")
          return
        }

        setIsConnected(true)
        setConnectionError(null)
        reconnectAttemptsRef.current = 0
        isConnectingRef.current = false
        owner.callbacks.onConnect?.()
      }

      socket.onclose = (event) => {
        if (!isCurrentOwner(owner)) return

        let closeDisposition: "handled" | "default"
        try {
          closeDisposition = owner.callbacks.onConnectionClose?.(event) ?? "default"
        } catch {
          const wasCurrent = retireOwner(owner, {
            pendingError: new Error("Connection closed before the message was accepted."),
            preparationError: new Error("Connection closed before the message was prepared."),
            notifyDisconnect: false,
          })
          if (wasCurrent) {
            const handlerError = new Error("WebSocket close handler failed.")
            console.error("WebSocket close handler failed")
            setConnectionError(handlerError)
            owner.callbacks.onError?.(handlerError)
          }
          return
        }

        const wasCurrent = retireOwner(owner, {
          pendingError: new Error("Connection closed before the message was accepted."),
          preparationError: new Error("Connection closed before the message was prepared."),
          notifyDisconnect: closeDisposition === "default",
        })
        if (!wasCurrent) return
        if (closeDisposition === "handled") return

        // Handle authentication errors (4001 = Authentication required)
        if (event.code === 4001) {
          if (owner.connection.credentialOwner.kind !== "auth-context") {
            owner.callbacks.onError?.(new Error("Authentication failed"))
            return
          }
          try {
            owner.refreshAccessToken(
              owner.connection.credentialOwner.accessToken,
              owner.connection.credentialOwner.userId,
            )
              .then((refreshSuccess) => {
                if (!isCurrentAttempt(owner)) return
                if (refreshSuccess) {
                  scheduleRetry(
                    connect,
                    1000,
                    owner.lifecycleEpoch,
                    owner.descriptorKey,
                    owner.attemptEpoch,
                  )
                } else {
                  owner.callbacks.onError?.(
                    new Error("Authentication failed and token refresh failed"),
                  )
                }
              })
              .catch(() => {
                if (!isCurrentAttempt(owner)) return
                console.error("Error refreshing auth token for WebSocket")
                owner.callbacks.onError?.(
                  new Error("Authentication failed and token refresh error"),
                )
              })
          } catch {
            if (!isCurrentAttempt(owner)) return
            console.error("Error refreshing auth token for WebSocket")
            owner.callbacks.onError?.(
              new Error("Authentication failed and token refresh error"),
            )
          }
          return
        }

        if (event.code === 4003) {
          const accessError = new Error("Access denied")
          setConnectionError(accessError)
          owner.callbacks.onError?.(accessError)
          return
        }

        // Don't reconnect if it's a 404 error or abnormal closure (1006)
        if (event.code === 1006) {
          return
        }

        // Don't reconnect if it's a clean close (might be intentional)
        if (event.code === 1000) {
          return
        }

        // Don't reconnect if the reason is component unmounting
        if (event.reason === 'Component unmounting') {
          return
        }

        // Only attempt to reconnect if under max attempts and the connection is task-bound
        if (reconnectAttemptsRef.current < maxReconnectAttempts && connection.taskId) {
          reconnectAttemptsRef.current++
          const delay = Math.min(1000 * reconnectAttemptsRef.current, 5000)
          scheduleRetry(
            connect,
            delay,
            owner.lifecycleEpoch,
            owner.descriptorKey,
            owner.attemptEpoch,
          )
        }
      }

      socket.onerror = () => {
        if (!isCurrentOwner(owner)) return
        console.error("WebSocket error")
        const connectionError = new Error("WebSocket connection failed. The backend WebSocket endpoint may not be available.")
        setConnectionError(connectionError)
        setIsConnected(false)
        isConnectingRef.current = false
        owner.callbacks.onError?.(connectionError)

        // Reset reconnect attempts to prevent immediate reconnection when backend is not available
        clearRetryTimers()
        reconnectAttemptsRef.current = maxReconnectAttempts
      }

      socket.onmessage = (event) => {
        if (!isCurrentOwner(owner)) return
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'message_accepted' || data.type === 'message_rejected') {
            const clientMessageId = data.client_message_id
            const pending = typeof clientMessageId === 'string'
              ? pendingDeliveriesRef.current.get(clientMessageId)
              : undefined
            if (
              pending
              && pending.socket === socket
              && pending.descriptorKey === owner.descriptorKey
              && pending.lifecycleEpoch === owner.lifecycleEpoch
              && pending.attemptEpoch === owner.attemptEpoch
              && pending.deliveryGeneration === deliveryGenerationRef.current
            ) {
              clearTimeout(pending.timeout)
              pendingDeliveriesRef.current.delete(clientMessageId)
              if (data.type === 'message_accepted') {
                pending.resolve({
                  client_message_id: clientMessageId,
                  turn_id: typeof data.turn_id === 'string' ? data.turn_id : clientMessageId,
                })
              } else {
                const error = new Error(data.message || 'Message was rejected.')
                Object.assign(error, {
                  retryWithNewId: data.retry_with_new_id === true,
                })
                pending.reject(error)
              }
            }
            return
          }

          // Handle different message types from the backend
          let message: WebSocketMessage

          if (isFinalAnswerStreamEventType(data.type)) {
            message = {
              type: data.type,
              data,
              timestamp: data.timestamp || new Date().toISOString(),
              task_id: data.task_id,
              event_id: data.event_id,
              message_id: data.message_id,
              delta: data.delta,
              content: data.content,
            }
          } else if (data.type === "trace_event") {
            // Ensure data.data is not an empty string
            const safeData = typeof data.data === 'string' && data.data === ''
              ? {}
              : data.data;

            message = {
              type: "trace_event",
              data: safeData,
              timestamp: data.timestamp,
              task_id: data.task_id,
              step_id: data.step_id,
              event_id: data.event_id,
              event_type: data.event_type,  // Keep event_type field!
            }
          } else if (data.type === "task_completed") {
            message = {
              type: "task_completed",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task?.id || data.task_id,
            }
          } else if (data.type === "dag_execution") {
            // Ensure data.data is not an empty string
            const safeData = typeof data.data === 'string' && data.data === ''
              ? {}
              : data.data;

            message = {
              type: "dag_execution",
              data: safeData,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "dag_step_info") {
            // Ensure data.data is not an empty string
            const safeData = typeof data.data === 'string' && data.data === ''
              ? {}
              : data.data;

            message = {
              type: "dag_step_info",
              data: safeData,
              timestamp: data.timestamp,
              task_id: data.task_id,
              step_id: safeData?.id,
            }
          } else if (data.type === "task_paused") {
            message = {
              type: "task_paused",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "task_waiting_for_user") {
            message = {
              type: "task_waiting_for_user",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "task_resumed") {
            message = {
              type: "task_resumed",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "agent_error") {
            message = {
              type: "agent_error",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "historical_data_complete") {
            message = {
              type: "historical_data_complete",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else {
            // Generic message handling
            const messageData = data.data || data;
            // Ensure we don't pass empty strings where objects are expected
            const safeData = typeof messageData === 'string' && messageData === ''
              ? {}
              : messageData;

            message = {
              type: data.type || "message",
              data: safeData,
              timestamp: data.timestamp || new Date().toISOString(),
              task_id: data.task_id,
              step_id: data.step_id,
            }
          }

          // Preserve the canonical task-control envelope even when a message
          // type normalizes its payload into ``data`` above.
          message.run_id = data.run_id
          message.state_version = data.state_version
          message.control_state = data.control_state
          message.status = data.status
          message.task = data.task

          setLastMessage(message)
          owner.callbacks.onMessage?.(message)
        } catch (error) {
          console.error("Error parsing WebSocket message", error)
        }
      }

    } catch {
      console.error("Failed to create WebSocket connection")
      const connectionError = new Error("Failed to create WebSocket connection.")
      isConnectingRef.current = false
      setConnectionError(connectionError)
      callbacksRef.current.onError?.(connectionError)
    }
  }, [
    clearRetryTimers,
    isCurrentAttempt,
    isCurrentOwner,
    retireOwner,
    scheduleRetry,
  ])

  const disconnect = useCallback(() => {
    const owner = socketOwnerRef.current
    invalidateLifecycle()
    if (owner) {
      retireOwner(owner, {
        pendingError: new Error("Disconnected before the message was accepted."),
        preparationError: new Error("Disconnected before the message was prepared."),
        close: {},
        notifyDisconnect: true,
      })
    } else {
      rejectPendingDeliveries(
        new Error("Disconnected before the message was accepted."),
      )
      rejectPreparations(
        new Error("Disconnected before the message was prepared."),
      )
    }
    setIsConnected(false)
    isConnectingRef.current = false
  }, [
    invalidateLifecycle,
    rejectPendingDeliveries,
    rejectPreparations,
    retireOwner,
  ])

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    const connection = connectionRef.current
    const socket = socketRef.current
    if (
      connection
      && socket?.readyState === WebSocket.OPEN
      && isCurrentSocket(socket, connection.identity)
    ) {
      socket.send(JSON.stringify(message))
    }
  }, [isCurrentSocket])

  const sendChatMessage = useCallback(async (
    message: string,
    files?: File[],
    force: boolean = false,
    requestedClientMessageId?: string,
  ): Promise<MessageDeliveryAck> => {
    const timestamp = Date.now()
    const owner = socketOwnerRef.current
    const connection = owner?.connection
    const socket = owner?.socket
    if (
      !connection
      || socket?.readyState !== WebSocket.OPEN
      || !owner
      || !isCurrentOwner(owner)
      || (connection.chatTaskIdMode === "required" && !connection.taskId)
    ) {
      throw new Error('Message not sent: the connection is not ready.')
    }
    if (connection.chatTaskIdMode === "omit" && files && files.length > 0) {
      throw new Error('File delivery is not supported for this connection.')
    }

    const currentTaskId = connection.taskId
    const currentDeliveryGeneration = deliveryGenerationRef.current
    const currentDescriptorKey = owner.descriptorKey
    const currentLifecycleEpoch = owner.lifecycleEpoch
    const currentAttemptEpoch = owner.attemptEpoch
    const clientMessageId = requestedClientMessageId || generateClientMessageId()
    if (
      pendingDeliveriesRef.current.has(clientMessageId)
      || preparationsRef.current.has(clientMessageId)
    ) {
      throw new Error("Message not sent: the client message id is already pending.")
    }
    const duplicateMessage = recentMessagesRef.current.find(
      msg => (
        msg.descriptorKey === currentDescriptorKey
        && msg.lifecycleEpoch === currentLifecycleEpoch
        && msg.attemptEpoch === currentAttemptEpoch
        && msg.deliveryGeneration === currentDeliveryGeneration
        && msg.message === message
        && msg.clientMessageId !== clientMessageId
        && (timestamp - msg.timestamp) < MESSAGE_DUPLICATE_THRESHOLD
      )
    )
    const duplicateIsPending = duplicateMessage
      ? pendingDeliveriesRef.current.has(duplicateMessage.clientMessageId)
      : false
    if (!force && duplicateIsPending) {
      throw new Error('Duplicate message ignored while the previous send is pending.')
    }

    let rejectCancellation!: (error: Error) => void
    const cancellation = new Promise<never>((_resolve, reject) => {
      rejectCancellation = reject
    })
    const claim: MessagePreparationClaim = {
      cancellation,
      cancel: (error) => {
        if (claim.cancelled) return
        claim.cancelled = true
        rejectCancellation(error)
      },
      cancelled: false,
      connectionIdentity: connection.identity,
      descriptorKey: currentDescriptorKey,
      lifecycleEpoch: currentLifecycleEpoch,
      attemptEpoch: currentAttemptEpoch,
      deliveryGeneration: currentDeliveryGeneration,
      socket,
    }
    preparationsRef.current.set(clientMessageId, claim)

    try {
      const messageData: Record<string, unknown> = {
        type: 'chat',
        message,
        client_message_id: clientMessageId,
        ...(connection.chatTaskIdMode === "required" ? { task_id: currentTaskId } : {}),
      }

      if (files && files.length > 0) {
        if (!currentTaskId) {
          throw new Error('File delivery requires a task-bound connection.')
        }
        type FileWithUploadId = File & { file_id?: string }
        const filesWithUploadIds = files as FileWithUploadId[]
        const filesToUpload = filesWithUploadIds.filter(file => !file.file_id)
        const preUploadedFiles = filesWithUploadIds
          .filter((file): file is FileWithUploadId & { file_id: string } => Boolean(file.file_id))
          .map(file => ({
            file_id: file.file_id,
            name: file.name,
            size: file.size,
            type: file.type || '',
          }))
        let uploadedFiles: Array<{ file_id: string; name?: string; size?: number; type?: string }> = []

        if (filesToUpload.length > 0 && uploadFiles) {
          uploadedFiles = await Promise.race([
            uploadFiles(filesToUpload, {
              taskId: currentTaskId,
              taskType: 'task',
            }),
            claim.cancellation,
          ])
        } else if (filesToUpload.length > 0) {
          const uploadRequest = (async () => {
            const formData = new FormData()
            filesToUpload.forEach(file => formData.append('files', file))
            formData.append('task_type', 'task')
            formData.append('task_id', currentTaskId.toString())
            const response = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${tokenRef.current ?? localStorage.getItem('token') ?? ''}`,
              },
              body: formData,
            })
            const parsed = await parseApiResponse(response)
            if (!response.ok || !isJsonRecord(parsed.data)) {
              throw new Error(getUploadErrorMessage(response, parsed, {
                generic: 'Upload failed',
                ...UPLOAD_ERROR_MESSAGES,
              }))
            }
            const data = parsed.data
            return data.success && Array.isArray(data.files)
              ? data.files
                .filter((file): file is { file_id: string; filename?: string; file_size?: number; mime_type?: string } => (
                  isJsonRecord(file) && typeof file.file_id === 'string'
                ))
                .map(file => ({
                  file_id: file.file_id,
                  name: typeof file.filename === 'string' ? file.filename : '',
                  size: typeof file.file_size === 'number' ? file.file_size : 0,
                  type: typeof file.mime_type === 'string' ? file.mime_type : '',
                }))
              : []
          })()
          uploadedFiles = await Promise.race([uploadRequest, claim.cancellation])
        }
        messageData.files = [...preUploadedFiles, ...uploadedFiles]
      }

      if (
        preparationsRef.current.get(clientMessageId) !== claim
        || claim.cancelled
        || socket.readyState !== WebSocket.OPEN
        || !isCurrentOwner(owner)
        || deliveryGenerationRef.current !== currentDeliveryGeneration
      ) {
        throw new Error("Message not sent: the connection changed before delivery.")
      }
      if (pendingDeliveriesRef.current.has(clientMessageId)) {
        throw new Error("Message not sent: the client message id is already pending.")
      }

      const delivery = new Promise<MessageDeliveryAck>((resolve, reject) => {
        const pendingDelivery: PendingDelivery = {
          resolve,
          reject,
          timeout: setTimeout(() => {
            if (pendingDeliveriesRef.current.get(clientMessageId) !== pendingDelivery) return
            pendingDeliveriesRef.current.delete(clientMessageId)
            reject(new Error('Message delivery was not acknowledged. Your draft was kept.'))
          }, 30000),
          connectionIdentity: connection.identity,
          descriptorKey: currentDescriptorKey,
          lifecycleEpoch: currentLifecycleEpoch,
          attemptEpoch: currentAttemptEpoch,
          deliveryGeneration: currentDeliveryGeneration,
          socket,
        }
        pendingDeliveriesRef.current.set(clientMessageId, pendingDelivery)
      })
      if (preparationsRef.current.get(clientMessageId) === claim) {
        preparationsRef.current.delete(clientMessageId)
      }

      try {
        socket.send(JSON.stringify(messageData))
      } catch (error) {
        const pending = pendingDeliveriesRef.current.get(clientMessageId)
        if (
          pending?.socket === socket
          && pending.descriptorKey === currentDescriptorKey
          && pending.lifecycleEpoch === currentLifecycleEpoch
          && pending.attemptEpoch === currentAttemptEpoch
          && pending.deliveryGeneration === currentDeliveryGeneration
        ) {
          clearTimeout(pending.timeout)
          pendingDeliveriesRef.current.delete(clientMessageId)
          pending.reject(error instanceof Error ? error : new Error(String(error)))
        }
        return delivery
      }

      recentMessagesRef.current.push({
        message,
        timestamp,
        connectionIdentity: connection.identity,
        descriptorKey: currentDescriptorKey,
        lifecycleEpoch: currentLifecycleEpoch,
        attemptEpoch: currentAttemptEpoch,
        deliveryGeneration: currentDeliveryGeneration,
        clientMessageId,
      })
      const cutoffTime = timestamp - 5000
      const firstKeepIndex = recentMessagesRef.current.findIndex(
        item => item.timestamp >= cutoffTime,
      )
      if (firstKeepIndex === -1) {
        recentMessagesRef.current = []
      } else if (firstKeepIndex > 0) {
        recentMessagesRef.current.splice(0, firstKeepIndex)
      }

      return delivery
    } finally {
      if (preparationsRef.current.get(clientMessageId) === claim) {
        preparationsRef.current.delete(clientMessageId)
      }
    }
  }, [isCurrentOwner, uploadFiles])

  const getCurrentTaskConnection = useCallback(() => {
    const connection = connectionRef.current
    const socket = socketRef.current
    if (
      !connection?.taskId
      || socket?.readyState !== WebSocket.OPEN
      || !isCurrentSocket(socket, connection.identity)
    ) {
      return null
    }
    return { socket, taskId: connection.taskId }
  }, [isCurrentSocket])

  const executeTask = useCallback((taskDescription: string, files?: Array<{ name: string; type: string; size: number; content?: string }>) => {
    const current = getCurrentTaskConnection()
    if (current) {
      const message = JSON.stringify({
        type: "execute_task",
        task_id: current.taskId,
        description: taskDescription,
        ...(files && files.length > 0 && { files })
      })
      current.socket.send(message)
    }
  }, [getCurrentTaskConnection])

  const pauseTask = useCallback(() => {
    const current = getCurrentTaskConnection()
    if (current) {
      const message = {
        type: "pause_task",
        task_id: current.taskId,
        command_id: generateClientMessageId(),
      }
      current.socket.send(JSON.stringify(message))
    }
  }, [getCurrentTaskConnection])

  const resumeTask = useCallback(() => {
    const current = getCurrentTaskConnection()
    if (current) {
      current.socket.send(JSON.stringify({
        type: "resume_task",
        task_id: current.taskId,
        command_id: generateClientMessageId(),
      }))
    }
  }, [getCurrentTaskConnection])

  const requestStatus = useCallback(() => {
    const current = getCurrentTaskConnection()
    if (current) {
      current.socket.send(JSON.stringify({
        type: "status_request",
        task_id: current.taskId,
      }))
    }
  }, [getCurrentTaskConnection])

  useEffect(() => {
    const ownedDescriptorKey = connectionDescriptorKey
    setConnectionError(null)
    reconnectAttemptsRef.current = 0
    if (autoConnect && ownedDescriptorKey !== null && !isConnectingRef.current) {
      connect()
    }

    return () => {
      invalidateLifecycle()
      const owner = socketOwnerRef.current
      const ownsCurrentSocket = Boolean(
        owner
        && owner.descriptorKey === ownedDescriptorKey,
      )
      if (owner && ownsCurrentSocket) {
        retireOwner(owner, {
          pendingError: new Error("Connection replaced before the message was accepted."),
          preparationError: new Error("Message not sent: the connection changed before delivery."),
          close: {
            code: 1000,
            reason: "Component unmounting",
          },
          notifyDisconnect: true,
        })
      }
      if (ownedDescriptorKey !== null) {
        rejectPendingDeliveries(
          new Error("Connection replaced before the message was accepted."),
          pending => pending.descriptorKey === ownedDescriptorKey,
        )
        rejectPreparations(
          new Error("Message not sent: the connection changed before delivery."),
          claim => claim.descriptorKey === ownedDescriptorKey,
        )
        clearRecentMessages(
          recent => recent.descriptorKey === ownedDescriptorKey,
        )
      }
      if (mountedRef.current) setIsConnected(false)
      isConnectingRef.current = false
    }
  }, [
    autoConnect,
    clearRecentMessages,
    connect,
    connectionDescriptorKey,
    invalidateLifecycle,
    rejectPendingDeliveries,
    rejectPreparations,
    retireOwner,
  ])

  // Separate effect to handle connection state changes
  useEffect(() => {
    if (isConnected) {
      reconnectAttemptsRef.current = 0 // Reset attempts on successful connection
    }
  }, [isConnected])

  return {
    isConnected,
    lastMessage,
    connectionError,
    connect,
    disconnect,
    sendMessage,
    sendChatMessage,
    executeTask,
    pauseTask,
    resumeTask,
    requestStatus,
  }
}
