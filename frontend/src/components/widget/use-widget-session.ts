"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { WebSocketConnectionFailure } from "@/hooks/use-websocket"

const TOKEN_REFRESH_THRESHOLD_MS = 60_000
const EXPIRY_WARNING_LEAD_MS = 10 * 60_000
const PARENT_RESPONSE_DEADLINE_MS = 10_000
const SESSION_STABILITY_WINDOW_MS = 15_000
const MAX_CONSECUTIVE_RECOVERY_ATTEMPTS = 3

interface RecoveryState {
  attempts: number
  reason: WidgetSessionReconnectReason
  responseTimer: ReturnType<typeof setTimeout> | null
  retryTimer: ReturnType<typeof setTimeout> | null
}

export type WidgetSessionStatus = "waiting" | "active" | "refreshing" | "terminal"
export type WidgetSessionReconnectReason = "ws_closed" | "token_expired"

export interface WidgetSessionAgent {
  id: number
  name: string
  description?: string
  logoUrl?: string
  suggestedPrompts: string[]
}

export interface WidgetSession {
  token: string
  tokenExpiresAt: string
  absoluteExpiresAt: string
  agent: WidgetSessionAgent
  generation: number
}

interface WidgetSessionBridgeState {
  status: WidgetSessionStatus
  session: WidgetSession | null
  agent: WidgetSessionAgent | null
  terminalCode: string | null
  isAbsoluteExpiryWarningVisible: boolean
}

interface ProtocolMessage extends Record<string, unknown> {
  xagent: true
  v: 1
  type: "session_update" | "session_terminal"
}

const initialState: WidgetSessionBridgeState = {
  status: "waiting",
  session: null,
  agent: null,
  terminalCode: null,
  isAbsoluteExpiryWarningVisible: false,
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const parseDate = (value: unknown): number | null => {
  if (typeof value !== "string" || !value) return null
  const timestamp = Date.parse(value)
  return Number.isFinite(timestamp) ? timestamp : null
}

const parseAgent = (value: unknown): WidgetSessionAgent | null => {
  if (!isRecord(value)) return null
  if (!Number.isInteger(value.id) || (value.id as number) <= 0) return null
  if (typeof value.name !== "string" || !value.name.trim()) return null
  if (value.description !== undefined && value.description !== null && typeof value.description !== "string") return null
  if (value.logo_url !== undefined && value.logo_url !== null && typeof value.logo_url !== "string") return null
  if (!Array.isArray(value.suggested_prompts) || !value.suggested_prompts.every((item) => typeof item === "string")) {
    return null
  }

  return {
    id: value.id as number,
    name: value.name,
    ...(typeof value.description === "string" ? { description: value.description } : {}),
    ...(typeof value.logo_url === "string" ? { logoUrl: value.logo_url } : {}),
    suggestedPrompts: value.suggested_prompts as string[],
  }
}

const isProtocolMessage = (value: unknown): value is ProtocolMessage => {
  if (!isRecord(value)) return false
  return value.xagent === true
    && value.v === 1
    && (value.type === "session_update" || value.type === "session_terminal")
}

export function buildWidgetSessionWebSocketUrl(origin: string): string {
  const url = new URL(origin)
  if (url.protocol === "https:") {
    url.protocol = "wss:"
  } else if (url.protocol === "http:") {
    url.protocol = "ws:"
  } else {
    throw new Error("Widget Session requires an HTTP(S) origin")
  }
  url.pathname = "/v1/external/chat/sessions/ws"
  url.search = ""
  url.hash = ""
  return url.toString()
}

export function useWidgetSession() {
  const [state, setState] = useState<WidgetSessionBridgeState>(initialState)
  const mountedRef = useRef(false)
  const targetOriginRef = useRef<string | null>(null)
  const recoveryRef = useRef<RecoveryState | null>(null)
  const consecutiveRecoveryAttemptsRef = useRef(0)
  const terminalRef = useRef(false)
  const generationRef = useRef(0)
  const activeSessionGenerationRef = useRef<number | null>(null)
  const warningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const stabilityTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const stabilityConnectionIdentityRef = useRef<string | null>(null)
  const issueReconnectRequestRef = useRef<(reason: WidgetSessionReconnectReason) => void>(() => undefined)

  const clearWarningTimer = useCallback(() => {
    if (warningTimerRef.current) {
      clearTimeout(warningTimerRef.current)
      warningTimerRef.current = null
    }
  }, [])

  const clearStabilityTimer = useCallback(() => {
    if (stabilityTimerRef.current) {
      clearTimeout(stabilityTimerRef.current)
      stabilityTimerRef.current = null
    }
    stabilityConnectionIdentityRef.current = null
  }, [])

  const clearRecoveryTimers = useCallback(() => {
    const recovery = recoveryRef.current
    if (!recovery) return
    if (recovery.responseTimer) clearTimeout(recovery.responseTimer)
    if (recovery.retryTimer) clearTimeout(recovery.retryTimer)
    recovery.responseTimer = null
    recovery.retryTimer = null
  }, [])

  const transitionTerminal = useCallback((code: string) => {
    if (terminalRef.current) return
    terminalRef.current = true
    activeSessionGenerationRef.current = null
    clearRecoveryTimers()
    recoveryRef.current = null
    clearStabilityTimer()
    clearWarningTimer()
    setState({
      status: "terminal",
      session: null,
      agent: null,
      terminalCode: code,
      isAbsoluteExpiryWarningVisible: false,
    })
  }, [clearRecoveryTimers, clearStabilityTimer, clearWarningTimer])

  const issueReconnectRequest = useCallback((reason: WidgetSessionReconnectReason) => {
    const targetOrigin = targetOriginRef.current
    if (!mountedRef.current || !targetOrigin || terminalRef.current) return

    const recovery = recoveryRef.current ?? {
      attempts: consecutiveRecoveryAttemptsRef.current,
      reason,
      responseTimer: null,
      retryTimer: null,
    }
    recoveryRef.current = recovery
    if (recovery.responseTimer || recovery.retryTimer) return
    if (recovery.attempts >= MAX_CONSECUTIVE_RECOVERY_ATTEMPTS) {
      transitionTerminal("network_unavailable")
      return
    }

    recovery.attempts += 1
    recovery.reason = reason
    consecutiveRecoveryAttemptsRef.current = recovery.attempts
    clearWarningTimer()
    setState((current) => ({
      status: "refreshing",
      session: null,
      agent: current.agent,
      terminalCode: null,
      isAbsoluteExpiryWarningVisible: false,
    }))
    window.parent.postMessage({ xagent: true, v: 1, type: "reconnect_request", reason }, targetOrigin)
    recovery.responseTimer = setTimeout(() => {
      if (recoveryRef.current !== recovery || terminalRef.current) return
      recovery.responseTimer = null
      if (recovery.attempts >= MAX_CONSECUTIVE_RECOVERY_ATTEMPTS) {
        transitionTerminal("network_unavailable")
        return
      }
      const delay = recovery.attempts === 1 ? 1_000 : 2_000
      recovery.retryTimer = setTimeout(() => {
        if (recoveryRef.current !== recovery || terminalRef.current) return
        recovery.retryTimer = null
        issueReconnectRequestRef.current(recovery.reason)
      }, delay)
    }, PARENT_RESPONSE_DEADLINE_MS)
  }, [clearWarningTimer, transitionTerminal])

  issueReconnectRequestRef.current = issueReconnectRequest

  const advanceRecovery = useCallback((reason: WidgetSessionReconnectReason) => {
    const recovery = recoveryRef.current
    if (!recovery) {
      issueReconnectRequest(reason)
      return
    }
    if (recovery.responseTimer) {
      clearTimeout(recovery.responseTimer)
      recovery.responseTimer = null
    }
    if (recovery.retryTimer) return
    if (recovery.attempts >= MAX_CONSECUTIVE_RECOVERY_ATTEMPTS) {
      transitionTerminal("network_unavailable")
      return
    }
    recovery.reason = reason
    const delay = recovery.attempts === 1 ? 1_000 : 2_000
    recovery.retryTimer = setTimeout(() => {
      if (recoveryRef.current !== recovery || terminalRef.current) return
      recovery.retryTimer = null
      issueReconnectRequestRef.current(recovery.reason)
    }, delay)
  }, [issueReconnectRequest, transitionTerminal])

  const requestReconnect = useCallback((reason: WidgetSessionReconnectReason) => {
    if (!mountedRef.current || terminalRef.current || recoveryRef.current) return
    clearStabilityTimer()
    activeSessionGenerationRef.current = null
    issueReconnectRequest(reason)
  }, [clearStabilityTimer, issueReconnectRequest])

  const armStabilityWindow = useCallback((connectionIdentity: string) => {
    clearStabilityTimer()
    stabilityConnectionIdentityRef.current = connectionIdentity
    stabilityTimerRef.current = setTimeout(() => {
      if (
        terminalRef.current
        || recoveryRef.current
        || stabilityConnectionIdentityRef.current !== connectionIdentity
      ) return
      stabilityTimerRef.current = null
      stabilityConnectionIdentityRef.current = null
      consecutiveRecoveryAttemptsRef.current = 0
    }, SESSION_STABILITY_WINDOW_MS)
  }, [clearStabilityTimer])

  const handleConnectionOpen = useCallback((connectionIdentity: string) => {
    const activeGeneration = activeSessionGenerationRef.current
    if (
      !mountedRef.current
      || terminalRef.current
      || recoveryRef.current
      || activeGeneration === null
      || connectionIdentity !== `widget-session:${activeGeneration}`
    ) return
    armStabilityWindow(connectionIdentity)
  }, [armStabilityWindow])

  const isActiveConnection = useCallback((connectionIdentity?: string) => {
    const activeGeneration = activeSessionGenerationRef.current
    return connectionIdentity === undefined || (
      activeGeneration !== null
      && connectionIdentity === `widget-session:${activeGeneration}`
    )
  }, [])

  const handleConnectionClose = useCallback((
    event: CloseEvent,
    connectionIdentity?: string,
  ): "handled" => {
    if (!isActiveConnection(connectionIdentity)) return "handled"
    clearStabilityTimer()
    if (event.code === 1000) {
      transitionTerminal("unexpected_error")
      return "handled"
    }

    if (event.code === 4403) {
      transitionTerminal("ws_4403")
      return "handled"
    }

    if (event.code === 4408) {
      transitionTerminal("ws_4408")
      return "handled"
    }

    requestReconnect("ws_closed")
    return "handled"
  }, [clearStabilityTimer, isActiveConnection, requestReconnect, transitionTerminal])

  const handleConnectionFailure = useCallback((
    failure: WebSocketConnectionFailure,
    connectionIdentity?: string,
  ) => {
    if (!isActiveConnection(connectionIdentity)) return
    clearStabilityTimer()
    if (failure.recoverable) {
      requestReconnect("ws_closed")
      return
    }
    transitionTerminal("unexpected_error")
  }, [clearStabilityTimer, isActiveConnection, requestReconnect, transitionTerminal])

  const scheduleExpiryWarning = useCallback((absoluteExpiresAt: number) => {
    clearWarningTimer()
    const delay = Math.max(0, absoluteExpiresAt - Date.now() - EXPIRY_WARNING_LEAD_MS)
    warningTimerRef.current = setTimeout(() => {
      warningTimerRef.current = null
      setState((current) => current.status === "active"
        ? { ...current, isAbsoluteExpiryWarningVisible: true }
        : current)
    }, delay)
  }, [clearWarningTimer])

  useEffect(() => {
    if (window.parent === window) {
      transitionTerminal("unexpected_error")
      return
    }
    mountedRef.current = true

    const onMessage = (event: MessageEvent<unknown>) => {
      if (event.source !== window.parent || !isProtocolMessage(event.data)) return
      if (targetOriginRef.current && event.origin !== targetOriginRef.current) return

      if (!targetOriginRef.current) {
        targetOriginRef.current = event.origin
      }

      if (terminalRef.current) return

      if (event.data.type === "session_terminal") {
        transitionTerminal(typeof event.data.code === "string" && event.data.code
          ? event.data.code
          : "unexpected_error")
        return
      }

      const token = event.data.session_token
      const tokenExpiresAt = parseDate(event.data.session_token_expires_at)
      const absoluteExpiresAt = parseDate(event.data.absolute_expires_at)
      const agent = parseAgent(event.data.agent)
      if (typeof token !== "string" || !token.trim() || tokenExpiresAt === null || absoluteExpiresAt === null || !agent) {
        transitionTerminal("unexpected_error")
        return
      }

      const now = Date.now()
      if (absoluteExpiresAt <= now || absoluteExpiresAt < tokenExpiresAt) {
        transitionTerminal("session_expired")
        return
      }

      if (tokenExpiresAt - now < TOKEN_REFRESH_THRESHOLD_MS) {
        if (recoveryRef.current) {
          advanceRecovery("token_expired")
        } else {
          requestReconnect("token_expired")
        }
        return
      }

      clearRecoveryTimers()
      recoveryRef.current = null
      clearStabilityTimer()
      generationRef.current += 1
      const session: WidgetSession = {
        token,
        tokenExpiresAt: event.data.session_token_expires_at as string,
        absoluteExpiresAt: event.data.absolute_expires_at as string,
        agent,
        generation: generationRef.current,
      }
      activeSessionGenerationRef.current = session.generation
      setState({
        status: "active",
        session,
        agent,
        terminalCode: null,
        isAbsoluteExpiryWarningVisible: false,
      })
      scheduleExpiryWarning(absoluteExpiresAt)
    }

    window.addEventListener("message", onMessage)
    window.parent.postMessage({ xagent: true, v: 1, type: "ready" }, "*")

    return () => {
      mountedRef.current = false
      activeSessionGenerationRef.current = null
      window.removeEventListener("message", onMessage)
      clearRecoveryTimers()
      recoveryRef.current = null
      clearStabilityTimer()
      clearWarningTimer()
    }
    // The bridge owns timers for its mounted lifetime; callback behavior reads
    // refs so a render must not retire an in-flight recovery attempt.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return {
    ...state,
    requestReconnect,
    handleConnectionOpen,
    handleConnectionClose,
    handleConnectionFailure,
  }
}
