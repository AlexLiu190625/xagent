"use client"

import { useCallback, useEffect, useRef, useState } from "react"

const TOKEN_REFRESH_THRESHOLD_MS = 60_000
const EXPIRY_WARNING_LEAD_MS = 10 * 60_000

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
  if (value.description !== undefined && typeof value.description !== "string") return null
  if (value.logo_url !== undefined && typeof value.logo_url !== "string") return null
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
  const [parentOrigin, setParentOrigin] = useState<string | null>(null)
  const mountedRef = useRef(false)
  const parentOriginRef = useRef<string | null>(null)
  const reconnectRequestedRef = useRef(false)
  const terminalRef = useRef(false)
  const generationRef = useRef(0)
  const warningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearWarningTimer = useCallback(() => {
    if (warningTimerRef.current) {
      clearTimeout(warningTimerRef.current)
      warningTimerRef.current = null
    }
  }, [])

  const transitionTerminal = useCallback((code: string) => {
    if (terminalRef.current) return
    terminalRef.current = true
    reconnectRequestedRef.current = false
    clearWarningTimer()
    setState({
      status: "terminal",
      session: null,
      terminalCode: code,
      isAbsoluteExpiryWarningVisible: false,
    })
  }, [clearWarningTimer])

  const requestReconnect = useCallback((reason: WidgetSessionReconnectReason) => {
    const targetOrigin = parentOriginRef.current
    if (!mountedRef.current || !targetOrigin || terminalRef.current || reconnectRequestedRef.current) return

    reconnectRequestedRef.current = true
    clearWarningTimer()
    setState({
      status: "refreshing",
      session: null,
      terminalCode: null,
      isAbsoluteExpiryWarningVisible: false,
    })
    window.parent.postMessage({ xagent: true, v: 1, type: "reconnect_request", reason }, targetOrigin)
  }, [clearWarningTimer])

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
    mountedRef.current = true

    const onMessage = (event: MessageEvent<unknown>) => {
      if (event.source !== window.parent || !isProtocolMessage(event.data)) return
      if (parentOriginRef.current && event.origin !== parentOriginRef.current) return

      if (!parentOriginRef.current) {
        parentOriginRef.current = event.origin
        setParentOrigin(event.origin)
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
        requestReconnect("token_expired")
        return
      }

      reconnectRequestedRef.current = false
      generationRef.current += 1
      const session: WidgetSession = {
        token,
        tokenExpiresAt: event.data.session_token_expires_at as string,
        absoluteExpiresAt: event.data.absolute_expires_at as string,
        agent,
        generation: generationRef.current,
      }
      setState({
        status: "active",
        session,
        terminalCode: null,
        isAbsoluteExpiryWarningVisible: false,
      })
      scheduleExpiryWarning(absoluteExpiresAt)
    }

    window.addEventListener("message", onMessage)
    window.parent.postMessage({ xagent: true, v: 1, type: "ready" }, "*")

    return () => {
      mountedRef.current = false
      window.removeEventListener("message", onMessage)
      clearWarningTimer()
    }
  }, [clearWarningTimer, requestReconnect, scheduleExpiryWarning, transitionTerminal])

  return {
    ...state,
    parentOrigin,
    requestReconnect,
  }
}
