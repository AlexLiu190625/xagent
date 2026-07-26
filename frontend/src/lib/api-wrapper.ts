"lib/api-wrapper"

import { getApiUrl } from "@/lib/utils"
import {
  AUTH_CACHE_KEY,
  AUTH_TOKEN_UPDATED_EVENT,
  AuthSessionSnapshot,
  clearStoredAuth,
  isAuthSessionCurrent,
  readAuthCache,
  readAuthSessionSnapshot,
  writeAuthCache,
} from "@/lib/auth-cache"

const AUTH_REFRESH_LOCK_NAME = "xagent-auth-refresh"
const AUTH_REFRESH_TIMEOUT_MS = 15_000

export type AuthRefreshResult =
  | { accessToken: string; session: AuthSessionSnapshot }
  | { accessToken: null; rejected: boolean }

const refreshPromises = new Map<string, Promise<AuthRefreshResult>>()
const REFRESH_EXCLUDED_AUTH_ENDPOINTS = [
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/setup-admin",
  "/api/auth/forgot-password",
  "/api/auth/reset-password",
]

function shouldSkipRefresh(url: string): boolean {
  if (url.includes("/api/auth/refresh")) {
    return true
  }

  try {
    const parsedUrl = new URL(url, window.location.origin)
    return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint =>
      parsedUrl.pathname.endsWith(endpoint)
    )
  } catch {
    return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint => url.includes(endpoint))
  }
}

// Fetch function with retry mechanism
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 2
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options)

      // If not a network error, return directly
      if (response.status !== 0 && !response.url.includes('net::ERR_')) {
        return response
      }

      // Network error, retry
      lastError = new Error(`Network error on attempt ${attempt + 1}`)

    } catch (error) {
      lastError = error as Error
      console.warn(`Network request failed (attempt ${attempt + 1}/${maxRetries + 1}):`, error)

      // Last attempt, no wait
      if (attempt < maxRetries) {
        // Exponential backoff, max wait 1 second
        await new Promise(resolve => setTimeout(resolve, Math.min(1000, 100 * Math.pow(2, attempt))))
      }
    }
  }

  // All retries failed, throw last error
  throw lastError || new Error('All retry attempts failed')
}

async function withAuthRefreshLock<T>(callback: () => Promise<T>): Promise<T> {
  if (typeof navigator !== "undefined" && navigator.locks) {
    return navigator.locks.request(AUTH_REFRESH_LOCK_NAME, callback)
  }

  return callback()
}

function dispatchAuthTokenUpdated() {
  window.dispatchEvent(new StorageEvent(AUTH_TOKEN_UPDATED_EVENT, {
    key: AUTH_CACHE_KEY,
    newValue: localStorage.getItem(AUTH_CACHE_KEY),
  }))
}

async function performTokenRefresh(
  session: AuthSessionSnapshot
): Promise<AuthRefreshResult> {
  return withAuthRefreshLock(async () => {
    if (!isAuthSessionCurrent(session)) {
      return { accessToken: null, rejected: false }
    }

    if (!session.refreshToken) {
      return { accessToken: null, rejected: true }
    }

    const abortController = new AbortController()
    const timeoutId = setTimeout(
      () => abortController.abort(),
      AUTH_REFRESH_TIMEOUT_MS
    )

    try {
      const response = await fetch(`${getApiUrl()}/api/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh_token: session.refreshToken }),
        signal: abortController.signal,
      })

      if (!isAuthSessionCurrent(session)) {
        return { accessToken: null, rejected: false }
      }

      if (!response.ok) {
        return {
          accessToken: null,
          rejected: response.status === 401 || response.status === 403,
        }
      }

      const data = await response.json()
      if (!isAuthSessionCurrent(session)) {
        return { accessToken: null, rejected: false }
      }
      if (!data.success || !data.access_token) {
        return { accessToken: null, rejected: true }
      }

      const authCache = readAuthCache()
      if (!authCache || !isAuthSessionCurrent(session)) {
        return { accessToken: null, rejected: false }
      }

      writeAuthCache(
        authCache.user,
        data.access_token,
        data.refresh_token || session.refreshToken,
        data.expires_in ? data.expires_in : undefined,
        data.refresh_expires_in ? data.refresh_expires_in : undefined
      )

      const refreshedSession = readAuthSessionSnapshot()
      dispatchAuthTokenUpdated()
      return { accessToken: data.access_token, session: refreshedSession }
    } catch (error) {
      console.error("Token refresh failed:", error)
      return { accessToken: null, rejected: false }
    } finally {
      clearTimeout(timeoutId)
    }
  })
}

export function refreshStoredAccessToken(
  expectedAccessToken?: string | null,
  expectedUserId?: string | number | null
): Promise<AuthRefreshResult> {
  const session = readAuthSessionSnapshot()
  const requestedAccessToken = expectedAccessToken === undefined
    ? session.accessToken
    : expectedAccessToken
  const requestedUserId = expectedUserId === undefined
    ? session.userId
    : expectedUserId === null
      ? null
      : String(expectedUserId)
  if (
    requestedAccessToken !== session.accessToken ||
    requestedUserId !== session.userId
  ) {
    return Promise.resolve({ accessToken: null, rejected: false })
  }

  const refreshKey = `${session.userId}::${session.accessToken}::${session.refreshToken}`
  const pendingRefresh = refreshPromises.get(refreshKey)
  if (pendingRefresh) {
    return pendingRefresh
  }

  const refreshPromise = performTokenRefresh(
    session
  ).finally(() => {
    refreshPromises.delete(refreshKey)
  })
  refreshPromises.set(refreshKey, refreshPromise)
  return refreshPromise
}

// API request wrapper
export async function apiRequest(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const session = readAuthSessionSnapshot()
  const { accessToken, userId } = session

  // If no token, request directly
  if (!accessToken) {
    return fetch(url, options)
  }

  // Add authorization header
  const headers = {
    ...options.headers,
    "Authorization": `Bearer ${accessToken}`,
  }

  // Fetch request with retry mechanism
  let response = await fetchWithRetry(url, { ...options, headers })

  if (!isAuthSessionCurrent(session)) {
    return response
  }

  // If 401 error and not a refresh request, try to refresh token
  if (response.status === 401 && !shouldSkipRefresh(url)) {
    // Check if token is expired or invalid
    const errorType = response.headers.get("Error-Type")
    const isExpired = errorType === "TokenExpired" || !errorType // Default to expired, try to refresh

    if (!isExpired) {
      // Explicitly invalid token, redirect to login page directly
      if (isAuthSessionCurrent(session)) {
        clearStoredAuth()
        window.location.href = "/login"
      }
      return response
    }
    const refreshResult = await refreshStoredAccessToken(accessToken, userId)

    if (refreshResult.accessToken !== null) {
      if (isAuthSessionCurrent(refreshResult.session)) {
        const retryHeaders = {
          ...options.headers,
          "Authorization": `Bearer ${refreshResult.accessToken}`,
        }
        response = await fetch(url, { ...options, headers: retryHeaders })
      }
    } else if (refreshResult.rejected && isAuthSessionCurrent(session)) {
      // Only a definitive refresh-token rejection should end the session.
      console.error("Refresh token was rejected, redirecting to login")
      clearStoredAuth()
      window.location.href = "/login"
    }
  }

  return response
}

const MAX_RAW_UPLOAD_MESSAGE_LENGTH = 200

function truncateUploadMessage(text: string): string {
  const trimmed = text.trim()
  if (trimmed.length <= MAX_RAW_UPLOAD_MESSAGE_LENGTH) {
    return trimmed
  }
  return `${trimmed.slice(0, MAX_RAW_UPLOAD_MESSAGE_LENGTH)}...`
}

type JsonRecord = Record<string, unknown>

export interface ParsedApiResponse {
  data: JsonRecord | JsonRecord[] | null
  text: string | null
  isHtml: boolean
}

export function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export async function parseApiResponse(response: Response): Promise<ParsedApiResponse> {
  const contentType = response.headers.get("content-type")?.toLowerCase() || ""
  const text = await response.text().catch(() => "")

  if (!text) {
    return {
      data: null,
      text: null,
      isHtml: contentType.includes("text/html"),
    }
  }

  try {
    return {
      data: JSON.parse(text),
      text,
      isHtml: /^\s*</.test(text),
    }
  } catch {
    return {
      data: null,
      text,
      isHtml: contentType.includes("text/html") || /^\s*</.test(text),
    }
  }
}

export const UPLOAD_ERROR_MESSAGES = {
  tooLarge: "File is too large. Please reduce the upload size and try again.",
  proxy: "Upload failed before reaching the application. Please check the server upload limit.",
}

export function getUploadErrorMessage(
  response: Response,
  parsed: ParsedApiResponse,
  messages: {
    generic: string
    tooLarge: string
    proxy: string
  }
): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) {
    return parsed.data.detail
  }

  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) {
    return parsed.data.message
  }

  if (response.status === 413) {
    return messages.tooLarge
  }

  if (parsed.isHtml) {
    return messages.proxy
  }

  if (parsed.text?.trim()) {
    return truncateUploadMessage(parsed.text)
  }

  return messages.generic
}

export function getApiErrorMessage(
  response: Response,
  parsed: ParsedApiResponse,
  generic: string
): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) {
    return parsed.data.detail
  }

  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) {
    return parsed.data.message
  }

  if (parsed.text?.trim() && !parsed.isHtml) {
    return truncateUploadMessage(parsed.text)
  }

  if (response.statusText?.trim()) {
    return response.statusText
  }

  return generic
}

// Convenience methods
export const api = {
  get: (url: string, options?: RequestInit) =>
    apiRequest(url, { ...options, method: "GET" }),

  post: (url: string, data?: unknown, options?: RequestInit) =>
    apiRequest(url, {
      ...options,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      body: data ? JSON.stringify(data) : undefined,
    }),

  put: (url: string, data?: unknown, options?: RequestInit) =>
    apiRequest(url, {
      ...options,
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      body: data ? JSON.stringify(data) : undefined,
    }),

  delete: (url: string, options?: RequestInit) =>
    apiRequest(url, { ...options, method: "DELETE" }),
}

// Check response status, if auth error redirect to login
export function handleAuthError(response: Response) {
  if (response.status === 401) {
    clearStoredAuth()
    window.location.href = "/login"
    return true
  }
  return false
}
