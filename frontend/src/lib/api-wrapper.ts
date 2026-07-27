"lib/api-wrapper"

import { getApiUrl } from "@/lib/utils"
import {
  type AuthSessionSnapshot,
  clearAuthSessionIfCurrent,
  compareAuthSession, compareCredentialSession,
  parseRefreshTokenPayload,
  readAuthSessionSnapshot,
  refreshAuthSession,
} from "@/lib/auth-cache"

const AUTH_REFRESH_TIMEOUT_MS = 15_000
export type AuthRefreshResult =
  | { accessToken: string; session: AuthSessionSnapshot }
  | { accessToken: null; rejected: boolean }
const refreshPromises = new Map<string, Promise<AuthRefreshResult>>()
const REFRESH_EXCLUDED_AUTH_ENDPOINTS = ["/api/auth/login", "/api/auth/register", "/api/auth/setup-admin", "/api/auth/forgot-password", "/api/auth/reset-password"]

function shouldSkipRefresh(url: string): boolean {
  if (url.includes("/api/auth/refresh")) return true
  try {
    const parsed = new URL(url, window.location.origin)
    return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint => parsed.pathname.endsWith(endpoint))
  } catch { return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint => url.includes(endpoint)) }
}
async function fetchWithRetry(url: string, options: RequestInit, maxRetries = 2): Promise<Response> {
  let lastError: Error | null = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options)
      if (response.status !== 0 && !response.url.includes("net::ERR_")) return response
      lastError = new Error(`Network error on attempt ${attempt + 1}`)
    } catch (error) {
      lastError = error as Error
      if (attempt < maxRetries) await new Promise(resolve => setTimeout(resolve, Math.min(1000, 100 * 2 ** attempt)))
    }
  }
  throw lastError || new Error("All retry attempts failed")
}
async function performTokenRefresh(session: AuthSessionSnapshot): Promise<AuthRefreshResult> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), AUTH_REFRESH_TIMEOUT_MS)
  try {
    const result = await refreshAuthSession(session, async refreshToken => {
      const response = await fetch(`${getApiUrl()}/api/auth/refresh`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }), signal: controller.signal,
      })
      if (!response.ok) return { ok: false, rejected: response.status === 401 || response.status === 403, payload: null }
      const payload = parseRefreshTokenPayload(await response.json(), refreshToken)
      return { ok: payload !== null, rejected: false, payload }
    })
    if (result.status === "updated" || result.status === "advanced") {
      return { accessToken: result.projection.snapshot.accessToken!, session: result.projection.snapshot }
    }
    return { accessToken: null, rejected: result.status === "rejected" }
  } catch (error) {
    console.error("Token refresh failed:", error)
    return { accessToken: null, rejected: false }
  } finally { clearTimeout(timeout) }
}
/** Refreshes only the immutable snapshot captured by the caller. */
export function refreshStoredAccessToken(expectedSession: AuthSessionSnapshot): Promise<AuthRefreshResult> {
  const comparison = compareAuthSession(expectedSession)
  if (comparison.status === "credentials_advanced" || comparison.status === "credentials_and_profile_advanced") {
    return Promise.resolve({ accessToken: comparison.projection.snapshot.accessToken!, session: comparison.projection.snapshot })
  }
  if (comparison.status !== "exact" && comparison.status !== "profile_advanced") return Promise.resolve({ accessToken: null, rejected: false })
  const key = `${expectedSession.sessionId}::${expectedSession.credentialRevision}::${expectedSession.accessToken}`
  const pending = refreshPromises.get(key)
  if (pending) return pending
  const promise = performTokenRefresh(expectedSession).finally(() => refreshPromises.delete(key))
  refreshPromises.set(key, promise)
  return promise
}

function withBearer(options: RequestInit, token: string): RequestInit {
  return { ...options, headers: { ...options.headers, Authorization: `Bearer ${token}` } }
}
/** A request has at most one post-401 replay, bound to an exact immutable credential snapshot. */
export async function apiRequest(url: string, options: RequestInit = {}): Promise<Response> {
  const session = readAuthSessionSnapshot()
  if (!session.accessToken) return fetch(url, options)
  let response = await fetchWithRetry(url, withBearer(options, session.accessToken))
  if (response.status !== 401 || shouldSkipRefresh(url)) return response
  const afterResponse = compareAuthSession(session)
  if (afterResponse.status === "credentials_advanced" || afterResponse.status === "credentials_and_profile_advanced") {
    const advanced = afterResponse.projection.snapshot
    if (compareCredentialSession(advanced).status === "exact_credentials") return fetch(url, withBearer(options, advanced.accessToken!))
    return response
  }
  if (afterResponse.status !== "exact" && afterResponse.status !== "profile_advanced") return response
  const errorType = response.headers.get("Error-Type")
  if (errorType && errorType !== "TokenExpired") {
    if (await clearAuthSessionIfCurrent(session)) window.location.href = "/login"
    return response
  }
  const refreshed = await refreshStoredAccessToken(session)
  if (refreshed.accessToken !== null && compareCredentialSession(refreshed.session).status === "exact_credentials") {
    return fetch(url, withBearer(options, refreshed.accessToken))
  }
  if (refreshed.accessToken === null && refreshed.rejected && await clearAuthSessionIfCurrent(session)) {
    console.error("Refresh token was rejected, redirecting to login")
    window.location.href = "/login"
  }
  return response
}

const MAX_RAW_UPLOAD_MESSAGE_LENGTH = 200
function truncateUploadMessage(text: string): string { const trimmed = text.trim(); return trimmed.length <= MAX_RAW_UPLOAD_MESSAGE_LENGTH ? trimmed : `${trimmed.slice(0, MAX_RAW_UPLOAD_MESSAGE_LENGTH)}...` }
type JsonRecord = Record<string, unknown>
export interface ParsedApiResponse { data: JsonRecord | JsonRecord[] | null; text: string | null; isHtml: boolean }
export function isJsonRecord(value: unknown): value is JsonRecord { return typeof value === "object" && value !== null && !Array.isArray(value) }
export async function parseApiResponse(response: Response): Promise<ParsedApiResponse> {
  const contentType = response.headers.get("content-type")?.toLowerCase() || ""
  const text = await response.text().catch(() => "")
  if (!text) return { data: null, text: null, isHtml: contentType.includes("text/html") }
  try { return { data: JSON.parse(text), text, isHtml: /^\s*</.test(text) } }
  catch { return { data: null, text, isHtml: contentType.includes("text/html") || /^\s*</.test(text) } }
}
export const UPLOAD_ERROR_MESSAGES = { tooLarge: "File is too large. Please reduce the upload size and try again.", proxy: "Upload failed before reaching the application. Please check the server upload limit." }
export function getUploadErrorMessage(response: Response, parsed: ParsedApiResponse, messages: { generic: string; tooLarge: string; proxy: string }): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) return parsed.data.detail
  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) return parsed.data.message
  if (response.status === 413) return messages.tooLarge
  if (parsed.isHtml) return messages.proxy
  return parsed.text?.trim() ? truncateUploadMessage(parsed.text) : messages.generic
}
export function getApiErrorMessage(response: Response, parsed: ParsedApiResponse, generic: string): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) return parsed.data.detail
  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) return parsed.data.message
  if (parsed.text?.trim() && !parsed.isHtml) return truncateUploadMessage(parsed.text)
  return response.statusText?.trim() || generic
}
export const api = {
  get: (url: string, options?: RequestInit) => apiRequest(url, { ...options, method: "GET" }),
  post: (url: string, data?: unknown, options?: RequestInit) => apiRequest(url, { ...options, method: "POST", headers: { "Content-Type": "application/json", ...options?.headers }, body: data ? JSON.stringify(data) : undefined }),
  put: (url: string, data?: unknown, options?: RequestInit) => apiRequest(url, { ...options, method: "PUT", headers: { "Content-Type": "application/json", ...options?.headers }, body: data ? JSON.stringify(data) : undefined }),
  delete: (url: string, options?: RequestInit) => apiRequest(url, { ...options, method: "DELETE" }),
}
export async function handleAuthError(response: Response): Promise<boolean> {
  if (response.status !== 401) return false
  const cleared = await clearAuthSessionIfCurrent(readAuthSessionSnapshot())
  if (cleared) window.location.href = "/login"
  return cleared
}
