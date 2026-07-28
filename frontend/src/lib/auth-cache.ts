export const AUTH_CACHE_KEY = "auth_cache"
export const AUTH_LOGIN_INTENT_KEY = "auth_login_intent"
export const AUTH_OIDC_INTENT_KEY = "auth_oidc_intent"
export const AUTH_TOKEN_UPDATED_EVENT = "auth-token-updated"
export const LEGACY_AUTH_TOKEN_KEY = "auth_token"
export const LEGACY_AUTH_USER_KEY = "auth_user"

export const AUTH_CACHE_DURATION_MS = 120 * 60 * 1000
const AUTH_CACHE_SCHEMA_VERSION = 2
const AUTH_LOGIN_INTENT_SCHEMA_VERSION = 1
const AUTH_MUTATION_LOCK = "xagent-auth-cache"
const MAX_REVISION = Number.MAX_SAFE_INTEGER

export interface AuthUser { id: string | number; username: string; email?: string | null; is_admin?: boolean }
export interface AuthCacheUser { id: string; username: string; email?: string | null; is_admin?: boolean }
export interface AuthTokenPayload {
  user: AuthUser
  access_token: string
  refresh_token?: string
  expires_in?: number
  refresh_expires_in?: number
}
export interface AuthCache {
  schemaVersion: typeof AUTH_CACHE_SCHEMA_VERSION
  sessionId: string
  credentialRevision: number
  profileRevision: number
  user: AuthCacheUser
  token: string
  refreshToken: string | null
  timestamp: number
  expiresAt?: number
  refreshExpiresAt?: number
}
export interface AuthSessionSnapshot {
  sessionId: string | null
  credentialRevision: number | null
  profileRevision: number | null
  userId: string | null
  accessToken: string | null
  refreshToken: string | null
  profileFingerprint: string | null
}
export interface AuthSessionProjection { cache: AuthCache; snapshot: AuthSessionSnapshot }
/** A short-lived capability that authorizes one password or OIDC response to create a bearer session. */
export interface AuthLoginIntent { id: string }
interface PersistedAuthLoginIntent extends AuthLoginIntent { schemaVersion: typeof AUTH_LOGIN_INTENT_SCHEMA_VERSION }
export type AuthSessionInspection =
  | { status: "valid"; projection: AuthSessionProjection }
  | { status: "absent" | "expired" | "invalid"; projection: null }
export type AuthSessionComparison =
  | { status: "exact" | "profile_advanced" | "credentials_advanced" | "credentials_and_profile_advanced"; projection: AuthSessionProjection }
  | { status: "replaced" | "absent" | "expired" | "invalid"; projection: null }
export type AuthCredentialComparison =
  | { status: "exact_credentials"; projection: AuthSessionProjection }
  | { status: "credentials_advanced"; projection: AuthSessionProjection }
  | { status: "replaced" | "absent" | "expired" | "invalid"; projection: null }
export type AuthMutationResult =
  | { status: "created" | "migrated" | "updated" | "advanced"; projection: AuthSessionProjection }
  | { status: "replaced" | "absent" | "expired" | "invalid" | "superseded" | "unavailable" }
export type AuthLogoutResult = {
  status: "cleared" | "unavailable"
  credentialsCleared: boolean
  barrier: "installed" | "removed" | "unavailable"
}
export type AuthIntentClaimResult =
  | { status: "claimed"; intent: AuthLoginIntent }
  | { status: "unavailable" }
type AuthStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">

const EMPTY_SNAPSHOT: AuthSessionSnapshot = {
  sessionId: null, credentialRevision: null, profileRevision: null,
  userId: null, accessToken: null, refreshToken: null,
  profileFingerprint: null,
}

/** The single capability boundary for browser-owned auth persistence. */
function getAuthStorage(): AuthStorage | null {
  if (typeof window === "undefined") return null
  try {
    const storage: unknown = window.localStorage
    if (typeof storage !== "object" || storage === null) return null
    const candidate = storage as Partial<AuthStorage>
    return typeof candidate.getItem === "function"
      && typeof candidate.setItem === "function"
      && typeof candidate.removeItem === "function"
      ? candidate as AuthStorage
      : null
  } catch {
    return null
  }
}
function getSessionStorage(): Pick<Storage, "getItem" | "setItem" | "removeItem"> | null {
  if (typeof window === "undefined") return null
  try {
    const storage: unknown = window.sessionStorage
    if (typeof storage !== "object" || storage === null) return null
    const candidate = storage as Partial<Pick<Storage, "getItem" | "setItem" | "removeItem">>
    return typeof candidate.getItem === "function" && typeof candidate.setItem === "function" && typeof candidate.removeItem === "function"
      ? candidate as Pick<Storage, "getItem" | "setItem" | "removeItem">
      : null
  } catch { return null }
}

function isFinitePositive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
}
function isSafeAbsoluteTimestamp(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
}
function isRevision(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
}
function normalizeUser(value: unknown): AuthCacheUser | null {
  if (typeof value !== "object" || value === null) return null
  const user = value as Partial<AuthUser>
  if (!(typeof user.id === "string" || typeof user.id === "number")) return null
  const id = String(user.id).trim()
  if (!id || typeof user.username !== "string" || !user.username.trim()) return null
  if (user.email !== undefined && user.email !== null && typeof user.email !== "string") return null
  if (user.is_admin !== undefined && typeof user.is_admin !== "boolean") return null
  return { id, username: user.username, email: user.email, is_admin: user.is_admin }
}
function normalizeToken(value: unknown): string | null {
  // Signed and bearer values are opaque. Trim only to reject an all-whitespace value.
  return typeof value === "string" && value.trim().length > 0 ? value : null
}
function normalizeExpiry(value: unknown): number | undefined | null {
  if (value === undefined) return undefined
  return isFinitePositive(value) ? value : null
}
function deadlineFromSeconds(now: number, seconds: number | undefined): number | undefined | null {
  if (seconds === undefined) return undefined
  const milliseconds = seconds * 1000
  const deadline = now + milliseconds
  return Number.isFinite(milliseconds) && Number.isSafeInteger(deadline) && deadline > now ? deadline : null
}
function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID()
  const bytes = new Uint32Array(4)
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes)
    return Array.from(bytes, value => value.toString(16).padStart(8, "0")).join("")
  }
  throw new Error("Secure session lineage generation is unavailable")
}
function parseAuthLoginIntent(raw: string | null): AuthLoginIntent | null {
  if (raw === null) return null
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== "object" || value === null) return null
    const intent = value as Partial<PersistedAuthLoginIntent>
    return intent.schemaVersion === AUTH_LOGIN_INTENT_SCHEMA_VERSION
      && typeof intent.id === "string" && intent.id.length > 0
      ? { id: intent.id }
      : null
  } catch { return null }
}
function serializeAuthLoginIntent(intent: AuthLoginIntent): string {
  const persisted: PersistedAuthLoginIntent = { schemaVersion: AUTH_LOGIN_INTENT_SCHEMA_VERSION, ...intent }
  return JSON.stringify(persisted)
}
function hasAuthLoginIntent(storage: AuthStorage, intent: AuthLoginIntent | null | undefined): boolean {
  return typeof intent?.id === "string" && intent.id.length > 0
    && parseAuthLoginIntent(storage.getItem(AUTH_LOGIN_INTENT_KEY))?.id === intent.id
}
function replaceAuthLoginIntent(storage: AuthStorage): AuthLoginIntent {
  const intent = { id: createSessionId() }
  storage.setItem(AUTH_LOGIN_INTENT_KEY, serializeAuthLoginIntent(intent))
  return intent
}
function projectionFromCache(cache: AuthCache): AuthSessionProjection {
  return { cache, snapshot: {
    sessionId: cache.sessionId, credentialRevision: cache.credentialRevision,
    profileRevision: cache.profileRevision, userId: cache.user.id,
    accessToken: cache.token, refreshToken: cache.refreshToken,
    profileFingerprint: JSON.stringify([cache.user.username, cache.user.email ?? null, cache.user.is_admin ?? null]),
  } }
}
function cacheUsable(cache: AuthCache, now: number): boolean {
  return cache.refreshToken && cache.refreshExpiresAt !== undefined
    ? cache.refreshExpiresAt > now
    : now - cache.timestamp <= AUTH_CACHE_DURATION_MS
}
export function isAuthCacheUsable(cache: AuthCache, now = Date.now()): boolean { return cacheUsable(cache, now) }

export function parseAuthTokenPayload(value: unknown): AuthTokenPayload | null {
  if (typeof value !== "object" || value === null) return null
  const data = value as Partial<AuthTokenPayload>
  const user = normalizeUser(data.user)
  const accessToken = normalizeToken(data.access_token)
  const refresh = data.refresh_token === undefined ? undefined : normalizeToken(data.refresh_token)
  const expiresIn = normalizeExpiry(data.expires_in)
  const refreshExpiresIn = normalizeExpiry(data.refresh_expires_in)
  if (!user || !accessToken || refresh === null || expiresIn === null || refreshExpiresIn === null
    || deadlineFromSeconds(Date.now(), expiresIn) === null || deadlineFromSeconds(Date.now(), refreshExpiresIn) === null) return null
  return { user, access_token: accessToken, refresh_token: refresh, expires_in: expiresIn, refresh_expires_in: refreshExpiresIn }
}
function parseRefreshCommitPayload(value: unknown, fallbackRefresh: string | null): Omit<AuthTokenPayload, "user"> | null {
  if (typeof value !== "object" || value === null) return null
  const data = value as Partial<Omit<AuthTokenPayload, "user">> & { success?: unknown }
  if (data.success !== true) return null
  const accessToken = normalizeToken(data.access_token)
  const refresh = data.refresh_token === undefined ? fallbackRefresh : normalizeToken(data.refresh_token)
  const expiresIn = normalizeExpiry(data.expires_in)
  const refreshExpiresIn = normalizeExpiry(data.refresh_expires_in)
  if (!accessToken || refresh === null || expiresIn === null || refreshExpiresIn === null
    || deadlineFromSeconds(Date.now(), expiresIn) === null || deadlineFromSeconds(Date.now(), refreshExpiresIn) === null) return null
  return { access_token: accessToken, refresh_token: refresh ?? undefined, expires_in: expiresIn, refresh_expires_in: refreshExpiresIn }
}
function parseCanonical(raw: string): AuthCache | null {
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== "object" || value === null) return null
    const cache = value as Partial<AuthCache>
    const user = normalizeUser(cache.user)
    if (cache.schemaVersion !== AUTH_CACHE_SCHEMA_VERSION || typeof cache.sessionId !== "string" || !cache.sessionId
      || !isRevision(cache.credentialRevision) || !isRevision(cache.profileRevision)
      || !user || !normalizeToken(cache.token) || !isSafeAbsoluteTimestamp(cache.timestamp)) return null
    if (cache.expiresAt !== undefined && !isSafeAbsoluteTimestamp(cache.expiresAt)) return null
    if (cache.refreshExpiresAt !== undefined && !isSafeAbsoluteTimestamp(cache.refreshExpiresAt)) return null
    if (cache.refreshToken !== null && cache.refreshToken !== undefined && !normalizeToken(cache.refreshToken)) return null
    return {
      schemaVersion: AUTH_CACHE_SCHEMA_VERSION, sessionId: cache.sessionId,
      credentialRevision: cache.credentialRevision, profileRevision: cache.profileRevision,
      user, token: cache.token!, refreshToken: cache.refreshToken || null,
      timestamp: cache.timestamp, expiresAt: cache.expiresAt, refreshExpiresAt: cache.refreshExpiresAt,
    }
  } catch { return null }
}
type LegacyCanonicalInspection =
  | { status: "valid"; cache: Omit<AuthCache, "schemaVersion" | "sessionId" | "credentialRevision" | "profileRevision"> }
  | { status: "expired" | "invalid" }
/** Recognize only the exact pre-v2 canonical cache shape; it is never a shadow fallback. */
function inspectLegacyCanonical(raw: string, now: number): LegacyCanonicalInspection {
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== "object" || value === null) return { status: "invalid" }
    const cache = value as Partial<AuthCache>
    if (cache.schemaVersion !== undefined || cache.sessionId !== undefined
      || cache.credentialRevision !== undefined || cache.profileRevision !== undefined) return { status: "invalid" }
    const user = normalizeUser(cache.user)
    const token = normalizeToken(cache.token)
    if (!user || !token || !isSafeAbsoluteTimestamp(cache.timestamp)) return { status: "invalid" }
    if (cache.expiresAt !== undefined && !isSafeAbsoluteTimestamp(cache.expiresAt)) return { status: "invalid" }
    if (cache.refreshExpiresAt !== undefined && !isSafeAbsoluteTimestamp(cache.refreshExpiresAt)) return { status: "invalid" }
    if (cache.refreshToken !== null && cache.refreshToken !== undefined && !normalizeToken(cache.refreshToken)) return { status: "invalid" }
    const legacy = {
      user, token, refreshToken: cache.refreshToken || null, timestamp: cache.timestamp,
      expiresAt: cache.expiresAt, refreshExpiresAt: cache.refreshExpiresAt,
    }
    const usable = legacy.refreshToken && legacy.refreshExpiresAt !== undefined
      ? legacy.refreshExpiresAt > now
      : now - legacy.timestamp <= AUTH_CACHE_DURATION_MS
    return usable ? { status: "valid", cache: legacy } : { status: "expired" }
  } catch { return { status: "invalid" } }
}
/** Pure storage inspection. It never evicts or migrates. */
export function inspectAuthSession(now = Date.now()): AuthSessionInspection {
  const storage = getAuthStorage()
  if (!storage) return { status: "absent", projection: null }
  let raw: string | null
  try {
    raw = storage.getItem(AUTH_CACHE_KEY)
  } catch {
    return { status: "absent", projection: null }
  }
  if (raw === null) return { status: "absent", projection: null }
  const cache = parseCanonical(raw)
  if (!cache) return { status: "invalid", projection: null }
  if (!cacheUsable(cache, now)) return { status: "expired", projection: null }
  return { status: "valid", projection: projectionFromCache(cache) }
}
export function readAuthCache(now = Date.now()): AuthCache | null {
  const inspection = inspectAuthSession(now)
  return inspection.status === "valid" ? inspection.projection.cache : null
}
export function readAuthSessionSnapshot(): AuthSessionSnapshot {
  const inspection = inspectAuthSession()
  return inspection.status === "valid" ? inspection.projection.snapshot : EMPTY_SNAPSHOT
}
export function compareAuthSession(captured: AuthSessionSnapshot, inspection = inspectAuthSession()): AuthSessionComparison {
  if (inspection.status !== "valid") return { status: inspection.status, projection: null }
  const { cache, snapshot } = inspection.projection
  if (!captured.sessionId || cache.sessionId !== captured.sessionId) return { status: "replaced", projection: null }
  if (cache.user.id !== captured.userId || cache.credentialRevision < (captured.credentialRevision ?? -1) || cache.profileRevision < (captured.profileRevision ?? -1)) {
    return { status: "invalid", projection: null }
  }
  if (cache.credentialRevision === captured.credentialRevision && (cache.token !== captured.accessToken || cache.refreshToken !== captured.refreshToken)) {
    return { status: "invalid", projection: null }
  }
  if (cache.profileRevision === captured.profileRevision && snapshot.profileFingerprint !== captured.profileFingerprint) {
    return { status: "invalid", projection: null }
  }
  const credentialExact = cache.credentialRevision === captured.credentialRevision && cache.token === captured.accessToken && cache.refreshToken === captured.refreshToken
  const profileExact = cache.profileRevision === captured.profileRevision
  if (credentialExact && profileExact) return { status: "exact", projection: inspection.projection }
  if (credentialExact) return { status: "profile_advanced", projection: inspection.projection }
  if (profileExact) return { status: "credentials_advanced", projection: inspection.projection }
  return { status: "credentials_and_profile_advanced", projection: inspection.projection }
}
export function compareCredentialSession(captured: AuthSessionSnapshot, inspection = inspectAuthSession()): AuthCredentialComparison {
  const comparison = compareAuthSession(captured, inspection)
  if (comparison.status === "exact" || comparison.status === "profile_advanced") {
    return { status: "exact_credentials", projection: comparison.projection }
  }
  if (comparison.status === "credentials_advanced" || comparison.status === "credentials_and_profile_advanced") {
    return { status: "credentials_advanced", projection: comparison.projection }
  }
  return { status: comparison.status, projection: null }
}
function isExactCredential(c: AuthCredentialComparison): boolean { return c.status === "exact_credentials" }
function isCredentialAdvanced(c: AuthCredentialComparison): boolean { return c.status === "credentials_advanced" }
function dispatchAuthTokenUpdated(storage: AuthStorage) {
  if (typeof window === "undefined" || typeof window.dispatchEvent !== "function") return
  try {
    window.dispatchEvent(new StorageEvent(AUTH_TOKEN_UPDATED_EVENT, { key: AUTH_CACHE_KEY, newValue: storage.getItem(AUTH_CACHE_KEY) }))
  } catch {
    // Storage persistence owns correctness; cross-context notification is best effort.
  }
}
type AuthMutationContext = {
  storage: AuthStorage
  markAuthUpdated: () => void
}
function persist(storage: AuthStorage, cache: AuthCache, markAuthUpdated: () => void) {
  storage.setItem(AUTH_CACHE_KEY, JSON.stringify(cache))
  markAuthUpdated()
}
function writeNewSession(storage: AuthStorage, payload: AuthTokenPayload, markAuthUpdated: () => void): AuthSessionProjection {
  const now = Date.now()
  const expiresAt = deadlineFromSeconds(now, payload.expires_in)
  const refreshExpiresAt = deadlineFromSeconds(now, payload.refresh_expires_in)
  if (expiresAt === null || refreshExpiresAt === null) throw new Error("Unsafe token expiry")
  const cache: AuthCache = {
    schemaVersion: AUTH_CACHE_SCHEMA_VERSION, sessionId: createSessionId(), credentialRevision: 0, profileRevision: 0,
    user: payload.user as AuthCacheUser, token: payload.access_token, refreshToken: payload.refresh_token || null,
    timestamp: now, expiresAt, refreshExpiresAt,
  }
  persist(storage, cache, markAuthUpdated)
  return projectionFromCache(cache)
}
async function withMutationLock<T>(conditional: boolean, action: (context: AuthMutationContext) => Promise<T>): Promise<T | null> {
  const storage = getAuthStorage()
  if (!storage) return null
  const locks = typeof navigator === "undefined" ? undefined : navigator.locks
  if (!locks && conditional) return null
  let changed = false
  const context: AuthMutationContext = {
    storage,
    markAuthUpdated: () => { changed = true },
  }
  try {
    return locks
      ? await locks.request(AUTH_MUTATION_LOCK, () => action(context))
      : await action(context)
  } catch {
    return null
  } finally {
    // A same-tab observer may synchronously read storage, so publication must
    // happen only after the Web Lock callback has returned.
    if (changed) dispatchAuthTokenUpdated(storage)
  }
}
function unavailable(): AuthMutationResult { return { status: "unavailable" } }
function nextRevision(value: number): number | null { return value < MAX_REVISION ? value + 1 : null }
/** Claims exclusive user intent before a password request or OIDC redirect begins. */
export async function claimAuthLoginIntent(): Promise<AuthIntentClaimResult> {
  const result = await withMutationLock(true, async ({ storage }) => ({ status: "claimed" as const, intent: replaceAuthLoginIntent(storage) }))
  return result ?? { status: "unavailable" }
}
/** Claims an intent and binds it to this tab before an OIDC full-page redirect. */
export async function claimOidcAuthLoginIntent(): Promise<AuthIntentClaimResult> {
  const claim = await claimAuthLoginIntent()
  if (claim.status !== "claimed") return claim
  const storage = getSessionStorage()
  if (!storage) return { status: "unavailable" }
  try {
    storage.setItem(AUTH_OIDC_INTENT_KEY, serializeAuthLoginIntent(claim.intent))
    return claim
  } catch { return { status: "unavailable" } }
}
/** Takes the OIDC intent created in this tab; callbacks cannot adopt a later intent. */
export function takeOidcAuthLoginIntent(): AuthLoginIntent | null {
  const storage = getSessionStorage()
  if (!storage) return null
  try {
    const intent = parseAuthLoginIntent(storage.getItem(AUTH_OIDC_INTENT_KEY))
    storage.removeItem(AUTH_OIDC_INTENT_KEY)
    return intent
  } catch { return null }
}
/** Password/OIDC session creation validates its originating intent under the same lock as the write. */
export async function createAuthSession(value: unknown, intent?: AuthLoginIntent | null): Promise<AuthMutationResult> {
  const payload = parseAuthTokenPayload(value)
  if (!payload) return { status: "invalid" }
  const result = await withMutationLock(true, async ({ storage, markAuthUpdated }) => {
    if (!intent || !hasAuthLoginIntent(storage, intent)) return { status: "superseded" as const }
    // Consume this one-shot capability before persisting its bearer response.
    replaceAuthLoginIntent(storage)
    return { status: "created" as const, projection: writeNewSession(storage, payload, markAuthUpdated) }
  })
  return result ?? unavailable()
}
/** Legacy migration is conditional: no lock means no unsafe migration. */
export async function migrateLegacyAuthSession(): Promise<AuthMutationResult> {
  const result = await withMutationLock(true, async ({ storage, markAuthUpdated }) => {
    const canonicalRaw = storage.getItem(AUTH_CACHE_KEY)
    if (canonicalRaw !== null) {
      const canonical = parseCanonical(canonicalRaw)
      if (canonical) {
        if (!cacheUsable(canonical, Date.now())) return { status: "expired" as const }
        return { status: "advanced" as const, projection: projectionFromCache(canonical) }
      }
      const legacyCanonical = inspectLegacyCanonical(canonicalRaw, Date.now())
      if (legacyCanonical.status !== "valid") return { status: legacyCanonical.status }
      const migrated: AuthCache = {
        schemaVersion: AUTH_CACHE_SCHEMA_VERSION, sessionId: createSessionId(), credentialRevision: 0, profileRevision: 0,
        ...legacyCanonical.cache,
      }
      persist(storage, migrated, markAuthUpdated)
      storage.removeItem(LEGACY_AUTH_TOKEN_KEY); storage.removeItem(LEGACY_AUTH_USER_KEY)
      return { status: "migrated" as const, projection: projectionFromCache(migrated) }
    }
    const token = storage.getItem(LEGACY_AUTH_TOKEN_KEY)
    const rawUser = storage.getItem(LEGACY_AUTH_USER_KEY)
    if (!token || !rawUser) return { status: "absent" as const }
    let user: unknown
    try { user = JSON.parse(rawUser) } catch { return { status: "invalid" as const } }
    const payload = parseAuthTokenPayload({ user, access_token: token })
    if (!payload) return { status: "invalid" as const }
    const projection = writeNewSession(storage, payload, markAuthUpdated)
    storage.removeItem(LEGACY_AUTH_TOKEN_KEY); storage.removeItem(LEGACY_AUTH_USER_KEY)
    return { status: "migrated" as const, projection }
  })
  return result ?? unavailable()
}
function advanceExact(storage: AuthStorage, cache: AuthCache, payload: Omit<AuthTokenPayload, "user">, markAuthUpdated: () => void): AuthMutationResult {
  const revision = nextRevision(cache.credentialRevision)
  if (revision === null) return { status: "invalid" }
  const now = Date.now()
  const expiresAt = deadlineFromSeconds(now, payload.expires_in)
  const refreshExpiresAt = deadlineFromSeconds(now, payload.refresh_expires_in)
  if (expiresAt === null || refreshExpiresAt === null) return { status: "invalid" }
  const updated: AuthCache = {
    ...cache, credentialRevision: revision, token: payload.access_token,
    refreshToken: payload.refresh_token || cache.refreshToken, timestamp: now,
    expiresAt,
    refreshExpiresAt: refreshExpiresAt ?? cache.refreshExpiresAt,
  }
  persist(storage, updated, markAuthUpdated); return { status: "updated", projection: projectionFromCache(updated) }
}
/** Commits a refresh response only when the captured credential lineage is still current. */
export async function commitAuthSessionRefresh(captured: AuthSessionSnapshot, value: unknown): Promise<AuthMutationResult> {
  const result = await withMutationLock(true, async ({ storage, markAuthUpdated }) => {
    const comparison = compareCredentialSession(captured)
    if (isCredentialAdvanced(comparison)) return { status: "advanced" as const, projection: comparison.projection! }
    if (!isExactCredential(comparison)) return { status: comparison.status as "replaced" | "absent" | "expired" | "invalid" }
    const payload = parseRefreshCommitPayload(value, comparison.projection!.cache.refreshToken)
    if (!payload) return { status: "invalid" as const }
    return advanceExact(storage, comparison.projection!.cache, payload, markAuthUpdated)
  })
  return result ?? unavailable()
}
export async function updateAuthSessionUser(captured: AuthSessionSnapshot, next: AuthUser | AuthCacheUser): Promise<AuthMutationResult> {
  const user = normalizeUser(next)
  if (!user || user.id !== captured.userId) return { status: "invalid" }
  const result = await withMutationLock(true, async ({ storage, markAuthUpdated }) => {
    const comparison = compareAuthSession(captured)
    if (comparison.status === "replaced" || comparison.status === "invalid" || comparison.status === "absent" || comparison.status === "expired") return { status: comparison.status }
    const cache = comparison.projection!.cache
    if (cache.user.id !== user.id || cache.profileRevision !== captured.profileRevision) return { status: "advanced" as const, projection: comparison.projection! }
    const revision = nextRevision(cache.profileRevision)
    if (revision === null) return { status: "invalid" as const }
    const updated = { ...cache, user: { ...cache.user, username: user.username, email: user.email, is_admin: user.is_admin }, profileRevision: revision }
    persist(storage, updated, markAuthUpdated); return { status: "updated" as const, projection: projectionFromCache(updated) }
  })
  return result ?? unavailable()
}
export async function clearAuthSessionIfCurrent(captured: AuthSessionSnapshot): Promise<boolean> {
  const result = await withMutationLock(true, async ({ storage, markAuthUpdated }) => {
    const comparison = compareCredentialSession(captured)
    if (!isExactCredential(comparison)) return false
    storage.removeItem(AUTH_CACHE_KEY); markAuthUpdated(); return true
  })
  return result ?? false
}
/** Explicit logout is serialized when possible, and remains available without Web Locks. */
export async function clearStoredAuth(): Promise<AuthLogoutResult> {
  const result = await withMutationLock(false, async ({ storage, markAuthUpdated }) => {
    let barrier: AuthLogoutResult["barrier"] = "unavailable"
    try {
      replaceAuthLoginIntent(storage)
      barrier = "installed"
    } catch {
      try {
        storage.removeItem(AUTH_LOGIN_INTENT_KEY)
        barrier = "removed"
      } catch {
        barrier = "unavailable"
      }
    }

    let credentialsCleared = true
    try {
      storage.removeItem(AUTH_CACHE_KEY)
      markAuthUpdated()
    } catch {
      credentialsCleared = false
    }
    for (const key of [LEGACY_AUTH_TOKEN_KEY, LEGACY_AUTH_USER_KEY]) {
      try {
        storage.removeItem(key)
      } catch {
        credentialsCleared = false
      }
    }
    return {
      status: credentialsCleared && barrier !== "unavailable" ? "cleared" as const : "unavailable" as const,
      credentialsCleared,
      barrier,
    }
  })
  return result ?? { status: "unavailable", credentialsCleared: false, barrier: "unavailable" }
}
