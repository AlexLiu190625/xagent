import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

// Mirrors the backend ``APIKeyMetadataResponse`` (no plaintext secret).
export interface AgentApiKeyMetadata {
  key_prefix: string
  masked_key: string
  created_at: string
}

// Mirrors the backend ``APIKeyGenerateResponse``. ``full_key`` is the
// plaintext secret returned exactly once -- the server only stores a hash.
export interface AgentApiKeyGenerated {
  full_key: string
  key_prefix: string
  created_at: string
}

export interface AgentApiKeyRevoke {
  revoked: boolean
  revoked_at: string | null
}

function apiKeyUrl(agentId: number): string {
  return `${getApiUrl()}/api/agents/${agentId}/api-key`
}

/**
 * Read the agent's active API key metadata. Returns ``null`` when no
 * active key exists (the backend answers 404 in that case) so the UI can
 * render the "no key yet" state without treating it as an error.
 */
export async function getAgentApiKeyMetadata(
  agentId: number
): Promise<AgentApiKeyMetadata | null> {
  const res = await apiRequest(apiKeyUrl(agentId), { method: "GET" })
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`Failed to load API key (${res.status})`)
  return (await res.json()) as AgentApiKeyMetadata
}

/**
 * Generate (or rotate) the agent's API key. Rotating revokes the existing
 * active key, so the caller must confirm intent before invoking this. The
 * returned ``full_key`` is shown to the user exactly once.
 */
export async function generateAgentApiKey(
  agentId: number
): Promise<AgentApiKeyGenerated> {
  const res = await apiRequest(apiKeyUrl(agentId), { method: "POST" })
  if (!res.ok) throw new Error(`Failed to generate API key (${res.status})`)
  return (await res.json()) as AgentApiKeyGenerated
}

/** Revoke the agent's active API key. Idempotent on the backend. */
export async function revokeAgentApiKey(
  agentId: number
): Promise<AgentApiKeyRevoke> {
  const res = await apiRequest(apiKeyUrl(agentId), { method: "DELETE" })
  if (!res.ok) throw new Error(`Failed to revoke API key (${res.status})`)
  return (await res.json()) as AgentApiKeyRevoke
}
