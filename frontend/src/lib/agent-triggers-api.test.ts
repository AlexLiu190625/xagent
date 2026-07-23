import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

import {
  StagedTrigger,
  createAgentTrigger,
  createStagedTriggers,
  listAgentTriggerRuns,
  stagedToCreatePayload,
  stagedToPseudoTrigger,
  updateAgentTrigger,
} from "./agent-triggers-api"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("agent trigger API client", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("creates webhook triggers with the expected endpoint and payload", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({
        id: 7,
        user_id: 1,
        agent_id: 42,
        type: "webhook",
        name: "CRM webhook",
        enabled: true,
        config: {},
        prompt_template: "Handle {{payload}}",
        webhook_token: "token-123",
        webhook_secret: "secret-123",
        next_run_at: null,
        last_run_at: null,
        last_error: null,
        created_at: null,
        updated_at: null,
      }),
    )

    const result = await createAgentTrigger(42, {
      type: "webhook",
      name: "CRM webhook",
      enabled: true,
      config: {},
      prompt_template: "Handle {{payload}}",
      secret: null,
    })

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/triggers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "webhook",
        name: "CRM webhook",
        enabled: true,
        config: {},
        prompt_template: "Handle {{payload}}",
        secret: null,
      }),
    })
    expect(result.webhook_secret).toBe("secret-123")
  })

  it("updates triggers with PATCH and surfaces backend validation details", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({ detail: "Trigger name must not be empty" }, { status: 400 }),
    )

    await expect(updateAgentTrigger(42, 7, { name: "   " })).rejects.toThrow(
      "Trigger name must not be empty",
    )

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/triggers/7", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "   " }),
    })
  })

  it("loads trigger runs from the agent-scoped route", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse([
        {
          id: 12,
          trigger_id: 7,
          task_id: 99,
          background_job_id: null,
          status: "completed",
          source_event_id: "event-1",
          payload_snapshot: {},
          idempotency_key: "key",
          error_message: null,
          started_at: null,
          finished_at: null,
          created_at: null,
          updated_at: null,
        },
      ]),
    )

    const runs = await listAgentTriggerRuns(42, 7)

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/42/triggers/7/runs",
    )
    expect(runs).toHaveLength(1)
    expect(runs[0].task_id).toBe(99)
  })
})

describe("staged triggers (agent creation flow)", () => {
  const staged: StagedTrigger = {
    clientId: -3,
    type: "scheduled",
    name: "Daily report",
    enabled: true,
    config: { interval_seconds: 3600 },
    prompt_template: "Run {{payload}}",
    secret: null,
  }

  it("maps a staged trigger to a pseudo AgentTrigger keyed by its negative clientId", () => {
    const pseudo = stagedToPseudoTrigger(staged)

    expect(pseudo.id).toBe(-3)
    expect(pseudo.type).toBe("scheduled")
    expect(pseudo.name).toBe("Daily report")
    expect(pseudo.enabled).toBe(true)
    expect(pseudo.config).toEqual({ interval_seconds: 3600 })
    expect(pseudo.prompt_template).toBe("Run {{payload}}")
    expect(pseudo.webhook_token).toBeNull()
    expect(pseudo.callback_id).toBeNull()
  })

  it("builds a create payload without leaking the client-side id", () => {
    const payload = stagedToCreatePayload(staged)

    expect(payload).toEqual({
      type: "scheduled",
      name: "Daily report",
      enabled: true,
      config: { interval_seconds: 3600 },
      prompt_template: "Run {{payload}}",
      secret: null,
    })
    expect("clientId" in payload).toBe(false)
  })

  const webhookNoSecret: StagedTrigger = {
    clientId: -1,
    type: "webhook",
    name: "Generated hook",
    enabled: true,
    config: {},
    prompt_template: null,
    secret: null,
  }
  const webhookCustomSecret: StagedTrigger = {
    clientId: -2,
    type: "webhook",
    name: "Custom hook",
    enabled: true,
    config: {},
    prompt_template: null,
    secret: "user-supplied",
  }

  function createdTrigger(id: number, overrides: Record<string, unknown> = {}) {
    return jsonResponse({
      id,
      user_id: 1,
      agent_id: 42,
      type: "webhook",
      name: "created",
      enabled: true,
      config: {},
      prompt_template: null,
      webhook_token: null,
      next_run_at: null,
      last_run_at: null,
      last_error: null,
      created_at: null,
      updated_at: null,
      ...overrides,
    })
  }

  it("keeps failed staged triggers (config intact) instead of dropping them", async () => {
    apiRequestMock
      .mockResolvedValueOnce(createdTrigger(11, { webhook_secret: "gen-secret" }))
      .mockResolvedValueOnce(
        jsonResponse({ detail: "Gmail account not found" }, { status: 404 }),
      )

    const outcome = await createStagedTriggers(42, [webhookNoSecret, staged])

    expect(outcome.failed).toHaveLength(1)
    expect(outcome.failed[0].staged).toBe(staged)
    expect(outcome.failed[0].error).toBe("Gmail account not found")
    expect(outcome.generatedSecrets).toEqual([
      { name: "Generated hook", secret: "gen-secret" },
    ])
  })

  it("does not report user-supplied webhook secrets as generated", async () => {
    apiRequestMock.mockResolvedValue(createdTrigger(12, { webhook_secret: "echoed-back" }))

    const outcome = await createStagedTriggers(42, [webhookCustomSecret])

    expect(outcome.failed).toHaveLength(0)
    expect(outcome.generatedSecrets).toEqual([])
  })
})
