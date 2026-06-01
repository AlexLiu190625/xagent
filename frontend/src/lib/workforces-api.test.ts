import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

import {
  archiveWorkforce,
  createWorkforce,
  listAgentOptions,
  listWorkforces,
  proposeWorkforceChanges,
  runWorkforce,
} from "./workforces-api"

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("workforces-api", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("uses the PR5 list pagination and visibility contract", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ items: [], total: 0, page: 2, size: 10, pages: 0 }),
    )

    const result = await listWorkforces({
      page: 2,
      size: 10,
      search: "launch",
      status: "active",
    })

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces?page=2&size=10&search=launch&status=active",
    )
    expect(result).toEqual({ items: [], total: 0, page: 2, size: 10, pages: 0 })
  })

  it("creates a draft workforce without sending unsupported status fields", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ id: 42, name: "Launch" }))

    await createWorkforce({
      name: "Launch",
      manager_agent_id: 7,
      manager_instructions: "Coordinate workers",
      workers: [
        {
          source_type: "existing",
          agent_id: 8,
          assignment_instructions: "Research competitors",
          sort_order: 1,
        },
      ],
    })

    const [, options] = apiRequestMock.mock.calls[0]
    expect(apiRequestMock.mock.calls[0][0]).toBe("http://api.local/api/workforces")
    expect(options.method).toBe("POST")
    expect(JSON.parse(String(options.body))).toEqual({
      name: "Launch",
      manager_agent_id: 7,
      manager_instructions: "Coordinate workers",
      workers: [
        {
          source_type: "existing",
          agent_id: 8,
          assignment_instructions: "Research competitors",
          sort_order: 1,
        },
      ],
    })
    expect(JSON.parse(String(options.body))).not.toHaveProperty("status")
  })

  it("loads workforce-selectable agents from the workforce options endpoint", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([]))

    await expect(listAgentOptions()).resolves.toEqual([])

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/agent-options",
    )
  })

  it("runs a workforce with the run payload shape", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        workforce_run_id: 9,
        task_id: 10,
        status: "running",
        redirect_url: "/task/10",
      }),
    )

    const result = await runWorkforce(5, {
      message: "Prepare the launch brief",
      files: ["file-1"],
      execution_mode: "react",
    })

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/5/runs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          message: "Prepare the launch brief",
          files: ["file-1"],
          execution_mode: "react",
        }),
      }),
    )
    expect(result.redirect_url).toBe("/task/10")
  })

  it("keeps the builder hybrid response intact", async () => {
    const response = {
      message_id: 12,
      user_message: {
        id: 11,
        role: "user",
        content: "Add Research",
        status: "stored",
        proposed_patch: null,
        created_at: null,
      },
      assistant_message: "I prepared a patch.",
      message: {
        id: 12,
        role: "assistant",
        content: "I prepared a patch.",
        status: "proposed",
        proposed_patch: { summary: "Add worker", operations: [], warnings: [] },
        created_at: null,
      },
      proposed_patch: { summary: "Add worker", operations: [], warnings: [] },
      requires_confirmation: true,
    }
    apiRequestMock.mockResolvedValueOnce(jsonResponse(response))

    await expect(
      proposeWorkforceChanges(5, { message: "Add Research" }),
    ).resolves.toEqual(response)
  })

  it("surfaces backend detail strings for archived edit boundaries", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Archived workforce cannot be edited" },
        { status: 409 },
      ),
    )

    await expect(archiveWorkforce(5)).rejects.toThrow(
      "Archived workforce cannot be edited",
    )
  })
})
