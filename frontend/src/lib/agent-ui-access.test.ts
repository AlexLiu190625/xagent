import { describe, expect, it } from "vitest"

import {
  canDeleteAgent,
  canEditAgent,
  canPublishAgent,
  canRunAgent,
  findRunnableAgentById,
  getAgentChatHref,
} from "./agent-ui-access"

describe("agent-ui-access", () => {
  it("allows readonly published agents to run without edit actions", () => {
    const agent = {
      id: 42,
      status: "published",
      readonly: true,
      can_edit: false,
      can_publish: false,
      can_delete: false,
    }

    expect(canRunAgent(agent)).toBe(true)
    expect(canEditAgent(agent)).toBe(false)
    expect(canPublishAgent(agent)).toBe(false)
    expect(canDeleteAgent(agent)).toBe(false)
    expect(getAgentChatHref(agent)).toBe("/task?agent=42")
  })

  it("keeps owner chat on the dedicated agent page", () => {
    const agent = {
      id: 7,
      status: "published",
      readonly: false,
      can_edit: true,
    }

    expect(canRunAgent(agent)).toBe(true)
    expect(canEditAgent(agent)).toBe(true)
    expect(getAgentChatHref(agent)).toBe("/agent/7")
  })

  it("only resolves runnable agents from query parameters", () => {
    const agents = [
      { id: 1, status: "draft" },
      { id: 2, status: "published" },
    ]

    expect(findRunnableAgentById(agents, "1")).toBeUndefined()
    expect(findRunnableAgentById(agents, "2")).toEqual(agents[1])
    expect(findRunnableAgentById(agents, null)).toBeUndefined()
  })
})
