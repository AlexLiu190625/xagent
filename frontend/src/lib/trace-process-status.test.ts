import { describe, expect, it } from "vitest"

import {
  normalizeTraceProcessStatus,
  resolveTraceProcessStatus,
} from "./trace-process-status"

describe("trace process status", () => {
  it("normalizes task status values", () => {
    expect(normalizeTraceProcessStatus("FAILED")).toBe("failed")
    expect(normalizeTraceProcessStatus("waiting_for_user")).toBe("waiting_for_user")
    expect(normalizeTraceProcessStatus("unknown")).toBeUndefined()
  })

  it("infers a failed process from terminal trace errors", () => {
    expect(
      resolveTraceProcessStatus({
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "llm_call_start",
            data: {},
          },
          {
            event_type: "trace_error",
            data: {
              error_type: "agent_error",
              status: "failed",
            },
          },
        ],
      })
    ).toBe("failed")
  })

  it("infers a failed process from step-local trace errors without a status field", () => {
    expect(
      resolveTraceProcessStatus({
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "llm_call_start",
            data: {},
          },
          {
            event_type: "trace_error",
            data: {
              error_type: "agent_pattern_error",
              error_message: "OpenAI bad request",
            },
          },
        ],
      })
    ).toBe("failed")
  })

  it("prefers explicit process status over trace inference", () => {
    expect(
      resolveTraceProcessStatus({
        processStatus: "completed",
        traceEvents: [
          {
            event_type: "trace_error",
            data: {
              error_type: "agent_error",
              status: "failed",
            },
          },
        ],
      })
    ).toBe("completed")
  })
})
