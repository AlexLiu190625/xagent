import {
  isStoppedTaskStatus,
  normalizeTaskStatus,
  type TaskStatus,
} from "./task-status"

export type TraceProcessStatus = TaskStatus

type TraceProcessEvent = {
  event_type?: string
  data?: unknown
}

const TERMINAL_FAILURE_EVENTS = new Set([
  "agent_error",
  "react_task_failed",
  "task_failed",
  "task_failed_react",
])

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : null

const isFailureTraceError = (eventData: Record<string, unknown> | null): boolean => {
  if (!eventData) return true

  const eventStatus = normalizeTraceProcessStatus(eventData.status)
  if (eventStatus === "failed") return true

  const errorType = typeof eventData.error_type === "string" ? eventData.error_type : ""
  if (errorType.endsWith("_error")) return true

  return (
    typeof eventData.error === "string" ||
    typeof eventData.message === "string" ||
    typeof eventData.error_message === "string"
  )
}

export const normalizeTraceProcessStatus = (
  status: unknown
): TraceProcessStatus | undefined => normalizeTaskStatus(status)

export const isStoppedTraceProcessStatus = (status: unknown): boolean =>
  isStoppedTaskStatus(status)

export const getTraceProcessStatusFromEvents = (
  events?: TraceProcessEvent[]
): TraceProcessStatus | undefined => {
  if (!Array.isArray(events)) return undefined

  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    const eventType = event?.event_type || ""
    const eventData = asRecord(event?.data)

    if (eventData) {
      const eventStatus = normalizeTraceProcessStatus(eventData.status)
      if (eventStatus && isStoppedTraceProcessStatus(eventStatus)) {
        return eventStatus
      }

      if (eventType === "trace_error" && isFailureTraceError(eventData)) {
        return "failed"
      }
    }

    if (TERMINAL_FAILURE_EVENTS.has(eventType)) {
      return "failed"
    }
  }

  return undefined
}

export const resolveTraceProcessStatus = ({
  processStatus,
  taskStatus,
  traceEvents,
}: {
  processStatus?: unknown
  taskStatus?: unknown
  traceEvents?: TraceProcessEvent[]
}): TraceProcessStatus | undefined =>
  normalizeTraceProcessStatus(processStatus) ||
  normalizeTraceProcessStatus(taskStatus) ||
  getTraceProcessStatusFromEvents(traceEvents)
