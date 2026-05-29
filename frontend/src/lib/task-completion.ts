import { normalizeTaskStatus } from "./task-status"

export type TaskTerminalStatus = "completed" | "failed"

type TaskCompletedRecord = {
  data?: unknown
  success?: unknown
  status?: unknown
  task?: {
    status?: unknown
    [key: string]: unknown
  }
  result?: unknown
  output?: unknown
  file_outputs?: unknown
  chat_response?: unknown
  metadata?: unknown
}

export type NormalizedTaskCompletion = {
  success: boolean
  status: TaskTerminalStatus
  task?: TaskCompletedRecord["task"]
  result?: unknown
  output?: unknown
  fileOutputs: Array<string | Record<string, unknown>>
  chatResponse?: unknown
  metadata?: unknown
}

const asRecord = (value: unknown): TaskCompletedRecord | null => {
  return value && typeof value === "object" ? (value as TaskCompletedRecord) : null
}

const normalizeTerminalStatus = (value: unknown): TaskTerminalStatus | null => {
  const normalized = normalizeTaskStatus(value)
  if (normalized === "completed") return "completed"
  if (normalized === "failed") return "failed"
  return null
}

export const normalizeTaskCompletedMessage = (
  message: unknown
): NormalizedTaskCompletion => {
  const root = asRecord(message) || {}
  const payload = asRecord(root.data) || root
  const taskStatus = normalizeTerminalStatus(payload.task?.status)
  const payloadStatus = normalizeTerminalStatus(payload.status)
  const explicitStatus = taskStatus || payloadStatus
  const success =
    explicitStatus
      ? explicitStatus === "completed"
      : typeof payload.success === "boolean"
      ? payload.success
      : false
  const status = explicitStatus || (success ? "completed" : "failed")
  const fileOutputs = Array.isArray(payload.file_outputs)
    ? (payload.file_outputs as Array<string | Record<string, unknown>>)
    : []

  return {
    success,
    status,
    task: payload.task,
    result: payload.result,
    output: payload.output,
    fileOutputs,
    chatResponse: payload.chat_response,
    metadata: payload.metadata,
  }
}
