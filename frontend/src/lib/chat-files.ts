export interface ChatFileInfo {
  file_id?: string
  name: string
  size: number
  type: string
  path?: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const asFileArray = (value: unknown): unknown[] =>
  Array.isArray(value) ? value : []

const normalizeFile = (value: unknown): ChatFileInfo | null => {
  if (!isRecord(value)) {
    return null
  }

  const rawFileId = value.file_id ?? value.id
  const fileId = typeof rawFileId === "string" ? rawFileId.trim() : ""
  const rawName = value.name ?? value.original_name ?? value.filename
  const name = typeof rawName === "string" ? rawName.trim() : ""
  if (!fileId && !name) {
    return null
  }

  const rawSize = value.size ?? value.file_size
  const size = typeof rawSize === "number"
    ? rawSize
    : Number.isFinite(Number(rawSize))
      ? Number(rawSize)
      : 0
  const rawType = value.type ?? value.mime_type
  const type = typeof rawType === "string" ? rawType : ""
  const rawPath = value.path

  return {
    ...(fileId ? { file_id: fileId } : {}),
    name,
    size,
    type,
    ...(typeof rawPath === "string" ? { path: rawPath } : {}),
  }
}

const addFiles = (
  target: ChatFileInfo[],
  seen: Set<string>,
  values: unknown,
) => {
  for (const value of asFileArray(values)) {
    const file = normalizeFile(value)
    if (!file) {
      continue
    }
    const key = file.file_id || `${file.name}:${file.size}:${file.type}`
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    target.push(file)
  }
}

const latestUserMessage = (messages: unknown): Record<string, unknown> | null => {
  const messageList = asFileArray(messages)
  for (let index = messageList.length - 1; index >= 0; index -= 1) {
    const message = messageList[index]
    if (isRecord(message) && message.role === "user") {
      return message
    }
  }
  return null
}

export function extractUserMessageFiles(eventData: unknown): ChatFileInfo[] {
  if (!isRecord(eventData)) {
    return []
  }

  const files: ChatFileInfo[] = []
  const seen = new Set<string>()

  addFiles(files, seen, eventData.files)

  const context = isRecord(eventData.context) ? eventData.context : null
  if (context) {
    addFiles(files, seen, context.files)
    addFiles(files, seen, context.file_info)

    const state = isRecord(context.state) ? context.state : null
    if (state) {
      addFiles(files, seen, state.file_info)
    }

    const metadata = isRecord(context.metadata) ? context.metadata : null
    if (metadata) {
      addFiles(files, seen, metadata.files)
      addFiles(files, seen, metadata.file_info)

      const requestContext = isRecord(metadata.request_context)
        ? metadata.request_context
        : null
      if (requestContext) {
        addFiles(files, seen, requestContext.files)
        addFiles(files, seen, requestContext.file_info)
      }
    }

    if (files.length > 0) {
      return files
    }

    const message = latestUserMessage(context.messages)
    const messageMetadata = message && isRecord(message.metadata)
      ? message.metadata
      : null
    if (messageMetadata) {
      addFiles(files, seen, messageMetadata.files)
      addFiles(files, seen, messageMetadata.file_info)
    }
  }

  return files
}
