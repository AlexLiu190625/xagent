export interface UserMessageAttachmentState {
  role: string
  content: unknown
  rawContent?: string
  hasFileAttachments?: boolean
}

export const userMessageText = (message: UserMessageAttachmentState): string => {
  if (message.rawContent) {
    return message.rawContent.trim()
  }
  return typeof message.content === "string" ? message.content.trim() : ""
}

export function upsertUserMessageAttachment<
  T extends UserMessageAttachmentState,
>(messages: T[], matchText: string, messageWithAttachments: T): T[] {
  const normalizedMatchText = matchText.trim()

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message.role === "user" &&
      !message.hasFileAttachments &&
      userMessageText(message) === normalizedMatchText
    ) {
      const nextMessages = [...messages]
      nextMessages[index] = messageWithAttachments
      return nextMessages
    }
  }

  const alreadyHasAttachments = messages.some(
    (message) =>
      message.role === "user" &&
      message.hasFileAttachments === true &&
      userMessageText(message) === normalizedMatchText,
  )
  if (alreadyHasAttachments) {
    return messages
  }

  return [...messages, messageWithAttachments]
}
