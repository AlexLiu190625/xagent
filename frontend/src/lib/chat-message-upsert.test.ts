import { describe, expect, it } from "vitest"

import {
  type UserMessageAttachmentState,
  upsertUserMessageAttachment,
} from "./chat-message-upsert"

describe("upsertUserMessageAttachment", () => {
  it("replaces a matching plain user message with the attachment version", () => {
    const result = upsertUserMessageAttachment(
      [
        { role: "assistant", content: "hello" },
        { role: "user", content: "Read file" },
      ],
      "Read file",
      {
        role: "user",
        content: "Read file with files",
        rawContent: "Read file",
        hasFileAttachments: true,
      },
    )

    expect(result).toEqual([
      { role: "assistant", content: "hello" },
      {
        role: "user",
        content: "Read file with files",
        rawContent: "Read file",
        hasFileAttachments: true,
      },
    ])
  })

  it("does not rewrite assistant messages or different user messages", () => {
    const plain: UserMessageAttachmentState[] = [
      { role: "assistant", content: "Read file" },
      { role: "user", content: "Other request" },
    ]
    const result = upsertUserMessageAttachment(plain, "Read file", {
      role: "user",
      content: "Read file with files",
      rawContent: "Read file",
      hasFileAttachments: true,
    })

    expect(result).toHaveLength(3)
    expect(result[0]).toBe(plain[0])
    expect(result[1]).toBe(plain[1])
    expect(result[2]).toEqual({
      role: "user",
      content: "Read file with files",
      rawContent: "Read file",
      hasFileAttachments: true,
    })
  })

  it("does not add another message when one already has attachments", () => {
    const messages = [
      {
        role: "user",
        content: "Read file with files",
        rawContent: "Read file",
        hasFileAttachments: true,
      },
    ]

    expect(
      upsertUserMessageAttachment(messages, "Read file", {
        role: "user",
        content: "replacement",
        rawContent: "Read file",
        hasFileAttachments: true,
      }),
    ).toBe(messages)
  })
})
