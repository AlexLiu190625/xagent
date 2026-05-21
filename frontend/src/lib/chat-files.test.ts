import { describe, expect, it } from "vitest"

import { extractUserMessageFiles } from "./chat-files"

describe("extractUserMessageFiles", () => {
  it("reads the stable top-level files shape first", () => {
    expect(
      extractUserMessageFiles({
        files: [
          {
            file_id: "file-1",
            name: "brief.md",
            size: 42,
            type: "text/markdown",
          },
        ],
      }),
    ).toEqual([
      {
        file_id: "file-1",
        name: "brief.md",
        size: 42,
        type: "text/markdown",
      },
    ])
  })

  it("falls back to historical context locations", () => {
    expect(
      extractUserMessageFiles({
        context: {
          messages: [
            {
              role: "user",
              metadata: {
                files: [
                  {
                    file_id: "file-2",
                    filename: "meta-image-upload.md",
                    file_size: "3552",
                    mime_type: "text/markdown",
                  },
                ],
              },
            },
          ],
        },
      }),
    ).toEqual([
      {
        file_id: "file-2",
        name: "meta-image-upload.md",
        size: 3552,
        type: "text/markdown",
      },
    ])
  })

  it("does not reuse files from historical user messages", () => {
    expect(
      extractUserMessageFiles({
        context: {
          messages: [
            {
              role: "user",
              metadata: {
                files: [
                  {
                    file_id: "old-file",
                    filename: "old.md",
                    file_size: 42,
                    mime_type: "text/markdown",
                  },
                ],
              },
            },
            {
              role: "assistant",
              content: "Done",
            },
            {
              role: "user",
              content: "Summarize this",
            },
          ],
        },
      }),
    ).toEqual([])
  })

  it("only falls back to files on the latest user message", () => {
    expect(
      extractUserMessageFiles({
        context: {
          messages: [
            {
              role: "user",
              metadata: {
                files: [
                  {
                    file_id: "old-file",
                    filename: "old.md",
                    file_size: 42,
                    mime_type: "text/markdown",
                  },
                ],
              },
            },
            {
              role: "user",
              metadata: {
                files: [
                  {
                    file_id: "new-file",
                    filename: "new.md",
                    file_size: 84,
                    mime_type: "text/markdown",
                  },
                ],
              },
            },
          ],
        },
      }),
    ).toEqual([
      {
        file_id: "new-file",
        name: "new.md",
        size: 84,
        type: "text/markdown",
      },
    ])
  })

  it("normalizes dirty data and deduplicates by file id", () => {
    expect(
      extractUserMessageFiles({
        files: [
          { file_id: "file-1", name: "brief.md", size: "bad" },
          { file_id: "file-1", name: "brief.md", size: 42 },
          { name: "loose.txt", size: 3, type: "text/plain" },
          { name: "loose.txt", size: 3, type: "text/plain" },
          { size: 99, type: "text/plain" },
        ],
      }),
    ).toEqual([
      {
        file_id: "file-1",
        name: "brief.md",
        size: 0,
        type: "",
      },
      {
        name: "loose.txt",
        size: 3,
        type: "text/plain",
      },
    ])
  })
})
