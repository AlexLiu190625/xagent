/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const clipboardWriteTextMock = vi.hoisted(() => vi.fn())
const appContextMock = vi.hoisted(() => ({
  filesDisabled: false,
  openFilePreview: vi.fn(),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => appContextMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    tDynamic: (key: string) => key,
  }),
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/components/file/docx-preview-renderer", () => ({
  DocxPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="docx-preview">{base64Content}</div>
  ),
}))

vi.mock("./TraceEventRenderer", () => ({
  TraceEventRenderer: () => <div data-testid="trace-renderer" />,
}))

import { ChatMessage } from "./ChatMessage"

describe("ChatMessage Session file capability", () => {
  beforeEach(() => {
    appContextMock.filesDisabled = false
    appContextMock.openFilePreview.mockReset()
    apiRequestMock.mockReset()
    clipboardWriteTextMock.mockReset()
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWriteTextMock },
    })
    vi.stubGlobal("ResizeObserver", class {
      observe() {}
      disconnect() {}
    })
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      callback(0)
      return 1
    })
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("renders assistant file markdown as inert text without preview egress", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <ChatMessage
        role="assistant"
        content={[
          "[assistant report.docx](file:assistant-doc-id)",
          "![assistant image](file:output/assistant.png)",
        ].join("\n\n")}
      />,
    )

    expect(screen.getByText("assistant report.docx")).not.toHaveAttribute("href")
    expect(screen.getByText("assistant image")).not.toHaveAttribute("src")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
    expect(screen.queryByRole("img")).not.toBeInTheDocument()
    expect(screen.queryByTestId("docx-preview")).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(container.innerHTML).not.toContain("assistant-doc-id")
    expect(container.innerHTML).not.toContain("output/assistant.png")

    fireEvent.click(screen.getByText("assistant report.docx"))
    fireEvent.click(screen.getByText("assistant image"))
    expect(appContextMock.openFilePreview).not.toHaveBeenCalled()
  })

  it("renders user file-chip syntax as plain non-clickable text when files are disabled", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <ChatMessage
        role="user"
        content={"Uploaded [report.txt](file:user-file-id) and `output/notes.txt`"}
      />,
    )

    expect(screen.getByText("Uploaded report.txt and notes.txt")).toBeInTheDocument()
    expect(container.innerHTML).not.toContain("user-file-id")
    expect(container.innerHTML).not.toContain("output/notes.txt")
    expect(apiRequestMock).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText("Uploaded report.txt and notes.txt"))
    expect(appContextMock.openFilePreview).not.toHaveBeenCalled()
  })

  it("preserves legacy assistant file preview behavior when files are enabled", () => {
    render(
      <ChatMessage
        role="assistant"
        content="[legacy archive.zip](file:legacy-file-id)"
      />,
    )

    fireEvent.click(screen.getByText("legacy archive.zip"))
    expect(appContextMock.openFilePreview).toHaveBeenCalledWith(
      "legacy-file-id",
      "legacy archive.zip",
      [{ fileName: "legacy archive.zip", fileId: "legacy-file-id" }],
    )
  })

  it("copies the files-disabled JSON representation instead of file metadata", () => {
    appContextMock.filesDisabled = true
    const content = JSON.stringify({
      artifact: {
        file_id: "clipboard-file-id",
        file_path: "/private/clipboard.pdf",
        download_url: "https://files.example/download/clipboard",
        url: "https://files.example/generic-clipboard-url",
        filename: "clipboard.pdf",
        mime_type: "application/pdf",
        text: "[open clipboard](file:clipboard-file-id)",
      },
      requestUrl: "https://api.example/tasks/42",
    })

    render(<ChatMessage role="assistant" content={content} />)

    fireEvent.click(screen.getByTitle("common.copy"))

    expect(clipboardWriteTextMock).toHaveBeenCalledTimes(1)
    const copied = clipboardWriteTextMock.mock.calls[0][0]
    expect(copied).not.toContain("clipboard-file-id")
    expect(copied).not.toContain("/private/clipboard.pdf")
    expect(copied).not.toContain("https://files.example/download/clipboard")
    expect(copied).not.toContain("https://files.example/generic-clipboard-url")
    expect(JSON.parse(copied)).toEqual({
      artifact: {
        filename: "clipboard.pdf",
        mime_type: "application/pdf",
        text: "open clipboard",
      },
      requestUrl: "https://api.example/tasks/42",
    })
  })
})
