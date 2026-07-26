import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const openFilePreviewMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const resetMentionMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  generateClientMessageId: vi.fn()
    .mockReturnValueOnce("client-message-1")
    .mockReturnValueOnce("client-message-2")
    .mockReturnValue("client-message-next"),
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://upload.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    openFilePreview: openFilePreviewMock,
  }),
}))

vi.mock("@/components/config-dialog", () => ({
  ConfigDialog: ({ trigger }: { trigger: unknown }) => trigger,
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: routerPushMock,
  }),
}))

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    dropdownPosition: null,
    fileList: [],
    filteredFiles: [],
    handleKeyDown: vi.fn(() => false),
    insertFile: vi.fn(),
    isLoadingFiles: false,
    resetMention: resetMentionMock,
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

import { ChatInput } from "./ChatInput"

const emptyJsonResponse = () =>
  new Response(JSON.stringify([]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })

describe("ChatInput", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation(() => Promise.resolve(emptyJsonResponse()))
    openFilePreviewMock.mockReset()
    routerPushMock.mockReset()
    resetMentionMock.mockReset()
  })

  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(document, "execCommand")
  })

  it("requires a model when submitting generic chat", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(screen.getByText("chatPage.input.noModelAlert")).toBeInTheDocument()
    })
    expect(onSend).not.toHaveBeenCalled()
  })

  it("suppresses preseeded files and restored file previews when files are disabled", () => {
    const onSend = vi.fn()
    const onFilesChange = vi.fn()
    const uploadFile = vi.fn()
    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn(),
    })
    const { container } = render(
      <ChatInput
        files={[file]}
        filesDisabled
        hideConfig
        inputValue="[secret.txt](file:file-secret)"
        onFilesChange={onFilesChange}
        onInputChange={vi.fn()}
        onSend={onSend}
        readOnlyConfig
        uploadFile={uploadFile}
      />
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(container.querySelector(".file-chip-preview")).toBeNull()
    expect(screen.queryByText("secret.txt")).not.toBeInTheDocument()

    const editor = screen.getByRole("textbox")
    editor.innerHTML = '<span class="file-chip-preview" data-file-path="file-secret">secret</span>'
    fireEvent.click(editor.querySelector(".file-chip-preview") as HTMLElement)

    fireEvent.drop(container.querySelector("form") as HTMLFormElement, {
      dataTransfer: { types: ["Files"] },
    })
    fireEvent.paste(editor, {
      clipboardData: {
        items: [{
          kind: "file",
          type: "text/plain",
          getAsFile: () => file,
        }],
        getData: () => "",
      },
    })

    expect(openFilePreviewMock).not.toHaveBeenCalled()
    expect(onFilesChange).not.toHaveBeenCalled()
    expect(uploadFile).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    expect(
      apiRequestMock.mock.calls.some(
        ([url]) => typeof url === "string" && url.includes("/api/files/"),
      )
    ).toBe(false)
  })

  it("keeps restored file preview semantics when only upload UI is hidden", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="[report.txt](file:file-report)"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        readOnlyConfig
      />
    )

    const chip = container.querySelector(".file-chip-preview")
    expect(chip).not.toBeNull()
    fireEvent.click(chip as HTMLElement)
    expect(openFilePreviewMock).toHaveBeenCalledWith(
      "file-report",
      "file-report",
      [{ fileName: "file-report", fileId: "file-report" }],
    )
  })

  it("allows selected agent submissions without a local model", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={onSend}
        readOnlyConfig
        selectedAgents={[{ id: 42, name: "Shared Agent" }]}
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "hello",
        expect.objectContaining({ model: "" })
      )
    })
    expect(resetMentionMock).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("chatPage.input.noModelAlert")).not.toBeInTheDocument()
  })

  it("does not show pause for uppercase terminal task status", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="next request"
        isLoading
        onInputChange={vi.fn()}
        onPause={vi.fn()}
        onSend={vi.fn()}
        taskStatus="FAILED"
      />
    )

    expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).not.toBeDisabled()
  })

  it("renders voice input as an inline toolbar action when ASR is configured", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(await screen.findByLabelText("voiceInput.start")).toBeInTheDocument()
  })

  it("allows live guidance while a task is running", async () => {
    const onSend = vi.fn()
    const onPause = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="please focus on the API contract"
        isLoading
        onInputChange={vi.fn()}
        onPause={onPause}
        onSend={onSend}
        taskStatus="running"
      />
    )

    expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).not.toBeDisabled()

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "please focus on the API contract",
        expect.objectContaining({ model: "" })
      )
    })
  })

  it("keeps the draft when durable delivery is rejected", async () => {
    const onInputChange = vi.fn()
    const onSend = vi.fn().mockRejectedValue(new Error("Message was rejected"))
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="keep this draft"
        onInputChange={onInputChange}
        onSend={onSend}
        readOnlyConfig
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    await waitFor(() => {
      expect(onInputChange).not.toHaveBeenCalledWith("")
    })
    expect(onSend.mock.calls[0][1]).toEqual(
      expect.objectContaining({ clientMessageId: expect.any(String) })
    )
  })

  it("keeps generic loading input disabled without a live task status", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="wait"
        isLoading
        onInputChange={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(container.querySelector('button[type="submit"]')).toBeDisabled()
  })

  it("shows pause for a running task when there is no draft to send", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        isLoading
        onInputChange={vi.fn()}
        onPause={vi.fn()}
        onSend={vi.fn()}
        taskStatus="running"
      />
    )

    expect(screen.getByTitle("agent.input.actions.pauseTask")).toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).toBeDisabled()
  })

  it("keeps pause hidden while running draft files are still uploading", async () => {
    const onPause = vi.fn()
    const uploadFile = vi.fn(() => new Promise<{ file_id: string }>(() => {}))

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])

      return (
        <ChatInput
          hideConfig
          inputValue=""
          files={files}
          isLoading
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onPause={onPause}
          onSend={vi.fn()}
          taskStatus="running"
          uploadFile={uploadFile}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(["draft"], "draft.txt", { type: "text/plain" })

    fireEvent.change(fileInput, { target: { files: [file] } })

    await waitFor(() => {
      expect(uploadFile).toHaveBeenCalledWith(
        file,
        expect.objectContaining({ taskType: "task" })
      )
    })
    await waitFor(() => {
      expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    })
    expect(container.querySelector('button[type="submit"]')).toBeDisabled()
  })
})
