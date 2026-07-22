import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const createPersonalApiKeyMock = vi.hoisted(() => vi.fn())
const listPersonalApiKeysMock = vi.hoisted(() => vi.fn())
const revokePersonalApiKeyMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/personal-api-keys-api", () => ({
  createPersonalApiKey: createPersonalApiKeyMock,
  listPersonalApiKeys: listPersonalApiKeysMock,
  revokePersonalApiKey: revokePersonalApiKeyMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string>) => {
      const translations: Record<string, string> = {
        "personalApiKeys.create": "Create Personal Key",
        "personalApiKeys.createForMe": "Create Personal Key for Me",
        "personalApiKeys.columns.owner": "Owner",
        "personalApiKeys.actions.revoke": "Revoke",
        "personalApiKeys.columns.key": "Secret Key",
        "personalApiKeys.columns.created": "Created",
        "personalApiKeys.reveal.title": "Personal API Key Created",
        "personalApiKeys.reveal.warning": "Copy this key now — it is shown only once.",
        "personalApiKeys.confirm.revokeTitle": "Revoke personal API key?",
        "personalApiKeys.confirm.revokeOwnDescription": "Revoking immediately invalidates this key.",
        "personalApiKeys.confirm.revokeOtherDescription": "Revoke this personal key for {owner}?",
      }
      return (translations[key] ?? key).replace("{owner}", vars?.owner ?? "")
    },
  }),
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: ({ isOpen, description, confirmText, onConfirm }: {
    isOpen: boolean
    description?: string
    confirmText?: string
    onConfirm: () => void
  }) => isOpen ? <div><p>{description}</p><button onClick={onConfirm}>{confirmText}</button></div> : null,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

import { PersonalApiKeysPanel } from "./personal-api-keys-panel"

function listResponse(canManageOthers: boolean) {
  const items = [
    {
      id: 1,
      key_prefix: "self123",
      masked_key: "xag_personal_self123_••••••••",
      revoked_at: null,
      expires_at: null,
      created_at: "2026-07-22T00:00:00Z",
      owner: { id: 1, username: "alice", email: "alice@example.com" },
    },
  ]

  if (canManageOthers) {
    items.push({
      id: 2,
      key_prefix: "other456",
      masked_key: "xag_personal_other456_••••••••",
      revoked_at: null,
      expires_at: null,
      created_at: "2026-07-22T00:00:00Z",
      owner: { id: 2, username: "bob", email: "bob@example.com" },
    })
  }

  return {
    can_manage_others: canManageOthers,
    items,
  }
}

describe("PersonalApiKeysPanel", () => {
  beforeEach(() => {
    createPersonalApiKeyMock.mockReset()
    listPersonalApiKeysMock.mockReset()
    revokePersonalApiKeyMock.mockReset()
  })

  afterEach(cleanup)

  it("renders a self-only list without owner controls", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(false))

    render(<PersonalApiKeysPanel />)

    expect(await screen.findByText("xag_personal_self123_••••••••")).toBeInTheDocument()
    expect(screen.queryByText("bob")).not.toBeInTheDocument()
    expect(screen.queryByText("Owner")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Personal Key" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Create Personal Key for Me" })).not.toBeInTheDocument()
  })

  it("renders owner data and the self-only creation copy for managed scopes", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(true))

    render(<PersonalApiKeysPanel />)

    expect(await screen.findByText("bob")).toBeInTheDocument()
    expect(screen.getByText("Owner")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Personal Key for Me" })).toBeInTheDocument()
  })

  it("reveals the plaintext key only after creation", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(false))
    createPersonalApiKeyMock.mockResolvedValue({
      id: 3,
      full_key: "xag_personal_created_secret",
      key_prefix: "created",
      created_at: "2026-07-22T00:00:00Z",
      expires_at: null,
    })

    render(<PersonalApiKeysPanel />)

    await screen.findByText("xag_personal_self123_••••••••")
    expect(screen.queryByText("xag_personal_created_secret")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create Personal Key" }))

    expect(await screen.findByText("xag_personal_created_secret")).toBeInTheDocument()
    expect(createPersonalApiKeyMock).toHaveBeenCalledOnce()
  })

  it("names another owner in the revoke confirmation", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(true))

    render(<PersonalApiKeysPanel />)

    await screen.findByText("bob")
    fireEvent.click(screen.getAllByRole("button", { name: "Revoke" })[1])

    expect(screen.getByText("Revoke this personal key for bob?")).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole("button", { name: "Revoke" })[2])
    await waitFor(() => expect(revokePersonalApiKeyMock).toHaveBeenCalledWith(2))
  })
})
