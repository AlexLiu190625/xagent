"use client"

import React, { useCallback, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { KeyRound, Copy, Check, AlertTriangle, Loader2 } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { copyToClipboard } from "@/lib/clipboard"
import {
  AgentApiKeyMetadata,
  generateAgentApiKey,
  getAgentApiKeyMetadata,
  revokeAgentApiKey,
} from "@/lib/agents-api"

interface AgentApiKeyDialogProps {
  agentId: number | null
  agentName?: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

type Mode = "view" | "confirmRegenerate" | "confirmRevoke"

export function AgentApiKeyDialog({
  agentId,
  agentName,
  open,
  onOpenChange,
}: AgentApiKeyDialogProps) {
  const { t } = useI18n()
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [metadata, setMetadata] = useState<AgentApiKeyMetadata | null>(null)
  // Plaintext key, held in memory only and shown exactly once after a
  // generate/rotate. Cleared on dialog close.
  const [fullKey, setFullKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [mode, setMode] = useState<Mode>("view")

  const loadMetadata = useCallback(async () => {
    if (agentId === null) return
    setLoading(true)
    try {
      setMetadata(await getAgentApiKeyMetadata(agentId))
    } catch (err) {
      console.error(err)
      toast.error(t("api_key.messages.load_failed") || "Failed to load API key")
    } finally {
      setLoading(false)
    }
  }, [agentId, t])

  useEffect(() => {
    if (open && agentId !== null) {
      // Clear any state carried over from a previously-opened agent so we
      // never flash a stale masked key before this agent's GET resolves.
      setFullKey(null)
      setMetadata(null)
      setMode("view")
      loadMetadata()
    }
  }, [open, agentId, loadMetadata])

  const handleGenerate = async () => {
    if (agentId === null) return
    setBusy(true)
    let result
    try {
      result = await generateAgentApiKey(agentId)
    } catch (err) {
      console.error(err)
      toast.error(t("api_key.messages.generate_failed") || "Failed to generate API key")
      setBusy(false)
      return
    }
    // Generation already succeeded (the old key, if any, is now rotated and
    // this plaintext is the only copy). Commit that state from the POST
    // response BEFORE any follow-up read, so a transient metadata-refresh
    // failure can never present a successful rotation as a failure and prompt
    // the user to retry -- which would invalidate the key they just copied.
    setFullKey(result.full_key)
    setMetadata({
      key_prefix: result.key_prefix,
      masked_key: "",
      created_at: result.created_at,
    })
    setMode("view")
    toast.success(t("api_key.messages.generated") || "API key generated")
    // Best-effort upgrade to the backend's canonical masked form; non-fatal.
    try {
      const meta = await getAgentApiKeyMetadata(agentId)
      if (meta) setMetadata(meta)
    } catch (err) {
      console.error(err)
    } finally {
      setBusy(false)
    }
  }

  const handleRevoke = async () => {
    if (agentId === null) return
    setBusy(true)
    try {
      await revokeAgentApiKey(agentId)
      setFullKey(null)
      setMetadata(null)
      setMode("view")
      toast.success(t("api_key.messages.revoked") || "API key revoked")
    } catch (err) {
      console.error(err)
      toast.error(t("api_key.messages.revoke_failed") || "Failed to revoke API key")
    } finally {
      setBusy(false)
    }
  }

  const handleCopy = async () => {
    if (!fullKey) return
    if (await copyToClipboard(fullKey)) {
      setCopied(true)
      toast.success(t("api_key.messages.copied") || "Copied to clipboard")
      setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error(t("api_key.messages.copy_failed") || "Failed to copy to clipboard")
    }
  }

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      // Drop the plaintext key from memory when the dialog closes.
      setFullKey(null)
      setMode("view")
    }
    onOpenChange(next)
  }

  const hasKey = metadata !== null

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="h-5 w-5" />
            {t("api_key.title") || "API Key"}
          </DialogTitle>
          <DialogDescription>
            {agentName
              ? `${agentName} · ${t("api_key.subtitle") || "SDK / REST API credential"}`
              : t("api_key.subtitle") || "SDK / REST API credential"}
          </DialogDescription>
        </DialogHeader>

        <div className="mt-2 space-y-4">
          {/* One-time plaintext reveal after generate/rotate. */}
          {fullKey && (
            <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 dark:bg-amber-950/30 p-3">
              <div className="flex items-center gap-2 text-sm font-medium text-amber-700 dark:text-amber-400">
                <AlertTriangle className="h-4 w-4" />
                {t("api_key.reveal.warning") || "Copy this key now — it is shown only once."}
              </div>
              <div className="flex items-center gap-2">
                <code className="flex-1 break-all rounded bg-muted px-2 py-1.5 text-xs font-mono">
                  {fullKey}
                </code>
                <Button size="icon" variant="secondary" onClick={handleCopy} title={t("api_key.reveal.copy") || "Copy"}>
                  {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-6 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" />
            </div>
          ) : mode === "confirmRegenerate" ? (
            <div className="space-y-3 rounded-md border p-3">
              <div className="text-sm text-destructive">
                {t("api_key.confirm.regenerate") ||
                  "Regenerating immediately invalidates the current key. Any app using it will stop working until updated."}
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => setMode("view")} disabled={busy}>
                  {t("api_key.actions.cancel") || "Cancel"}
                </Button>
                <Button variant="destructive" size="sm" onClick={handleGenerate} disabled={busy}>
                  {t("api_key.actions.regenerate") || "Regenerate"}
                </Button>
              </div>
            </div>
          ) : mode === "confirmRevoke" ? (
            <div className="space-y-3 rounded-md border p-3">
              <div className="text-sm text-destructive">
                {t("api_key.confirm.revoke") ||
                  "Revoking invalidates the current key. Any app using it will stop working."}
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => setMode("view")} disabled={busy}>
                  {t("api_key.actions.cancel") || "Cancel"}
                </Button>
                <Button variant="destructive" size="sm" onClick={handleRevoke} disabled={busy}>
                  {t("api_key.actions.revoke") || "Revoke"}
                </Button>
              </div>
            </div>
          ) : hasKey ? (
            <div className="space-y-3">
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  {t("api_key.active_key") || "Active key"}
                </Label>
                <div className="rounded-md bg-muted px-3 py-2 font-mono text-sm">
                  {metadata?.masked_key || `${metadata?.key_prefix}••••••••`}
                </div>
                {metadata?.created_at && (
                  <div className="text-xs text-muted-foreground">
                    {t("api_key.created_at") || "Created"}:{" "}
                    {new Date(metadata.created_at).toLocaleString()}
                  </div>
                )}
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setMode("confirmRegenerate")} disabled={busy}>
                  {t("api_key.actions.regenerate") || "Regenerate"}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                  onClick={() => setMode("confirmRevoke")}
                  disabled={busy}
                >
                  {t("api_key.actions.revoke") || "Revoke"}
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="text-sm text-muted-foreground">
                {t("api_key.empty") || "No API key yet. Generate one to call this agent via the SDK or REST API."}
              </div>
              <Button size="sm" onClick={handleGenerate} disabled={busy}>
                {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {t("api_key.actions.generate") || "Generate API Key"}
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
