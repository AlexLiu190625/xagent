"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { AlertTriangle, Check, Copy, KeyRound, Plus, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { copyToClipboard } from "@/lib/clipboard"
import {
  PersonalApiKeyCreated,
  PersonalApiKeyListItem,
  createPersonalApiKey,
  listPersonalApiKeys,
  revokePersonalApiKey,
} from "@/lib/personal-api-keys-api"

export function PersonalApiKeysPanel() {
  const { t } = useI18n()
  const [keys, setKeys] = useState<PersonalApiKeyListItem[]>([])
  const [canManageOthers, setCanManageOthers] = useState(false)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [reveal, setReveal] = useState<PersonalApiKeyCreated | null>(null)
  const [copied, setCopied] = useState(false)
  const [confirmKey, setConfirmKey] = useState<PersonalApiKeyListItem | null>(null)
  const [revoking, setRevoking] = useState(false)
  const listGeneration = useRef(0)

  const loadKeys = useCallback(async () => {
    const generation = ++listGeneration.current
    setLoading(true)
    try {
      const response = await listPersonalApiKeys()
      if (generation !== listGeneration.current) return
      setKeys(response.items)
      setCanManageOthers(response.can_manage_others)
    } catch (error) {
      if (generation !== listGeneration.current) return
      console.error(error)
      toast.error(t("personalApiKeys.messages.loadFailed") || "Failed to load personal API keys")
    } finally {
      if (generation === listGeneration.current) setLoading(false)
    }
  }, [t])

  useEffect(() => {
    loadKeys()
  }, [loadKeys])

  const handleCreate = async () => {
    setCreating(true)
    try {
      const created = await createPersonalApiKey()
      setReveal(created)
      toast.success(t("personalApiKeys.messages.created") || "Personal API key created")
      loadKeys()
    } catch (error) {
      console.error(error)
      toast.error(t("personalApiKeys.messages.createFailed") || "Failed to create personal API key")
    } finally {
      setCreating(false)
    }
  }

  const handleCopyReveal = async () => {
    if (!reveal) return
    if (await copyToClipboard(reveal.full_key)) {
      setCopied(true)
      toast.success(t("personalApiKeys.messages.copied") || "Copied to clipboard")
      setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error(t("personalApiKeys.messages.copyFailed") || "Failed to copy to clipboard")
    }
  }

  const handleRevoke = async () => {
    if (!confirmKey) return
    setRevoking(true)
    try {
      await revokePersonalApiKey(confirmKey.id)
      setConfirmKey(null)
      toast.success(t("personalApiKeys.messages.revoked") || "Personal API key revoked")
      loadKeys()
    } catch (error) {
      console.error(error)
      toast.error(t("personalApiKeys.messages.revokeFailed") || "Failed to revoke personal API key")
    } finally {
      setRevoking(false)
    }
  }

  const formatDate = (value: string) => new Date(value).toLocaleDateString()
  const revokeDescription = confirmKey
    ? canManageOthers
      ? t("personalApiKeys.confirm.revokeOtherDescription", { owner: confirmKey.owner.username }) ||
        `Revoke this personal key for ${confirmKey.owner.username}?`
      : t("personalApiKeys.confirm.revokeOwnDescription") || "Revoking immediately invalidates this key."
    : ""

  return (
    <div className="px-6 md:px-8 pb-8 mt-6">
      <Card className="shadow-sm">
        <CardHeader className="pb-3 border-b flex flex-row items-center justify-between gap-4 space-y-0">
          <div>
            <h2 className="text-lg font-semibold">{t("personalApiKeys.title") || "Personal Keys"}</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {t("personalApiKeys.description") || "Manage your personal SDK and REST API keys."}
            </p>
          </div>
          <Button onClick={handleCreate} disabled={creating} className="shrink-0">
            <Plus className="w-4 h-4 mr-1" />
            {canManageOthers
              ? t("personalApiKeys.createForMe") || "Create Personal Key for Me"
              : t("personalApiKeys.create") || "Create Personal Key"}
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="text-xs font-semibold text-muted-foreground">
                  {t("personalApiKeys.columns.key") || "Secret Key"}
                </TableHead>
                {canManageOthers && (
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("personalApiKeys.columns.owner") || "Owner"}
                  </TableHead>
                )}
                <TableHead className="text-xs font-semibold text-muted-foreground">
                  {t("personalApiKeys.columns.created") || "Created"}
                </TableHead>
                <TableHead className="w-[100px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {keys.map((key) => (
                <TableRow key={key.id} className={key.revoked_at ? "opacity-50" : ""}>
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {key.masked_key}
                  </TableCell>
                  {canManageOthers && <TableCell className="text-sm">{key.owner.username}</TableCell>}
                  <TableCell className="text-sm text-muted-foreground">{formatDate(key.created_at)}</TableCell>
                  <TableCell className="text-right">
                    {!key.revoked_at && <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive"
                      onClick={() => setConfirmKey(key)}
                      title={t("personalApiKeys.actions.revoke") || "Revoke"}
                      aria-label={t("personalApiKeys.actions.revoke") || "Revoke"}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>}
                  </TableCell>
                </TableRow>
              ))}
              {keys.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={canManageOthers ? 4 : 3} className="text-center text-muted-foreground h-32">
                    {t("personalApiKeys.noData") || "No personal API keys yet."}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={reveal !== null} onOpenChange={(open) => !open && setReveal(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="h-5 w-5" />
              {t("personalApiKeys.reveal.title") || "Personal API Key Created"}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 dark:bg-amber-950/30 p-3">
            <DialogDescription className="flex items-center gap-2 text-sm font-medium text-amber-700 dark:text-amber-400">
              <AlertTriangle className="h-4 w-4" />
              {t("personalApiKeys.reveal.warning") || "Copy this key now — it is shown only once."}
            </DialogDescription>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all rounded bg-muted px-2 py-1.5 text-xs font-mono">{reveal?.full_key}</code>
              <Button
                size="icon"
                variant="secondary"
                onClick={handleCopyReveal}
                title={t("personalApiKeys.actions.copy") || "Copy personal API key"}
                aria-label={t("personalApiKeys.actions.copy") || "Copy personal API key"}
              >
                {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => setReveal(null)}>{t("common.done") || "Done"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        isOpen={confirmKey !== null}
        onOpenChange={(open) => !open && setConfirmKey(null)}
        onConfirm={handleRevoke}
        isLoading={revoking}
        title={t("personalApiKeys.confirm.revokeTitle") || "Revoke personal API key?"}
        description={revokeDescription}
        confirmText={t("personalApiKeys.actions.revoke") || "Revoke"}
      />
    </div>
  )
}
