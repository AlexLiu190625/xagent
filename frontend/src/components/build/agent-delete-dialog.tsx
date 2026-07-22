"use client"

import React, { useEffect, useState } from "react"
import Link from "next/link"
import { ExternalLink, Loader2 } from "lucide-react"

import { WorkforceStatusBadge } from "@/components/workforce"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import type {
  AgentDeleteConflictDetail,
  AgentDeleteWorkforceReference,
} from "@/lib/agent-delete"

export type AgentDeletePendingAction =
  | { kind: "delete" }
  | { kind: "discard"; workforceId: number }
  | null

interface AgentDeleteDialogProps {
  target: { id: number; name: string } | null
  conflict: AgentDeleteConflictDetail | null
  pendingAction: AgentDeletePendingAction
  onOpenChange: (open: boolean) => void
  onConfirmDelete: () => void
  onDiscardWorkforce: (reference: AgentDeleteWorkforceReference) => void
}

export function AgentDeleteDialog({
  target,
  conflict,
  pendingAction,
  onOpenChange,
  onConfirmDelete,
  onDiscardWorkforce,
}: AgentDeleteDialogProps) {
  const { t } = useI18n()
  const [confirmDiscardId, setConfirmDiscardId] = useState<number | null>(null)
  const isPending = pendingAction !== null

  useEffect(() => {
    setConfirmDiscardId(null)
  }, [target?.id, conflict])

  const handleOpenChange = (open: boolean) => {
    if (!isPending) {
      onOpenChange(open)
    }
  }

  const handleDelete = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault()
    onConfirmDelete()
  }

  return (
    <AlertDialog open={target !== null} onOpenChange={handleOpenChange}>
      <AlertDialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
        <AlertDialogHeader>
          <AlertDialogTitle>
            {t(
              conflict
                ? "builds.list.deleteDialog.blockedTitle"
                : "builds.list.deleteDialog.title",
            )}
          </AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-4 text-left">
              <p>
                {target
                  ? t(
                      conflict
                        ? "builds.list.deleteDialog.blockedDescription"
                        : "builds.list.deleteDialog.description",
                      { name: target.name },
                    )
                  : null}
              </p>

              {conflict ? (
                <>
                  {conflict.references.length > 0 ? (
                    <ul className="space-y-3" aria-label={t("builds.list.deleteDialog.referencesLabel")}>
                      {conflict.references.map((reference) => {
                        const isDiscardPending =
                          pendingAction?.kind === "discard" &&
                          pendingAction.workforceId === reference.workforce_id
                        const isConfirmingDiscard =
                          confirmDiscardId === reference.workforce_id

                        return (
                          <li
                            key={reference.workforce_id}
                            className="space-y-3 rounded-md border p-3"
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-medium text-foreground">
                                {reference.name}
                              </span>
                              <WorkforceStatusBadge status={reference.status} />
                              {reference.roles.map((role) => (
                                <Badge key={role} variant="outline">
                                  {t(`builds.list.deleteDialog.roles.${role}`)}
                                </Badge>
                              ))}
                              {!reference.can_edit ? (
                                <Badge variant="secondary">
                                  {t("workforces.actions.readOnly")}
                                </Badge>
                              ) : null}
                            </div>

                            <div className="flex flex-wrap gap-2">
                              <Button variant="outline" size="sm" asChild>
                                <Link
                                  href={`/workforces/${reference.workforce_id}`}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {t("builds.list.deleteDialog.openWorkforce", {
                                    name: reference.name,
                                  })}
                                  <ExternalLink aria-hidden="true" />
                                </Link>
                              </Button>

                              {reference.can_discard ? (
                                <Button
                                  type="button"
                                  variant={isConfirmingDiscard ? "destructive" : "outline"}
                                  size="sm"
                                  disabled={isPending}
                                  onClick={() => {
                                    if (!isConfirmingDiscard) {
                                      setConfirmDiscardId(reference.workforce_id)
                                      return
                                    }
                                    onDiscardWorkforce(reference)
                                  }}
                                >
                                  {isDiscardPending ? (
                                    <Loader2 className="animate-spin" aria-hidden="true" />
                                  ) : null}
                                  {t(
                                    isConfirmingDiscard
                                      ? "builds.list.deleteDialog.confirmDiscardDraft"
                                      : "builds.list.deleteDialog.discardDraft",
                                    { name: reference.name },
                                  )}
                                </Button>
                              ) : null}
                            </div>
                          </li>
                        )
                      })}
                    </ul>
                  ) : null}

                  {conflict.has_hidden_references ? (
                    <p className="rounded-md bg-muted p-3" role="note">
                      {t("builds.list.deleteDialog.hiddenReferences")}
                    </p>
                  ) : null}

                  {conflict.references.length === 0 &&
                  !conflict.has_hidden_references ? (
                    <p className="rounded-md bg-muted p-3" aria-live="polite">
                      {t("builds.list.deleteDialog.readyToRetry")}
                    </p>
                  ) : null}
                </>
              ) : null}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>
            {t("common.cancel")}
          </AlertDialogCancel>
          <AlertDialogAction
            disabled={isPending}
            className="bg-destructive text-white hover:bg-destructive/90"
            onClick={handleDelete}
          >
            {pendingAction?.kind === "delete" ? (
              <Loader2 className="animate-spin" aria-hidden="true" />
            ) : null}
            {t(
              conflict
                ? "builds.list.deleteDialog.retryDelete"
                : "builds.list.deleteDialog.confirm",
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
