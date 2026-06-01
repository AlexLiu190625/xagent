"use client"

import React from "react"
import { AlertTriangle, CheckCircle2, Loader2, Sparkles } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceBuilderPatch } from "@/types/workforce"

interface ProposedPatchCardProps {
  patch: WorkforceBuilderPatch | null
  messageId: number | null
  status?: string | null
  applying?: boolean
  readOnly?: boolean
  onApply: (messageId: number, patch: WorkforceBuilderPatch) => Promise<void> | void
}

function formatOperationTitle(op: string) {
  return op
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

function operationTitle(op: string, t: (key: string) => string) {
  const key = `workforces.builder.operations.${op}`
  const value = t(key)
  return value === key ? formatOperationTitle(op) : value
}

function operationSubtitle(operation: Record<string, unknown>) {
  if (operation.op === "add_existing_worker") {
    return `agent_id=${String(operation.agent_id ?? "")}`
  }
  if (operation.op === "update_worker" || operation.op === "remove_worker") {
    return `member_id=${String(operation.member_id ?? "")}`
  }
  return null
}

export function ProposedPatchCard({
  patch,
  messageId,
  status,
  applying = false,
  readOnly = false,
  onApply,
}: ProposedPatchCardProps) {
  const { t } = useI18n()
  const alreadyApplied = status === "applied"
  const canApply = Boolean(
    patch
      && messageId
      && patch.operations.length > 0
      && !applying
      && !alreadyApplied
      && !readOnly,
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.builder.patchTitle")}</CardTitle>
        <CardDescription>{t("workforces.builder.patchDescription")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!patch ? (
          <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
            {t("workforces.builder.noProposal")}
          </div>
        ) : (
          <>
            <div className="rounded-xl border bg-muted/20 p-4">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 size-4 text-primary" />
                <div>
                  <div className="font-medium">{t("workforces.builder.summary")}</div>
                  <p className="mt-1 text-sm text-muted-foreground">{patch.summary}</p>
                </div>
              </div>
            </div>

            {patch.clarification ? (
              <Alert>
                <AlertTriangle className="size-4" />
                <AlertTitle>{t("workforces.builder.clarificationNeeded")}</AlertTitle>
                <AlertDescription>{patch.clarification}</AlertDescription>
              </Alert>
            ) : null}

            {patch.warnings.length > 0 ? (
              <Alert className="border-amber-200 bg-amber-50 text-amber-900">
                <AlertTriangle className="size-4" />
                <AlertTitle>{t("workforces.builder.warnings")}</AlertTitle>
                <AlertDescription>
                  <div className="space-y-1">
                    {patch.warnings.map((warning, index) => (
                      <p key={`${warning}-${index}`}>{warning}</p>
                    ))}
                  </div>
                </AlertDescription>
              </Alert>
            ) : (
              <Alert>
                <CheckCircle2 className="size-4 text-emerald-600" />
                <AlertTitle>{t("workforces.builder.readyToApply")}</AlertTitle>
                <AlertDescription>
                  {t("workforces.builder.noDestructiveWarning")}
                </AlertDescription>
              </Alert>
            )}

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="font-medium">{t("workforces.builder.operationsTitle")}</div>
                <Badge variant="outline">
                  {t("workforces.builder.changeCount", { count: patch.operations.length })}
                </Badge>
              </div>
              {patch.operations.length === 0 ? (
                <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                  {t("workforces.builder.noOperations")}
                </div>
              ) : (
                patch.operations.map((operation, index) => (
                  <div key={`${operation.op}-${index}`} className="rounded-lg border bg-background p-4">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <div>
                        <div className="font-medium">{operationTitle(operation.op, t)}</div>
                        {operationSubtitle(operation) ? (
                          <div className="mt-1 text-xs text-muted-foreground">
                            {operationSubtitle(operation)}
                          </div>
                        ) : null}
                      </div>
                      <Badge variant="secondary">#{index + 1}</Badge>
                    </div>
                    <pre className="overflow-x-auto whitespace-pre-wrap break-words text-xs leading-6 text-muted-foreground">
                      {JSON.stringify(operation, null, 2)}
                    </pre>
                  </div>
                ))
              )}
            </div>

            <Button
              className="w-full"
              disabled={!canApply}
              onClick={() => {
                if (patch && messageId) {
                  void onApply(messageId, patch)
                }
              }}
            >
              {applying ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  {t("workforces.loading.applyingChanges")}
                </>
              ) : alreadyApplied ? (
                t("workforces.actions.alreadyApplied")
              ) : readOnly ? (
                t("workforces.actions.readOnly")
              ) : (
                t("workforces.actions.applyChanges")
              )}
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  )
}
