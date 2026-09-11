"use client"

import React, { useEffect, useRef, useState } from "react"
import { usePathname } from "next/navigation"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { toast } from "@/components/ui/sonner"
import { Textarea } from "@/components/ui/textarea"
import { useApp } from "@/contexts/app-context-chat"
import {
  useConnectorRuntimeDialog,
  type ConnectorRuntimeDialogRequest,
} from "@/contexts/connector-runtime-dialog-context"
import { useI18n } from "@/contexts/i18n-context"
import type { TranslationKey, TranslationVariables } from "@/i18n/translations"
import {
  buildSubmitItems,
  classifySubmitFailure,
  connectorRuntimeInputDraftKey,
  fetchTaskConnectorRuntimeRequirements,
  isAcceptedRuntimeKeyName,
  isConnectorRuntimeDialogHostPath,
  isSubmitEnabled,
  resolveDialogActions,
  resolveDialogOutcome,
  submitTaskConnectorRuntimeValues,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeErrorMessageKey,
  type ConnectorRuntimeFailureDisposition,
  type ConnectorRuntimeReport,
  type DialogOutcome,
} from "@/lib/connector-runtime-api"
import { generateClientMessageId } from "@/lib/utils"

// An exhaustive lookup (not a template-literal key) so a source scan can
// verify every key this dialog renders exists in both locales without
// having to interpret string concatenation.
const ERROR_MESSAGE_TRANSLATION_KEYS: Record<ConnectorRuntimeErrorMessageKey, TranslationKey> = {
  network: "connectorRuntime.errors.network",
  busyRetry: "connectorRuntime.errors.busyRetry",
  contactAdmin: "connectorRuntime.errors.contactAdmin",
  conflict: "connectorRuntime.errors.conflict",
  typeString: "connectorRuntime.errors.typeString",
  typeObject: "connectorRuntime.errors.typeObject",
  emptyValue: "connectorRuntime.errors.emptyValue",
  keyNameRejected: "connectorRuntime.errors.keyNameRejected",
  configChanged: "connectorRuntime.errors.configChanged",
  notInSession: "connectorRuntime.errors.notInSession",
  connectorUnavailable: "connectorRuntime.errors.connectorUnavailable",
  tooLarge: "connectorRuntime.errors.tooLarge",
}

function translateFailure(
  t: (key: TranslationKey, vars?: TranslationVariables) => string,
  messageKey: ConnectorRuntimeErrorMessageKey,
  vars?: TranslationVariables,
): string {
  return t(ERROR_MESSAGE_TRANSLATION_KEYS[messageKey], vars)
}

/**
 * The mount point for both halves of the connector-runtime dialog. Every
 * `AppProvider` in the tree renders this, including the two widget/share
 * ones -- there the dialog's own context read always returns the no-provider
 * default (no request, every action a no-op), so nothing below ever mounts.
 * Only two responsibilities live here: reading the current request, and the
 * two cleanup effects that narrow what the provider retains as the viewed
 * task changes or this tree unmounts. Everything about reading the report,
 * rendering rows, submitting and resending lives in the inner component
 * below, which only exists while there is a request to act on.
 */
export function ConnectorRuntimeDialog() {
  const { request, retainOnlyTask } = useConnectorRuntimeDialog()
  const { state } = useApp()
  const cleanupRef = useRef(retainOnlyTask)
  cleanupRef.current = retainOnlyTask

  // Task-switch cleanup: no cleanup function of its own. Combining this with
  // the unmount effect below into one effect would run the unmount cleanup
  // on every task change too, which would erase a same-render first-gate
  // snapshot before anything could read it.
  useEffect(() => {
    retainOnlyTask(state.taskId)
  }, [state.taskId, retainOnlyTask])

  // Unmount cleanup: a separate effect with an empty dependency array, read
  // through a ref so it always calls the latest function without needing to
  // be in that array (matches the workforce pages' own unmount-cleanup shape).
  useEffect(() => () => cleanupRef.current(null), [])

  if (!request) return null
  return <ConnectorRuntimeDialogBody key={request.taskId} request={request} />
}

function findConnector(
  report: ConnectorRuntimeReport,
  ref: { connector_type: string; connector_id: number } | undefined,
): ConnectorRuntimeConnector | null {
  if (!ref) return null
  return (
    report.connectors.find(
      c =>
        c.connector_ref.connector_type === ref.connector_type
        && c.connector_ref.connector_id === ref.connector_id,
    ) ?? null
  )
}

function connectorKeyOf(ref: { connector_type: string; connector_id: number }): string {
  return `${ref.connector_type}:${ref.connector_id}`
}

type FieldErrorLocation =
  | { scope: "dialog" }
  | { scope: "connector"; connectorKey: string }
  | { scope: "field"; connectorKey: string; draftKey: string }

/**
 * Where a failed save's error attaches, given the report the dialog is
 * currently showing. A location the current report can no longer find (the
 * connector's declaration changed, or a 409 refresh already collapsed the row
 * into "already filled") falls back to the whole dialog rather than being
 * silently dropped.
 */
function locateFieldError(
  report: ConnectorRuntimeReport,
  disposition: ConnectorRuntimeFailureDisposition,
): FieldErrorLocation {
  const { connectorRef, key } = disposition.locate
  if (!connectorRef) return { scope: "dialog" }
  const connector = findConnector(report, connectorRef)
  if (!connector) return { scope: "dialog" }
  const connectorKey = connectorKeyOf(connectorRef)
  if (key === undefined) return { scope: "connector", connectorKey }
  const input = connector.inputs.find(i => i.key === key)
  if (!input) return { scope: "dialog" }
  return { scope: "field", connectorKey, draftKey: connectorRuntimeInputDraftKey(connectorRef, key) }
}

function uniqueKeys(locations: Array<{ key: string }>): string[] {
  return Array.from(new Set(locations.map(l => l.key)))
}

interface FieldErrorState {
  disposition: ConnectorRuntimeFailureDisposition
  location: FieldErrorLocation
}

function ConnectorRuntimeDialogBody({ request }: { request: ConnectorRuntimeDialogRequest }) {
  const pathname = usePathname()
  const pathnameRef = useRef(pathname)
  pathnameRef.current = pathname

  const { close } = useConnectorRuntimeDialog()
  const { sendMessage } = useApp()
  const { t } = useI18n()

  const aliveRef = useRef(true)
  useEffect(() => () => { aliveRef.current = false }, [])

  const requestRef = useRef(request)
  requestRef.current = request

  const [report, setReport] = useState<ConnectorRuntimeReport | null>(null)
  const [visible, setVisible] = useState(false)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [invalidDraftKeys, setInvalidDraftKeys] = useState<Set<string>>(new Set())
  const [submitting, setSubmitting] = useState(false)
  const [fieldError, setFieldError] = useState<FieldErrorState | null>(null)
  const [lastAlsoResend, setLastAlsoResend] = useState(false)
  const [sendFailed, setSendFailed] = useState(false)
  const [resending, setResending] = useState(false)

  // Read on mount and on every subsequent request for this same task (the
  // dialog is already open and a new terminal frame retargeted it): the
  // route gate runs before the request even goes out, and again right
  // before the dialog would become visible, since the user is free to
  // navigate away from a host page while this read is in flight.
  useEffect(() => {
    if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
      close("not-shown")
      return
    }
    const seqAtStart = request.seq
    let cancelled = false
    fetchTaskConnectorRuntimeRequirements(request.taskId).then((result) => {
      if (cancelled || !aliveRef.current || requestRef.current.seq !== seqAtStart) return
      if (!result.ok) {
        console.warn(
          "[connector-runtime] requirements read failed",
          result.kind === "http" ? result.status : result.kind,
        )
        close("not-shown")
        return
      }
      if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
        close("not-shown")
        return
      }
      const outcome = resolveDialogOutcome(result.report)
      if (outcome.kind === "met") {
        close("not-shown")
        return
      }
      setReport(result.report)
      setFieldError(null)
      setSendFailed(false)
      setVisible(true)
    })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.seq])

  // Leaving the host pages closes a dialog the user has already seen. A move
  // between two host pages is handled by the outer task-switch cleanup, not
  // by this effect noticing the path changed.
  useEffect(() => {
    if (visible && !isConnectorRuntimeDialogHostPath(pathname)) close("left-host")
  }, [visible, pathname, close])

  const outcome: DialogOutcome | null = report ? resolveDialogOutcome(report) : null
  const submitItems = report ? buildSubmitItems(report, drafts) : []
  const hasInvalidObjectDraft = invalidDraftKeys.size > 0
  const canSubmit = isSubmitEnabled(submitItems, hasInvalidObjectDraft)
  const hasResendPayload = request.resendPayload !== null
  const actions = outcome ? resolveDialogActions(outcome, hasResendPayload) : []

  const handleDraftChange = (connector: ConnectorRuntimeConnector, key: string, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, key)
    setDrafts(prev => ({ ...prev, [draftKey]: value }))
  }

  const handleObjectBlur = (connector: ConnectorRuntimeConnector, key: string, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, key)
    let invalid = false
    if (value.trim() !== "") {
      try {
        const parsed: unknown = JSON.parse(value)
        invalid = typeof parsed !== "object" || parsed === null || Array.isArray(parsed)
      } catch {
        invalid = true
      }
    }
    setInvalidDraftKeys((prev) => {
      const next = new Set(prev)
      if (invalid) next.add(draftKey)
      else next.delete(draftKey)
      return next
    })
  }

  const doResend = async (): Promise<boolean> => {
    const snapshot = requestRef.current.resendPayload
    if (!snapshot) return true
    try {
      await sendMessage(snapshot.text, { clientMessageId: generateClientMessageId() }, snapshot.files)
      return true
    } catch {
      return false
    }
  }

  const handleSave = async (alsoResend: boolean) => {
    if (!report || submitting) return
    const seqAtStart = request.seq
    const items = buildSubmitItems(report, drafts)
    setSubmitting(true)
    setLastAlsoResend(alsoResend)
    const result = await submitTaskConnectorRuntimeValues(request.taskId, items)
    if (!aliveRef.current) return
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the save
      // was in flight; the result is stale, but `submitting` must still
      // reset or the save buttons and close handlers stay stuck forever.
      setSubmitting(false)
      return
    }

    if (!result.ok) {
      const disposition = classifySubmitFailure(result, report)
      setFieldError({ disposition, location: locateFieldError(report, disposition) })
      setSubmitting(false)
      if (disposition.refresh) {
        const refreshed = await fetchTaskConnectorRuntimeRequirements(request.taskId)
        if (!aliveRef.current || requestRef.current.seq !== seqAtStart) return
        if (refreshed.ok) setReport(refreshed.report)
      }
      return
    }

    setReport(result.report)
    const newOutcome = resolveDialogOutcome(result.report)
    if (newOutcome.kind === "fillable" || newOutcome.kind === "nothing_fillable") {
      // Still blocked on something this dialog can collect (or, for
      // nothing_fillable, on nothing the user can act on beyond "Got it"):
      // stay open, re-render from the fresh report.
      setSubmitting(false)
      setFieldError(null)
      return
    }

    if (newOutcome.kind === "unsupported_only") {
      const keys = uniqueKeys(newOutcome.blocking)
      toast(t("connectorRuntime.onlyUnsupportedRemaining", { keys: keys.join(", ") }))
    }

    if (alsoResend) {
      const sent = await doResend()
      if (!aliveRef.current) return
      if (requestRef.current.seq !== seqAtStart) {
        // Same reason as the earlier seq check: a newer request retargeted
        // this dialog instance while the resend was in flight, so this
        // result is stale, but `submitting` must still reset.
        setSubmitting(false)
        return
      }
      if (!sent) {
        setSubmitting(false)
        setSendFailed(true)
        return
      }
    }
    setSubmitting(false)
    close(alsoResend ? "resent" : "dismissed")
  }

  const handleRetryResend = async () => {
    // A resend is one billed model call plus a possibly side-effecting tool
    // run; a double click here must not fire it twice.
    if (resending) return
    const seqAtStart = request.seq
    setResending(true)
    const sent = await doResend()
    if (!aliveRef.current) return
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the
      // resend was in flight; the result is stale, but `resending` must
      // still reset or the retry button stays stuck forever.
      setResending(false)
      return
    }
    setResending(false)
    if (sent) {
      setSendFailed(false)
      close("resent")
    }
  }

  const handleDismiss = () => {
    if (submitting) return
    close("dismissed")
  }

  const handleOpenChange = (open: boolean) => {
    if (open || submitting) return
    handleDismiss()
  }

  const dialogFieldError = fieldError && fieldError.location.scope === "dialog" ? fieldError.disposition : null

  if (!visible || !report || !outcome) return null

  return (
    <Dialog open={visible} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("connectorRuntime.title")}</DialogTitle>
          <DialogDescription>{t("connectorRuntime.description")}</DialogDescription>
        </DialogHeader>

        {outcome.kind === "unsupported_only" && (
          <p className="text-sm text-muted-foreground">{t("connectorRuntime.onlyUnsupportedNotice")}</p>
        )}

        {dialogFieldError && (
          <p className="text-sm text-destructive" role="alert">
            {translateFailure(t, dialogFieldError.messageKey)}
          </p>
        )}

        {sendFailed ? (
          <div className="space-y-3">
            <p className="text-sm text-destructive">{t("connectorRuntime.sendFailed")}</p>
            <Button disabled={resending} onClick={handleRetryResend}>{t("connectorRuntime.actions.resend")}</Button>
          </div>
        ) : (
          <div className="space-y-4">
            {report.connectors.map((connector) => {
              const connectorKey = connectorKeyOf(connector.connector_ref)
              const connectorError =
                fieldError
                && fieldError.location.scope === "connector"
                && fieldError.location.connectorKey === connectorKey
                  ? fieldError.disposition
                  : null
              const stillMissing =
                outcome.kind === "fillable"
                  ? uniqueKeys(outcome.blocking.filter(b => connectorKeyOf(b.connectorRef) === connectorKey))
                  : []
              return (
                <div key={connectorKey} className="space-y-2 rounded-md border p-3">
                  <div className="text-sm font-medium">{connector.name}</div>
                  {connectorError && (
                    <p className="text-sm text-destructive" role="alert">
                      {translateFailure(t, connectorError.messageKey)}
                    </p>
                  )}
                  {connector.inputs.map((input) => {
                    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.key)
                    const acceptedKeyName = input.section !== "context" || isAcceptedRuntimeKeyName(input.key)
                    const fieldLevelError =
                      fieldError
                      && fieldError.location.scope === "field"
                      && fieldError.location.draftKey === draftKey
                        ? fieldError.disposition
                        : null

                    if (input.section === "context" && input.satisfied) {
                      return (
                        <div key={input.key} className="text-sm">
                          <span className="font-medium">{input.key}</span>{" "}
                          <span className="text-muted-foreground">{t("connectorRuntime.filled")}</span>
                        </div>
                      )
                    }

                    if (input.section !== "context") {
                      return (
                        <div key={input.key} className="text-sm">
                          <div className="font-medium">{input.key}</div>
                          <p className="text-muted-foreground">{t("connectorRuntime.unsupportedNote")}</p>
                        </div>
                      )
                    }

                    // Unfilled context row while only "Got it" is offered:
                    // no input control and no "saved, cannot be changed"
                    // hint, since there is no save entry point in this shape.
                    if (outcome.kind === "unsupported_only" || outcome.kind === "nothing_fillable") {
                      return (
                        <div key={input.key} className="text-sm">
                          <span className="font-medium">{input.key}</span>
                          {!acceptedKeyName && (
                            <p className="text-destructive">{t("connectorRuntime.keyNameWarning")}</p>
                          )}
                        </div>
                      )
                    }

                    return (
                      <div key={input.key} className="space-y-1">
                        <Label htmlFor={`connector-runtime-${draftKey}`}>{input.key}</Label>
                        {input.type === "object" ? (
                          <Textarea
                            id={`connector-runtime-${draftKey}`}
                            value={drafts[draftKey] ?? ""}
                            aria-invalid={invalidDraftKeys.has(draftKey)}
                            onChange={e => handleDraftChange(connector, input.key, e.target.value)}
                            onBlur={e => handleObjectBlur(connector, input.key, e.target.value)}
                          />
                        ) : (
                          <Input
                            id={`connector-runtime-${draftKey}`}
                            type="text"
                            value={drafts[draftKey] ?? ""}
                            onChange={e => handleDraftChange(connector, input.key, e.target.value)}
                          />
                        )}
                        {invalidDraftKeys.has(draftKey) && (
                          <p className="text-sm text-destructive">{t("connectorRuntime.objectInvalid")}</p>
                        )}
                        <p className="text-sm text-muted-foreground">{t("connectorRuntime.contextNote")}</p>
                        {!acceptedKeyName && (
                          <p className="text-sm text-destructive">{t("connectorRuntime.keyNameWarning")}</p>
                        )}
                        {fieldLevelError && (
                          <p className="text-sm text-destructive" role="alert">
                            {translateFailure(t, fieldLevelError.messageKey, { key: input.key })}
                          </p>
                        )}
                      </div>
                    )
                  })}
                  {stillMissing.length > 0 && (
                    <p className="text-sm text-muted-foreground">
                      {t("connectorRuntime.stillMissingAfterSave", { keys: stillMissing.join(", ") })}
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {!sendFailed && (
          <DialogFooter>
            {actions.includes("acknowledge") && (
              <Button variant="outline" onClick={handleDismiss}>
                {t("connectorRuntime.actions.acknowledge")}
              </Button>
            )}
            {actions.includes("saveOnly") && (
              <Button
                variant={actions.includes("saveAndResend") ? "outline" : "default"}
                disabled={!canSubmit || submitting}
                onClick={() => handleSave(false)}
              >
                {t("connectorRuntime.actions.saveOnly")}
              </Button>
            )}
            {actions.includes("saveAndResend") && (
              <Button disabled={!canSubmit || submitting} onClick={() => handleSave(true)}>
                {t("connectorRuntime.actions.saveAndResend")}
              </Button>
            )}
            {dialogFieldError?.retry && (
              <Button variant="outline" onClick={() => handleSave(lastAlsoResend)}>
                {t("connectorRuntime.actions.retry")}
              </Button>
            )}
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}
