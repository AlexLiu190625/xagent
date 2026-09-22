"use client"

import React, { useEffect, useRef, useState } from "react"
import { usePathname } from "next/navigation"

import { readRetryWithNewId, readSendDisposition } from "@/components/chat/clarification-delivery"
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
  NOOP_ACTIONS,
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
  isSubmittableObjectValue,
  isTypeMismatchDispositionStale,
  resolveDialogActions,
  resolveDialogOutcome,
  submitTaskConnectorRuntimeValues,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeErrorMessageKey,
  type ConnectorRuntimeFailureDisposition,
  type ConnectorRuntimeInput,
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
  typeUnknown: "connectorRuntime.errors.typeUnknown",
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
 * The same failure, worded for whole-dialog scope. That scope is where
 * locateFieldError falls back when the row a failure named is not one the
 * current report renders an editable control for -- the report dropped it,
 * or a refresh collapsed it into "already filled" -- so two kinds of text
 * stop being true there and get a variant instead of going through
 * translateFailure:
 *
 * - `conflict` is the only text carrying a `{key}` placeholder, and there
 *   is no key here to fill it with.
 * - `typeObject` and `typeString` each name the type one specific field
 *   needs. With no such field on screen that sentence points at nothing
 *   the user can act on, so the variant says only the part that stays
 *   true: the save was rejected over this value. The hint itself is not
 *   dropped -- a rejection the user saw must not vanish on a refresh they
 *   did not ask for -- only reworded, and a report that brings the row
 *   back brings the named text back with it.
 *
 * `typeUnknown` needs no variant: it already names neither field nor type.
 */
function translateDialogScopeFailure(
  t: (key: TranslationKey, vars?: TranslationVariables) => string,
  messageKey: ConnectorRuntimeErrorMessageKey,
): string {
  if (messageKey === "conflict") return t("connectorRuntime.errors.conflictNoKey")
  if (messageKey === "typeObject" || messageKey === "typeString") {
    return t("connectorRuntime.errors.typeNoField")
  }
  return translateFailure(t, messageKey)
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

  // Widget and share pages mount this component with no
  // ConnectorRuntimeDialogProvider above it by design (see the docstring
  // above), so retainOnlyTask here is the shared no-op default -- reference-
  // equal to the module's own constant, since a real provider's action is a
  // distinct function from useMemo. Neither effect below calls it in that
  // case: the call would be harmless, but it would also trip the dev-only
  // "called outside provider" warning that exists to catch an actual wiring
  // mistake, not this expected shape.
  const hasProvider = retainOnlyTask !== NOOP_ACTIONS.retainOnlyTask
  const hasProviderRef = useRef(hasProvider)
  hasProviderRef.current = hasProvider

  // Task-switch cleanup: no cleanup function of its own. Combining this with
  // the unmount effect below into one effect would run the unmount cleanup
  // on every task change too, which would erase a same-render first-gate
  // snapshot before anything could read it.
  useEffect(() => {
    if (!hasProvider) return
    retainOnlyTask(state.taskId)
  }, [state.taskId, retainOnlyTask, hasProvider])

  // Unmount cleanup: a separate effect with an empty dependency array, read
  // through a ref so it always calls the latest function without needing to
  // be in that array (matches the workforce pages' own unmount-cleanup shape).
  // hasProviderRef is read the same way for the same reason.
  useEffect(() => () => {
    if (hasProviderRef.current) cleanupRef.current(null)
  }, [])

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

// Why a draft failed the object-field blur check: "invalid" for anything
// that is not JSON-object-shaped, "empty" for `{}`, which parses fine but
// isSubmittableObjectValue (connector-runtime-api.ts) rejects because the
// server treats it as a blank context value. The row's error message reads
// this to show the reason-specific hint instead of a generic one.
type InvalidObjectDraftReason = "invalid" | "empty"

/**
 * Whether an invalid-object mark for this input is still live. Only a
 * `context` row the current report leaves unsatisfied and still declares
 * `object`-typed renders the textarea whose blur handler can clear such a
 * mark; against any other row the mark is unreachable. Both the submit gate
 * and the row's own error message read this one predicate, so the button can
 * never be disabled by an error the row does not show, and the row can never
 * show an error that leaves the button enabled.
 */
function hasLiveInvalidObjectMark(
  connector: ConnectorRuntimeConnector,
  input: ConnectorRuntimeInput,
  invalidDraftKeys: Map<string, InvalidObjectDraftReason>,
): boolean {
  return (
    input.section === "context"
    && !input.satisfied
    && input.type === "object"
    && invalidDraftKeys.has(connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type))
  )
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
 *
 * Called fresh from render against whatever report the dialog currently
 * holds, never cached alongside the disposition that produced it. Every
 * disposition that asks for a refresh (`refresh: true`) is followed by
 * exactly one `setReport` call from one of three places -- the failed
 * save's own post-refresh install, the read effect's same-task re-request,
 * or that same effect's already-visible "met" branch -- and none of them
 * needs to also re-derive or clear a location: recomputing this on every
 * render against whatever `report` state currently holds means all three
 * land on the right answer without any of them knowing this function
 * exists.
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
  // Every key-bearing failure reason this locates is section-scoped to
  // "context" (type_mismatch.context., empty_value.context., conflict.context.
  // in connector-runtime-api.ts), and the draft key below is keyed by section.
  // Matching key alone would let a same-named row in another section (e.g.
  // "secrets") win the find, producing a draft key that no context row
  // holds — the error would then attach to nothing instead of falling back
  // to the whole-dialog scope this function otherwise guarantees.
  const input = connector.inputs.find(i => i.section === "context" && i.key === key)
  // A row the current report already reports satisfied renders the
  // "already filled" shortcut below instead of an editable control (or any
  // field-level error), most often because a refresh this same disposition
  // asked for just collapsed it: attaching here would pick a draft key
  // nothing on screen renders, so this falls back to the whole dialog
  // instead, same as a row it cannot find at all.
  if (!input || input.satisfied) return { scope: "dialog" }
  return {
    scope: "field",
    connectorKey,
    draftKey: connectorRuntimeInputDraftKey(connectorRef, input.section, key, input.type),
  }
}

function uniqueKeys(locations: Array<{ key: string }>): string[] {
  return Array.from(new Set(locations.map(l => l.key)))
}

interface FieldErrorState {
  disposition: ConnectorRuntimeFailureDisposition
}

function ConnectorRuntimeDialogBody({ request }: { request: ConnectorRuntimeDialogRequest }) {
  const pathname = usePathname()
  const pathnameRef = useRef(pathname)
  pathnameRef.current = pathname

  const { close } = useConnectorRuntimeDialog()
  const { sendMessage } = useApp()
  const { t } = useI18n()

  const aliveRef = useRef(true)
  useEffect(() => {
    // The assignment is not redundant with useRef(true): React 18 StrictMode
    // double-invokes this effect in development (mount -> cleanup -> mount),
    // and the cleanup below runs in between. Without resetting here, every
    // guard that reads this ref would short-circuit for a component that is
    // genuinely still mounted, and the dialog would never become visible.
    aliveRef.current = true
    return () => { aliveRef.current = false }
  }, [])

  const requestRef = useRef(request)
  requestRef.current = request

  const [report, setReport] = useState<ConnectorRuntimeReport | null>(null)
  const [visible, setVisible] = useState(false)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [invalidDraftKeys, setInvalidDraftKeys] = useState<Map<string, InvalidObjectDraftReason>>(new Map())
  const [submitting, setSubmitting] = useState(false)
  const [fieldError, setFieldError] = useState<FieldErrorState | null>(null)
  const [lastAlsoResend, setLastAlsoResend] = useState(false)
  // The clientMessageId of the snapshot the "saved but not sent" panel is
  // about, or null when no send has failed. It carries that id rather than
  // being a bare flag because the panel names one message while its retry
  // button sends whichever snapshot the request currently holds, and the
  // request can stop carrying that snapshot underneath it: a same-task
  // retarget swaps in a newer candidate (openForTask), and a settlement
  // frame for this task takes it away without moving `seq` at all
  // (forgetDelivery).
  const [sendFailedSnapshotId, setSendFailedSnapshotId] = useState<string | null>(null)
  // Derived every render against the snapshot the request currently
  // carries, the same way `activeFieldError` below is re-derived rather
  // than cached: the panel and its retry button must be about the same
  // message in every painted frame, including the first frame after a
  // retarget commits. An effect that noticed the two had come apart and
  // reset the panel afterwards left that first frame actionable, and never
  // ran at all for a removal that does not move `seq`.
  const sendFailed = sendFailedSnapshotId !== null
    && sendFailedSnapshotId === request.resendPayload?.clientMessageId
  const [resending, setResending] = useState(false)
  // The client message id the most recent unresolved resend attempt used,
  // together with the clientMessageId of the snapshot it was sent for, so a
  // further retry can reuse the id only while it is still retrying that same
  // snapshot -- see doResend, which is the only reader and writer.
  const resendMessageIdRef = useRef<{ forSnapshotId: string, clientMessageId: string } | null>(null)

  // Read on mount and on every subsequent request for this same task (the
  // dialog is already open and a new terminal frame retargeted it): the
  // route gate runs before the request even goes out, and again right
  // before the dialog would become visible, since the user is free to
  // navigate away from a host page while this read is in flight.
  //
  // `visible` is read at the moment this effect starts, which is exactly
  // right here: setVisible is only ever called with `true` in this
  // component, so if the dialog was already showing something when this
  // request came in, it is still showing it by the time the fetch below
  // resolves (any path that would make it stop -- unmount, a task switch, a
  // host-page departure -- clears `request` and is caught by the seq/alive
  // checks first). A once-visible dialog must not vanish out from under a
  // user who is mid-draft: a re-read for the same task only happens because
  // another terminal frame retargeted this instance, not because the user
  // did anything, so it must never read as a decision the user made. The
  // same reasoning covers a still-live rejection message and a still-live
  // "saved but not sent" panel below: neither is cleared just because this
  // re-read ran. Whether that panel still has a snapshot to be about is a
  // separate question this effect does not answer -- `sendFailed` above
  // derives it from the request on every render, including the retargets
  // that never reach this effect at all.
  useEffect(() => {
    if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
      close("not-shown")
      return
    }
    const wasVisible = visible
    const seqAtStart = request.seq
    let cancelled = false
    fetchTaskConnectorRuntimeRequirements(request.taskId).then((result) => {
      if (cancelled || !aliveRef.current || requestRef.current.seq !== seqAtStart) return
      if (!result.ok) {
        console.warn(
          "[connector-runtime] requirements read failed",
          result.kind === "http" ? result.status : result.kind,
        )
        // Keep whatever the user is already looking at (report and draft)
        // rather than discarding it over a transient read failure.
        if (!wasVisible) close("not-shown")
        return
      }
      if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
        close("not-shown")
        return
      }
      const outcome = resolveDialogOutcome(result.report)
      if (outcome.kind === "met") {
        if (!wasVisible) {
          close("not-shown")
          return
        }
        // Same path handleSave's post-save refresh already takes when a
        // refresh finds nothing left to fill (see "leaves a way out..."
        // below): install the report and let the footer collapse to "Got
        // it" instead of silently discarding the user's in-progress draft.
        setReport(result.report)
        return
      }
      setReport(result.report)
      // This branch only runs because another terminal frame for the same
      // task retargeted an already-open dialog (see this effect's opening
      // comment) -- never because the user resolved anything -- so a live
      // rejection or send failure must survive it. A type-mismatch hint is
      // cleared only once this fresher report proves it stale (the row's
      // declared type changed under it, via isTypeMismatchDispositionStale,
      // the same check handleSave's own post-failure refresh uses below); a
      // 409 conflict hint has no such report-derived staleness condition, so
      // it is left alone here the same way handleSave's refresh already
      // leaves it alone. The "saved but not sent" panel is not about the
      // report at all -- nothing a re-read can show would make a send
      // failure no longer have happened -- so nothing the read returns
      // clears it either. The only things that do are a resend that
      // actually completes (handleRetryResend), unmounting, and the
      // request no longer carrying the snapshot the panel is about, which
      // `sendFailed` derives during render rather than any effect here.
      setFieldError(prev => (prev && isTypeMismatchDispositionStale(prev.disposition, result.report) ? null : prev))
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
  // Re-derived every render against the report currently on screen, rather
  // than resolved once into `fieldError` and cached there -- see
  // locateFieldError's own docstring for why a cached location goes stale
  // the moment any of this dialog's three setReport call sites installs a
  // fresher report.
  const activeFieldError = fieldError && report
    ? { disposition: fieldError.disposition, location: locateFieldError(report, fieldError.disposition) }
    : null
  const submitItems = report ? buildSubmitItems(report, drafts) : []
  // Only a mark on a row the current report still renders an editable control
  // for may gate submission. A key a refreshed report reports satisfied loses
  // its textarea, so its mark could never be cleared again -- submit would
  // stay disabled with no error anywhere on screen. Derived from the report
  // during render rather than pruned at each point that installs one, because
  // there are three such points today and a fourth would silently reintroduce
  // this. Reads the same `hasLiveInvalidObjectMark` predicate the row
  // renderer reads for its error message, so the two can never disagree.
  const hasInvalidObjectDraft = report !== null && report.connectors.some(connector =>
    connector.inputs.some(input => hasLiveInvalidObjectMark(connector, input, invalidDraftKeys)),
  )
  const canSubmit = isSubmitEnabled(submitItems, hasInvalidObjectDraft)
  // Whether any submission-shaped action is in flight: an explicit save
  // (`submitting`, which may itself run a resend as part of "save and
  // resend") or a standalone retry resend from the send-failed panel
  // (`resending`). The footer `canSubmitNow` gates below is only ever
  // rendered while `sendFailed` is false, and `resending` only runs while
  // `sendFailed` is true -- its own button lives inside that panel -- so
  // folding `resending` into `busy` makes no difference to the footer
  // buttons today. What it does gate is `handleDismiss`/`handleOpenChange`
  // further down, which must keep the dialog open while either kind of
  // submission has not yet settled.
  const busy = submitting || resending
  // The one value every entry point into a submission reads: both footer
  // save buttons, the retry button a retryable failure offers, and
  // handleSave itself. The retry button used to be rendered off
  // `dialogFieldError.retry` alone, so a draft edited into an invalid object
  // after a failed save stayed submittable through it while the save buttons
  // were correctly disabled. That matters because buildSubmitItems drops an
  // unparsable object draft instead of failing: such a batch writes every
  // other field, silently loses that one and closes the dialog -- and a
  // stored context value is immutable, so there is no correcting it
  // afterwards. handleSave re-checks rather than trusting its callers, so a
  // fourth entry point cannot reintroduce the same bypass.
  const canSubmitNow = canSubmit && !busy
  const hasResendPayload = request.resendPayload !== null
  const actions = outcome ? resolveDialogActions(outcome, hasResendPayload) : []
  // Whether this shape offers any way to submit. The row renderer asks this
  // instead of listing the outcome kinds that offer none, because that list
  // was one kind short: a `met` report reaches the render whenever one is
  // installed into a dialog that stays open -- the refresh a failed save
  // triggers, a same-task re-request's read that finds nothing missing, and
  // a save-and-resend's own successful save for as long as its resend is in
  // flight -- and an unfilled *optional* context key inside one was still
  // drawn as an editable field with no button able to send it. Derived from
  // the action set, so the rows and the footer cannot disagree about whether
  // saving is possible.
  const hasSaveEntryPoint = actions.includes("saveOnly")
  // A met report that reached the render still carries the snapshot of the
  // message that failed, and this shape offers no way to send it: the
  // footer collapses to "Got it", and the save-and-resend button a
  // fillable report offers cannot be reused here because a met report
  // produces no submittable items, which leaves it permanently disabled
  // (isSubmitEnabled in connector-runtime-api.ts). Nothing is missing any
  // more, so the header must stop saying one is and must say instead what
  // did not happen and what the user can still do. Not while the
  // send-failed panel is up: that panel is about a send this dialog
  // already attempted and carries its own retry button, so pointing the
  // user back at the message box there would contradict the button
  // directly under it. Nor while this dialog's own save-and-resend is
  // still sending that same message: saying it was not resent would be
  // false and would invite a second send while the first is still on the
  // wire.
  const metHoldingSnapshot = outcome?.kind === "met" && hasResendPayload && !sendFailed && !busy

  const handleDraftChange =(connector: ConnectorRuntimeConnector, input: ConnectorRuntimeInput, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
    setDrafts(prev => ({ ...prev, [draftKey]: value }))
  }

  const handleObjectBlur = (connector: ConnectorRuntimeConnector, input: ConnectorRuntimeInput, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
    let reason: InvalidObjectDraftReason | null = null
    if (value.trim() !== "") {
      try {
        const parsed: unknown = JSON.parse(value)
        // Same predicate buildSubmitItems filters on, so a draft this marks
        // valid is never the one buildSubmitItems silently drops. A value
        // that fails it for being array/null/non-object is "invalid"; one
        // that is object-shaped but empty is "empty" -- the row's error
        // message tells the two apart.
        if (!isSubmittableObjectValue(parsed)) {
          reason = typeof parsed === "object" && parsed !== null && !Array.isArray(parsed) ? "empty" : "invalid"
        }
      } catch {
        reason = "invalid"
      }
    }
    setInvalidDraftKeys((prev) => {
      const next = new Map(prev)
      if (reason) next.set(draftKey, reason)
      else next.delete(draftKey)
      return next
    })
  }

  type ResendOutcome = "sent" | "failed" | "nothing-to-send"

  const doResend = async (): Promise<ResendOutcome> => {
    const snapshot = requestRef.current.resendPayload
    if (!snapshot) {
      // Unreachable today: both callers only reach this after a precondition
      // that implies a snapshot exists -- handleSave's canResendNow (which
      // itself requires the "met" outcome the saveAndResend button promised
      // to be resendable) and handleRetryResend's sendFailed precondition
      // (set only right after a resend that read a snapshot). Kept distinct
      // from "sent" and "failed" so a future caller that does reach it is
      // not misreported as either a completed resend or a failed one.
      console.warn("[connector-runtime] resend attempted with no snapshot to send")
      return "nothing-to-send"
    }
    // Reuse the id the last unresolved attempt for this snapshot used,
    // unless that attempt's own outcome already proved it, or the snapshot
    // itself is not the one that id was minted for any more: the very first
    // attempt (ref starts null on a fresh dialog instance), any attempt an
    // earlier one proved not delivered, and any attempt whose carried id was
    // minted for a different snapshot, all mint fresh below. The comparison
    // is against the snapshot's own clientMessageId, not request.seq: a
    // same-task retarget that leaves this snapshot in place (openForTask's
    // "kept") bumps seq without invalidating this id, while one that swaps in
    // a different snapshot (a newer candidate staged while this dialog was
    // open, openForTask's "staged"/"stashed") must not let the old id carry
    // over onto different text. This is never the id of the original send
    // that opened this dialog -- the server has recorded that one as FAILED
    // and would bounce a same-id retry of it -- only ever an id this same
    // doResend minted.
    const carriedId = resendMessageIdRef.current
    const clientMessageId = carriedId && carriedId.forSnapshotId === snapshot.clientMessageId
      ? carriedId.clientMessageId
      : generateClientMessageId()
    try {
      // Matches every other programmatic resend call site in the app
      // (clarification-form.tsx, workforce-builder.tsx, agent-builder.tsx):
      // without force, a duplicate of this exact text still pending from an
      // earlier send on this same connection throws instead of sending. That
      // duplicate check only ever matches a *different* clientMessageId than
      // the one it is scanning for -- its own match condition excludes the
      // id under retry -- so a same-id retry never reaches it either way and
      // this flag changes nothing for it; force is what lets a fresh-id retry
      // (the original send still pending, unacknowledged) go out at all.
      // targetTaskId names the task this dialog is for rather than letting
      // sendMessage default to whichever task the page is currently
      // showing, the same way the app's other cross-task send sites
      // (workforce-builder.tsx, agent-builder.tsx) name theirs. The two
      // can differ: the effect that drops a request belonging to a task
      // the user has navigated away from is a plain useEffect, so it runs
      // after the browser has painted, and the frame where the new task is
      // already current while this dialog still holds the old one's
      // snapshot is a frame the user can click in. Naming the task turns
      // that click from a message delivered into the wrong conversation
      // into a send that fails and says so on the panel below.
      // request.taskId rather than snapshot.taskId: the two are always
      // equal (every candidate openForTask can hand over is filtered by
      // task id), and this body is mounted keyed on request.taskId, so it
      // cannot change for the life of this instance.
      await sendMessage(
        snapshot.text,
        { clientMessageId, force: true, targetTaskId: request.taskId },
        snapshot.files,
      )
      resendMessageIdRef.current = null
      return "sent"
    } catch (error) {
      // Matches the read path's warn so a failing resend leaves the same
      // diagnostic signal. Carries the fixed prefix alone: unlike the read
      // path there is no closed-set status to report here, and the rejection
      // value is arbitrary, so logging it could carry message content.
      console.warn("[connector-runtime] resend failed")
      // A definite not_sent/rejected disposition, or the server explicitly
      // demanding a new id, means this id is spent -- the next retry mints
      // fresh. Every other case, including an outcome_unknown disposition
      // and a plain exception that carries no disposition at all, leaves the
      // possibility open that the server already durably accepted this
      // attempt, so the next retry reuses this same id rather than risking
      // the same turn running twice under a second one.
      const mustMintNewId = (
        readRetryWithNewId(error)
        || readSendDisposition(error) === "not_sent"
        || readSendDisposition(error) === "rejected"
      )
      resendMessageIdRef.current = mustMintNewId ? null : { forSnapshotId: snapshot.clientMessageId, clientMessageId }
      return "failed"
    }
  }

  const handleSave = async (alsoResend: boolean) => {
    if (!report || !canSubmitNow) return
    const seqAtStart = request.seq
    const items = buildSubmitItems(report, drafts)
    setSubmitting(true)
    setLastAlsoResend(alsoResend)
    const result = await submitTaskConnectorRuntimeValues(request.taskId, items)
    if (!aliveRef.current) {
      // The dialog unmounted while the save was in flight (a task switch,
      // or leaving the host pages) -- submitTaskConnectorRuntimeValues has
      // no abort signal, so a result.ok here already wrote an immutable
      // value server-side, and the resend the user asked for is never
      // going to run. Nothing else in this render tree still holds the
      // state to report that; this toast is the only place left to say
      // so, matching the sibling "superseded" branch right below.
      if (alsoResend && result.ok) toast(t("connectorRuntime.savedNotResentUnmounted"))
      return
    }
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the save
      // was in flight; the result is stale, but `submitting` must still
      // reset or the save buttons and close handlers stay stuck forever.
      // A successful "save and resend" whose resend never got to run needs
      // to say so, matching the other two "did not resend" paths below --
      // this one only fires when the save itself landed, since a rejected
      // save has nothing that was "saved but not resent" to report.
      if (alsoResend && result.ok) {
        toast(t("connectorRuntime.savedNotResentSuperseded"))
      }
      setSubmitting(false)
      return
    }

    if (!result.ok) {
      const disposition = classifySubmitFailure(result, report)
      setFieldError({ disposition })
      if (!disposition.refresh) {
        setSubmitting(false)
        return
      }
      // The save buttons stay disabled across the refresh: re-enabling before
      // it settles lets a second submit go out built from the report this
      // refresh is about to replace. Every path that settles the dialog below
      // still resets `submitting`, or the buttons and the close handlers stay
      // stuck forever; only an unmounted instance skips it, since it has no
      // buttons left to re-enable.
      const refreshed = await fetchTaskConnectorRuntimeRequirements(request.taskId)
      if (!aliveRef.current) {
        // This branch only runs after the save itself failed a few lines
        // above, so nothing was written server-side -- there is no "saved
        // but not resent" fact to report here, unlike the early return
        // right after the save POST above.
        return
      }
      if (requestRef.current.seq !== seqAtStart) {
        setSubmitting(false)
        return
      }
      if (refreshed.ok) {
        setReport(refreshed.report)
        // A type-mismatch hint names a specific declared type; once the
        // refreshed report shows this row now declares the other type, that
        // hint no longer describes the row it is attached to and must be
        // cleared outright rather than left to describe a type this row no
        // longer has.
        if (isTypeMismatchDispositionStale(disposition, refreshed.report)) setFieldError(null)
      }
      setSubmitting(false)
      return
    }

    setReport(result.report)
    // A save that landed has no rejection left to show, even on the one path
    // below that renders before this dialog settles (a "save and resend"
    // whose report comes back met, which awaits the resend before closing):
    // without this, that rejection would re-derive against the fresh report
    // and land at whole-dialog scope, next to a send-failed panel for a save
    // that in fact succeeded.
    setFieldError(null)
    const newOutcome = resolveDialogOutcome(result.report)
    // Only a met report can carry the resend the primary button promised.
    // `unsupported_only` still lacks a required secret this dialog cannot
    // collect, and `nothing_fillable` is a connector the server still reports
    // unavailable with nothing left for the user to fill; the backend rejects
    // either while it builds the turn's tool list, so a resend would fail on
    // the same gate and put a second failure in the conversation. Neither
    // resends, and because the button promised one, both say so.
    const canResendNow = newOutcome.kind === "met"

    if (newOutcome.kind === "unsupported_only") {
      const keys = uniqueKeys(newOutcome.blocking).join(", ")
      toast(alsoResend
        ? t("connectorRuntime.savedNotResentUnsupported", { keys })
        : t("connectorRuntime.onlyUnsupportedRemaining", { keys }))
    } else if (alsoResend && newOutcome.kind === "nothing_fillable") {
      toast(t("connectorRuntime.savedNotResentUnavailable"))
    } else if (alsoResend && newOutcome.kind === "fillable") {
      // The save landed and the refreshed report still leaves a required
      // context value unfilled, so canResendNow below is false and the
      // resend the primary button promised never runs. The rows this
      // report re-renders show what is still missing; none of them says
      // the message did not go out, and this dialog is the only thing
      // that knows it did not.
      toast(t("connectorRuntime.savedNotResentIncomplete"))
    }

    if (newOutcome.kind === "fillable" || newOutcome.kind === "nothing_fillable") {
      // Still blocked on something this dialog can collect (or, for
      // nothing_fillable, on nothing the user can act on beyond "Got it"):
      // stay open, re-render from the fresh report.
      setSubmitting(false)
      setFieldError(null)
      return
    }

    if (alsoResend && canResendNow) {
      const resendOutcome = await doResend()
      if (!aliveRef.current) {
        // The save has landed and the resend has run to completion. A sent
        // message shows up in the transcript on its own, so that outcome
        // stays silent. A failed one would normally surface in this dialog's
        // send-failed panel, which an unmounted instance can never render --
        // and doResend's console.warn reaches no user -- so say it once,
        // globally, without touching state or the provider.
        if (resendOutcome !== "sent") toast(t("connectorRuntime.sendFailed"))
        return
      }
      if (requestRef.current.seq !== seqAtStart) {
        // Same reason as the earlier seq check: a newer request retargeted
        // this dialog instance while the resend was in flight, so this
        // result is stale, but `submitting` must still reset. The resend's
        // own outcome is not stale: the values are stored either way, and
        // the message either went out or did not. Nothing the fresher
        // request renders carries that fact -- this branch leaves
        // `sendFailed` false, so no panel says it -- and the footer it
        // draws next offers "Save and resend this message" again, which a
        // user who was told nothing would press on a turn that already
        // went out. A toast rather than the send-failed panel: that
        // panel's retry button reads whichever snapshot the fresher
        // request now carries, and once that snapshot has been replaced
        // the retry goes out under a new client message id rather than the
        // one the attempt that just settled here used
        // (xorbitsai/xagent#2502).
        // "nothing-to-send" maps with "failed" on purpose -- both mean no
        // message went out -- and is unreachable from here anyway, since
        // this block only runs when doResend was called with a snapshot.
        setSubmitting(false)
        toast(resendOutcome === "sent"
          ? t("connectorRuntime.resendSupersededSent")
          : t("connectorRuntime.sendFailed"))
        return
      }
      if (resendOutcome !== "sent") {
        setSubmitting(false)
        // The seq check just above proves no retarget landed while the
        // resend was in flight, so the snapshot the request carries here
        // is still the one doResend read -- which is what the panel this
        // raises is about, and what its retry button would send.
        const failedSnapshotId = requestRef.current.resendPayload?.clientMessageId ?? null
        if (failedSnapshotId === null) {
          // A settlement frame for this task arrived while the save was in
          // flight and took the snapshot with it (forgetDelivery), without
          // reopening the dialog and so without moving `seq`. There is
          // nothing left to retry, and a panel here would draw a retry
          // button with nothing behind it -- so say the same thing the
          // panel says, once, and leave the dialog on its report.
          toast(t("connectorRuntime.sendFailed"))
          return
        }
        setSendFailedSnapshotId(failedSnapshotId)
        return
      }
    }
    setSubmitting(false)
    close(alsoResend && canResendNow ? "resent" : "dismissed")
  }

  const handleRetryResend = async () => {
    // A resend is one billed model call plus a possibly side-effecting tool
    // run; a double click here must not fire it twice.
    if (resending) return
    // The panel this button lives on is about one message, and doResend
    // below sends whichever snapshot the request carries when it runs: the
    // two must be the same message. Unreachable today -- `sendFailed`
    // derives the panel's visibility from exactly this comparison during
    // render, so a frame that draws this button has already proved them
    // equal -- but the handler re-checks rather than trusting the render
    // that drew it, the same way handleSave re-checks `canSubmitNow`. The
    // panel's own state is dropped here too: the send it was about can no
    // longer be retried from this dialog, so leaving the id behind would
    // make the panel reappear if that snapshot ever came back.
    if (
      sendFailedSnapshotId === null
      || sendFailedSnapshotId !== requestRef.current.resendPayload?.clientMessageId
    ) {
      setSendFailedSnapshotId(null)
      return
    }
    const seqAtStart = request.seq
    setResending(true)
    const resendOutcome = await doResend()
    if (!aliveRef.current) {
      // Unlike handleSave's unmounted exit above, this one says nothing in
      // either outcome. A retry that went out shows up in the transcript on
      // its own. A retry that failed leaves things as the user last saw
      // them on the send-failed panel -- saved, not sent -- but nothing
      // tells them the retry they clicked did not change that. Left as is
      // here; routing every exit through one place that has to account for
      // it is tracked in xorbitsai/xagent#2478.
      return
    }
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the
      // resend was in flight; the result is stale, but `resending` must
      // still reset or the retry button stays stuck forever. A resend that
      // did go out needs to say so: the send-failed panel this button
      // lives on is about to be replaced by whatever the fresher request
      // renders next, and without a toast the user has no way to tell
      // that clicking a resend button there would send this same turn a
      // second time.
      setResending(false)
      if (resendOutcome === "sent") toast(t("connectorRuntime.resendSupersededSent"))
      return
    }
    setResending(false)
    if (resendOutcome === "sent") {
      setSendFailedSnapshotId(null)
      close("resent")
    }
  }

  // A resend in flight holds the dialog open for the same reason a save does:
  // dismissing mid-send drops the request (and with it the snapshot the retry
  // button reads), leaving nothing to retry from if that send fails.
  const handleDismiss = () => {
    if (busy) return
    close("dismissed")
  }

  const handleOpenChange = (open: boolean) => {
    if (open || busy) return
    handleDismiss()
  }

  const dialogFieldError = activeFieldError?.location.scope === "dialog" ? activeFieldError.disposition : null

  if (!visible || !report || !outcome) return null

  return (
    <Dialog open={visible} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {t(metHoldingSnapshot ? "connectorRuntime.metTitle" : "connectorRuntime.title")}
          </DialogTitle>
          <DialogDescription>
            {t(metHoldingSnapshot ? "connectorRuntime.metNotResent" : "connectorRuntime.description")}
          </DialogDescription>
        </DialogHeader>

        {outcome.kind === "unsupported_only" && (
          <p className="text-sm text-muted-foreground">{t("connectorRuntime.onlyUnsupportedNotice")}</p>
        )}

        {dialogFieldError && (
          <p className="text-sm text-destructive" role="alert">
            {/* This scope has no row identity in hand -- it is where
                locateFieldError falls back when the row a failure named is
                gone or unrecognized, most often a refresh that just
                dropped that row or collapsed it into "already filled".
                Which reasons that makes untrue, and what they say instead,
                is translateDialogScopeFailure's own business. */}
            {translateDialogScopeFailure(t, dialogFieldError.messageKey)}
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
                activeFieldError
                && activeFieldError.location.scope === "connector"
                && activeFieldError.location.connectorKey === connectorKey
                  ? activeFieldError.disposition
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
                    // Doubles as this row's React key: it carries the
                    // input's full identity (connector, section, key name,
                    // declared type), so two rows that legitimately share a
                    // key name across sections never collide, and a row
                    // whose declared type changes across a refresh is
                    // treated as a new row rather than reusing the old
                    // one's draft and error state under a new meaning.
                    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
                    const acceptedKeyName = input.section !== "context" || isAcceptedRuntimeKeyName(input.key)
                    const markLive = hasLiveInvalidObjectMark(connector, input, invalidDraftKeys)
                    const fieldLevelError =
                      activeFieldError
                      && activeFieldError.location.scope === "field"
                      && activeFieldError.location.draftKey === draftKey
                        ? activeFieldError.disposition
                        : null

                    if (input.section === "context" && input.satisfied) {
                      return (
                        <div key={draftKey} className="text-sm">
                          <span className="font-medium">{input.key}</span>{" "}
                          <span className="text-muted-foreground">{t("connectorRuntime.filled")}</span>
                        </div>
                      )
                    }

                    if (input.section !== "context") {
                      return (
                        <div key={draftKey} className="text-sm">
                          <div className="font-medium">{input.key}</div>
                          <p className="text-muted-foreground">{t("connectorRuntime.unsupportedNote")}</p>
                        </div>
                      )
                    }

                    // Unfilled context row while only "Got it" is offered:
                    // no input control and no "saved, cannot be changed"
                    // hint, since there is no save entry point in this shape.
                    if (!hasSaveEntryPoint) {
                      return (
                        <div key={draftKey} className="text-sm">
                          <span className="font-medium">{input.key}</span>
                          {!acceptedKeyName && (
                            <p className="text-destructive">{t("connectorRuntime.keyNameWarning")}</p>
                          )}
                        </div>
                      )
                    }

                    return (
                      <div key={draftKey} className="space-y-1">
                        <Label htmlFor={`connector-runtime-${draftKey}`}>{input.key}</Label>
                        {input.type === "object" ? (
                          <Textarea
                            id={`connector-runtime-${draftKey}`}
                            value={drafts[draftKey] ?? ""}
                            aria-invalid={markLive}
                            onChange={e => handleDraftChange(connector, input, e.target.value)}
                            onBlur={e => handleObjectBlur(connector, input, e.target.value)}
                          />
                        ) : (
                          <Input
                            id={`connector-runtime-${draftKey}`}
                            type="text"
                            value={drafts[draftKey] ?? ""}
                            onChange={e => handleDraftChange(connector, input, e.target.value)}
                          />
                        )}
                        {markLive && (
                          <p className="text-sm text-destructive">
                            {t(
                              invalidDraftKeys.get(draftKey) === "empty"
                                ? "connectorRuntime.objectEmpty"
                                : "connectorRuntime.objectInvalid",
                            )}
                          </p>
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
              // handleDismiss already refuses while a submission is in
              // flight; without this the button still looks pressable and
              // does nothing when pressed. A met report renders this as the
              // only button, and a save-and-resend whose save came back met
              // is still submitting for as long as its resend runs, so this
              // is a state the user can reach. The save buttons reach the
              // same guard through canSubmitNow, which folds busy in.
              <Button variant="outline" disabled={busy} onClick={handleDismiss}>
                {t("connectorRuntime.actions.acknowledge")}
              </Button>
            )}
            {actions.includes("saveOnly") && (
              <Button
                variant={actions.includes("saveAndResend") ? "outline" : "default"}
                disabled={!canSubmitNow}
                onClick={() => handleSave(false)}
              >
                {t("connectorRuntime.actions.saveOnly")}
              </Button>
            )}
            {actions.includes("saveAndResend") && (
              <Button disabled={!canSubmitNow} onClick={() => handleSave(true)}>
                {t("connectorRuntime.actions.saveAndResend")}
              </Button>
            )}
            {dialogFieldError?.retry && (
              <Button variant="outline" disabled={!canSubmitNow} onClick={() => handleSave(lastAlsoResend)}>
                {t("connectorRuntime.actions.retry")}
              </Button>
            )}
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}
