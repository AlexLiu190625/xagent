// Pure predicates and small value types the connector-runtime dialog uses to
// decide what its state means, kept apart from connector-runtime-dialog.tsx
// so they can be data for a table-driven test instead of only reachable
// through rendering. No React, no i18n, no translation keys: nothing here
// may depend on how the dialog draws itself or what it says.

import { sendOutcomeMayHaveLanded } from "@/components/chat/clarification-delivery"
// Type-only, like clarification-delivery's own import of it: naming the
// disposition union here adds no runtime dependency on the websocket hook.
import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import {
  buildSubmitItems,
  connectorRuntimeInputDraftKey,
  isSubmitEnabled,
  resolveDialogActions,
  resolveDialogOutcome,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeDialogAction,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
  type ConnectorRuntimeSubmitItem,
  type DialogOutcome,
} from "@/lib/connector-runtime-api"

// Why a draft failed the object-field blur check: "invalid" for anything
// that is not JSON-object-shaped, "empty" for `{}`, which parses fine but
// isSubmittableObjectValue (connector-runtime-api.ts) rejects because the
// server treats it as a blank context value. The row's error message reads
// this to show the reason-specific hint instead of a generic one.
export type InvalidObjectDraftReason = "invalid" | "empty"

/**
 * Whether an invalid-object mark for this input is still live. Only a
 * `context` row the current report leaves unsatisfied and still declares
 * `object`-typed renders the textarea whose blur handler can clear such a
 * mark; against any other row the mark is unreachable. Both the submit gate
 * and the row's own error message read this one predicate, so the button can
 * never be disabled by an error the row does not show, and the row can never
 * show an error that leaves the button enabled.
 */
export function hasLiveInvalidObjectMark(
  connector: ConnectorRuntimeConnector,
  input: ConnectorRuntimeInput,
  invalidDraftKeys: ReadonlyMap<string, InvalidObjectDraftReason>,
): boolean {
  return (
    input.section === "context"
    && !input.satisfied
    && input.type === "object"
    && invalidDraftKeys.has(connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type))
  )
}

export function uniqueKeys(locations: Array<{ key: string }>): string[] {
  return Array.from(new Set(locations.map(l => l.key)))
}

/**
 * The disposition the send-failed panel carries after one more attempt for
 * the same snapshot. Uncertainty only ever accumulates: once any attempt
 * ended with its outcome unknown, a later one the server definitely refused
 * does not make the earlier one un-sent, so neither the panel nor anything
 * standing in for it may fall back to saying the message never went out.
 *
 * One function rather than one rule in the panel and another wherever a
 * toast reports the same attempt: the two are read by the same user, seconds
 * apart, about one message.
 */
export function mergeSendFailureDisposition(
  previous: MessageDeliveryDisposition | null,
  attempt: MessageDeliveryDisposition | null,
): MessageDeliveryDisposition | null {
  if (sendOutcomeMayHaveLanded(previous) || sendOutcomeMayHaveLanded(attempt)) return "outcome_unknown"
  return attempt
}

/**
 * Whether the report currently in hand can carry a resend of the message
 * this dialog is holding. Only a met report can: `unsupported_only` still
 * lacks a required secret this dialog cannot collect, `nothing_fillable` is a
 * connector the server still reports unavailable with nothing left for the
 * user to fill, and `fillable` still has a required context value missing.
 * The backend rejects all three while it builds the turn's tool list, so a
 * resend would fail on the same gate and put a second failure in the
 * conversation.
 *
 * Both entry points into a resend ask this -- handleSave right after its own
 * save lands, and handleRetryResend against the report on screen -- so the
 * two cannot disagree about whether the same snapshot is sendable.
 */
export function canResendReport(report: ConnectorRuntimeReport): boolean {
  return resolveDialogOutcome(report).kind === "met"
}

/**
 * The failed send the "saved but not sent" panel is about: the
 * clientMessageId of the snapshot it names, and the disposition that failure
 * carried, held together so the panel can never word one send's outcome with
 * another's. See `sendFailed` in connector-runtime-dialog.tsx for how the id
 * half is read.
 */
export interface SendFailureState {
  snapshotId: string
  disposition: MessageDeliveryDisposition | null
}

/**
 * Everything deriveGates needs, read fresh off the dialog's own state and
 * the request it is currently showing. Flat facts rather than the dialog's
 * storage shape itself: a caller holding twelve independent pieces of state
 * today can feed this the same way a caller holding one combined state
 * value will tomorrow, and neither has to restate the formulas below --
 * only how to read its own storage into this shape.
 */
export interface GateFacts {
  report: ConnectorRuntimeReport | null
  reportSeq: number | null
  settledReadKey: string | null
  readNonce: number
  // Whether any submission-shaped action is in flight: an explicit save
  // (which may itself run a resend as part of "save and resend") or a
  // standalone retry resend from the send-failed panel. The caller keeps
  // this rather than deriveGates folding it from two flags of its own,
  // because what those flags are and how many there are is the caller's
  // storage question, not this module's.
  //
  // The caller must keep `retrying` implying `busy`: a retry resend is one
  // of the submission-shaped actions `busy` covers. A fact set with
  // `retrying` true and `busy` false describes no state the dialog can be
  // in, and nothing below is written to give it a meaning.
  busy: boolean
  // The send the "saved but not sent" panel is about, or null when no send
  // has failed. See `liveSendFailure` below for why this is joined against
  // the request's own resend payload on every call rather than trusted as
  // it stands.
  heldFailure: SendFailureState | null
  // Whether the send-failed panel's own retry resend is in flight. Implies
  // `busy`; see there.
  retrying: boolean
  drafts: Readonly<Record<string, string>>
  invalidDraftKeys: ReadonlyMap<string, InvalidObjectDraftReason>
  request: { seq: number; resendPayload: { clientMessageId: string } | null }
}

export interface Gates {
  liveSendFailure: SendFailureState | null
  sendFailed: boolean
  needsSnapshotRecycle: boolean
  readKey: string
  reading: boolean
  reportIsStale: boolean
  readFailed: boolean
  outcome: DialogOutcome | null
  submitItems: ConnectorRuntimeSubmitItem[]
  hasInvalidObjectDraft: boolean
  canSubmit: boolean
  canSubmitNow: boolean
  hasResendPayload: boolean
  actions: ConnectorRuntimeDialogAction[]
  hasSaveEntryPoint: boolean
  metHoldingSnapshot: boolean
  retryResendDisabled: boolean
}

/**
 * Every value the connector-runtime dialog derives from its own state and
 * the request it is currently showing, gathered in one place so the same
 * formulas cannot drift between whatever state produces them and whatever
 * renders off them. Everything here is computed fresh from `facts` on every
 * call -- nothing is cached across calls -- so it is the caller's choice how
 * often that happens (today, every render).
 */
export function deriveGates(facts: GateFacts): Gates {
  // Joined against the request's own resend payload on every call, rather
  // than trusted as it stands, because the panel and its retry button must
  // be about the same message on every call, including the first one after
  // a same-task retarget swaps in a newer resend candidate, or a
  // settlement frame takes the snapshot away without moving `seq` at all.
  // Today nothing brings a clientMessageId back once it has stopped
  // matching (a snapshot's own id is written once at send time), so a held
  // failure that stops matching is unreachable rather than wrong.
  // `needsSnapshotRecycle` below tells the caller, which drops it so it
  // cannot become wrong if an id ever does come back.
  const liveSendFailure = facts.heldFailure !== null
    && facts.heldFailure.snapshotId === facts.request.resendPayload?.clientMessageId
    ? facts.heldFailure
    : null
  const sendFailed = liveSendFailure !== null
  // True while `heldFailure` no longer matches the request's resend
  // payload. The caller must drop it -- from render, not an effect, since
  // an effect notices a frame late and never runs at all for a removal that
  // does not move `seq` -- so that the next call's facts no longer carry
  // it. How many calls this stays true for depends on the caller doing so.
  const needsSnapshotRecycle = facts.heldFailure !== null && liveSendFailure === null

  // The read attempt the dialog is currently on: the request it is for, and
  // which try for that request it is. Both halves are needed -- `seq` alone
  // cannot tell a fresh attempt for the same request from the one that just
  // failed, so a "read again" press would leave `reading` false and the
  // screen would say nothing while the retry was out.
  const readKey = `${facts.request.seq}:${facts.readNonce}`
  const reading = facts.settledReadKey !== readKey
  // Whether the report on hand belongs to some earlier request than the one
  // the dialog is now for. A same-task retarget bumps `seq` and starts a
  // fresh read while the previous report is still on screen; submitting or
  // resending against that report acts on rows the current one may no
  // longer declare, and a stored context value is immutable, so there is no
  // correcting it afterwards.
  //
  // This is the gate rather than `reading`, because the two come apart in
  // exactly the case that matters: a read that fails settles without
  // installing anything, so `reading` goes false while the report on hand
  // is still the previous request's. `reportSeq` can only catch up when the
  // attempt that installs a fresher report also records itself settled, so
  // this is true whenever `reading` is -- one gate covers the pending read
  // and the failed one both.
  const reportIsStale = facts.reportSeq !== facts.request.seq
  // The read for this request settled and left the previous request's
  // report on hand: it failed. Derived rather than stored, so it cannot
  // disagree with the two facts it is made of.
  const readFailed = !reading && reportIsStale

  const outcome: DialogOutcome | null = facts.report ? resolveDialogOutcome(facts.report) : null
  const submitItems = facts.report ? buildSubmitItems(facts.report, facts.drafts) : []
  // Only a mark on a row the current report still renders an editable
  // control for may gate submission. A key a refreshed report reports
  // satisfied loses its textarea, so its mark could never be cleared again
  // -- submit would stay disabled with no error anywhere on screen. Reads
  // the same `hasLiveInvalidObjectMark` predicate the row renderer reads
  // for its error message, so the two can never disagree. `invalidDraftKeys`
  // is read-only here the same way it is everywhere else this module reads
  // it: nothing in this module ever needs to write it back.
  const hasInvalidObjectDraft = facts.report !== null && facts.report.connectors.some(connector =>
    connector.inputs.some(input => hasLiveInvalidObjectMark(
      connector,
      input,
      facts.invalidDraftKeys,
    )),
  )
  const canSubmit = isSubmitEnabled(submitItems, hasInvalidObjectDraft)
  // The one value every entry point into a submission reads: both footer
  // save buttons, the retry button a retryable failure offers, and the save
  // handler itself. That matters because buildSubmitItems drops an
  // unparsable object draft instead of failing: such a batch writes every
  // other field, silently loses that one and closes the dialog -- and a
  // stored context value is immutable, so there is no correcting it
  // afterwards.
  //
  // `reportIsStale` is folded in here and not into `busy`, because the
  // caller also uses `busy` to gate dismissal: a read still out for a
  // retargeted request must not stand between the user and closing the
  // dialog. What `busy` itself holds open is the caller's business; see
  // where the dialog computes it.
  const canSubmitNow = canSubmit && !facts.busy && !reportIsStale
  const hasResendPayload = facts.request.resendPayload !== null
  const actions = outcome ? resolveDialogActions(outcome, hasResendPayload) : []
  // Whether this shape offers any way to submit. The row renderer asks this
  // instead of listing the outcome kinds that offer none, because that list
  // was one kind short: a `met` report reaches the render whenever one is
  // installed into a dialog that stays open, and an unfilled *optional*
  // context key inside one was still drawn as an editable field with no
  // button able to send it. Derived from the action set, so the rows and
  // the footer cannot disagree about whether saving is possible.
  const hasSaveEntryPoint = actions.includes("saveOnly")
  // A met report that reached the render still carries the snapshot of the
  // message that failed, and this shape offers no way to send it: the
  // footer collapses to "Got it", and the save-and-resend button a
  // fillable report offers cannot be reused here because a met report
  // produces no submittable items, which leaves it permanently disabled.
  // Not while the send-failed panel is up, and not while a submission for
  // this same message is still in flight: saying it was not resent would
  // be false in both cases.
  const metHoldingSnapshot = outcome?.kind === "met" && hasResendPayload && !sendFailed && !facts.busy
  // Mirrors the guard the send-failed panel's own retry button carries: no
  // second attempt while one is already out, and none against a report the
  // current request no longer produced.
  const retryResendDisabled = facts.retrying || reportIsStale

  return {
    liveSendFailure,
    sendFailed,
    needsSnapshotRecycle,
    readKey,
    reading,
    reportIsStale,
    readFailed,
    outcome,
    submitItems,
    hasInvalidObjectDraft,
    canSubmit,
    canSubmitNow,
    hasResendPayload,
    actions,
    hasSaveEntryPoint,
    metHoldingSnapshot,
    retryResendDisabled,
  }
}
