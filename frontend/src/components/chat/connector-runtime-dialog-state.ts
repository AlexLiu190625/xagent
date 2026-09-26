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
  connectorRuntimeInputDraftKey,
  resolveDialogOutcome,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
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
  invalidDraftKeys: Map<string, InvalidObjectDraftReason>,
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
