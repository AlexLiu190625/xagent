import { describe, expect, it } from "vitest"

import {
  deriveGates,
  type GateFacts,
  type InvalidObjectDraftReason,
  type SendFailureState,
} from "./connector-runtime-dialog-state"
import type {
  ConnectorRuntimeConnector,
  ConnectorRuntimeInput,
  ConnectorRuntimeReport,
  ConnectorRuntimeSection,
  ConnectorRuntimeType,
} from "@/lib/connector-runtime-api"

const REF = { connector_type: "custom_api", connector_id: 1 }

function input(overrides: Partial<ConnectorRuntimeInput> & { section: ConnectorRuntimeSection; key: string; type: ConnectorRuntimeType }): ConnectorRuntimeInput {
  return { required: false, satisfied: false, expired: false, ...overrides }
}
function connector(inputs: ConnectorRuntimeInput[]): ConnectorRuntimeConnector {
  return { connector_ref: REF, name: "Example", inputs }
}
function report(satisfied: boolean, connectors: ConnectorRuntimeConnector[] = []): ConnectorRuntimeReport {
  return { satisfied, secrets_expires_at: null, connectors }
}

const MET_REPORT = report(true)
const NOTHING_FILLABLE_REPORT = report(false)
const UNSUPPORTED_ONLY_REPORT = report(false, [
  connector([input({ section: "secrets", key: "s", type: "string", required: true })]),
])
const FILLABLE_REPORT = report(false, [
  connector([input({ section: "context", key: "k", type: "string", required: true })]),
])

const DRAFT_KEY = `${REF.connector_type}:${REF.connector_id}:context:k:string`

function baseFacts(overrides: Partial<GateFacts> = {}): GateFacts {
  return {
    report: null,
    reportSeq: null,
    settledReadKey: null,
    readNonce: 0,
    busy: false,
    heldFailure: null,
    retrying: false,
    drafts: {},
    invalidDraftKeys: new Map<string, InvalidObjectDraftReason>(),
    request: { seq: 1, resendPayload: null },
    ...overrides,
  }
}

const FAILURE: SendFailureState = { snapshotId: "cid-1", disposition: "not_sent" }

describe("deriveGates", () => {
  it.each([
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: no failure held",
      facts: baseFacts(),
      expect: { liveSendFailure: null, sendFailed: false, needsSnapshotRecycle: false },
    },
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: held failure matches the request's resend payload",
      facts: baseFacts({ heldFailure: FAILURE, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { liveSendFailure: FAILURE, sendFailed: true, needsSnapshotRecycle: false },
    },
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: held failure no longer matches (snapshot moved on)",
      facts: baseFacts({ heldFailure: FAILURE, request: { seq: 1, resendPayload: { clientMessageId: "cid-2" } } }),
      expect: { liveSendFailure: null, sendFailed: false, needsSnapshotRecycle: true },
    },
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: held failure but the request now carries no resend payload at all",
      facts: baseFacts({ heldFailure: FAILURE, request: { seq: 1, resendPayload: null } }),
      expect: { liveSendFailure: null, sendFailed: false, needsSnapshotRecycle: true },
    },
    {
      name: "readKey/reading: never settled",
      facts: baseFacts({ request: { seq: 3, resendPayload: null }, readNonce: 2 }),
      expect: { readKey: "3:2", reading: true },
    },
    {
      name: "readKey/reading: settled for this exact attempt",
      facts: baseFacts({ request: { seq: 3, resendPayload: null }, readNonce: 2, settledReadKey: "3:2" }),
      expect: { readKey: "3:2", reading: false },
    },
    {
      name: "readKey/reading: settled for an earlier nonce of the same request (a \"read again\" press is out)",
      facts: baseFacts({ request: { seq: 3, resendPayload: null }, readNonce: 2, settledReadKey: "3:1" }),
      expect: { readKey: "3:2", reading: true },
    },
    {
      name: "reportIsStale/readFailed: report matches the current request",
      facts: baseFacts({ report: MET_REPORT, reportSeq: 5, settledReadKey: "5:0", request: { seq: 5, resendPayload: null } }),
      expect: { reportIsStale: false, readFailed: false },
    },
    {
      name: "reportIsStale/readFailed: report is from an earlier request and the read has settled (the read failed)",
      facts: baseFacts({ report: MET_REPORT, reportSeq: 5, settledReadKey: "6:0", request: { seq: 6, resendPayload: null } }),
      expect: { reportIsStale: true, readFailed: true },
    },
    {
      name: "reportIsStale/readFailed: report is from an earlier request but a re-read for the new one is still out (not yet a failure)",
      facts: baseFacts({ report: MET_REPORT, reportSeq: 5, settledReadKey: null, request: { seq: 6, resendPayload: null } }),
      expect: { reportIsStale: true, readFailed: false },
    },
    {
      name: "outcome: no report yet",
      facts: baseFacts(),
      expect: { outcome: null },
    },
    {
      name: "outcome: met",
      facts: baseFacts({ report: MET_REPORT }),
      expect: { outcome: { kind: "met" } },
    },
    {
      name: "outcome: nothing_fillable",
      facts: baseFacts({ report: NOTHING_FILLABLE_REPORT }),
      expect: { outcome: { kind: "nothing_fillable" } },
    },
    {
      name: "outcome/actions/hasSaveEntryPoint: unsupported_only offers only acknowledge",
      facts: baseFacts({ report: UNSUPPORTED_ONLY_REPORT }),
      expect: {
        outcome: { kind: "unsupported_only", blocking: [{ connectorRef: REF, key: "s" }] },
        actions: ["acknowledge"],
        hasSaveEntryPoint: false,
      },
    },
    {
      name: "actions/hasSaveEntryPoint: fillable without a resend payload offers only saveOnly",
      facts: baseFacts({ report: FILLABLE_REPORT, request: { seq: 1, resendPayload: null } }),
      expect: { actions: ["saveOnly"], hasSaveEntryPoint: true, hasResendPayload: false },
    },
    {
      name: "actions/hasSaveEntryPoint/hasResendPayload: fillable with a resend payload offers both",
      facts: baseFacts({ report: FILLABLE_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { actions: ["saveAndResend", "saveOnly"], hasSaveEntryPoint: true, hasResendPayload: true },
    },
    {
      name: "canSubmitNow: a fillable report with its required row still empty cannot submit",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1 }),
      expect: { canSubmitNow: false },
    },
    {
      name: "canSubmitNow: filling the required row enables it",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, drafts: { [DRAFT_KEY]: "value" } }),
      expect: { canSubmitNow: true },
    },
    {
      name: "canSubmitNow: busy holds it closed even once the row is filled",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, drafts: { [DRAFT_KEY]: "value" }, busy: true }),
      expect: { canSubmitNow: false },
    },
    {
      name: "canSubmitNow: a stale report holds it closed even once the row is filled",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, drafts: { [DRAFT_KEY]: "value" }, request: { seq: 2, resendPayload: null } }),
      expect: { canSubmitNow: false },
    },
    {
      name: "hasInvalidObjectDraft/canSubmitNow: a live invalid-object mark on the only required row blocks submission even though it is \"filled\"",
      facts: baseFacts({
        report: report(false, [connector([input({ section: "context", key: "k", type: "object", required: true })])]),
        reportSeq: 1,
        drafts: { [`${REF.connector_type}:${REF.connector_id}:context:k:object`]: "{}" },
        invalidDraftKeys: new Map([[`${REF.connector_type}:${REF.connector_id}:context:k:object`, "empty"]]),
      }),
      expect: { hasInvalidObjectDraft: true, canSubmitNow: false },
    },
    {
      name: "metHoldingSnapshot: met, holding a resend payload, nothing failed, not busy",
      facts: baseFacts({ report: MET_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { metHoldingSnapshot: true },
    },
    {
      name: "metHoldingSnapshot: false without a resend payload to hold",
      facts: baseFacts({ report: MET_REPORT, request: { seq: 1, resendPayload: null } }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "metHoldingSnapshot: false while the send-failed panel is live for that same snapshot",
      facts: baseFacts({
        report: MET_REPORT,
        heldFailure: FAILURE,
        request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } },
      }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "metHoldingSnapshot: false while busy (a save-and-resend for this same message is still on the wire)",
      facts: baseFacts({ report: MET_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } }, busy: true }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "metHoldingSnapshot: false when the report is fillable rather than met, even holding a resend payload, nothing failed, not busy",
      facts: baseFacts({ report: FILLABLE_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "retryResendDisabled: neither retrying nor stale",
      facts: baseFacts({ reportSeq: 1, request: { seq: 1, resendPayload: null } }),
      expect: { retryResendDisabled: false },
    },
    {
      name: "retryResendDisabled: a retry is already out",
      facts: baseFacts({ reportSeq: 1, request: { seq: 1, resendPayload: null }, retrying: true }),
      expect: { retryResendDisabled: true },
    },
    {
      name: "retryResendDisabled: the report on hand is stale",
      facts: baseFacts({ reportSeq: 1, request: { seq: 2, resendPayload: null } }),
      expect: { retryResendDisabled: true },
    },
  ])("$name", ({ facts, expect: partial }) => {
    const gates = deriveGates(facts)
    expect(gates).toMatchObject(partial)
  })

  // The invariant this exercises: the dialog's lifecycle (busy or not) and
  // the report's own outcome (met, fillable, ...) are independent
  // dimensions, not one flattened display phase. A same-task retarget's
  // re-read installs whatever report comes back regardless of what is in
  // flight, so a report that reads "still needs a value" can land while a
  // save-and-resend for an earlier, now-superseded snapshot is still out.
  // Both facts must show up in the gates at once: busy holds every submit
  // gate closed, and the fillable outcome still drives the row and footer
  // content underneath it.
  it("holds submission closed while busy, even when a same-task re-read replaces the report with one still asking for a value", () => {
    const gates = deriveGates(baseFacts({
      report: FILLABLE_REPORT,
      reportSeq: 7,
      settledReadKey: "7:0",
      busy: true,
      request: { seq: 7, resendPayload: { clientMessageId: "cid-9" } },
    }))
    expect(gates.reportIsStale).toBe(false)
    expect(gates.reading).toBe(false)
    // "blocking" on a fillable outcome lists the still-unsupported secrets
    // group, not the context row the report is fillable *because* of --
    // there is none here, so it is empty even though "k" is what makes this
    // report fillable at all (resolveDialogOutcome, connector-runtime-api.ts).
    expect(gates.outcome).toEqual({ kind: "fillable", blocking: [] })
    expect(gates.actions).toEqual(["saveAndResend", "saveOnly"])
    expect(gates.hasSaveEntryPoint).toBe(true)
    expect(gates.canSubmitNow).toBe(false)
    expect(gates.metHoldingSnapshot).toBe(false)
  })
})
