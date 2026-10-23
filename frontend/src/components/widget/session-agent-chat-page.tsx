"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2 } from "lucide-react"
import type { ComposerAttachmentStatus } from "@/components/chat/ChatInput"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { toast } from "@/components/ui/sonner"
import { WidgetChromeControls } from "@/components/widget/widget-chrome-controls"
import {
  AppProvider,
  type AppProviderTransportConfig,
  type SessionAttachmentsConfig,
  useApp,
} from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import type { WebSocketConnection } from "@/hooks/use-websocket"
import { readSendErrorCode } from "@/components/chat/clarification-delivery"
import {
  clientErrorTranslationKey,
  type ClientErrorCode,
} from "@/lib/client-errors"
import {
  uploadWidgetSessionFile,
  type SessionUploadedFile,
} from "@/lib/widget-session-upload"
import {
  buildWidgetSessionWebSocketUrl,
  useWidgetSession,
  type WidgetSession,
  type WidgetSessionAgent,
  type WidgetSessionFileUploads,
  type WidgetSessionStatus,
} from "./use-widget-session"

const SESSION_PROTOCOL = "xagent-session-v1"
// Asks the session server to confirm and send heartbeat frames. A server that
// does not know the marker ignores it, and the connection then runs without
// dead-connection detection.
const SESSION_LIVENESS_PROTOCOL = "xagent-session-liveness-v1"
type SessionTerminalPresentation = "expired" | "unavailable"

const SESSION_TERMINAL_PRESENTATION: Readonly<Record<string, SessionTerminalPresentation>> = {
  session_expired: "expired",
  grant_expired: "expired",
  grant_already_used: "expired",
  reconnect_invalid: "unavailable",
  identity_mismatch: "unavailable",
  ws_4408: "expired",
  ws_4403: "unavailable",
  unexpected_error: "unavailable",
}

const getSessionTerminalPresentation = (code: string | null): SessionTerminalPresentation => (
  code ? SESSION_TERMINAL_PRESENTATION[code] ?? "unavailable" : "unavailable"
)

// What the page needs from the bridge to upload files before a send.
interface SessionUploadControls {
  ensureFreshSession: () => Promise<WidgetSession>
  refreshSession: (rejectedToken: string) => Promise<WidgetSession>
  waitForOpenConnection: () => Promise<void>
}

// While any of these holds, sendMessage itself refuses the turn; an upload
// started before it would be wasted.
const SEND_BLOCKED_PHASES: ReadonlySet<string> = new Set([
  "reset_requested",
  "reload_required",
  "replacement_sending",
  "replacement_awaiting_task",
])

const isAbortError = (error: unknown): boolean =>
  typeof error === "object" && error !== null
  && (error as { name?: unknown }).name === "AbortError"

// The error that ends a send whose file the visitor removed. The visitor did it
// on purpose, so the composer must not announce a failure.
const removedFileError = (): Error =>
  Object.assign(
    new DOMException("A file was removed before the message was sent.", "AbortError"),
    { notificationHandled: true },
  )

// Resolves with `promise`, or rejects as soon as `signal` aborts. Only this
// caller's wait ends; `promise` itself keeps running.
const raceSend = <T,>(promise: Promise<T>, signal: AbortSignal): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(removedFileError())
    if (signal.aborted) {
      onAbort()
      return
    }
    signal.addEventListener("abort", onAbort, { once: true })
    promise.then(
      (value) => { signal.removeEventListener("abort", onAbort); resolve(value) },
      (error) => { signal.removeEventListener("abort", onAbort); reject(error) },
    )
  })

const isNotificationHandled = (error: unknown): boolean =>
  typeof error === "object" && error !== null
  && (error as { notificationHandled?: unknown }).notificationHandled === true

const userFacingError = (message: string, errorCode?: ClientErrorCode): Error =>
  Object.assign(new Error(message), {
    userFacing: true,
    ...(errorCode ? { errorCode } : {}),
  })

interface SessionConversationContentProps {
  status: WidgetSessionStatus
  agent: WidgetSessionAgent | null
  terminalCode: string | null
  isAbsoluteExpiryWarningVisible: boolean
  // Lifted to the never-unmounting SessionAgentChatPage below: this
  // component's own "active" JSX branch unmounts on a degraded/reconnect
  // transition (see the status check further down), which would otherwise
  // silently reset an accepted expand back to the collapsed label/icon while
  // the panel itself is still actually expanded.
  isExpanded: boolean
  onExpandedChange: (expanded: boolean) => void
  // Present only while the session server advertises file uploads.
  attachments?: SessionAttachmentsConfig
  uploads: SessionUploadControls
}

function SessionConversationContent({
  status,
  agent,
  terminalCode,
  isAbsoluteExpiryWarningVisible,
  isExpanded,
  onExpandedChange,
  attachments,
  uploads,
}: SessionConversationContentProps) {
  const {
    state,
    filesDisabled,
    voiceInputEnabled,
    isConnected,
    sendMessage,
    startNewConversation,
    isConversationResetPending,
    isMessageDeliveryPending,
    isSessionInteractionLocked,
    sessionConversationState,
  } = useApp()
  const { t } = useI18n()
  const [startFiles, setStartFiles] = useState<File[]>([])
  const [attachmentStatuses, setAttachmentStatuses] =
    useState<ReadonlyMap<File, ComposerAttachmentStatus>>(() => new Map())
  // Counts attachment work that must keep the composer locked: each upload and
  // each send with files, from its first upload until its chat frame is out.
  const [attachmentWork, setAttachmentWork] = useState(0)
  const isAttachmentBusy = attachmentWork > 0
  // Keyed by the File the visitor picked, so a message sent again with the
  // same files (edited or not) reuses what already reached the server.
  const uploadedFilesRef = useRef(new WeakMap<File, SessionUploadedFile>())
  const uploadControllersRef = useRef(new Map<File, AbortController>())
  // One entry per send with files that is still running; removing any of its
  // files aborts that send.
  const activeSendsRef = useRef(new Set<{ files: ReadonlySet<File>, controller: AbortController }>())
  const panelFilesRef = useRef<File[]>([])
  const [resetError, setResetError] = useState(false)
  // The failure is stored as a code, not as text, so the line is translated
  // at render time and follows a locale switch. It is only a note on the
  // start page; the toast raised by the composer is the failure signal.
  const [startMessageFailure, setStartMessageFailure] =
    useState<{ code: ClientErrorCode | null } | null>(null)
  const conversationEstablished = state.taskId !== null

  // The failure line belongs to the current start-page visit only: whenever
  // the conversation becomes established or returns to the start page, the
  // stored failure is dropped, so a failure that arrived while a conversation
  // was established never shows on a later start page. The drop happens while
  // rendering rather than in an effect: React re-runs this component with the
  // cleared failure before committing, so the stale line under role="alert"
  // never reaches the DOM, not even for a single commit.
  const [lastSeenConversationEstablished, setLastSeenConversationEstablished] =
    useState(conversationEstablished)
  if (lastSeenConversationEstablished !== conversationEstablished) {
    setLastSeenConversationEstablished(conversationEstablished)
    setStartMessageFailure(null)
  }

  const setAttachmentStatus = useCallback((
    file: File,
    next: ComposerAttachmentStatus | null,
  ) => {
    setAttachmentStatuses((current) => {
      const updated = new Map(current)
      if (next === null) updated.delete(file)
      else updated.set(file, next)
      return updated
    })
  }, [])

  // Forgets everything held for a file the visitor no longer wants: cancels a
  // request still in flight and drops what was already uploaded. A request
  // that had reached the server may still leave the file stored there; it is
  // never attached to a message.
  const releaseAttachment = useCallback((file: File) => {
    uploadControllersRef.current.get(file)?.abort()
    uploadControllersRef.current.delete(file)
    activeSendsRef.current.forEach((send) => {
      if (send.files.has(file)) send.controller.abort()
    })
    uploadedFilesRef.current.delete(file)
    setAttachmentStatus(file, null)
  }, [setAttachmentStatus])

  const releaseRemovedAttachments = useCallback((previous: File[], next: File[]) => {
    const kept = new Set(next)
    previous.filter((file) => !kept.has(file)).forEach(releaseAttachment)
  }, [releaseAttachment])

  const abortAllUploads = useCallback(() => {
    uploadControllersRef.current.forEach((controller) => controller.abort())
    uploadControllersRef.current.clear()
    activeSendsRef.current.forEach((send) => send.controller.abort())
  }, [])

  useEffect(() => abortAllUploads, [abortAllUploads])
  useEffect(() => {
    if (status === "terminal" || status === "degraded") abortAllUploads()
  }, [abortAllUploads, status])

  const handleStartFilesChange = useCallback((next: File[]) => {
    releaseRemovedAttachments(startFiles, next)
    setStartFiles(next)
  }, [releaseRemovedAttachments, startFiles])

  const handlePanelFilesChange = useCallback((next: File[]) => {
    releaseRemovedAttachments(panelFilesRef.current, next)
    panelFilesRef.current = next
  }, [releaseRemovedAttachments])

  // Uploads one file with the session token. A send uploads its files one
  // after another, and the file cards' retry buttons are disabled while any
  // upload or send with files is in flight. A retry only counts as in flight
  // once its token is ready, so a retry clicked while the token is still being
  // renewed can overlap another upload; the bridge then shares or skips the
  // token refresh after a 401.
  const uploadAttachment = useCallback(async (file: File): Promise<SessionUploadedFile> => {
    const controller = new AbortController()
    uploadControllersRef.current.set(file, controller)
    setAttachmentWork((count) => count + 1)
    setAttachmentStatus(file, "uploading")
    try {
      const uploaded = await uploadWidgetSessionFile({
        origin: window.location.origin,
        getToken: async () => (await uploads.ensureFreshSession()).token,
        refreshToken: async (rejected) => (await uploads.refreshSession(rejected)).token,
        file,
        signal: controller.signal,
      })
      uploadedFilesRef.current.set(file, uploaded)
      setAttachmentStatus(file, "uploaded")
      return uploaded
    } catch (error) {
      // A released file has no card left to mark, and the visitor who removed
      // it needs no "not sent" notice for it.
      if (controller.signal.aborted) {
        throw typeof error === "object" && error !== null
          ? Object.assign(error, { notificationHandled: true })
          : error
      }
      setAttachmentStatus(file, "failed")
      throw error
    } finally {
      if (uploadControllersRef.current.get(file) === controller) {
        uploadControllersRef.current.delete(file)
      }
      setAttachmentWork((count) => count - 1)
    }
  }, [setAttachmentStatus, uploads])

  const handleRetryAttachment = useCallback(async (file: File) => {
    try {
      await uploads.ensureFreshSession()
      await uploadAttachment(file)
    } catch (error) {
      if (isAbortError(error)) return
      const code = readSendErrorCode(error) ?? "upload_failed"
      toast.error(t(clientErrorTranslationKey(code)))
    }
  }, [t, uploadAttachment, uploads])

  // The checks that cost nothing: a send that is certain to be refused is
  // refused before any file leaves the browser.
  const assertAttachmentsSendable = useCallback((
    files: File[],
    config: Record<string, unknown> | undefined,
    limits: SessionAttachmentsConfig,
  ) => {
    const metadata = config?.metadata as { request_id?: unknown } | undefined
    if (typeof metadata?.request_id === "string" && metadata.request_id) {
      throw userFacingError(t("widgetSession.attachments.noFilesOnAnswer"))
    }
    const allowedExtensions = limits.accept.split(",")
    for (const file of files) {
      const dot = file.name.lastIndexOf(".")
      const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : ""
      if (!allowedExtensions.includes(extension)) {
        throw userFacingError(t("widgetSession.attachments.typeNotSupported"))
      }
      if (file.size > limits.maxFileBytes) {
        throw userFacingError(t(clientErrorTranslationKey("upload_too_large")), "upload_too_large")
      }
    }
    if (files.length > limits.maxFiles) {
      throw userFacingError(
        t("widgetSession.attachments.tooManyFiles", { count: limits.maxFiles }),
      )
    }
  }, [t])

  // Sends a message with the files the visitor attached. Files go to the
  // session first, one after another, and the chat frame then names them by id.
  const sendWithAttachments = useCallback(async (
    message: string,
    files: File[],
    config?: Record<string, unknown>,
  ) => {
    if (files.length === 0 || !attachments) {
      await sendMessage(message, config, files)
      return
    }
    if (
      SEND_BLOCKED_PHASES.has(sessionConversationState)
      || isSessionInteractionLocked
      || isConversationResetPending
    ) {
      // sendMessage would refuse the turn in these states, but calling it
      // with the files dropped could send the text alone if its own view of
      // the state is newer than this render's, so the send stops here.
      throw new Error("Message not sent: the conversation is not ready for attachments.")
    }
    assertAttachmentsSendable(files, config, attachments)

    setAttachmentWork((count) => count + 1)
    const send = { files: new Set(files), controller: new AbortController() }
    activeSendsRef.current.add(send)
    try {
      const uploaded: SessionUploadedFile[] = []
      for (const file of files) {
        if (send.controller.signal.aborted) throw removedFileError()
        const known = uploadedFilesRef.current.get(file)
        uploaded.push(known ?? await uploadAttachment(file))
      }
      await raceSend(uploads.waitForOpenConnection(), send.controller.signal)
      // Removing a file aborts the send above; this check is the backstop for
      // a file that was released without reaching the abort.
      if (files.some((file) => !uploadedFilesRef.current.has(file))) {
        throw removedFileError()
      }

      try {
        await sendMessage(message, config, [], { uploadedFiles: uploaded })
      } catch (error) {
        if (readSendErrorCode(error) === "message_attachment_unavailable") {
          // The server no longer has one of these files: upload them all again.
          files.forEach((file) => {
            uploadedFilesRef.current.delete(file)
            setAttachmentStatus(file, null)
          })
        }
        throw error
      }
      files.forEach((file) => {
        uploadedFilesRef.current.delete(file)
        setAttachmentStatus(file, null)
      })
    } finally {
      activeSendsRef.current.delete(send)
      setAttachmentWork((count) => count - 1)
    }
  }, [
    assertAttachmentsSendable,
    attachments,
    isConversationResetPending,
    isSessionInteractionLocked,
    sendMessage,
    sessionConversationState,
    setAttachmentStatus,
    uploadAttachment,
    uploads,
  ])

  const handlePanelSend = useCallback(async (
    message: string,
    config?: Record<string, unknown>,
    files?: File[],
  ) => {
    await sendWithAttachments(message, files ?? [], config)
  }, [sendWithAttachments])

  const handleStartMessage = useCallback(async (
    message: string,
    files: File[],
    config?: Record<string, unknown>,
  ) => {
    setStartMessageFailure(null)
    try {
      await sendWithAttachments(message, files, config)
      setStartFiles([])
    } catch (error) {
      // An error that is already announced, or that the visitor caused, leaves
      // no failure line behind.
      if (!isNotificationHandled(error)) {
        setStartMessageFailure({ code: readSendErrorCode(error) })
      }
      throw error
    }
  }, [sendWithAttachments])

  const handleStartNewConversation = useCallback(async () => {
    setResetError(false)
    try {
      await startNewConversation()
    } catch {
      setResetError(true)
    }
  }, [startNewConversation])

  if (status === "waiting" || (status === "refreshing" && !agent)) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          {status === "waiting"
            ? t("widgetChat.status.initializing")
            : t("widgetChat.status.connecting")}
        </div>
      </div>
    )
  }

  if (status === "degraded" || status === "terminal" || !agent) {
    const isExpired = status !== "degraded"
      && getSessionTerminalPresentation(terminalCode) === "expired"
    return (
      <div className="flex h-screen items-center justify-center bg-background p-6">
        <div className="max-w-md text-center">
          <h1 className="text-lg font-semibold">
            {isExpired
              ? t("widgetSession.expired.title")
              : t("widgetSession.unavailable.title")}
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            {isExpired
              ? t("widgetSession.expired.description")
              : t("widgetSession.unavailable.description")}
          </p>
        </div>
      </div>
    )
  }

  const reloadRequired = isSessionInteractionLocked
  const resetDisabled =
    reloadRequired
    || isConversationResetPending
    || isMessageDeliveryPending
    || isAttachmentBusy

  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="flex-none border-b bg-card p-4 text-card-foreground shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full bg-primary/10 text-primary">
            {agent.logoUrl ? (
              // Session metadata may name an arbitrary remote logo host.
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={agent.logoUrl}
                alt={agent.name}
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="h-5 w-5 rounded-full bg-primary/20" />
            )}
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold">
              {agent.name}
            </h1>
            <p className="text-xs text-muted-foreground">
              {status === "active" && isConnected
                ? t("widgetChat.status.online")
                : t("widgetChat.status.connecting")}
            </p>
          </div>
          <WidgetChromeControls
            newConversation={conversationEstablished ? {
              label: isConversationResetPending
                ? t("widgetSession.resetting")
                : t("widgetSession.startNewConversation"),
              onClick: () => void handleStartNewConversation(),
              disabled: resetDisabled,
              pending: isConversationResetPending,
            } : undefined}
            expanded={isExpanded}
            onExpandedChange={onExpandedChange}
          />
        </div>
        {isAbsoluteExpiryWarningVisible ? (
          <p
            className="mt-3 rounded-md border border-amber-300/60 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200"
            role="status"
          >
            {t("widgetSession.expiryWarning")}
          </p>
        ) : null}
        {resetError && !reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.resetFailed")}
          </p>
        ) : null}
        {startMessageFailure && !reloadRequired && !conversationEstablished ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {startMessageFailure.code
              ? t(clientErrorTranslationKey(startMessageFailure.code))
              : t("widgetSession.startMessageFailed")}
          </p>
        ) : null}
        {reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.reloadRequired")}
          </p>
        ) : null}
      </header>

      <div className="min-h-0 flex-1">
        {conversationEstablished ? (
          <TaskConversationPanel
            mode="page"
            showTaskActions={false}
            showTokenUsage={false}
            showDagPreview={false}
            showTaskFiles={false}
            hideFileUpload={true}
            hideConfig={true}
            compactInput={true}
            deferFileUpload={true}
            onSend={attachments ? handlePanelSend : undefined}
            attachments={attachments}
            attachmentStatuses={attachments ? attachmentStatuses : undefined}
            onRetryAttachment={attachments ? handleRetryAttachment : undefined}
            attachmentsBusy={isAttachmentBusy}
            onAttachmentFilesChange={attachments ? handlePanelFilesChange : undefined}
            isBusy={isAttachmentBusy}
          />
        ) : (
          <div className="h-full overflow-y-auto">
            <main className="container mx-auto max-w-4xl px-4 py-8">
              <ChatStartScreen
                title={agent.name}
                description={agent.description}
                prompts={agent.suggestedPrompts}
                onSend={handleStartMessage}
                isSending={
                  state.isProcessing
                  || isConversationResetPending
                  || isMessageDeliveryPending
                  || reloadRequired
                  || isAttachmentBusy
                }
                files={attachments ? startFiles : []}
                onFilesChange={attachments ? handleStartFilesChange : () => undefined}
                attachments={attachments}
                attachmentStatuses={attachments ? attachmentStatuses : undefined}
                onRetryAttachment={attachments ? handleRetryAttachment : undefined}
                attachmentsBusy={isAttachmentBusy}
                readOnlyConfig={true}
                hideConfig={true}
                compactInput={true}
                deferFileUpload={true}
                filesDisabled={filesDisabled}
                voiceInputEnabled={voiceInputEnabled}
                autoFocus={true}
                inputMinHeightClass="min-h-[44px]"
              />
            </main>
          </div>
        )}
      </div>
    </div>
  )
}

const toAttachments = (
  fileUploads: WidgetSessionFileUploads | undefined,
): SessionAttachmentsConfig | undefined => fileUploads
  ? {
    accept: fileUploads.allowedExtensions.join(","),
    maxFiles: fileUploads.maxFilesPerMessage,
    maxFileBytes: fileUploads.maxFileBytes,
  }
  : undefined

export function SessionAgentChatPage() {
  const bridge = useWidgetSession()
  const [isExpanded, setIsExpanded] = useState(false)

  // The advertised capabilities follow every session_update. While the bridge
  // refreshes the token it has no session, and the composer keeps the last
  // advertisement so the visitor's files do not disappear for the refresh.
  const advertisedUploads = bridge.session?.capabilities?.fileUploads
  const [heldUploads, setHeldUploads] = useState(advertisedUploads)
  if (bridge.session && heldUploads !== advertisedUploads) {
    setHeldUploads(advertisedUploads)
  }
  const fileUploads = bridge.session
    ? advertisedUploads
    : bridge.status === "refreshing" ? heldUploads : undefined
  const attachments = useMemo(() => toAttachments(fileUploads), [fileUploads])
  const uploads = useMemo<SessionUploadControls>(() => ({
    ensureFreshSession: bridge.ensureFreshSession,
    refreshSession: bridge.refreshSession,
    waitForOpenConnection: bridge.waitForOpenConnection,
  }), [bridge.ensureFreshSession, bridge.refreshSession, bridge.waitForOpenConnection])

  const connection = useMemo<WebSocketConnection | null>(() => {
    if (bridge.status !== "active" || !bridge.session) return null

    return {
      identity: `widget-session:${bridge.session.generation}`,
      url: buildWidgetSessionWebSocketUrl(window.location.origin),
      protocols: [
        SESSION_PROTOCOL,
        `xagent-session-token.${bridge.session.token}`,
        SESSION_LIVENESS_PROTOCOL,
      ],
      expectedProtocol: SESSION_PROTOCOL,
      chatTaskIdMode: "omit",
      taskBindingMode: "session-subprotocol",
      credentialOwner: { kind: "external" },
    }
  }, [bridge.session, bridge.status])

  const transport = useMemo<AppProviderTransportConfig>(() => ({
    legacyErrorProse: "untrusted",
    capabilities: {
      // This page only renders for the embedded widget's session-resume
      // route, so an in-tab navigation always abandons the visitor's iframe.
      linksOpenInNewTab: "enabled",
      taskStop: "enabled",
    },
    session: {
      connection,
      onConnectionClose: bridge.handleConnectionClose,
      onConnectionFailure: bridge.handleConnectionFailure,
      onConnectionOpen: bridge.handleConnectionOpen,
      allowTasklessChat: true,
      supportsConversationReset: true,
      history: "none",
      // Uploading is the separate `attachments` option: "disabled" here keeps
      // previews, downloads and file mentions closed.
      files: "disabled",
      ...(attachments ? { attachments } : {}),
      agentCards: "disabled",
      voice: "disabled",
      taskControls: "disabled",
    },
  }), [
    attachments,
    bridge.handleConnectionClose,
    bridge.handleConnectionFailure,
    bridge.handleConnectionOpen,
    connection,
  ])

  return (
    <AppProvider transport={transport}>
      <SessionConversationContent
        status={bridge.status}
        agent={bridge.agent}
        terminalCode={bridge.terminalCode}
        isAbsoluteExpiryWarningVisible={
          bridge.isAbsoluteExpiryWarningVisible
        }
        isExpanded={isExpanded}
        onExpandedChange={setIsExpanded}
        attachments={attachments}
        uploads={uploads}
      />
    </AppProvider>
  )
}
