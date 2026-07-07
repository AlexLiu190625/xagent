"use client"

import React, { useEffect, useMemo, useState } from "react"
import { Check, ChevronLeft, Code2, Copy, KeyRound, RefreshCw, X, Zap } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import {
  buildWidgetSnippet,
  fetchAgentWidgetKey,
  isValidAllowedDomain,
  normalizeAllowedDomain,
  rotateAgentWidgetKey,
  updateAgentWidgetConfig,
} from "@/lib/agent-widget-config"
import { getBrowserLocationOrigin } from "@/lib/browser-location"
import { copyToClipboard } from "@/lib/clipboard"

export interface AgentWidgetConfig {
  widget_enabled: boolean
  allowed_domains: string[]
}

interface AgentWidgetSettingsDialogProps {
  agentId: number | null
  agentName?: string
  open: boolean
  onOpenChange: (open: boolean) => void
  widgetConfig: AgentWidgetConfig
  onWidgetConfigUpdated?: (updatedAgent: Record<string, unknown>) => void
}

export function AgentWidgetSettingsDialog({
  agentId,
  agentName,
  open,
  onOpenChange,
  widgetConfig,
  onWidgetConfigUpdated,
}: AgentWidgetSettingsDialogProps) {
  const { t } = useI18n()
  const [appOrigin, setAppOrigin] = useState(() => getBrowserLocationOrigin())
  const [widgetState, setWidgetState] = useState<AgentWidgetConfig>(widgetConfig)
  const [newDomain, setNewDomain] = useState("")
  const [domainError, setDomainError] = useState<string | null>(null)
  const [isUpdating, setIsUpdating] = useState(false)
  const [copiedSnippet, setCopiedSnippet] = useState(false)
  const [widgetKey, setWidgetKey] = useState<string | null>(null)
  const [isRotatingKey, setIsRotatingKey] = useState(false)
  const [copiedKey, setCopiedKey] = useState(false)

  useEffect(() => {
    if (open) {
      setAppOrigin(getBrowserLocationOrigin())
    }
  }, [open])

  useEffect(() => {
    if (!open || !agentId) {
      setWidgetKey(null)
      return
    }
    let cancelled = false
    fetchAgentWidgetKey(agentId, t("appWidget.messages.widgetKeyLoadFailed"))
      .then((state) => {
        if (!cancelled) setWidgetKey(state.widget_key)
      })
      .catch((error) => {
        if (!cancelled) {
          toast.error(
            error instanceof Error ? error.message : t("appWidget.messages.widgetKeyLoadFailed"),
          )
        }
      })
    return () => {
      cancelled = true
    }
    // `t` is intentionally excluded: it is only used for the error toast, and
    // depending on it would re-fetch (and clobber a freshly rotated key) on
    // every render where the i18n function identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, agentId])

  useEffect(() => {
    if (!open) return
    setWidgetState({
      widget_enabled: Boolean(widgetConfig.widget_enabled),
      allowed_domains: Array.isArray(widgetConfig.allowed_domains) ? widgetConfig.allowed_domains : [],
    })
    setNewDomain("")
    setDomainError(null)
    setCopiedSnippet(false)
  }, [open, widgetConfig.allowed_domains, widgetConfig.widget_enabled])

  const allowedDomains = Array.isArray(widgetState.allowed_domains)
    ? widgetState.allowed_domains
    : []

  const widgetSnippet = useMemo(
    () => buildWidgetSnippet(widgetKey ?? "", appOrigin),
    [widgetKey, appOrigin],
  )

  const handleWidgetConfigUpdate = async (updates: Partial<AgentWidgetConfig>): Promise<boolean> => {
    if (!agentId) return false
    setIsUpdating(true)
    try {
      const updatedAgent = await updateAgentWidgetConfig(
        agentId,
        updates,
        t("appWidget.messages.updateFailed"),
      )
      const nextState = {
        widget_enabled:
          typeof updatedAgent.widget_enabled === "boolean"
            ? updatedAgent.widget_enabled
            : updates.widget_enabled ?? widgetState.widget_enabled,
        allowed_domains: Array.isArray(updatedAgent.allowed_domains)
          ? updatedAgent.allowed_domains
          : updates.allowed_domains ?? widgetState.allowed_domains,
      }
      setWidgetState(nextState)
      onWidgetConfigUpdated?.(updatedAgent)
      toast.success(t("appWidget.messages.updated"))
      return true
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("appWidget.messages.updateFailed"))
      return false
    } finally {
      setIsUpdating(false)
    }
  }

  const handleCopyWidgetKey = async () => {
    if (!widgetKey) return
    if (await copyToClipboard(widgetKey)) {
      setCopiedKey(true)
      toast.success(t("common.copied"))
      window.setTimeout(() => setCopiedKey(false), 2000)
    } else {
      toast.error(t("appWidget.messages.copyFailed"))
    }
  }

  const handleRotateWidgetKey = async () => {
    if (!agentId) return
    if (!window.confirm(t("appWidget.dialog.rotateWidgetKeyConfirm"))) return
    setIsRotatingKey(true)
    try {
      const state = await rotateAgentWidgetKey(
        agentId,
        t("appWidget.messages.widgetKeyRotateFailed"),
      )
      setWidgetKey(state.widget_key)
      toast.success(t("appWidget.messages.widgetKeyRotated"))
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t("appWidget.messages.widgetKeyRotateFailed"),
      )
    } finally {
      setIsRotatingKey(false)
    }
  }

  const handleCopySnippet = async () => {
    if (!widgetSnippet) return
    if (await copyToClipboard(widgetSnippet)) {
      setCopiedSnippet(true)
      toast.success(t("common.copied"))
      window.setTimeout(() => setCopiedSnippet(false), 2000)
    } else {
      toast.error(t("appWidget.messages.copyFailed"))
    }
  }

  const handleAddDomain = async () => {
    const domain = normalizeAllowedDomain(newDomain)
    if (!domain) return
    if (!isValidAllowedDomain(domain)) {
      setDomainError(t("appWidget.dialog.invalidDomain"))
      return
    }
    const existingDomains = new Set(allowedDomains.map((item) => item.toLowerCase()))
    if (existingDomains.has(domain)) {
      setNewDomain("")
      return
    }
    const updated = await handleWidgetConfigUpdate({
      allowed_domains: [...allowedDomains, domain],
    })
    if (updated) {
      setNewDomain("")
    }
  }

  const handleRemoveDomain = (domain: string) => {
    void handleWidgetConfigUpdate({
      allowed_domains: allowedDomains.filter((item) => item !== domain),
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="flex max-h-[88vh] w-[calc(100vw-2rem)] max-w-none flex-col overflow-hidden p-0 sm:max-w-[680px]"
      >
        <DialogHeader className="border-b px-5 py-4 pr-12">
          <DialogTitle className="flex items-center gap-2 text-base">
            <Zap className="h-4 w-4 text-primary" />
            {t("triggers.title")}
          </DialogTitle>
          <DialogDescription>
            {agentName ? `${agentName} · ${t("triggers.subtitle")}` : t("triggers.subtitle")}
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-3 border-b pb-4">
              <div className="flex min-w-0 items-center gap-3">
                <Button variant="ghost" size="sm" className="-ml-2" onClick={() => onOpenChange(false)}>
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  {t("common.back")}
                </Button>
                <div className="flex min-w-0 items-center gap-2 border-l pl-3">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted">
                    <Code2 className="h-4 w-4" />
                  </div>
                  <div className="truncate text-sm font-semibold">
                    {t("appWidget.dialog.title")}
                  </div>
                </div>
              </div>
              <Switch
                aria-label={t("appWidget.dialog.enabledLabel")}
                checked={widgetState.widget_enabled}
                disabled={isUpdating || !agentId}
                onCheckedChange={(checked) => void handleWidgetConfigUpdate({ widget_enabled: checked })}
              />
            </div>

            <section className="space-y-3">
              <div>
                <h3 className="text-sm font-medium">{t("appWidget.dialog.allowedDomains")}</h3>
                <p className="text-xs text-muted-foreground">
                  {t("deploy_agent.access_control.allowed_domains_desc")}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("appWidget.dialog.allowedDomainsSecurityNote")}
                </p>
              </div>
              <div className="flex gap-2">
                <Input
                  value={newDomain}
                  onChange={(event) => {
                    setNewDomain(event.target.value)
                    setDomainError(null)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault()
                      void handleAddDomain()
                    }
                  }}
                  placeholder={t("appWidget.dialog.domainPlaceholder")}
                  disabled={isUpdating || !agentId}
                  aria-invalid={domainError ? true : undefined}
                />
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => void handleAddDomain()}
                  disabled={isUpdating || !agentId || !newDomain.trim()}
                >
                  {t("appWidget.dialog.addDomain")}
                </Button>
              </div>
              {domainError && (
                <p className="text-xs text-destructive" role="alert">
                  {domainError}
                </p>
              )}
              <div className="flex flex-wrap gap-2">
                {allowedDomains.length > 0 ? (
                  allowedDomains.map((domain) => (
                    <Badge key={domain} variant="secondary" className="gap-1 px-2.5 py-1">
                      {domain}
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        onClick={() => handleRemoveDomain(domain)}
                        disabled={isUpdating}
                        aria-label={t("appWidget.dialog.removeDomain")}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">
                    {t("appWidget.dialog.noDomains")}
                  </span>
                )}
              </div>
            </section>

            <section className="space-y-3">
              <div>
                <h3 className="flex items-center gap-1.5 text-sm font-medium">
                  <KeyRound className="h-3.5 w-3.5" />
                  {t("appWidget.dialog.widgetKeyTitle")}
                </h3>
                <p className="text-xs text-muted-foreground">
                  {t("appWidget.dialog.widgetKeyDescription")}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <code className="min-w-0 flex-1 truncate rounded-md bg-muted px-3 py-2 font-mono text-xs text-muted-foreground">
                  {widgetKey ?? "…"}
                </code>
                <Button
                  type="button"
                  variant="secondary"
                  size="icon"
                  onClick={() => void handleCopyWidgetKey()}
                  disabled={!widgetKey}
                  aria-label={t("appWidget.dialog.copyWidgetKey")}
                  title={t("appWidget.dialog.copyWidgetKey")}
                  className="h-8 w-8 shrink-0"
                >
                  {copiedKey ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void handleRotateWidgetKey()}
                  disabled={!agentId || !widgetKey || isRotatingKey}
                  className="shrink-0"
                >
                  <RefreshCw className={`mr-1 h-3.5 w-3.5 ${isRotatingKey ? "animate-spin" : ""}`} />
                  {t("appWidget.dialog.rotateWidgetKey")}
                </Button>
              </div>
            </section>

            <section className="space-y-3">
              <div>
                <h3 className="text-sm font-medium">{t("appWidget.dialog.embedTitle")}</h3>
                <p className="text-xs text-muted-foreground">
                  {t("appWidget.dialog.embedDescription")}
                </p>
              </div>
              <div className="rounded-md bg-muted p-3">
                <div className="flex items-start gap-2">
                  <pre className="min-w-0 flex-1 max-h-56 overflow-auto whitespace-pre-wrap break-all text-xs text-muted-foreground">
                    {widgetSnippet}
                  </pre>
                  <Button
                    type="button"
                    variant="secondary"
                    size="icon"
                    onClick={() => void handleCopySnippet()}
                    disabled={!widgetSnippet}
                    aria-label={t("appWidget.dialog.copySnippet")}
                    title={t("appWidget.dialog.copySnippet")}
                    className="h-8 w-8 shrink-0"
                  >
                    {copiedSnippet ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            </section>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
