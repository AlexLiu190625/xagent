"use client"

import React, { useState } from "react"
import { AlertTriangle, Loader2 } from "lucide-react"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"

interface DeploymentConfigFallbackAlertProps {
  /**
   * Reload the runtime deployment configuration and update the owning dialog.
   * The alert remains mounted when the retry rejects, so degraded state is
   * visible until a caller confirms that the regional target was recovered.
   */
  onRetry: () => Promise<void>
}

/**
 * Persistent warning for deployment dialogs using the browser-origin fallback.
 *
 * A toast is easy to miss and can leave a user copying a target that is wrong
 * for a regional deployment. This shared alert keeps that state visible and
 * gives every deployment surface the same explicit recovery action.
 */
export function DeploymentConfigFallbackAlert({
  onRetry,
}: DeploymentConfigFallbackAlertProps) {
  const { t } = useI18n()
  const [isRetrying, setIsRetrying] = useState(false)

  const retry = async () => {
    setIsRetrying(true)
    try {
      await onRetry()
    } finally {
      setIsRetrying(false)
    }
  }

  return (
    <Alert className="border-amber-200 bg-amber-50 text-amber-900">
      <AlertTriangle className="text-amber-700" aria-hidden="true" />
      <AlertDescription className="flex w-full items-center gap-3 text-amber-800">
        <span className="flex-1">
          {t("deployment_config.messages.load_failed")
            || "Failed to load deployment configuration; using this browser's origin."}
        </span>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={isRetrying}
          onClick={() => void retry()}
        >
          {isRetrying && <Loader2 className="h-4 w-4 animate-spin" />}
          {t("deployment_config.actions.retry") || "Retry"}
        </Button>
      </AlertDescription>
    </Alert>
  )
}
