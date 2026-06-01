"use client"

import React, { useState } from "react"
import { ArrowLeft, Loader2, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import { createWorkforceFromPrompt } from "@/lib/workforces-api"
import type { WorkforceDetail } from "@/types/workforce"

interface WorkforcePromptCreatorProps {
  onManualSetup: () => void
  onCreated: (workforce: WorkforceDetail) => void
  onBack?: () => void
}

export function WorkforcePromptCreator({
  onManualSetup,
  onCreated,
  onBack,
}: WorkforcePromptCreatorProps) {
  const { t } = useI18n()
  const [prompt, setPrompt] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleCreate = async () => {
    const value = prompt.trim()
    if (!value || submitting) return
    try {
      setSubmitting(true)
      setError(null)
      const workforce = await createWorkforceFromPrompt({ prompt: value })
      onCreated(workforce)
    } catch (err) {
      setError(err instanceof Error ? err.message : t("workforces.errors.create"))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-4 sm:p-8">
        <div className="space-y-4">
          {onBack ? (
            <Button variant="ghost" className="w-fit gap-2 px-0" onClick={onBack}>
              <ArrowLeft className="h-4 w-4" />
              {t("workforces.create.backToWorkforces")}
            </Button>
          ) : null}
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5" />
              {t("workforces.create.prompt.badge")}
            </div>
            <h1 className="mt-4 text-3xl font-bold">
              {t("workforces.create.prompt.title")}
            </h1>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              {t("workforces.create.prompt.description")}
            </p>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t("workforces.create.prompt.cardTitle")}</CardTitle>
            <CardDescription>{t("workforces.create.prompt.cardDescription")}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder={t("workforces.create.prompt.placeholder")}
              rows={10}
            />
            {error ? <div className="text-sm text-red-500">{error}</div> : null}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <Button variant="outline" onClick={onManualSetup} disabled={submitting}>
                {t("workforces.create.prompt.manualSetup")}
              </Button>
              <Button onClick={handleCreate} disabled={submitting || !prompt.trim()}>
                {submitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {t("workforces.loading.creating")}
                  </>
                ) : (
                  t("workforces.create.prompt.generate")
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
