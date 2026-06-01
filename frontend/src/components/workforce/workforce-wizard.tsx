"use client"

import React, { useEffect, useMemo, useState } from "react"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import {
  createWorkforce,
  listAgentOptions,
} from "@/lib/workforces-api"
import type {
  WorkforceAgentOption,
  WorkforceDetail,
  WorkforceWorkerDraft,
} from "@/types/workforce"
import { ManagerStep } from "./manager-step"
import { ReviewStep } from "./review-step"
import { WorkersStep } from "./workers-step"

const STEPS = [
  "workforces.create.steps.basics",
  "workforces.create.steps.workers",
  "workforces.create.steps.review",
] as const

interface WorkforceWizardProps {
  onCreated: (workforce: WorkforceDetail) => void
  onBack?: () => void
  onPromptSetup?: () => void
}

export function WorkforceWizard({
  onCreated,
  onBack,
  onPromptSetup,
}: WorkforceWizardProps) {
  const { t } = useI18n()
  const [step, setStep] = useState(0)
  const [loadingAgents, setLoadingAgents] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [agents, setAgents] = useState<WorkforceAgentOption[]>([])

  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [managerAgentId, setManagerAgentId] = useState("")
  const [managerInstructions, setManagerInstructions] = useState("")
  const [workers, setWorkers] = useState<WorkforceWorkerDraft[]>([])

  const managerWorkerConflict = useMemo(() => {
    if (!managerAgentId) return null
    const managerId = Number(managerAgentId)
    return workers.find((worker) => worker.agent_id === managerId) ?? null
  }, [managerAgentId, workers])

  const managerWorkerConflictMessage = useMemo(() => {
    if (!managerWorkerConflict) return null
    const agent = agents.find((item) => item.id === managerWorkerConflict.agent_id)
    return t("workforces.review.warnings.managerCannotBeWorker", {
      name:
        managerWorkerConflict.alias
        || agent?.name
        || t("workforces.workers.aWorker"),
    })
  }, [agents, managerWorkerConflict, t])

  const workersAreValid = useMemo(
    () =>
      workers.length > 0
      && !managerWorkerConflict
      && workers.every((worker) => {
        if (!worker.assignment_instructions.trim()) return false
        return Boolean(worker.agent_id)
      }),
    [managerWorkerConflict, workers],
  )

  useEffect(() => {
    const loadAgents = async () => {
      try {
        setLoadingAgents(true)
        const agentData = await listAgentOptions()
        setAgents(agentData.filter((agent) => agent.status === "published"))
      } catch (err) {
        setError(err instanceof Error ? err.message : t("workforces.errors.loadAgents"))
      } finally {
        setLoadingAgents(false)
      }
    }
    void loadAgents()
  }, [t])

  const canMoveForward = useMemo(() => {
    if (step === 0) {
      return Boolean(name.trim() && managerAgentId)
    }
    if (step === 1) {
      return workersAreValid
    }
    return true
  }, [step, name, managerAgentId, workersAreValid])

  const handleCreate = async () => {
    if (!name.trim() || !managerAgentId || !workersAreValid) return
    setSubmitting(true)
    setError(null)
    try {
      const workforce = await createWorkforce({
        name: name.trim(),
        description: description.trim() || undefined,
        manager_agent_id: Number(managerAgentId),
        manager_instructions: managerInstructions.trim() || undefined,
        workers: workers.map((worker) => ({
          source_type: worker.source_type,
          agent_id: worker.agent_id,
          alias: worker.alias.trim() || undefined,
          assignment_instructions: worker.assignment_instructions.trim(),
          enabled: worker.enabled,
          sort_order: worker.sort_order,
          canvas_position: worker.canvas_position,
        })),
      })
      onCreated(workforce)
    } catch (err) {
      setError(err instanceof Error ? err.message : t("workforces.errors.create"))
    } finally {
      setSubmitting(false)
    }
  }

  const handleBack = () => {
    if (step === 0) {
      onBack?.()
      return
    }
    setStep((current) => Math.max(0, current - 1))
  }

  if (loadingAgents) {
    return (
      <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">
        {t("workforces.loading.agents")}
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-4 sm:p-8">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            {onBack ? (
              <Button variant="ghost" className="w-fit gap-2 px-0" onClick={onBack}>
                <ArrowLeft className="h-4 w-4" />
                {t("workforces.create.backToWorkforces")}
              </Button>
            ) : null}
            {onPromptSetup ? (
              <Button variant="outline" onClick={onPromptSetup}>
                {t("workforces.create.manual.backToPrompt")}
              </Button>
            ) : null}
          </div>
          <div>
            <h1 className="text-3xl font-bold">{t("workforces.create.manual.title")}</h1>
            <p className="mt-2 text-muted-foreground">
              {t("workforces.create.manual.description")}
            </p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {STEPS.map((labelKey, index) => (
            <Card key={labelKey} className={index === step ? "border-primary" : undefined}>
              <CardContent className="flex items-center gap-3 p-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-full border text-sm font-medium">
                  {index + 1}
                </div>
                <div className="font-medium">{t(labelKey)}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {step === 0 ? (
          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <Card>
              <CardHeader>
                <CardTitle>{t("workforces.create.steps.basics")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>{t("workforces.fields.name")}</Label>
                  <Input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder={t("workforces.create.placeholders.name")}
                  />
                </div>
                <div className="space-y-2">
                  <Label>{t("workforces.fields.description")}</Label>
                  <Textarea
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                    placeholder={t("workforces.create.placeholders.description")}
                    rows={4}
                  />
                </div>
                <div className="space-y-2">
                  <Label>{t("workforces.fields.managerInstructions")}</Label>
                  <Textarea
                    value={managerInstructions}
                    onChange={(event) => setManagerInstructions(event.target.value)}
                    placeholder={t("workforces.create.placeholders.managerInstructions")}
                    rows={5}
                  />
                </div>
              </CardContent>
            </Card>
            <ManagerStep
              managerAgentId={managerAgentId}
              onManagerAgentIdChange={setManagerAgentId}
              agents={agents}
            />
          </div>
        ) : null}

        {step === 1 ? (
          <WorkersStep
            managerAgentId={managerAgentId}
            agents={agents}
            workers={workers}
            onWorkersChange={setWorkers}
          />
        ) : null}

        {step === 2 ? (
          <ReviewStep
            name={name}
            description={description}
            managerAgentId={managerAgentId}
            managerInstructions={managerInstructions}
            workers={workers}
            agents={agents}
          />
        ) : null}

        {managerWorkerConflictMessage ? (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            {managerWorkerConflictMessage}
          </div>
        ) : null}

        {error ? <div className="text-sm text-red-500">{error}</div> : null}

        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={handleBack}
            disabled={submitting || (step === 0 && !onBack)}
          >
            {step === 0 ? t("workforces.create.backToWorkforces") : t("common.back")}
          </Button>
          <div className="flex items-center gap-3">
            {step < STEPS.length - 1 ? (
              <Button
                onClick={() => setStep((current) => current + 1)}
                disabled={!canMoveForward}
              >
                {t("common.next")}
              </Button>
            ) : (
              <Button
                onClick={handleCreate}
                disabled={submitting || !canMoveForward || !workersAreValid}
              >
                {submitting
                  ? t("workforces.loading.creating")
                  : t("workforces.actions.create")}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
