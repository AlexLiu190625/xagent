"use client"

import React from "react"
import Link from "next/link"
import { AlertTriangle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useI18n } from "@/contexts/i18n-context"
import { canEditAgent } from "@/lib/agent-ui-access"
import type {
  WorkforceAgentOption,
  WorkforceWorkerDraft,
} from "@/types/workforce"

interface ReviewStepProps {
  name: string
  description: string
  managerAgentId: string
  managerInstructions: string
  workers: WorkforceWorkerDraft[]
  agents: WorkforceAgentOption[]
  getAgentHref?: (agentId: number) => string
}

export function ReviewStep({
  name,
  description,
  managerAgentId,
  managerInstructions,
  workers,
  agents,
  getAgentHref = (agentId) => `/build/${agentId}`,
}: ReviewStepProps) {
  const { t } = useI18n()
  const manager = agents.find((agent) => String(agent.id) === managerAgentId)

  const warnings: string[] = []
  if (manager && manager.status !== "published") {
    warnings.push(t("workforces.review.warnings.managerNotPublished"))
  }
  for (const worker of workers) {
    const agent = agents.find((item) => item.id === worker.agent_id)
    if (agent && agent.status !== "published") {
      warnings.push(
        t("workforces.review.warnings.workerNotPublished", {
          name: worker.alias || agent.name,
        }),
      )
    }
    if (!worker.assignment_instructions.trim()) {
      warnings.push(
        t("workforces.review.warnings.missingInstructions", {
          name: worker.alias || t("workforces.workers.aWorker"),
        }),
      )
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.create.steps.review")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {warnings.length > 0 ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
            <div className="flex items-center gap-2 font-medium">
              <AlertTriangle className="size-4" />
              {t("workforces.review.potentialRisks")}
            </div>
            <div className="mt-2 space-y-1 text-sm">
              {warnings.map((warning, index) => (
                <p key={`${warning}-${index}`}>{warning}</p>
              ))}
            </div>
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.name")}
            </div>
            <div className="mt-1 font-medium">
              {name || t("workforces.review.untitled")}
            </div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.manager")}
            </div>
            <div className="mt-1 flex items-center gap-2 font-medium">
              <span>{manager?.name || t("workforces.common.notSelected")}</span>
              {manager ? (
                <Badge variant="outline">{t(`workforces.status.${manager.status}`)}</Badge>
              ) : null}
              {manager && !canEditAgent(manager) ? (
                <Badge variant="secondary">{t("workforces.actions.readOnly")}</Badge>
              ) : null}
            </div>
            {manager && canEditAgent(manager) ? (
              <Link
                href={getAgentHref(manager.id)}
                target="_blank"
                className="mt-2 inline-block text-sm text-primary hover:underline"
              >
                {t("workforces.actions.openAgentEditor")}
              </Link>
            ) : null}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            {t("workforces.fields.description")}
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            {description || t("workforces.common.noDescription")}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            {t("workforces.fields.managerInstructions")}
          </div>
          <div className="mt-1 whitespace-pre-wrap text-sm text-muted-foreground">
            {managerInstructions || t("workforces.review.noManagerInstructions")}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            {t("workforces.fields.workers")}
          </div>
          <div className="mt-3 space-y-3">
            {workers.length === 0 ? (
              <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                {t("workforces.workers.noneConfigured")}
              </div>
            ) : (
              workers.map((worker, index) => {
                const agent = agents.find((item) => item.id === worker.agent_id)
                const title = worker.alias
                  || agent?.name
                  || t("workforces.workers.fallbackName", { index: index + 1 })

                return (
                  <div key={`${worker.source_type}-${index}`} className="rounded-lg border p-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="font-medium">{title}</div>
                      <Badge variant="outline">
                        {t(`workforces.sourceTypes.${worker.source_type}`)}
                      </Badge>
                      {agent ? (
                        <Badge variant="secondary">{t(`workforces.status.${agent.status}`)}</Badge>
                      ) : null}
                      {agent && !canEditAgent(agent) ? (
                        <Badge variant="secondary">{t("workforces.actions.readOnly")}</Badge>
                      ) : null}
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      {agent?.description || t("workforces.workers.publishedAgent")}
                    </div>
                    <div className="mt-3 text-sm text-muted-foreground">
                      {worker.assignment_instructions}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
