"use client"

import React from "react"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useI18n } from "@/contexts/i18n-context"
import { canEditAgent } from "@/lib/agent-ui-access"
import type { WorkforceDetail } from "@/types/workforce"
import { WorkforceStatusBadge } from "./workforce-status-badge"

interface WorkforceSummaryProps {
  workforce: WorkforceDetail
  getAgentHref?: (agentId: number) => string
}

export function WorkforceSummary({
  workforce,
  getAgentHref = (agentId) => `/build/${agentId}`,
}: WorkforceSummaryProps) {
  const { locale, t } = useI18n()
  const enabledWorkers = workforce.workers.filter((item) => item.enabled).length

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>{workforce.name}</span>
          <WorkforceStatusBadge status={workforce.status} />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.manager")}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-2 font-medium">
              <span>{workforce.manager.name}</span>
              {!canEditAgent(workforce.manager) ? (
                <Badge variant="secondary">{t("workforces.actions.readOnly")}</Badge>
              ) : null}
            </div>
            <div className="text-sm text-muted-foreground">
              {workforce.manager.description || t("workforces.common.noDescription")}
            </div>
            {canEditAgent(workforce.manager) ? (
              <Link
                href={getAgentHref(workforce.manager.id)}
                target="_blank"
                className="mt-2 inline-block text-sm text-primary hover:underline"
              >
                {t("workforces.actions.openAgentEditor")}
              </Link>
            ) : null}
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.workers")}
            </div>
            <div className="mt-1 font-medium">{workforce.workers.length}</div>
            <div className="text-sm text-muted-foreground">
              {t("workforces.summary.enabledCount", { count: enabledWorkers })}
            </div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.updated")}
            </div>
            <div className="mt-1 font-medium" suppressHydrationWarning>
              {workforce.updated_at
                ? new Date(workforce.updated_at).toLocaleString(locale)
                : t("workforces.common.notAvailable")}
            </div>
          </div>
        </div>

        {workforce.description ? (
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.description")}
            </div>
            <p className="mt-1 text-sm text-muted-foreground">{workforce.description}</p>
          </div>
        ) : null}

        {workforce.manager_instructions ? (
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {t("workforces.fields.managerInstructions")}
            </div>
            <p className="mt-1 whitespace-pre-wrap text-sm text-muted-foreground">
              {workforce.manager_instructions}
            </p>
          </div>
        ) : null}

        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            {t("workforces.fields.workers")}
          </div>
          <div className="mt-3 grid gap-3">
            {workforce.workers.length === 0 ? (
              <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                {t("workforces.workers.noneYet")}
              </div>
            ) : (
              workforce.workers.map((worker) => (
                <div key={worker.id} className="rounded-lg border bg-background/40 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium">{worker.alias || worker.agent.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {worker.agent.description || t("workforces.common.noDescription")}
                      </div>
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        <Badge variant="outline">
                          {t(`workforces.sourceTypes.${worker.source_type}`)}
                        </Badge>
                        {worker.template_id ? (
                          <Badge variant="outline">{worker.template_id}</Badge>
                        ) : null}
                        {!canEditAgent(worker.agent) ? (
                          <Badge variant="secondary">{t("workforces.actions.readOnly")}</Badge>
                        ) : null}
                        {canEditAgent(worker.agent) ? (
                          <Link
                            href={getAgentHref(worker.agent.id)}
                            target="_blank"
                            className="text-sm text-primary hover:underline"
                          >
                            {t("workforces.actions.editAgent")}
                          </Link>
                        ) : null}
                      </div>
                    </div>
                    <Badge variant={worker.enabled ? "default" : "secondary"}>
                      {worker.enabled
                        ? t("workforces.status.enabled")
                        : t("workforces.status.disabled")}
                    </Badge>
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    {worker.assignment_instructions}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
