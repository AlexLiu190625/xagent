"use client"

import React from "react"
import { ArrowRight, Network } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceCanvasNode, WorkforceCanvasResponse } from "@/types/workforce"

interface WorkforceCanvasProps {
  canvas: WorkforceCanvasResponse
}

export function WorkforceCanvas({ canvas }: WorkforceCanvasProps) {
  const { t } = useI18n()
  const nodesById = React.useMemo(
    () => new Map(canvas.nodes.map((node) => [node.id, node])),
    [canvas.nodes],
  )

  const nodeTypeLabel = (node: WorkforceCanvasNode) => {
    const key = `workforces.canvas.nodeTypes.${node.type}`
    const translated = t(key)
    return translated === key ? node.type : translated
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.actions.canvas")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-3 md:grid-cols-3">
          {canvas.nodes.map((node) => (
            <div key={node.id} className="rounded-lg border bg-background/40 p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                {nodeTypeLabel(node)}
              </div>
              <div className="mt-2 font-medium">{node.label}</div>
              {node.enabled === false ? (
                <div className="mt-2 text-xs text-muted-foreground">
                  {t("workforces.status.disabled")}
                </div>
              ) : null}
            </div>
          ))}
        </div>
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
            <Network className="h-3.5 w-3.5" />
            <span>{t("workforces.canvas.connections")}</span>
          </div>
          {canvas.edges.length > 0 ? (
            <div className="grid gap-2 lg:grid-cols-2">
              {canvas.edges.map((edge) => {
                const sourceNode = nodesById.get(edge.source)
                const targetNode = nodesById.get(edge.target)
                return (
                  <div
                    key={edge.id}
                    className="flex min-w-0 items-center gap-3 rounded-lg border bg-background/40 px-3 py-2 text-sm"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-medium">
                        {sourceNode?.label || edge.source}
                      </div>
                      {sourceNode ? (
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                          {nodeTypeLabel(sourceNode)}
                        </div>
                      ) : null}
                    </div>
                    <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-medium">
                        {targetNode?.label || edge.target}
                      </div>
                      {targetNode ? (
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                          {nodeTypeLabel(targetNode)}
                        </div>
                      ) : null}
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed px-3 py-4 text-sm text-muted-foreground">
              {t("workforces.canvas.noConnections")}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
