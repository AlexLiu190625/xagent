"use client"

import { Suspense, useEffect } from "react"
import { ArrowLeft, Loader2 } from "lucide-react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"

function TaskDetailContent() {
  const { state, setTaskId, closeFilePreview } = useApp()
  const { t } = useI18n()
  const params = useParams()
  const router = useRouter()
  const taskIdFromUrl = params.id

  useEffect(() => {
    if (taskIdFromUrl && typeof taskIdFromUrl === "string") {
      const taskIdNum = parseInt(taskIdFromUrl, 10)
      if (!Number.isNaN(taskIdNum) && taskIdNum !== state.taskId) {
        setTaskId(taskIdNum)
      }
    }
  }, [taskIdFromUrl, setTaskId, state.taskId])

  useEffect(() => {
    return () => {
      closeFilePreview()
    }
  }, [closeFilePreview])

  return (
    <div className="h-full relative">
      {state.currentTask?.agentId && (
        <div className="absolute top-4 left-4 z-50">
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full bg-background/50 hover:bg-background/80 backdrop-blur border shadow-sm"
            onClick={() => {
              const agentId = state.currentTask?.agentId
              router.push(agentId ? `/agent/${agentId}` : "/task")
            }}
            title={t("common.back")}
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </div>
      )}
      <TaskConversationPanel mode="page" />
    </div>
  )
}

export default function TaskDetailPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>}>
      <TaskDetailContent />
    </Suspense>
  )
}
