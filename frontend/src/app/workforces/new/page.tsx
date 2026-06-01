"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { WorkforcePromptCreator, WorkforceWizard } from "@/components/workforce"

export default function NewWorkforcePage() {
  const router = useRouter()
  const [mode, setMode] = useState<"prompt" | "manual">("prompt")
  const onCreated = (workforce: { id: number }) => {
    router.push(`/workforces/${workforce.id}`)
  }
  const onBack = () => {
    router.push("/workforces")
  }

  if (mode === "manual") {
    return (
      <WorkforceWizard
        onCreated={onCreated}
        onBack={onBack}
        onPromptSetup={() => setMode("prompt")}
      />
    )
  }

  return (
    <WorkforcePromptCreator
      onCreated={onCreated}
      onBack={onBack}
      onManualSetup={() => setMode("manual")}
    />
  )
}
