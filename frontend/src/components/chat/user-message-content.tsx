"use client"

import React from "react"

import { FileAttachment } from "@/components/file/file-attachment"
import type { ChatFileInfo } from "@/lib/chat-files"

interface BuildUserMessageContentOptions {
  scrollableText?: boolean
  onPreview?: (file: ChatFileInfo) => void
}

export function buildUserMessageContent(
  message: string,
  files: ChatFileInfo[],
  options: BuildUserMessageContentOptions = {},
): React.ReactNode {
  if (files.length === 0) {
    return message
  }

  const textClassName = options.scrollableText
    ? "whitespace-pre-wrap max-h-60 overflow-y-auto"
    : undefined

  return (
    <div className="space-y-2">
      <div className={textClassName}>{message}</div>
      <FileAttachment
        files={files}
        variant="user-message"
        onPreview={options.onPreview}
      />
    </div>
  )
}

export function userMessagePreviewFiles(files: ChatFileInfo[]) {
  return files
    .map((file) => ({
      fileId: file.file_id || "",
      fileName: file.name,
    }))
    .filter((file) => !!file.fileId)
}
