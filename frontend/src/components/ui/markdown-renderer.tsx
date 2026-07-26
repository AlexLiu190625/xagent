import React, { useEffect, useState } from 'react'
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components, ExtraProps } from 'react-markdown'
import { apiRequest } from '@/lib/api-wrapper'
import { AgentCard } from '@/components/chat/AgentCard'
import { useI18n } from '@/contexts/i18n-context'
import { InlineFilePreview } from '@/components/file/inline-file-preview'
import {
  getInlineFilePreviewKind,
  getInlineFilePreviewMimeType,
  isPreviewableInlineFileKind,
  resolveInlineFileId,
  type PreviewableInlineFileKind,
} from '@/components/file/inline-file-preview-utils'
import { getApiUrl } from '@/lib/utils'


interface AgentInfo {
  id: number
  name: string
  description?: string
  status: 'draft' | 'published'
  instructions?: string
}

// Enhanced Markdown detection function: covers broader Markdown features not limited to starting with #
const isLikelyMarkdown = (s: string): boolean => {
  const t = s.trim()
  if (!t) return false
  return (
    t.startsWith('#') || // Heading
    s.includes('```') || // Code block
    s.includes('**') || // Bold
    /(\n|^)\s*(-|\*|\d+\.)\s/.test(s) || // List (unordered/ordered)
    (s.includes('|') && s.includes('---')) || // Table
    /\[[^\]]+\]\([^\)]+\)/.test(s) || // Link [text](url)
    /!\[[^\]]*\]\([^\)]+\)/.test(s) || // Image ![alt](url)
    /(\n|^)\s*>\s/.test(s) || // Blockquote
    /(\n|^)\s*---\s*(\n|$)/.test(s) // Horizontal rule
  )
}

const FILE_NAME_KEYS = new Set(['filename', 'file_name', 'name'])
const FILE_DESCRIPTOR_KEYS = new Set(['mime_type', 'type'])
const GENERIC_IDENTITY_AND_LOCATION_KEYS = new Set(['id', 'url', 'href', 'uri'])
const FILE_RECORD_COLLECTION_KEYS = new Set(['artifacts', 'documents', 'files'])
const KNOWN_FILE_DESCRIPTOR_VALUES = new Set([
  'audio',
  'document',
  'file',
  'image',
  'presentation',
  'spreadsheet',
  'video',
])
const LOCAL_FILE_LOCATION_KEYS = new Set([
  // Backend artifacts.LOCAL_PATH_KEYS.
  'absolute_path',
  'file_path',
  'image_path',
  'local_path',
  'output_dir',
  'output_path',
  // Additional local locations emitted by current tool-result producers.
  'audio_path',
  'backup_path',
  'base_dir',
  'current_path',
  'full_path',
  'json_path',
  'marked_image_path',
  'relative_path',
  'source_path',
  'storage_path',
  'transcription_path',
  'translation_path',
  'uploads_directory',
  'video_path',
  'workspace_dir',
])
const AMBIGUOUS_FILE_LOCATION_KEYS = new Set(['path', 'directory'])
const FILE_MARKDOWN_REFERENCE_RE = /!?\[([^\]]*)\]\(\s*file:(?:\/\/)?[^)]*\)/g
const BACKTICK_FILE_PATH_RE = /`([^`\n]*[\\/][^`\n]*)`/g

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeFileMetadataKey(key: string): string {
  return key.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase()
}

function isFileIdKey(key: string): boolean {
  const normalized = normalizeFileMetadataKey(key)
  return normalized === 'file_id' || normalized.endsWith('_file_id')
}

function isFileAccessKey(key: string): boolean {
  const normalized = normalizeFileMetadataKey(key)
  return /(^|_)(preview|download|signed|file)_url$/.test(normalized)
}

function hasStrongFileIdentity(value: Record<string, unknown>): boolean {
  const entries = Object.entries(value)
  const keys = entries.map(([key]) => normalizeFileMetadataKey(key))
  if (keys.some(isFileIdKey)) {
    return true
  }
  if (keys.some((key) => key === 'filename' || key === 'file_name')) {
    return true
  }

  const hasName = keys.some((key) => FILE_NAME_KEYS.has(key))
  const descriptor = entries.find(([key]) => (
    FILE_DESCRIPTOR_KEYS.has(normalizeFileMetadataKey(key))
  ))?.[1]
  return (
    hasName
    && typeof descriptor === 'string'
    && (
      descriptor.includes('/')
      || KNOWN_FILE_DESCRIPTOR_VALUES.has(descriptor.toLowerCase())
    )
  )
}

function hasFileCopyEvidence(value: Record<string, unknown>): boolean {
  const keys = new Set(Object.keys(value).map(normalizeFileMetadataKey))
  return (
    keys.has('source')
    && keys.has('destination')
    && typeof value.extracted === 'boolean'
  )
}

function isLocalFileLocationKey(
  key: string,
  owner: Record<string, unknown>,
  fileRecordContext = false,
): boolean {
  const normalized = normalizeFileMetadataKey(key)
  if (LOCAL_FILE_LOCATION_KEYS.has(normalized)) {
    return true
  }
  if (AMBIGUOUS_FILE_LOCATION_KEYS.has(normalized)) {
    return fileRecordContext || hasStrongFileIdentity(owner)
  }
  if (normalized === 'html_src') {
    return hasStrongFileIdentity(owner)
  }
  if (normalized === 'source' || normalized === 'destination') {
    return hasFileCopyEvidence(owner)
  }
  return false
}

function basename(value: string): string | null {
  const parts = value.split(/[\\/]/).filter(Boolean)
  return parts.at(-1) || null
}

function rawFileLabel(value: Record<string, unknown>): string | null {
  for (const key of ['filename', 'file_name', 'fileName', 'name']) {
    const candidate = value[key]
    if (typeof candidate === 'string' && candidate.trim()) {
      return basename(candidate) || candidate
    }
  }
  return null
}

function collectKnownLocalPaths(value: unknown): Map<string, string> {
  const paths = new Map<string, string>()

  const visit = (item: unknown, fileRecordContext = false): void => {
    if (Array.isArray(item)) {
      item.forEach((child) => visit(child, fileRecordContext))
      return
    }
    if (!isRecord(item)) {
      return
    }

    const label = rawFileLabel(item)
    Object.entries(item).forEach(([key, child]) => {
      if (
        isLocalFileLocationKey(key, item, fileRecordContext)
        && typeof child === 'string'
        && child.trim()
      ) {
        const fallback = basename(child)
        if (label || fallback) {
          paths.set(child, label ?? fallback ?? child)
        }
      }
      visit(
        child,
        fileRecordContext
        || FILE_RECORD_COLLECTION_KEYS.has(normalizeFileMetadataKey(key)),
      )
    })
  }

  visit(value)
  return paths
}

function replaceKnownLocalPaths(
  value: string,
  knownPaths: Map<string, string>,
): string {
  return [...knownPaths.entries()]
    .sort(([left], [right]) => right.length - left.length)
    .reduce(
      (result, [path, replacement]) => result.split(path).join(replacement),
      value,
    )
}

function projectFilesDisabledValueWithPaths(
  value: unknown,
  knownPaths: Map<string, string>,
  fileRecordContext = false,
): unknown {
  if (typeof value === 'string') {
    return sanitizeFilesDisabledText(replaceKnownLocalPaths(value, knownPaths))
  }

  if (Array.isArray(value)) {
    return value.map((item) => (
      projectFilesDisabledValueWithPaths(item, knownPaths, fileRecordContext)
    ))
  }

  if (!isRecord(value)) {
    return value
  }

  const strongFileIdentity = hasStrongFileIdentity(value)
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, child]) => {
      const normalizedKey = normalizeFileMetadataKey(key)
      if (
        isFileIdKey(normalizedKey)
        || isFileAccessKey(normalizedKey)
        || isLocalFileLocationKey(normalizedKey, value, fileRecordContext)
      ) {
        return []
      }
      if (
        strongFileIdentity
        && GENERIC_IDENTITY_AND_LOCATION_KEYS.has(normalizedKey)
      ) {
        return []
      }
      return [[
        key,
        projectFilesDisabledValueWithPaths(
          child,
          knownPaths,
          fileRecordContext || FILE_RECORD_COLLECTION_KEYS.has(normalizedKey),
        ),
      ]]
    }),
  )
}

export function sanitizeFilesDisabledText(value: string): string {
  return value
    .replace(FILE_MARKDOWN_REFERENCE_RE, (_match, label: string) => label)
    .replace(BACKTICK_FILE_PATH_RE, (match, path: string) => {
      const scheme = /^([a-z][a-z0-9+.-]*):\/\//i.exec(path.trim())
      if (scheme && scheme[1].toLowerCase() !== 'file') {
        return match
      }
      return path.split(/[\\/]/).pop() || path
    })
}

/**
 * Produces the inert, display-safe representation of structured file metadata.
 * Local file locations are removed from any tool-result record. Generic
 * identifiers and URLs are removed only when that record has explicit file
 * identity; unrelated business records retain those fields.
 */
export function projectFilesDisabledValue(value: unknown): unknown {
  return projectFilesDisabledValueWithPaths(value, collectKnownLocalPaths(value))
}

export function projectFilesDisabledToolResultOutput(value: unknown): unknown {
  const projected = projectFilesDisabledValue(value)
  if (isRecord(projected)) {
    if ('output' in projected) {
      return projected.output
    }
    if ('message' in projected) {
      return projected.message
    }
  }
  return projected
}

export function getFilesDisabledFileLabel(value: unknown): string | null {
  if (isRecord(value)) {
    const label = rawFileLabel(value)
    if (label) {
      return label
    }
    for (const [key, child] of Object.entries(value)) {
      if (isLocalFileLocationKey(key, value, true) && typeof child === 'string') {
        const pathLabel = basename(child)
        if (pathLabel) {
          return pathLabel
        }
      }
    }
  }

  const projected = projectFilesDisabledValue(value)
  if (!isRecord(projected)) {
    return null
  }

  for (const key of ['filename', 'file_name', 'fileName', 'name']) {
    const candidate = projected[key]
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate
    }
  }
  return null
}

export function serializeFilesDisabledValue(value: unknown): string {
  if (typeof value === 'string') {
    try {
      return JSON.stringify(projectFilesDisabledValue(JSON.parse(value)), null, 2)
    } catch {
      return sanitizeFilesDisabledText(value)
    }
  }

  try {
    const serialized = JSON.stringify(projectFilesDisabledValue(value), null, 2)
    return serialized ?? String(value)
  } catch {
    return String(value)
  }
}

interface MarkdownRendererProps {
  content: string
  className?: string
  filesDisabled?: boolean
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
}

const safeUrlTransform = (url: string): string => {
  if (!url) return ''
  if (url.startsWith('file:')) return url
  if (url.startsWith('agent:')) return url
  return defaultUrlTransform(url)
}

// Hook to fetch agent details
function useAgentInfo(agentId: string) {
  const [agentInfo, setAgentInfo] = useState<AgentInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false

    async function fetchAgentInfo() {
      try {
        setLoading(true)
        setError(null)

        const apiUrl = getApiUrl()
        const response = await apiRequest(`${apiUrl}/api/agents/${agentId}`)

        if (!response.ok) {
          throw new Error(`Failed to fetch agent: ${response.statusText}`)
        }

        const data: AgentInfo = await response.json()

        if (!cancelled) {
          setAgentInfo(data)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    fetchAgentInfo()

    return () => {
      cancelled = true
    }
  }, [agentId])

  return { agentInfo, loading, error }
}


// Agent Card Container component that fetches data
function AgentCardContainer({
  agentId,
  agentName: initialAgentName,
  onAgentClick,
}: {
  agentId: string
  agentName: string
  onAgentClick?: (agentId: string, agentName: string) => void
}) {
  const { t } = useI18n()
  const { agentInfo, loading, error } = useAgentInfo(agentId)

  // Show loading state
  if (loading) {
    return (
      <div className="inline-flex items-center gap-2 bg-muted/50 border border-border rounded-lg p-3 my-2 max-w-sm">
        <div className="w-8 h-8 rounded-md bg-muted animate-pulse" />
        <div className="flex-1">
          <div className="h-4 bg-muted rounded animate-pulse w-32 mb-1" />
          <div className="h-3 bg-muted rounded animate-pulse w-24" />
        </div>
      </div>
    )
  }

  // Show error state with fallback name
  if (error || !agentInfo) {
    return (
      <AgentCard
        agentId={agentId}
        agentName={initialAgentName}
        description={t("markdownRenderer.loadAgentDetailsFailed")}
        status="draft"
      />
    )
  }

  // Show agent info
  // Don't pass onClick - let AgentCard handle navigation internally based on status
  return (
    <AgentCard
      agentId={agentId}
      agentName={agentInfo.name}
      description={agentInfo.description || agentInfo.instructions}
      status={agentInfo.status}
    />
  )
}

function containsAgentCardElement(children: React.ReactNode): boolean {
  return React.Children.toArray(children).some((child) => {
    if (!React.isValidElement(child)) {
      return false
    }

    if (child.props?.['data-agent-card-wrapper']) {
      return true
    }

    return containsAgentCardElement(child.props?.children)
  })
}

function containsBlockPreviewElement(children: React.ReactNode): boolean {
  return React.Children.toArray(children).some((child) => {
    if (!React.isValidElement(child)) {
      return false
    }

    if (child.props?.['data-inline-file-preview-wrapper']) {
      return true
    }

    return containsBlockPreviewElement(child.props?.children)
  })
}

function hastText(node: any): string {
  if (!node) return ''
  if (typeof node.value === 'string') return node.value
  if (!Array.isArray(node.children)) return ''
  return node.children.map(hastText).join('')
}

const nodeText = (children: React.ReactNode): string => {
  return React.Children.toArray(children)
    .map((child) => {
      if (typeof child === 'string' || typeof child === 'number') {
        return String(child)
      }
      if (React.isValidElement(child)) {
        return nodeText(child.props?.children)
      }
      return ''
    })
    .join('')
}

function resolvePreviewableFileLink({
  fileNameFromPath,
  fileName,
}: {
  fileNameFromPath: string
  fileName: string
}): { previewKind: PreviewableInlineFileKind; displayFilename: string } | null {
  const pathKind = getInlineFilePreviewKind({ filename: fileNameFromPath })
  if (isPreviewableInlineFileKind(pathKind)) {
    return { previewKind: pathKind, displayFilename: fileName }
  }

  const labelKind = getInlineFilePreviewKind({ filename: fileName })
  if (isPreviewableInlineFileKind(labelKind)) {
    return { previewKind: labelKind, displayFilename: fileName }
  }

  return null
}

function containsPreviewFileLinkNode(node: any): boolean {
  if (!node) return false
  const href = node.properties?.href
  if (typeof href === 'string' && href.startsWith('file:')) {
    const filePath = href.replace(/^file:/, '')
    const fileNameFromPath = filePath.split('/').pop() || filePath
    const title = typeof node.properties?.title === 'string' ? node.properties.title : ''
    const label = title || hastText(node)
    if (resolvePreviewableFileLink({ fileNameFromPath, fileName: label })) return true
  }
  const src = node.properties?.src
  if (typeof src === 'string' && src.startsWith('file:')) {
    return true
  }
  if (!Array.isArray(node.children)) return false
  return node.children.some(containsPreviewFileLinkNode)
}

type MarkdownRendererContextValue = {
  filesDisabled: boolean
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
  openLabel: string
  loadErrorText: string
}

type MarkdownComponentProps<Tag extends keyof React.JSX.IntrinsicElements> =
  React.ComponentPropsWithoutRef<Tag> & ExtraProps

const MarkdownRendererContext = React.createContext<MarkdownRendererContextValue | null>(null)

function useMarkdownRendererContext(): MarkdownRendererContextValue {
  const context = React.useContext(MarkdownRendererContext)
  if (!context) {
    throw new Error('Markdown components must be rendered within MarkdownRenderer')
  }
  return context
}

function MarkdownParagraph({
  node,
  children,
  ...props
}: MarkdownComponentProps<'p'>) {
  if (
    containsAgentCardElement(children) ||
    containsBlockPreviewElement(children) ||
    containsPreviewFileLinkNode(node)
  ) {
    return (
      <div className="my-4" {...props}>
        {children}
      </div>
    )
  }

  return <p {...props}>{children}</p>
}

function MarkdownLink({
  node,
  href,
  title,
  children,
  ...props
}: MarkdownComponentProps<'a'>) {
  const {
    filesDisabled,
    onFileClick,
    onAgentClick,
    openLabel,
    loadErrorText,
  } =
    useMarkdownRendererContext()

  if (href && href.startsWith('file:')) {
    const filePath = href.replace(/^file:/, '')
    const fileNameFromPath = filePath.split('/').pop() || filePath
    const linkText = (node ? hastText(node) : nodeText(children)).trim()
    const fileName = title || linkText || fileNameFromPath
    const preview = resolvePreviewableFileLink({ fileNameFromPath, fileName })
    const fileId = resolveInlineFileId(filePath)

    if (filesDisabled) {
      return <span>{children}</span>
    }

    if (preview) {
      return (
        <InlineFilePreview
          source={{
            fileId,
            filename: preview.displayFilename,
            type: preview.previewKind,
            mimeType: getInlineFilePreviewMimeType(preview.previewKind),
          }}
          openLabel={openLabel}
          loadErrorText={loadErrorText}
          onFileClick={onFileClick}
        />
      )
    }

    const handleClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
      if (onFileClick) {
        event.preventDefault()
        const fallbackTitle = title || linkText || fileNameFromPath
        onFileClick(fileId, fallbackTitle)
      }
    }

    return (
      <a
        href="#"
        data-file-path={filePath}
        className="file-link"
        title={title || undefined}
        onClick={handleClick}
        {...props}
      >
        {children}
      </a>
    )
  }

  if (href && href.startsWith('agent:')) {
    const agentId = href.replace(/^agent:\/\//, '')
    const agentNameFromLink =
      (node ? hastText(node) : nodeText(children)).trim() || `Agent ${agentId}`

    return React.createElement('div', {
      className: 'my-2',
      key: `agent-${agentId}-wrapper`,
      'data-agent-card-wrapper': true,
    }, React.createElement(AgentCardContainer, {
      key: `agent-${agentId}`,
      agentId,
      agentName: agentNameFromLink,
      onAgentClick,
    }))
  }

  return (
    <a href={href || undefined} title={title || undefined} {...props}>
      {children}
    </a>
  )
}

function MarkdownImage({
  node: _node,
  src,
  alt,
  title,
  ...props
}: MarkdownComponentProps<'img'>) {
  const { filesDisabled, onFileClick, openLabel, loadErrorText } =
    useMarkdownRendererContext()
  const resolvedSrc = src || ''

  if (resolvedSrc.startsWith('file:')) {
    const filePath = resolvedSrc.replace(/^file:/, '')
    const fileNameFromPath = filePath.split('/').pop() || filePath
    const fileName = title || alt || fileNameFromPath
    const preview = resolvePreviewableFileLink({ fileNameFromPath, fileName })
    const previewKind = preview?.previewKind ?? 'image'
    if (filesDisabled) {
      return <span>{fileName}</span>
    }

    return (
      <InlineFilePreview
        source={{
          fileId: resolveInlineFileId(filePath),
          filename: preview?.displayFilename ?? fileName,
          type: previewKind,
          mimeType: getInlineFilePreviewMimeType(previewKind),
        }}
        openLabel={openLabel}
        loadErrorText={loadErrorText}
        onFileClick={onFileClick}
        imageClassName="file-image cursor-pointer"
      />
    )
  }

  return <img src={resolvedSrc} alt={alt || ''} title={title || alt || ''} {...props} />
}

// Keep these component identities stable across chat/trace updates. Replacing
// them makes React remount every custom Markdown node, including a playing
// <audio> element whose playback state would then be lost.
const markdownComponents: Components = {
  p: MarkdownParagraph,
  a: MarkdownLink,
  img: MarkdownImage,
}

export function MarkdownRenderer({
  content,
  className = '',
  filesDisabled = false,
  onFileClick,
  onAgentClick,
}: MarkdownRendererProps) {
  const { t } = useI18n()
  const contextValue = React.useMemo<MarkdownRendererContextValue>(
    () => ({
      filesDisabled,
      onFileClick,
      onAgentClick,
      openLabel: t('files.previewDialog.buttons.open'),
      loadErrorText: t('files.previewDialog.errors.loadFailed'),
    }),
    [filesDisabled, onFileClick, onAgentClick, t]
  )

  return (
    <MarkdownRendererContext.Provider value={contextValue}>
      <div className={`prose prose-invert max-w-none break-words [overflow-wrap:anywhere] ${className}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={markdownComponents}
          urlTransform={safeUrlTransform}
        >
          {content}
        </ReactMarkdown>
      </div>
    </MarkdownRendererContext.Provider>
  )
}

interface JsonRendererProps {
  data: any
  className?: string
  filesDisabled?: boolean
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
}

export function JsonRenderer({
  data,
  className = '',
  filesDisabled = false,
  onFileClick,
  onAgentClick,
}: JsonRendererProps) {
  const [expanded, setExpanded] = React.useState(true)

  if (typeof data === 'string') {
    // Try to parse as JSON first
    try {
      const parsed = JSON.parse(data)
      return (
        <JsonRenderer
          data={parsed}
          className={className}
          filesDisabled={filesDisabled}
          onFileClick={onFileClick}
          onAgentClick={onAgentClick}
        />
      )
    } catch {
      // If not JSON, try to identify Markdown more comprehensively
      const displayText = filesDisabled ? sanitizeFilesDisabledText(data) : data
      if (isLikelyMarkdown(displayText)) {
        return (
          <MarkdownRenderer
            content={displayText}
            className={className}
            filesDisabled={filesDisabled}
            onFileClick={onFileClick}
            onAgentClick={onAgentClick}
          />
        )
      }
      // Otherwise display as plain text
      return (
        <pre className={`py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
          {displayText}
        </pre>
      )
    }
  }

  const displayData = filesDisabled ? projectFilesDisabledValue(data) : data

  if (typeof displayData === 'object' && displayData !== null) {
    // Check if it's a result object with output that might be markdown
    if (
      isRecord(data) &&
      typeof data.output === 'string' &&
      isLikelyMarkdown(data.output.trim()) &&
      isRecord(displayData) &&
      typeof displayData.output === 'string'
    ) {
      return (
        <div className={`space-y-3 ${className}`}>
          <div className="bg-muted p-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap">
            <div className="text-green-400 mb-2">✅ Task completed successfully</div>
            <div className="text-gray-400">Goal: {displayData.goal as React.ReactNode}</div>
          </div>
          <div className="border-t border-border pt-3">
            <div className="text-sm font-medium text-foreground mb-2">Result:</div>
            <MarkdownRenderer
              content={displayData.output}
              filesDisabled={filesDisabled}
              onFileClick={onFileClick}
              onAgentClick={onAgentClick}
            />
          </div>
        </div>
      )
    }

    // For other objects, display as formatted JSON
    return (
      <div className={`space-y-2 ${className}`}>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          {expanded ? '▼' : '▶'} JSON Data
        </button>
        {expanded && (
          <pre className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(displayData, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  // For other types, display as string
  return (
    <pre className={`bg-muted py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
      {String(data)}
    </pre>
  )
}
