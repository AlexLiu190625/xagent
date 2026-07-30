import type { ReactNode } from "react"

export type HomePageExtension = () => ReactNode

export interface BuildAgentCardExtensionProps {
  agentId: number
}

export interface BuildPageExtension {
  renderAgentCardSupplement:
    (props: BuildAgentCardExtensionProps) => ReactNode
}

export type UseBuildPageExtension = () => BuildPageExtension
