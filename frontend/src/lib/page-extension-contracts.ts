import type { ComponentType, ReactNode } from "react"

// Embedding distributions may replace default implementation modules during frontend composition.
export type HomePageExtensionComponent = ComponentType

// The page guarantees a stable Provider lifetime and agentId join key.
// The paired replacement implementation owns data loading, sharing, and invalidation.
export interface BuildAgentCardExtensionProps {
  // Stable key for joining Provider-owned page data to an Agent card.
  agentId: number
}

export interface BuildPageExtensionProviderProps {
  children: ReactNode
}

export type BuildPageExtensionProviderComponent =
  ComponentType<BuildPageExtensionProviderProps>

export type BuildAgentCardExtensionComponent =
  ComponentType<BuildAgentCardExtensionProps>
