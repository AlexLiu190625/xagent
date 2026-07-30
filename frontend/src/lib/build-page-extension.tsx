import type {
  BuildAgentCardExtensionProps,
  UseBuildPageExtension,
} from "@/lib/page-extension-contracts"

export const useBuildPageExtension = (() => ({
  renderAgentCardSupplement: (props: BuildAgentCardExtensionProps) => {
    void props
    return null
  },
})) satisfies UseBuildPageExtension
