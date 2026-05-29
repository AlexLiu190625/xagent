type Translate = (key: string) => string

export function getRunDisabledReason(status: string | null | undefined, t: Translate) {
  if (status === "active") return null
  if (status === "archived") return t("workforces.run.archivedDisabled")
  return t("workforces.run.inactiveDisabled")
}

export function getBuilderReadOnlyReason(status: string | null | undefined, t: Translate) {
  if (status === "archived") return t("workforces.builder.archivedReadOnly")
  return null
}
