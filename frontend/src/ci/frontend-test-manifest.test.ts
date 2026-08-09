import { readFileSync } from "node:fs"
import { describe, expect, it, vi } from "vitest"
import vitestConfig from "../../vitest.config"

vi.mock("vitest/config", () => ({
  defineConfig: <T>(config: T) => config,
}))

const manifestCommand =
  "vitest run --config vitest.config.ts src/ci/frontend-test-manifest.test.ts"
const pagesCommand =
  "vitest run --config vitest.config.ts src/components/pages src/ci/frontend-test-manifest.test.ts"
const appPagesCommand = "vitest run --config vitest.config.ts src/app"
const homeBuildContractsCommand =
  "vitest run --config vitest.config.ts src/lib/models.test.ts src/lib/task-create.test.ts src/i18n/translations.test.ts src/lib/utils.test.ts src/lib/time-utils.test.ts"

function exactLineCount(lines: string[], value: string) {
  return lines.filter((line) => line === value).length
}

describe("frontend CI test manifest", () => {
  it("keeps App Router discovery and required frontend CI lanes source-locked", () => {
    const packageJson = JSON.parse(readFileSync("package.json", "utf8")) as {
      scripts: Record<string, string>
    }
    const workflowSource = readFileSync("../.github/workflows/ci.yml", "utf8")
    const startMarker = "  frontend-build:\n"
    const endMarker = "\n  ci-summary:\n"

    expect(workflowSource.split(startMarker)).toHaveLength(2)
    expect(workflowSource.split(endMarker)).toHaveLength(2)

    const startIndex = workflowSource.indexOf(startMarker)
    const endIndex = workflowSource.indexOf(endMarker)
    expect(endIndex).toBeGreaterThan(startIndex)

    const frontendBuildJob = workflowSource.slice(startIndex, endIndex)
    const frontendBuildLines = frontendBuildJob.split("\n").map((line) => line.trim())

    expect(packageJson.scripts["test:ci-manifest"]).toBe(manifestCommand)
    expect(packageJson.scripts["test:pages"]).toBe(pagesCommand)
    expect(packageJson.scripts["test:app-pages"]).toBe(appPagesCommand)
    expect(packageJson.scripts["test:home-build-contracts"]).toBe(
      homeBuildContractsCommand,
    )
    expect(packageJson.scripts["test:home-build-pages"]).toBeUndefined()

    for (const command of [
      "run: npm run test:widget:coverage",
      "run: npm run test:ci-manifest",
      "run: npm run test:pages",
      "run: npm run test:app-pages",
      "run: npm run test:home-build-contracts",
    ]) {
      expect(exactLineCount(frontendBuildLines, command)).toBe(1)
    }

    expect(frontendBuildLines).not.toContain("run: npm run test:home-build-pages")
    expect(frontendBuildJob).not.toMatch(/^\s+continue-on-error:/m)

    const testConfig = vitestConfig.test
    expect([...(testConfig?.include ?? [])].sort()).toEqual([
      "src/**/*.test.ts",
      "src/**/*.test.tsx",
    ])
    expect(testConfig?.exclude).toBeUndefined()
    expect(testConfig?.passWithNoTests).not.toBe(true)
  })
})
