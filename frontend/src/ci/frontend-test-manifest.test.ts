import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
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
const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const packageJsonPath = path.resolve(moduleDir, "../../package.json")
const workflowPath = path.resolve(moduleDir, "../../../.github/workflows/ci.yml")

function exactLineCount(lines: string[], value: string) {
  return lines.filter((line) => line === value).length
}

function extractFrontendBuildJob(source: string) {
  source = source.replace(/\r\n/g, "\n")
  const jobHeader = /^  (?:([A-Za-z_][A-Za-z0-9_-]*)|'([A-Za-z_][A-Za-z0-9_-]*)'|"([A-Za-z_][A-Za-z0-9_-]*)"):[ \t]*(?:#.*)?$/gm
  const jobHeaders = Array.from(source.matchAll(jobHeader), (match) => ({
    id: match[1] ?? match[2] ?? match[3] ?? "",
    index: match.index!,
  }))
  const frontendHeaders = jobHeaders.filter(
    ({ id }) => id === "frontend-build",
  )

  expect(frontendHeaders).toHaveLength(1)

  const frontendHeader = frontendHeaders[0]!
  const frontendHeaderIndex = jobHeaders.indexOf(frontendHeader)
  const nextHeader = jobHeaders[frontendHeaderIndex + 1]
  expect(nextHeader).toBeDefined()

  return source.slice(frontendHeader.index, nextHeader!.index)
}

describe("frontend CI test manifest", () => {
  it("extracts the same frontend-build job from LF and CRLF workflow text", () => {
    const workflowSource = readFileSync(workflowPath, "utf8")
    const crlfWorkflowSource = workflowSource.replace(/\r?\n/g, "\r\n")

    expect(extractFrontendBuildJob(crlfWorkflowSource)).toBe(
      extractFrontendBuildJob(workflowSource),
    )
  })

  it.each([
    [
      "a renamed successor",
      (source: string) =>
        source.replace("\n  ci-summary:\n", "\n  post-frontend:\n"),
    ],
    [
      "an inserted unquoted successor",
      (source: string) =>
        source.replace(
          "\n  ci-summary:\n",
          "\n  manifest-sibling:\n    continue-on-error: true\n\n  ci-summary:\n",
        ),
    ],
    [
      "an inserted single-quoted successor",
      (source: string) =>
        source.replace(
          "\n  ci-summary:\n",
          "\n  'manifest-sibling':\n    runs-on: ubuntu-latest\n\n  ci-summary:\n",
        ),
    ],
    [
      "an inserted double-quoted successor",
      (source: string) =>
        source.replace(
          "\n  ci-summary:\n",
          "\n  \"manifest-sibling\":\n    runs-on: ubuntu-latest\n\n  ci-summary:\n",
        ),
    ],
  ])("keeps the frontend-build slice for %s", (_, updateSource) => {
    const workflowSource = readFileSync(workflowPath, "utf8")
    const frontendBuildJob = extractFrontendBuildJob(workflowSource)

    expect(extractFrontendBuildJob(updateSource(workflowSource))).toBe(
      frontendBuildJob,
    )
  })

  it("keeps the frontend-build slice when e2e becomes its successor", () => {
    const workflowSource = readFileSync(workflowPath, "utf8")
    const frontendBuildJob = extractFrontendBuildJob(workflowSource)
    const reorderedWorkflowSource = workflowSource
      .replace(frontendBuildJob, "")
      .replace("  e2e:\n", `${frontendBuildJob}  e2e:\n`)

    expect(extractFrontendBuildJob(reorderedWorkflowSource)).toBe(
      frontendBuildJob,
    )
  })

  it("keeps App Router discovery and required frontend CI lanes source-locked", () => {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
      scripts: Record<string, string>
    }
    const workflowSource = readFileSync(workflowPath, "utf8")
    const frontendBuildJob = extractFrontendBuildJob(workflowSource)
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
