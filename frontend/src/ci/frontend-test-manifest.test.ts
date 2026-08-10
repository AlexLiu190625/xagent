import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { parseDocument } from "yaml"
import { describe, expect, it, vi } from "vitest"
import vitestConfig from "../../vitest.config"

vi.mock("vitest/config", () => ({
  defineConfig: <T>(config: T) => config,
}))

const manifestCommand =
  "vitest run --config vitest.config.ts src/ci/frontend-test-manifest.test.ts"
const pagesCommand =
  "vitest run --config vitest.config.ts src/components/pages src/ci/frontend-test-manifest.test.ts"
const kbComponentsCommand =
  "vitest run --config vitest.config.ts src/components/kb"
const appPagesCommand = "vitest run --config vitest.config.ts src/app"
const homeBuildContractsCommand =
  "vitest run --config vitest.config.ts src/lib/models.test.ts src/lib/task-create.test.ts src/i18n/translations.test.ts src/lib/utils.test.ts src/lib/time-utils.test.ts"
const ciSummaryCondition =
  "always() && (github.event_name != 'pull_request' || github.event.pull_request.draft == false)"
const frontendSummaryCheckCommand =
  "check_job \"frontend-build\" \"${{ needs['frontend-build'].result }}\""
const requiredFrontendSteps = [
  { command: "npm run test:widget:coverage", shell: undefined },
  { command: "npm run test:ci-manifest", shell: "bash" },
  { command: "npm run test:pages", shell: "bash" },
  { command: "npm run test:kb-components", shell: undefined },
  { command: "npm run test:app-pages", shell: undefined },
  { command: "npm run test:home-build-contracts", shell: undefined },
] as const
const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const packageJsonPath = path.resolve(moduleDir, "../../package.json")
const workflowPath = path.resolve(moduleDir, "../../../.github/workflows/ci.yml")

function requireRecord(
  value: unknown,
  owner: string,
): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${owner} must be an object`)
  }
}

function assertNoDefaultShell(value: Record<string, unknown>, owner: string) {
  const defaults = value.defaults
  if (defaults === undefined) {
    return
  }

  requireRecord(defaults, `${owner}.defaults`)
  const run = defaults.run
  if (run === undefined) {
    return
  }

  requireRecord(run, `${owner}.defaults.run`)
  if (run.shell !== undefined) {
    throw new Error(`${owner} must not set defaults.run.shell`)
  }
}

function assertCiSummaryContract(jobs: Record<string, unknown>) {
  const ciSummary = jobs["ci-summary"]
  requireRecord(ciSummary, "jobs.ci-summary")

  if (!Array.isArray(ciSummary.needs)) {
    throw new Error("jobs.ci-summary.needs must be an array")
  }
  if (ciSummary.needs.filter((job) => job === "frontend-build").length !== 1) {
    throw new Error("jobs.ci-summary.needs must contain frontend-build exactly once")
  }
  if (ciSummary["continue-on-error"] !== undefined) {
    throw new Error("jobs.ci-summary must not set continue-on-error")
  }
  if (ciSummary.if !== ciSummaryCondition) {
    throw new Error("jobs.ci-summary has an unexpected if policy")
  }
  if (!Array.isArray(ciSummary.steps)) {
    throw new Error("jobs.ci-summary.steps must be an array")
  }

  const matchingSteps = ciSummary.steps.filter((step) => {
    requireRecord(step, "jobs.ci-summary.steps entry")
    return step.name === "Check required jobs"
  })
  if (matchingSteps.length !== 1) {
    throw new Error("Check required jobs must appear in exactly one ci-summary step")
  }

  const checkStep = matchingSteps[0]!
  if (checkStep.shell !== "bash") {
    throw new Error("Check required jobs must use bash")
  }
  if (checkStep.if !== undefined) {
    throw new Error("Check required jobs must not set if")
  }
  if (checkStep["continue-on-error"] !== undefined) {
    throw new Error("Check required jobs must not set continue-on-error")
  }
  const run = checkStep.run
  if (typeof run !== "string") {
    throw new Error("Check required jobs run must be a string")
  }

  const frontendCheckLines = run
    .split(/\r?\n/)
    .filter((line) => line === frontendSummaryCheckCommand)
  if (frontendCheckLines.length !== 1) {
    throw new Error("Check required jobs must check frontend-build exactly once")
  }
}

function assertSemanticWorkflowManifest(source: string) {
  const document = parseDocument(source, {
    version: "1.2",
    uniqueKeys: true,
    merge: false,
  })

  if (document.errors.length > 0) {
    throw document.errors[0]
  }
  if (document.warnings.length > 0) {
    throw document.warnings[0]
  }

  const workflow = document.toJS({ maxAliasCount: 100 })
  requireRecord(workflow, "workflow root")
  requireRecord(workflow.jobs, "jobs")
  assertCiSummaryContract(workflow.jobs)
  const frontendBuild = workflow.jobs["frontend-build"]
  requireRecord(frontendBuild, "jobs.frontend-build")
  if (!Array.isArray(frontendBuild.steps)) {
    throw new Error("jobs.frontend-build.steps must be an array")
  }

  assertNoDefaultShell(workflow, "workflow root")
  assertNoDefaultShell(frontendBuild, "jobs.frontend-build")
  if (frontendBuild["continue-on-error"] !== undefined) {
    throw new Error("jobs.frontend-build must not set continue-on-error")
  }

  for (const step of frontendBuild.steps) {
    requireRecord(step, "jobs.frontend-build.steps entry")
    if (step["continue-on-error"] !== undefined) {
      throw new Error("frontend-build steps must not set continue-on-error")
    }
  }

  for (const requiredStep of requiredFrontendSteps) {
    const matchingSteps = frontendBuild.steps.filter((step) => {
      requireRecord(step, "jobs.frontend-build.steps entry")
      return step.run === requiredStep.command
    })

    if (matchingSteps.length !== 1) {
      throw new Error(`${requiredStep.command} must appear in exactly one frontend step`)
    }

    const step = matchingSteps[0]!
    if (step["working-directory"] !== "./frontend") {
      throw new Error(`${requiredStep.command} must use ./frontend`)
    }
    if (step.if !== undefined) {
      throw new Error(`${requiredStep.command} must not set if`)
    }
    if (step["continue-on-error"] !== undefined) {
      throw new Error(`${requiredStep.command} must not set continue-on-error`)
    }
    if (step.shell !== requiredStep.shell) {
      throw new Error(`${requiredStep.command} has an unexpected shell policy`)
    }
  }

  if (frontendBuild.steps.some((step) => step.run === "npm run test:home-build-pages")) {
    throw new Error("legacy test:home-build-pages must not run in frontend-build")
  }
}

function readWorkflowSource() {
  return readFileSync(workflowPath, "utf8")
}

describe("frontend CI test manifest", () => {
  it.each([["LF", readWorkflowSource()], ["CRLF", readWorkflowSource().replace(/\r?\n/g, "\r\n")]])(
    "accepts the real workflow with %s line endings",
    (_, source) => {
      expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
    },
  )

  it.each([
    [
      "an anchored sibling",
      (source: string) =>
        source.replace(
          "\n  ci-summary:\n",
          "\n  manifest-anchor: &manifest_base\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo manifest anchor\n\n  manifest-alias: *manifest_base\n\n  ci-summary:\n",
        ),
    ],
    [
      "an aliased sibling",
      (source: string) =>
        source
          .replace("  prepare-deepdoc-cache:\n", "  prepare-deepdoc-cache: &manifest_base\n")
          .replace("\n  ci-summary:\n", "\n  manifest-alias: *manifest_base\n\n  ci-summary:\n"),
    ],
  ])("accepts %s", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).toContain("&manifest_base")
    expect(source).toContain("*manifest_base")
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("accepts a folded required command with the same semantic value", () => {
    const source = readWorkflowSource().replace(
      "        run: npm run test:app-pages\n",
      "        run: >-\n          npm run\n          test:app-pages\n",
    )

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("accepts a colon-shaped line in a block scalar", () => {
    const source = readWorkflowSource().replace(
      "        run: npm run build\n",
      "        run: |\n          echo 'label: value'\n          npm run build\n",
    )

    expect(source).toContain("          echo 'label: value'\n")
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("rejects an unknown YAML directive warning before conversion", () => {
    const workflowSource = readWorkflowSource()
    const source = `%BAD_DIRECTIVE\n---\n${workflowSource}`

    expect(source).not.toBe(workflowSource)
    expect(source).toMatch(/^%BAD_DIRECTIVE\n---\n/)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Unknown directive %BAD_DIRECTIVE",
    )
  })

  it("rejects an anchored job expanded through more than 100 aliases", () => {
    const aliases = Array.from(
      { length: 101 },
      (_, index) => `  manifest-alias-${index + 1}: *manifest_base`,
    ).join("\n")
    const workflowSource = readWorkflowSource()
    const source = workflowSource.replace(
      "\n  ci-summary:\n",
      `\n  manifest-anchor: &manifest_base\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo manifest anchor\n${aliases}\n\n  ci-summary:\n`,
    )

    expect(source).not.toBe(workflowSource)
    expect(source).toContain("  manifest-anchor: &manifest_base\n")
    expect(source.match(/manifest-alias-\d+: \*manifest_base/g)).toHaveLength(101)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    ["an empty source", ""],
    ["malformed YAML", "jobs:\n  frontend-build: ["],
    ["missing jobs", "name: Manifest fixture\n"],
    ["a non-mapping jobs owner", "jobs: []\n"],
    ["missing frontend-build", "jobs: {}\n"],
    ["a non-mapping frontend-build owner", "jobs:\n  frontend-build: []\n"],
    ["a non-sequence steps owner", "jobs:\n  frontend-build:\n    steps: {}\n"],
    ["an unresolved alias", "jobs:\n  frontend-build: *missing\n"],
  ])("rejects %s", (_, source) => {
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects semantically complete duplicate jobs and frontend-build mappings", () => {
    const workflowSource = readWorkflowSource()
    const duplicateJobs = `${workflowSource
      .replace("\njobs:\n", "\njobs: &manifest_jobs\n")
      .trimEnd()}\n'jobs': *manifest_jobs\n`
    const duplicateFrontendBuild = workflowSource
      .replace(
        "  frontend-build:\n",
        "  frontend-build: &manifest_frontend_build\n",
      )
      .replace(
        "\n  ci-summary:\n",
        "\n  'frontend-build': *manifest_frontend_build\n\n  ci-summary:\n",
      )

    expect(duplicateJobs).not.toBe(workflowSource)
    expect(duplicateJobs).toContain("jobs: &manifest_jobs\n")
    expect(duplicateJobs).toContain("'jobs': *manifest_jobs\n")
    expect(duplicateFrontendBuild).not.toBe(workflowSource)
    expect(duplicateFrontendBuild).toContain(
      "frontend-build: &manifest_frontend_build\n",
    )
    expect(duplicateFrontendBuild).toContain(
      "'frontend-build': *manifest_frontend_build\n",
    )
    expect(() => assertSemanticWorkflowManifest(duplicateJobs)).toThrow()
    expect(() => assertSemanticWorkflowManifest(duplicateFrontendBuild)).toThrow()
  })

  it("rejects a required command moved to a sibling job", () => {
    const source = readWorkflowSource()
      .replace(
        "\n      - name: Run App Router tests\n        working-directory: ./frontend\n        run: npm run test:app-pages\n",
        "",
      )
      .replace(
        "\n  ci-summary:\n",
        "\n  manifest-sibling:\n    runs-on: ubuntu-latest\n    steps:\n      - working-directory: ./frontend\n        run: npm run test:app-pages\n\n  ci-summary:\n",
      )

    expect(source).toContain("  manifest-sibling:\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects a same-job heredoc decoy after the real pages step is removed", () => {
    const source = readWorkflowSource()
      .replace(
        "\n      - name: Run page component tests\n        working-directory: ./frontend\n        shell: bash\n        run: npm run test:pages\n",
        "",
      )
      .replace(
        "\n      - name: Run App Router tests\n",
        "\n      - name: Preserve page launcher text as a shell heredoc\n        run: |\n          run: npm run test:pages\n\n      - name: Run App Router tests\n",
      )

    expect(source).not.toContain("      - name: Run page component tests\n")
    expect(source).toContain("          run: npm run test:pages\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:pages must appear in exactly one frontend step",
    )
  })

  it("rejects missing and duplicate required semantic steps", () => {
    const source = readWorkflowSource()
    const missing = source.replace("        run: npm run test:app-pages\n", "")
    const duplicate = source.replace(
      "\n      - name: Run App Router tests\n",
      "\n      - name: Duplicate App Router tests\n        working-directory: ./frontend\n        run: npm run test:app-pages\n\n      - name: Run App Router tests\n",
    )

    expect(missing).not.toContain("        run: npm run test:app-pages\n")
    expect(duplicate.match(/run: npm run test:app-pages/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow()
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow()
  })

  it("rejects missing and duplicate KB directory steps", () => {
    const workflowSource = readWorkflowSource()
    const kbStep =
      "\n      - name: Run knowledge base component tests\n        working-directory: ./frontend\n        run: npm run test:kb-components\n"
    const missing = workflowSource.replace(kbStep, "")
    const duplicate = workflowSource.replace(kbStep, kbStep.repeat(2))

    expect(missing).not.toContain("run: npm run test:kb-components")
    expect(duplicate.match(/run: npm run test:kb-components/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow()
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow()
  })

  it("rejects a required step with the wrong working directory", () => {
    const source = readWorkflowSource().replace(
      "      - name: Run App Router tests\n        working-directory: ./frontend\n",
      "      - name: Run App Router tests\n        working-directory: .\n",
    )

    expect(source).toContain("      - name: Run App Router tests\n        working-directory: .\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects a required step-level condition", () => {
    const source = readWorkflowSource().replace(
      "      - name: Run App Router tests\n",
      "      - name: Run App Router tests\n        if: github.event_name == 'schedule'\n",
    )

    expect(source).toContain("        if: github.event_name == 'schedule'\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects custom shells on non-launcher required steps", () => {
    const source = readWorkflowSource().replace(
      "      - name: Run App Router tests\n        working-directory: ./frontend\n",
      "      - name: Run App Router tests\n        working-directory: ./frontend\n        shell: echo {0}\n",
    )

    expect(source).toContain("        shell: echo {0}\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    [
      "test:ci-manifest",
      (source: string) =>
        source.replace(
          "        shell: bash\n        run: npm run test:ci-manifest\n",
          "        shell: echo {0}\n        run: npm run test:ci-manifest\n",
        ),
    ],
    [
      "test:pages",
      (source: string) =>
        source.replace(
          "      - name: Run page component tests\n        working-directory: ./frontend\n        shell: bash\n",
          "      - name: Run page component tests\n        working-directory: ./frontend\n        shell: echo {0}\n",
        ),
    ],
  ])("rejects a non-bash %s launcher", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).toContain("        shell: echo {0}\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    ["a step continue-on-error", (source: string) => source.replace("        run: npm run test:app-pages\n", "        continue-on-error: true\n        run: npm run test:app-pages\n")],
    ["a job continue-on-error", (source: string) => source.replace("  frontend-build:\n", "  frontend-build:\n    continue-on-error: true\n")],
    ["a workflow default shell", (source: string) => source.replace("jobs:\n", "defaults:\n  run:\n    shell: echo {0}\n\njobs:\n")],
    ["a job default shell", (source: string) => source.replace("  frontend-build:\n", "  frontend-build:\n    defaults:\n      run:\n        shell: echo {0}\n")],
  ])("rejects %s", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    [
      "a missing ci-summary",
      (source: string) => source.replace(/\n  ci-summary:\n[\s\S]*$/, "\n"),
    ],
    [
      "a non-mapping ci-summary",
      (source: string) => source.replace(/\n  ci-summary:\n[\s\S]*$/, "\n  ci-summary: []\n"),
    ],
    [
      "non-array ci-summary needs",
      (source: string) =>
        source.replace(
          "    needs:\n      - prepare-deepdoc-cache\n",
          "    needs: prepare-deepdoc-cache\n",
        ),
    ],
  ])("rejects %s", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects removing frontend-build from ci-summary.needs", () => {
    const source = readWorkflowSource().replace("      - frontend-build\n", "")

    expect(source).not.toContain("      - frontend-build\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects duplicate frontend-build entries in ci-summary.needs", () => {
    const source = readWorkflowSource().replace(
      "      - frontend-build\n",
      "      - frontend-build\n      - frontend-build\n",
    )

    expect(source.match(/^      - frontend-build$/gm)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects ci-summary job-level continue-on-error", () => {
    const source = readWorkflowSource().replace(
      "  ci-summary:\n",
      "  ci-summary:\n    continue-on-error: true\n",
    )

    expect(source).toContain("  ci-summary:\n    continue-on-error: true\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    [
      "missing always()",
      "if: github.event_name != 'pull_request' || github.event.pull_request.draft == false",
    ],
    ["a different condition", "if: always()"],
  ])("rejects ci-summary with %s", (_, replacement) => {
    const source = readWorkflowSource().replace(
      "if: always() && (github.event_name != 'pull_request' || github.event.pull_request.draft == false)",
      replacement,
    )

    expect(source).toContain(`    ${replacement}\n`)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects missing and duplicate Check required jobs owner steps", () => {
    const workflowSource = readWorkflowSource()
    const missing = workflowSource.replace(
      "      - name: Check required jobs\n",
      "      - name: Renamed required jobs\n",
    )
    const duplicate = workflowSource.replace(
      "      - name: Check required jobs\n",
      `      - name: Check required jobs\n        shell: bash\n        run: |\n          ${frontendSummaryCheckCommand}\n\n      - name: Check required jobs\n`,
    )

    expect(missing).not.toContain("      - name: Check required jobs\n")
    expect(duplicate.match(/name: Check required jobs/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "Check required jobs must appear in exactly one ci-summary step",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "Check required jobs must appear in exactly one ci-summary step",
    )
  })

  it.each([
    ["a non-bash shell", "      - name: Check required jobs\n        shell: sh\n"],
    [
      "a step-level condition",
      "      - name: Check required jobs\n        shell: bash\n        if: success()\n",
    ],
    [
      "step-level continue-on-error",
      "      - name: Check required jobs\n        shell: bash\n        continue-on-error: true\n",
    ],
  ])("rejects the summary check step with %s", (_, replacement) => {
    const source = readWorkflowSource().replace(
      "      - name: Check required jobs\n        shell: bash\n        run: |\n",
      `${replacement}        run: |\n`,
    )

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("rejects a missing, duplicate, or non-full-line frontend summary check", () => {
    const workflowSource = readWorkflowSource()
    const checkLine =
      "check_job \"frontend-build\" \"${{ needs['frontend-build'].result }}\""
    const indentedCheckLine = `          ${checkLine}\n`
    const missing = workflowSource.replace(indentedCheckLine, "")
    const duplicate = workflowSource.replace(indentedCheckLine, indentedCheckLine.repeat(2))
    const embedded = workflowSource.replace(indentedCheckLine, `          echo '${checkLine}'\n`)

    expect(missing).not.toContain(indentedCheckLine)
    expect(duplicate.match(/check_job "frontend-build"/g)).toHaveLength(2)
    expect(embedded).toContain(`          echo '${checkLine}'\n`)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow()
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow()
    expect(() => assertSemanticWorkflowManifest(embedded)).toThrow()
  })

  it("does not accept a frontend summary check decoy outside the owned step", () => {
    const workflowSource = readWorkflowSource()
    const checkLine =
      "check_job \"frontend-build\" \"${{ needs['frontend-build'].result }}\""
    const source = workflowSource
      .replace(`          ${checkLine}\n`, "")
      .replace(
        '          exit "$failed"\n',
        `          exit "$failed"\n\n      - name: Preserve frontend summary text outside owner\n        shell: bash\n        run: |\n          ${checkLine}\n`,
      )

    expect(source).toContain("      - name: Preserve frontend summary text outside owner\n")
    expect(source.match(/check_job "frontend-build"/g)).toHaveLength(1)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it("keeps package launchers and Vitest discovery contracts source-locked", () => {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
      scripts: Record<string, string>
    }

    expect(packageJson.scripts["test:ci-manifest"]).toBe(manifestCommand)
    expect(packageJson.scripts["test:pages"]).toBe(pagesCommand)
    expect(packageJson.scripts["test:kb-components"]).toBe(kbComponentsCommand)
    expect(packageJson.scripts["test:app-pages"]).toBe(appPagesCommand)
    expect(packageJson.scripts["test:home-build-contracts"]).toBe(homeBuildContractsCommand)
    expect(packageJson.scripts["test:home-build-pages"]).toBeUndefined()

    const testConfig = vitestConfig.test
    expect([...(testConfig?.include ?? [])].sort()).toEqual([
      "src/**/*.test.ts",
      "src/**/*.test.tsx",
    ])
    expect(testConfig?.exclude).toBeUndefined()
    expect(testConfig?.passWithNoTests).not.toBe(true)
  })
})
