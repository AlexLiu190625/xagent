/**
 * Guards accidental drift across .github/workflows/ci.yml, package.json launchers,
 * and vitest.config.ts discovery. test:ci-manifest and test:pages are independent
 * launchers; legitimate changes update the owner files and these invariants together.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { isMap, isScalar, parseDocument } from "yaml"
import { describe, expect, it, vi } from "vitest"
import vitestConfig from "../../vitest.config"

// jsdom can load Vitest config in a second realm where vitest/config cannot initialize.
vi.mock("vitest/config", () => ({
  defineConfig: <T>(config: T) => config,
}))

const manifestCommand = "vitest run --config vitest.config.ts src/ci/frontend-test-manifest.test.ts"
const pagesCommand =
  "vitest run --config vitest.config.ts src/components/pages src/ci/frontend-test-manifest.test.ts"
const kbComponentsCommand = "vitest run --config vitest.config.ts src/components/kb"
const appPagesCommand = "vitest run --config vitest.config.ts src/app"
const homeBuildContractsCommand =
  "vitest run --config vitest.config.ts src/lib/models.test.ts src/lib/task-create.test.ts src/i18n/translations.test.ts src/lib/utils.test.ts src/lib/time-utils.test.ts"
const ciSummaryCondition =
  "always() && (github.event_name != 'pull_request' || github.event.pull_request.draft == false)"
const frontendSummaryCheckCommand =
  'check_job "frontend-build" "${{ needs[\'frontend-build\'].result }}"'
const requiredFrontendSteps = [
  { command: "npm run test:widget:coverage", requiresExplicitBash: false },
  { command: "npm run test:ci-manifest", requiresExplicitBash: true },
  { command: "npm run test:pages", requiresExplicitBash: true },
  { command: "npm run test:kb-components", requiresExplicitBash: false },
  { command: "npm run test:app-pages", requiresExplicitBash: false },
  { command: "npm run test:home-build-contracts", requiresExplicitBash: false },
] as const
const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const packageJsonPath = path.resolve(moduleDir, "../../package.json")
const workflowPath = path.resolve(moduleDir, "../../../.github/workflows/ci.yml")

function replaceExactlyOnce(source: string, search: string, replacement: string, owner: string) {
  if (search.length === 0) {
    throw new Error(`${owner} search marker must not be empty`)
  }

  const count = countOccurrences(source, search)
  if (count !== 1) {
    throw new Error(`${owner} must appear exactly once; found ${count}`)
  }

  const mutated = source.replace(search, replacement)
  if (mutated === source) {
    throw new Error(`${owner} mutation must change the source`)
  }
  return mutated
}

function countOccurrences(source: string, search: string) {
  let count = 0
  let offset = 0
  while ((offset = source.indexOf(search, offset)) !== -1) {
    count += 1
    offset += search.length
  }
  return count
}

function replaceWorkflowJob(source: string, jobName: string, replacement: string, owner: string) {
  const document = parseDocument(source, {
    version: "1.2",
    uniqueKeys: true,
    merge: false,
    keepSourceTokens: true,
  })
  if (document.errors.length > 0) {
    throw document.errors[0]
  }

  const workflow = document.contents
  const jobs = isMap(workflow) ? workflow.get("jobs", true) : undefined
  if (!isMap(jobs)) {
    throw new Error(`${owner} requires a jobs mapping`)
  }

  const matches = jobs.items.filter(
    (pair) => isScalar(pair.key) && pair.key.value === jobName,
  )
  if (matches.length !== 1) {
    throw new Error(`${owner} must appear exactly once; found ${matches.length}`)
  }

  const pair = matches[0]!
  if (!isScalar(pair.key) || pair.key.range == null) {
    throw new Error(`${owner} key must have a source range`)
  }
  const jobStart = source.lastIndexOf("\n", pair.key.range[0] - 1) + 1
  const nextPair = jobs.items[jobs.items.indexOf(pair) + 1]
  const nextKey = nextPair?.key
  const nextPairStart = nextPair?.srcToken?.start[0]?.offset
  let jobEnd = source.length
  if (nextPairStart !== undefined) {
    jobEnd = nextPairStart
  } else if (nextKey !== undefined && isScalar(nextKey) && nextKey.range != null) {
    jobEnd = source.lastIndexOf("\n", nextKey.range[0] - 1) + 1
  }
  const mutated = `${source.slice(0, jobStart)}${replacement}${source.slice(jobEnd)}`
  if (mutated === source) {
    throw new Error(`${owner} mutation must change the source`)
  }
  return mutated
}

function requireRecord(value: unknown, owner: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${owner} must be an object`)
  }
}

function assertBashCompatibleDefaultShell(value: Record<string, unknown>, owner: string) {
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
  if (run.shell !== undefined && run.shell !== "bash") {
    throw new Error(`${owner} defaults.run.shell must be bash when set`)
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
    const warning = document.warnings[0]!
    throw new Error(`workflow YAML warning [${warning.code ?? "UNKNOWN"}]`)
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

  assertBashCompatibleDefaultShell(workflow, "workflow root")
  assertBashCompatibleDefaultShell(frontendBuild, "jobs.frontend-build")
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
    const hasAllowedShell = requiredStep.requiresExplicitBash
      ? step.shell === "bash"
      : step.shell === undefined || step.shell === "bash"
    if (!hasAllowedShell) {
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
  it.each([
    ["LF", readWorkflowSource()],
    ["CRLF", readWorkflowSource().replace(/\r?\n/g, "\r\n")],
  ])("accepts the real workflow with %s line endings", (_, source) => {
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it.each([
    [
      "an anchored sibling",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "\n  ci-summary:\n",
          "\n  manifest-anchor: &manifest_base\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo manifest anchor\n\n  manifest-alias: *manifest_base\n\n  ci-summary:\n",
          "ci-summary insertion point",
        ),
    ],
    [
      "an aliased sibling",
      (source: string) => {
        const anchored = replaceExactlyOnce(
          source,
          "  prepare-deepdoc-cache:\n",
          "  prepare-deepdoc-cache: &manifest_base\n",
          "prepare-deepdoc-cache anchor",
        )
        return replaceExactlyOnce(
          anchored,
          "\n  ci-summary:\n",
          "\n  manifest-alias: *manifest_base\n\n  ci-summary:\n",
          "ci-summary insertion point",
        )
      },
    ],
  ])("accepts %s", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).toContain("&manifest_base")
    expect(source).toContain("*manifest_base")
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("accepts a folded required command with the same semantic value", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "        run: npm run test:app-pages\n",
      "        run: >-\n          npm run\n          test:app-pages\n",
      "App Router command",
    )

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it.each([
    "npm run test:widget:coverage",
    "npm run test:kb-components",
    "npm run test:app-pages",
    "npm run test:home-build-contracts",
  ])("accepts an explicit bash shell on the %s non-launcher step", (command) => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      `        run: ${command}\n`,
      `        shell: bash\n        run: ${command}\n`,
      `${command} shell insertion`,
    )

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it.each([
    [
      "workflow",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "jobs:\n",
          "defaults:\n  run:\n    shell: bash\n\njobs:\n",
          "workflow jobs owner",
        ),
    ],
    [
      "frontend-build job",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  frontend-build:\n",
          "  frontend-build:\n    defaults:\n      run:\n        shell: bash\n",
          "frontend-build owner",
        ),
    ],
  ])("accepts an explicit bash default at %s scope", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it.each([
    [
      "manifest launcher with workflow defaults",
      (source: string) =>
        replaceExactlyOnce(
          replaceExactlyOnce(
            source,
            "jobs:\n",
            "defaults:\n  run:\n    shell: bash\n\njobs:\n",
            "workflow jobs owner",
          ),
          "        shell: bash\n        run: npm run test:ci-manifest\n",
          "        run: npm run test:ci-manifest\n",
          "manifest launcher shell",
        ),
      "npm run test:ci-manifest has an unexpected shell policy",
    ],
    [
      "pages launcher with job defaults",
      (source: string) =>
        replaceExactlyOnce(
          replaceExactlyOnce(
            source,
            "  frontend-build:\n",
            "  frontend-build:\n    defaults:\n      run:\n        shell: bash\n",
            "frontend-build owner",
          ),
          "        shell: bash\n        run: npm run test:pages\n",
          "        run: npm run test:pages\n",
          "pages launcher shell",
        ),
      "npm run test:pages has an unexpected shell policy",
    ],
  ])("requires explicit bash for the %s", (_, transform, expectedError) => {
    const source = transform(readWorkflowSource())

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it("accepts a colon-shaped line in a block scalar", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "        run: npm run build\n",
      "        run: |\n          echo 'label: value'\n          npm run build\n",
      "frontend build command",
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
      "workflow YAML warning [BAD_DIRECTIVE]",
    )
  })

  it("fails a mutation when its owner marker is missing or duplicated", () => {
    expect(() => replaceExactlyOnce("alpha", "missing", "replacement", "fixture marker")).toThrow(
      "fixture marker must appear exactly once; found 0",
    )
    expect(() =>
      replaceExactlyOnce("alpha alpha", "alpha", "replacement", "fixture marker"),
    ).toThrow("fixture marker must appear exactly once; found 2")
  })

  it("rejects an anchored job expanded through more than 100 aliases", () => {
    const aliases = Array.from(
      { length: 101 },
      (_, index) => `  manifest-alias-${index + 1}: *manifest_base`,
    ).join("\n")
    const workflowSource = readWorkflowSource()
    const source = replaceExactlyOnce(
      workflowSource,
      "\n  ci-summary:\n",
      `\n  manifest-anchor: &manifest_base\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo manifest anchor\n${aliases}\n\n  ci-summary:\n`,
      "ci-summary insertion point",
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
    ["an unresolved alias", "jobs:\n  frontend-build: *missing\n"],
  ])("rejects %s", (_, source) => {
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    [
      "missing frontend-build",
      (source: string) => replaceWorkflowJob(source, "frontend-build", "", "frontend-build job"),
      "jobs.frontend-build must be an object",
    ],
    [
      "a non-mapping frontend-build owner",
      (source: string) =>
        replaceWorkflowJob(
          source,
          "frontend-build",
          "  frontend-build: []\n",
          "frontend-build job",
        ),
      "jobs.frontend-build must be an object",
    ],
    [
      "a non-sequence frontend-build.steps owner",
      (source: string) =>
        replaceWorkflowJob(
          source,
          "frontend-build",
          "  frontend-build:\n    steps: {}\n",
          "frontend-build job",
        ),
      "jobs.frontend-build.steps must be an array",
    ],
  ])("rejects %s at its intended owner", (_, transform, expectedError) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it("rejects semantically complete duplicate jobs and frontend-build mappings", () => {
    const workflowSource = readWorkflowSource()
    const duplicateJobs = `${replaceExactlyOnce(
      workflowSource,
      "\njobs:\n",
      "\njobs: &manifest_jobs\n",
      "jobs owner",
    ).trimEnd()}\n'jobs': *manifest_jobs\n`
    const anchoredFrontendBuild = replaceExactlyOnce(
      workflowSource,
      "  frontend-build:\n",
      "  frontend-build: &manifest_frontend_build\n",
      "frontend-build owner",
    )
    const duplicateFrontendBuild = replaceExactlyOnce(
      anchoredFrontendBuild,
      "\n  ci-summary:\n",
      "\n  'frontend-build': *manifest_frontend_build\n\n  ci-summary:\n",
      "ci-summary insertion point",
    )

    expect(duplicateJobs).not.toBe(workflowSource)
    expect(duplicateJobs).toContain("jobs: &manifest_jobs\n")
    expect(duplicateJobs).toContain("'jobs': *manifest_jobs\n")
    expect(duplicateFrontendBuild).not.toBe(workflowSource)
    expect(duplicateFrontendBuild).toContain("frontend-build: &manifest_frontend_build\n")
    expect(duplicateFrontendBuild).toContain("'frontend-build': *manifest_frontend_build\n")
    expect(() => assertSemanticWorkflowManifest(duplicateJobs)).toThrow()
    expect(() => assertSemanticWorkflowManifest(duplicateFrontendBuild)).toThrow()
  })

  it("rejects a required command moved to a sibling job", () => {
    const appStep =
      "\n      - name: Run App Router tests\n        working-directory: ./frontend\n        run: npm run test:app-pages\n"
    const withoutAppStep = replaceExactlyOnce(
      readWorkflowSource(),
      appStep,
      "",
      "App Router step removal",
    )
    const source = replaceExactlyOnce(
      withoutAppStep,
      "\n  ci-summary:\n",
      "\n  manifest-sibling:\n    runs-on: ubuntu-latest\n    steps:\n      - working-directory: ./frontend\n        run: npm run test:app-pages\n\n  ci-summary:\n",
      "ci-summary insertion point",
    )

    expect(source).not.toContain(appStep)
    expect(source).toContain("  manifest-sibling:\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:app-pages must appear in exactly one frontend step",
    )
  })

  it("fails the moved-to-sibling mutation when the App step owner drifts", () => {
    const workflowSource = readWorkflowSource()
    const driftedSource = replaceExactlyOnce(
      workflowSource,
      "      - name: Run App Router tests\n",
      "      - name: Renamed App Router tests\n",
      "App Router step name",
    )
    const appStep =
      "\n      - name: Run App Router tests\n        working-directory: ./frontend\n        run: npm run test:app-pages\n"

    expect(driftedSource).not.toBe(workflowSource)
    expect(() => replaceExactlyOnce(driftedSource, appStep, "", "App Router step removal")).toThrow(
      "App Router step removal must appear exactly once; found 0",
    )
  })

  it("rejects a same-job heredoc decoy after the real pages step is removed", () => {
    const withoutPagesStep = replaceExactlyOnce(
      readWorkflowSource(),
      "\n      - name: Run page component tests\n        working-directory: ./frontend\n        shell: bash\n        run: npm run test:pages\n",
      "",
      "pages launcher step removal",
    )
    const source = replaceExactlyOnce(
      withoutPagesStep,
      "\n      - name: Run App Router tests\n",
      "\n      - name: Preserve page launcher text as a shell heredoc\n        run: |\n          run: npm run test:pages\n\n      - name: Run App Router tests\n",
      "App Router step insertion point",
    )

    expect(source).not.toContain("      - name: Run page component tests\n")
    expect(source).toContain("          run: npm run test:pages\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:pages must appear in exactly one frontend step",
    )
  })

  it("rejects missing and duplicate required semantic steps", () => {
    const source = readWorkflowSource()
    const missing = replaceExactlyOnce(
      source,
      "        run: npm run test:app-pages\n",
      "",
      "App Router command removal",
    )
    const duplicate = replaceExactlyOnce(
      source,
      "\n      - name: Run App Router tests\n",
      "\n      - name: Duplicate App Router tests\n        working-directory: ./frontend\n        run: npm run test:app-pages\n\n      - name: Run App Router tests\n",
      "App Router duplicate insertion point",
    )

    expect(missing).not.toContain("        run: npm run test:app-pages\n")
    expect(duplicate.match(/run: npm run test:app-pages/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "npm run test:app-pages must appear in exactly one frontend step",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "npm run test:app-pages must appear in exactly one frontend step",
    )
  })

  it("rejects missing and duplicate KB directory steps", () => {
    const workflowSource = readWorkflowSource()
    const kbStep =
      "\n      - name: Run knowledge base component tests\n        working-directory: ./frontend\n        run: npm run test:kb-components\n"
    const missing = replaceExactlyOnce(workflowSource, kbStep, "", "KB test step removal")
    const duplicate = replaceExactlyOnce(
      workflowSource,
      kbStep,
      kbStep.repeat(2),
      "KB test step duplication",
    )

    expect(missing).not.toContain("run: npm run test:kb-components")
    expect(duplicate.match(/run: npm run test:kb-components/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "npm run test:kb-components must appear in exactly one frontend step",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "npm run test:kb-components must appear in exactly one frontend step",
    )
  })

  it("rejects a required step with the wrong working directory", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "      - name: Run App Router tests\n        working-directory: ./frontend\n",
      "      - name: Run App Router tests\n        working-directory: .\n",
      "App Router working directory",
    )

    expect(source).toContain("      - name: Run App Router tests\n        working-directory: .\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:app-pages must use ./frontend",
    )
  })

  it("rejects a required step-level condition", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "      - name: Run App Router tests\n",
      "      - name: Run App Router tests\n        if: github.event_name == 'schedule'\n",
      "App Router step condition",
    )

    expect(source).toContain("        if: github.event_name == 'schedule'\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:app-pages must not set if",
    )
  })

  it("rejects custom shells on non-launcher required steps", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "      - name: Run App Router tests\n        working-directory: ./frontend\n",
      "      - name: Run App Router tests\n        working-directory: ./frontend\n        shell: echo {0}\n",
      "App Router step shell",
    )

    expect(source).toContain("        shell: echo {0}\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:app-pages has an unexpected shell policy",
    )
  })

  it.each([
    [
      "test:ci-manifest",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "        shell: bash\n        run: npm run test:ci-manifest\n",
          "        shell: echo {0}\n        run: npm run test:ci-manifest\n",
          "manifest launcher shell",
        ),
    ],
    [
      "test:pages",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "      - name: Run page component tests\n        working-directory: ./frontend\n        shell: bash\n",
          "      - name: Run page component tests\n        working-directory: ./frontend\n        shell: echo {0}\n",
          "pages launcher shell",
        ),
    ],
  ])("rejects a non-bash %s launcher", (_, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).toContain("        shell: echo {0}\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      `npm run ${_} has an unexpected shell policy`,
    )
  })

  it.each([
    [
      "a step continue-on-error",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "        run: npm run test:app-pages\n",
          "        continue-on-error: true\n        run: npm run test:app-pages\n",
          "App Router continue-on-error insertion",
        ),
      "frontend-build steps must not set continue-on-error",
    ],
    [
      "a job continue-on-error",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  frontend-build:\n",
          "  frontend-build:\n    continue-on-error: true\n",
          "frontend-build continue-on-error insertion",
        ),
      "jobs.frontend-build must not set continue-on-error",
    ],
  ])("rejects %s", (_, transform, expectedError) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it.each([
    [
      "workflow root",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "jobs:\n",
          "defaults:\n  run:\n    shell: echo {0}\n\njobs:\n",
          "workflow jobs owner",
        ),
    ],
    [
      "jobs.frontend-build",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  frontend-build:\n",
          "  frontend-build:\n    defaults:\n      run:\n        shell: echo {0}\n",
          "frontend-build owner",
        ),
    ],
  ])("rejects a custom default shell at %s scope", (owner, transform) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      `${owner} defaults.run.shell must be bash when set`,
    )
  })

  it.each([
    [
      "a missing ci-summary",
      (source: string) => replaceWorkflowJob(source, "ci-summary", "", "ci-summary job"),
      "jobs.ci-summary must be an object",
    ],
    [
      "a non-mapping ci-summary",
      (source: string) =>
        replaceWorkflowJob(source, "ci-summary", "  ci-summary: []\n", "ci-summary job"),
      "jobs.ci-summary must be an object",
    ],
    [
      "non-array ci-summary needs",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "    needs:\n      - prepare-deepdoc-cache\n",
          "    needs: prepare-deepdoc-cache\n",
          "ci-summary needs owner",
        ),
      "jobs.ci-summary.needs must be an array",
    ],
  ])("rejects %s", (_, transform, expectedError) => {
    const source = transform(readWorkflowSource())

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it.each([
    [
      "plain",
      "\n\n  after-summary:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n",
      (source: string) => source,
    ],
    [
      "single-quoted",
      "\n\n  'after-summary':\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n",
      (source: string) => source,
    ],
    [
      "double-quoted",
      '\n\n  "after-summary":\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n',
      (source: string) => source,
    ],
    [
      "anchored",
      "\n\n  # anchored following job\n  after-summary: &after_summary\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n",
      (source: string) => source,
    ],
    [
      "aliased",
      "\n\n  # aliased following job\n  after-summary: *after_summary\n",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  prepare-deepdoc-cache:\n",
          "  prepare-deepdoc-cache: &after_summary\n",
          "prepare-deepdoc-cache anchor",
        ),
    ],
  ])("preserves a complete %s following sibling when mutating ci-summary", (_, block, prepare) => {
    const workflowSource = `${prepare(readWorkflowSource()).trimEnd()}${block}`
    const source = replaceWorkflowJob(workflowSource, "ci-summary", "", "ci-summary job")

    expect(source.slice(-block.length)).toBe(block)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary must be an object",
    )
  })

  it("rejects removing frontend-build from ci-summary.needs", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "      - frontend-build\n",
      "",
      "ci-summary frontend-build need",
    )

    expect(source).not.toContain("      - frontend-build\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary.needs must contain frontend-build exactly once",
    )
  })

  it("rejects duplicate frontend-build entries in ci-summary.needs", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "      - frontend-build\n",
      "      - frontend-build\n      - frontend-build\n",
      "ci-summary frontend-build need",
    )

    expect(source.match(/^      - frontend-build$/gm)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary.needs must contain frontend-build exactly once",
    )
  })

  it("rejects ci-summary job-level continue-on-error", () => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "  ci-summary:\n",
      "  ci-summary:\n    continue-on-error: true\n",
      "ci-summary owner",
    )

    expect(source).toContain("  ci-summary:\n    continue-on-error: true\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary must not set continue-on-error",
    )
  })

  it.each([
    [
      "missing always()",
      "if: github.event_name != 'pull_request' || github.event.pull_request.draft == false",
    ],
    ["a different condition", "if: always()"],
  ])("rejects ci-summary with %s", (_, replacement) => {
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "if: always() && (github.event_name != 'pull_request' || github.event.pull_request.draft == false)",
      replacement,
      "ci-summary if policy",
    )

    expect(source).toContain(`    ${replacement}\n`)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary has an unexpected if policy",
    )
  })

  it("rejects missing and duplicate Check required jobs owner steps", () => {
    const workflowSource = readWorkflowSource()
    const missing = replaceExactlyOnce(
      workflowSource,
      "      - name: Check required jobs\n",
      "      - name: Renamed required jobs\n",
      "ci-summary check step name",
    )
    const duplicate = replaceExactlyOnce(
      workflowSource,
      "      - name: Check required jobs\n",
      `      - name: Check required jobs\n        shell: bash\n        run: |\n          ${frontendSummaryCheckCommand}\n\n      - name: Check required jobs\n`,
      "ci-summary check step duplication",
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
    const source = replaceExactlyOnce(
      readWorkflowSource(),
      "      - name: Check required jobs\n        shell: bash\n        run: |\n",
      `${replacement}        run: |\n`,
      "ci-summary check step contract",
    )

    expect(source).not.toBe(readWorkflowSource())
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      _ === "a non-bash shell"
        ? "Check required jobs must use bash"
        : _ === "a step-level condition"
          ? "Check required jobs must not set if"
          : "Check required jobs must not set continue-on-error",
    )
  })

  it("rejects a missing, duplicate, or non-full-line frontend summary check", () => {
    const workflowSource = readWorkflowSource()
    const checkLine = 'check_job "frontend-build" "${{ needs[\'frontend-build\'].result }}"'
    const indentedCheckLine = `          ${checkLine}\n`
    const missing = replaceExactlyOnce(
      workflowSource,
      indentedCheckLine,
      "",
      "frontend summary check line removal",
    )
    const duplicate = replaceExactlyOnce(
      workflowSource,
      indentedCheckLine,
      indentedCheckLine.repeat(2),
      "frontend summary check line duplication",
    )
    const embedded = replaceExactlyOnce(
      workflowSource,
      indentedCheckLine,
      `          echo '${checkLine}'\n`,
      "frontend summary check line embedding",
    )

    expect(missing).not.toContain(indentedCheckLine)
    expect(duplicate.match(/check_job "frontend-build"/g)).toHaveLength(2)
    expect(embedded).toContain(`          echo '${checkLine}'\n`)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
    expect(() => assertSemanticWorkflowManifest(embedded)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
  })

  it("does not accept a frontend summary check decoy outside the owned step", () => {
    const workflowSource = readWorkflowSource()
    const checkLine = 'check_job "frontend-build" "${{ needs[\'frontend-build\'].result }}"'
    const withoutOwnedCheck = replaceExactlyOnce(
      workflowSource,
      `          ${checkLine}\n`,
      "",
      "owned frontend summary check line",
    )
    const source = replaceExactlyOnce(
      withoutOwnedCheck,
      '          exit "$failed"\n',
      `          exit "$failed"\n\n      - name: Preserve frontend summary text outside owner\n        shell: bash\n        run: |\n          ${checkLine}\n`,
      "ci-summary decoy insertion point",
    )

    expect(source).toContain("      - name: Preserve frontend summary text outside owner\n")
    expect(source.match(/check_job "frontend-build"/g)).toHaveLength(1)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
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
    expect(testConfig?.passWithNoTests).toBeFalsy()
  })
})
