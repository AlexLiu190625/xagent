import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { apiRequest } from "@/lib/api-wrapper"
import {
  abortableSleep,
  isCanonicalUuid,
  uploadWidgetSessionFile,
  WidgetSessionUploadError,
} from "@/lib/widget-session-upload"

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))

const ORIGIN = "https://chat.example.test"
const FILE_ID = "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c"

const fetchMock = vi.fn()

const jsonResponse = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  })

const created = (overrides: Record<string, unknown> = {}) =>
  jsonResponse(201, {
    file_id: FILE_ID,
    filename: "report.pdf",
    file_size: 11,
    mime_type: "application/pdf",
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  })

const rateLimited = (headers: Record<string, string> = {}) =>
  jsonResponse(429, { error: { code: "rate_limited", message: "slow down" } }, headers)

const makeFile = () => new File(["hello world"], "report.pdf", { type: "application/pdf" })

function baseOptions(overrides: Partial<Parameters<typeof uploadWidgetSessionFile>[0]> = {}) {
  const sleeps: number[] = []
  return {
    sleeps,
    options: {
      origin: ORIGIN,
      getToken: vi.fn(async () => "st_current"),
      refreshToken: vi.fn(async () => "st_refreshed"),
      file: makeFile(),
      sleep: vi.fn(async (ms: number) => { sleeps.push(ms) }),
      random: () => 0,
      ...overrides,
    },
  }
}

const expectUploadError = async (promise: Promise<unknown>, code: string) => {
  const error = await promise.then(() => null, (reason: unknown) => reason)
  expect(error).toBeInstanceOf(WidgetSessionUploadError)
  expect((error as WidgetSessionUploadError).errorCode).toBe(code)
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal("fetch", fetchMock)
  vi.mocked(apiRequest).mockReset()
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe("uploadWidgetSessionFile", () => {
  it("retries once after a rate_limited response and then succeeds", async () => {
    fetchMock.mockResolvedValueOnce(rateLimited()).mockResolvedValueOnce(created())
    const { options } = baseOptions()

    const result = await uploadWidgetSessionFile(options)

    expect(result).toEqual({
      kind: "uploaded",
      fileId: FILE_ID,
      name: "report.pdf",
      size: 11,
      type: "application/pdf",
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("gives up with upload_rate_limited after exactly three retries", async () => {
    fetchMock.mockImplementation(async () => rateLimited())
    const { options, sleeps } = baseOptions()

    await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(sleeps).toHaveLength(3)
  })

  it("does not retry quota_exceeded", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(429, { error: { code: "quota_exceeded", message: "full" } }),
    )
    const { options } = baseOptions()

    await expectUploadError(uploadWidgetSessionFile(options), "upload_quota_exceeded")

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(options.sleep).not.toHaveBeenCalled()
  })

  describe("backoff", () => {
    it("waits at least Retry-After before the first retry", async () => {
      fetchMock
        .mockResolvedValueOnce(rateLimited({ "Retry-After": "5" }))
        .mockResolvedValueOnce(created())
      const { options, sleeps } = baseOptions()

      await uploadWidgetSessionFile(options)

      expect(sleeps).toEqual([5000])
    })

    it("grows exponentially from a short Retry-After and adds jitter", async () => {
      fetchMock.mockImplementation(async () => rateLimited({ "Retry-After": "2" }))
      const { options, sleeps } = baseOptions({ random: () => 0.5 })

      await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

      expect(sleeps).toEqual([2250, 4250, 8250])
    })

    it("caps a large Retry-After at sixty seconds plus jitter", async () => {
      fetchMock
        .mockResolvedValueOnce(rateLimited({ "Retry-After": "600" }))
        .mockResolvedValueOnce(created())
      const { options, sleeps } = baseOptions({ random: () => 0.999 })

      await uploadWidgetSessionFile(options)

      expect(sleeps).toEqual([60_499])
    })

    it.each([
      ["missing", {}],
      ["not a number", { "Retry-After": "soon" }],
      ["zero", { "Retry-After": "0" }],
    ])("falls back to 2^attempt seconds when Retry-After is %s", async (_label, headers) => {
      fetchMock.mockImplementation(async () => rateLimited(headers))
      const { options, sleeps } = baseOptions()

      await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

      expect(sleeps).toEqual([2000, 4000, 8000])
    })

    it("asks for a fresh token before every request, including retries", async () => {
      fetchMock.mockResolvedValueOnce(rateLimited()).mockResolvedValueOnce(created())
      const tokens = ["st_one", "st_two"]
      const { options } = baseOptions({ getToken: vi.fn(async () => tokens.shift() as string) })

      await uploadWidgetSessionFile(options)

      expect(fetchMock.mock.calls.map((call) => call[1].headers.Authorization)).toEqual([
        "Bearer st_one",
        "Bearer st_two",
      ])
    })
  })

  describe("credentials", () => {
    it("sends only the session token, omits cookies and posts a single file field", async () => {
      localStorage.setItem("token", "main-site-token")
      fetchMock.mockResolvedValueOnce(created())
      const { options } = baseOptions()

      await uploadWidgetSessionFile(options)

      const [url, init] = fetchMock.mock.calls[0]
      expect(url).toBe(`${ORIGIN}/v1/external/chat/sessions/files`)
      expect(init.method).toBe("POST")
      expect(init.credentials).toBe("omit")
      expect(init.headers).toEqual({ Authorization: "Bearer st_current" })
      expect(JSON.stringify(init.headers)).not.toContain("main-site-token")
      const entries = Array.from((init.body as FormData).entries())
      expect(entries).toHaveLength(1)
      expect(entries[0][0]).toBe("file")
      expect((entries[0][1] as File).name).toBe("report.pdf")
      expect(apiRequest).not.toHaveBeenCalled()
    })
  })

  describe("failure classification", () => {
    it("maps 413 to upload_too_large", async () => {
      fetchMock.mockResolvedValueOnce(jsonResponse(413, { error: { code: "invalid_input", message: "big" } }))
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_too_large")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("maps 422 to upload_rejected", async () => {
      fetchMock.mockResolvedValueOnce(jsonResponse(422, { error: { code: "invalid_input", message: "type" } }))
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_rejected")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it.each([
      ["400", () => jsonResponse(400, { error: { code: "invalid_input", message: "bad" } })],
      ["404 without an error envelope", () => jsonResponse(404, { detail: "Not Found" })],
      ["500", () => jsonResponse(500, { error: { code: "internal_error", message: "x" } })],
      ["429 with an unknown code", () => jsonResponse(429, { error: { code: "other", message: "x" } })],
      ["429 without a JSON body", () => new Response("<html></html>", { status: 429 })],
      ["a 502 error page", () => new Response("<html></html>", { status: 502 })],
      ["a 201 with a non-JSON body", () => new Response("created", { status: 201 })],
      ["a 201 with a non-object body", () => jsonResponse(201, "ok")],
      ["a 201 without file_id", () => created({ file_id: undefined })],
      ["a 201 with an uppercase UUID", () => created({ file_id: FILE_ID.toUpperCase() })],
      ["a 201 with a non-UUID file_id", () => created({ file_id: "file-1" })],
    ])("maps %s to upload_failed after a single request", async (_label, makeResponse) => {
      fetchMock.mockResolvedValueOnce(makeResponse())
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("maps a network error to upload_failed after a single request", async () => {
      fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("maps an unclassified token failure to upload_failed and keeps a coded one", async () => {
      const { options } = baseOptions({ getToken: vi.fn(async () => { throw new Error("boom") }) })
      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")

      const coded = baseOptions({
        getToken: vi.fn(async () => { throw new WidgetSessionUploadError("connection_not_ready") }),
      })
      await expectUploadError(uploadWidgetSessionFile(coded.options), "connection_not_ready")
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it("falls back to the File's own metadata when the response omits it", async () => {
      fetchMock.mockResolvedValueOnce(created({ filename: 3, file_size: "x", mime_type: null }))
      const { options } = baseOptions()
      await expect(uploadWidgetSessionFile(options)).resolves.toMatchObject({
        name: "report.pdf",
        size: 11,
        type: "application/pdf",
      })
    })
  })

  describe("401 handling", () => {
    it("refreshes the token once and retries the same file", async () => {
      fetchMock
        .mockResolvedValueOnce(jsonResponse(401, { error: { code: "invalid_api_key", message: "no" } }))
        .mockResolvedValueOnce(created())
      const { options } = baseOptions()

      await expect(uploadWidgetSessionFile(options)).resolves.toMatchObject({ fileId: FILE_ID })

      expect(options.refreshToken).toHaveBeenCalledTimes(1)
      expect(fetchMock.mock.calls.map((call) => call[1].headers.Authorization)).toEqual([
        "Bearer st_current",
        "Bearer st_refreshed",
      ])
    })

    it("fails with upload_failed after a second 401", async () => {
      fetchMock.mockImplementation(async () => jsonResponse(401, { error: { code: "invalid_api_key", message: "no" } }))
      const { options } = baseOptions()

      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")

      expect(fetchMock).toHaveBeenCalledTimes(2)
      expect(options.refreshToken).toHaveBeenCalledTimes(1)
    })
  })

  describe("abort", () => {
    it("stops waiting and never sends another request when aborted during a backoff", async () => {
      vi.useFakeTimers()
      fetchMock.mockImplementation(async () => rateLimited())
      const controller = new AbortController()
      const { options } = baseOptions({
        signal: controller.signal,
        sleep: abortableSleep,
      })

      const outcome = uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)
      await vi.advanceTimersByTimeAsync(100)
      expect(fetchMock).toHaveBeenCalledTimes(1)
      controller.abort()
      const error = await outcome

      expect((error as Error).name).toBe("AbortError")
      await vi.advanceTimersByTimeAsync(120_000)
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("rethrows the AbortError raised by an aborted request", async () => {
      const controller = new AbortController()
      fetchMock.mockImplementationOnce(async () => {
        controller.abort()
        throw new DOMException("aborted", "AbortError")
      })
      const { options } = baseOptions({ signal: controller.signal })

      const error = await uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)

      expect((error as Error).name).toBe("AbortError")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("does not send anything when already aborted", async () => {
      const controller = new AbortController()
      controller.abort()
      const { options } = baseOptions({ signal: controller.signal })

      const error = await uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)

      expect((error as Error).name).toBe("AbortError")
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it("treats a non-AbortError thrown after an abort as an abort", async () => {
      const controller = new AbortController()
      fetchMock.mockImplementationOnce(async () => {
        controller.abort()
        throw new TypeError("network down")
      })
      const { options } = baseOptions({ signal: controller.signal })

      const error = await uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)

      expect((error as Error).name).toBe("AbortError")
    })

    it("abortableSleep resolves after the delay and rejects when already aborted", async () => {
      vi.useFakeTimers()
      const sleeping = abortableSleep(1000)
      await vi.advanceTimersByTimeAsync(1000)
      await expect(sleeping).resolves.toBeUndefined()

      const controller = new AbortController()
      controller.abort()
      await expect(abortableSleep(1000, controller.signal)).rejects.toMatchObject({ name: "AbortError" })
    })
  })

  it("returns a plain object and never writes onto the File", async () => {
    fetchMock.mockResolvedValueOnce(created())
    const { options } = baseOptions()
    const before = Object.keys(options.file)

    const result = await uploadWidgetSessionFile(options)

    expect(Object.keys(options.file)).toEqual(before)
    expect(options.file).not.toHaveProperty("file_id")
    expect(Object.getPrototypeOf(result)).toBe(Object.prototype)
  })
})

describe("isCanonicalUuid", () => {
  it("accepts only lowercase canonical UUIDs", () => {
    expect(isCanonicalUuid(FILE_ID)).toBe(true)
    expect(isCanonicalUuid(FILE_ID.toUpperCase())).toBe(false)
    expect(isCanonicalUuid(FILE_ID.replace(/-/g, ""))).toBe(false)
    expect(isCanonicalUuid(` ${FILE_ID}`)).toBe(false)
    expect(isCanonicalUuid(42)).toBe(false)
  })
})
