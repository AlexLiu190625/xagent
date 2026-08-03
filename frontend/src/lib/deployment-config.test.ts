import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const getApiUrlMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: getApiUrlMock,
}))

import {
  __resetDeploymentConfigCache,
  browserDeploymentConfig,
  buildDeploymentShareUrl,
  fetchDeploymentConfig,
  resolveDeploymentOrigin,
} from "./deployment-config"

describe("deployment config", () => {
  beforeEach(() => {
    __resetDeploymentConfigCache()
    apiRequestMock.mockReset()
    getApiUrlMock.mockReset()
    getApiUrlMock.mockReturnValue("")
  })

  it("lets tests isolate successful memoized requests", async () => {
    apiRequestMock.mockImplementation(() =>
      Promise.resolve(new Response(
        JSON.stringify({
          deployment_origin: "https://sg-origin.cloud.example.test",
          app_origin: "https://cloud.example.test",
          region: "sg",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )),
    )

    await fetchDeploymentConfig()
    __resetDeploymentConfigCache()
    await fetchDeploymentConfig()

    expect(apiRequestMock).toHaveBeenCalledTimes(2)
  })

  it("retries a failed request and caches the successful deployment targets", async () => {
    apiRequestMock
      .mockRejectedValueOnce(new Error("network unavailable"))
      .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          deployment_origin: "https://sg-origin.cloud.example.test",
          app_origin: "https://cloud.example.test",
          region: "sg",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    await expect(fetchDeploymentConfig()).rejects.toThrow("network unavailable")

    const [first, second] = await Promise.all([
      fetchDeploymentConfig(),
      fetchDeploymentConfig(),
    ])
    expect(first).toEqual({
      deployment_origin: "https://sg-origin.cloud.example.test",
      app_origin: "https://cloud.example.test",
      region: "sg",
    })
    expect(second).toEqual(first)
    expect(apiRequestMock).toHaveBeenCalledTimes(2)
    expect(apiRequestMock).toHaveBeenLastCalledWith("/api/deployment-config")
  })

  it.each([
    [
      "missing fields",
      {
        deployment_origin: "https://sg-origin.cloud.example.test",
        app_origin: "https://cloud.example.test",
      },
    ],
    [
      "non-string fields",
      {
        deployment_origin: { origin: "https://sg-origin.cloud.example.test" },
        app_origin: "https://cloud.example.test",
        region: "sg",
      },
    ],
    [
      "invalid origins",
      {
        deployment_origin: "not-an-origin",
        app_origin: "https://cloud.example.test",
        region: "sg",
      },
    ],
    [
      "origins with query parameters",
      {
        deployment_origin: "https://sg-origin.cloud.example.test?region=sg",
        app_origin: "https://cloud.example.test",
        region: "sg",
      },
    ],
  ])("rejects a 200 response with %s", async (_case, payload) => {
    apiRequestMock.mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(fetchDeploymentConfig()).rejects.toThrow(
      "Invalid deployment configuration",
    )
  })

  it("accepts an application base URL with a path prefix", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          deployment_origin: "https://sg-origin.cloud.example.test",
          app_origin: "https://cloud.example.test/xagent",
          region: "sg",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    await expect(fetchDeploymentConfig()).resolves.toEqual({
      deployment_origin: "https://sg-origin.cloud.example.test",
      app_origin: "https://cloud.example.test/xagent",
      region: "sg",
    })
  })

  it("uses the explicit deployment origin after configuration loads", () => {
    expect(
      resolveDeploymentOrigin(
        {
          deployment_origin: "https://sg-origin.cloud.example.test/",
          app_origin: "https://cloud.example.test",
          region: "sg",
        },
        "https://cloud.example.test",
      ),
    ).toBe("https://sg-origin.cloud.example.test")
  })

  it("keeps client targets empty while deployment configuration is loading", () => {
    expect(
      resolveDeploymentOrigin(null, "https://cloud.example.test"),
    ).toBe("")
    expect(
      buildDeploymentShareUrl(
        "share-token",
        null,
        "https://cloud.example.test",
      ),
    ).toBe("")
  })

  it("represents browser fallback without duplicating the browser origin", () => {
    expect(browserDeploymentConfig()).toEqual({
      deployment_origin: null,
      app_origin: null,
      region: null,
    })
  })

  it("uses browser origins for standalone deployment configuration", () => {
    const standalone = {
      deployment_origin: null,
      app_origin: null,
      region: null,
    }

    expect(
      resolveDeploymentOrigin(standalone, "https://self-hosted.example.test"),
    ).toBe("https://self-hosted.example.test")
    expect(
      buildDeploymentShareUrl(
        "share-token",
        standalone,
        "https://self-hosted.example.test",
      ),
    ).toBe("https://self-hosted.example.test/share/share-token")
  })

  it("rejects a standalone share target without any valid origin", () => {
    expect(
      buildDeploymentShareUrl(
        "share-token",
        {
          deployment_origin: null,
          app_origin: "",
          region: null,
        },
        "",
      ),
    ).toBe("")
  })

  it("keeps standalone share links direct when no region is advertised", () => {
    expect(
      buildDeploymentShareUrl(
        "share-token",
        {
          deployment_origin: "https://api.example.test",
          app_origin: "https://app.example.test",
          region: null,
        },
        "https://app.example.test",
      ),
    ).toBe("https://app.example.test/share/share-token")
  })

  it("preserves the application path prefix in standalone share links", () => {
    expect(
      buildDeploymentShareUrl(
        "share-token",
        {
          deployment_origin: "https://api.example.test",
          app_origin: "https://app.example.test/xagent",
          region: null,
        },
        "https://app.example.test",
      ),
    ).toBe("https://app.example.test/xagent/share/share-token")
  })

  it("routes regional share links through the canonical region bootstrap", () => {
    expect(
      buildDeploymentShareUrl(
        "share-token",
        {
          deployment_origin: "https://sg-origin.cloud.example.test",
          app_origin: "https://cloud.example.test",
          region: "sg",
        },
        "https://cloud.example.test",
      ),
    ).toBe(
      "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fshare-token",
    )
  })

  it("preserves the application path prefix in regional share links", () => {
    expect(
      buildDeploymentShareUrl(
        "share-token",
        {
          deployment_origin: "https://sg-origin.cloud.example.test",
          app_origin: "https://cloud.example.test/xagent",
          region: "sg",
        },
        "https://cloud.example.test",
      ),
    ).toBe(
      "https://cloud.example.test/xagent/change-region?region=sg&next=%2Fxagent%2Fshare%2Fshare-token",
    )
  })
})
