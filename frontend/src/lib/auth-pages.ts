export const AUTH_PUBLIC_PATHS = [
  "/login",
  "/register",
  "/setup",
  "/forgot-password",
  "/reset-password",
  "/auth/oidc/callback",
] as const

export function isExternalRoutePath(pathname: string | null): boolean {
  return pathname === "/widget"
    || pathname?.startsWith("/widget/") === true
    || pathname === "/share"
    || pathname?.startsWith("/share/") === true
}

export function isAuthPublicPath(pathname: string | null): boolean {
  if (!pathname) {
    return false
  }
  if (isExternalRoutePath(pathname)) {
    return true
  }
  return AUTH_PUBLIC_PATHS.includes(pathname as (typeof AUTH_PUBLIC_PATHS)[number])
}
