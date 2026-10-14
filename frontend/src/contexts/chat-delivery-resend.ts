import { MAX_RECONNECT_CHAIN_MS, MessageDeliveryError } from "@/hooks/use-websocket"

// The reconnect chain plus slack for the open itself.
export const RESEND_RECONNECT_DEADLINE_MS = MAX_RECONNECT_CHAIN_MS + 4_000

// Where a send was made from, captured with the first attempt.
export interface ResendOwnership {
  loginIdentity: string | null
  boundTaskId: number | null
  // Present only for the Session transport.
  session: { deliveryGeneration: number; phase: string } | null
}

// Every check must hold; an unreadable login never matches.
export const mayResend = (first: ResendOwnership, now: ResendOwnership): boolean => {
  if (first.loginIdentity === null || now.loginIdentity !== first.loginIdentity) return false
  if (first.session === null) return now.boundTaskId === first.boundTaskId
  if (now.session === null || now.session.deliveryGeneration !== first.session.deliveryGeneration) return false
  if (now.session.phase !== "unbound" && now.session.phase !== "bound") return false
  return now.boundTaskId === first.boundTaskId
    || (first.boundTaskId === null && now.boundTaskId !== null)
}

// Sends that lost their connection wait here, first in first out, for the next
// open. Each wait ends at that open, when the owner calls `release`, or at the
// deadline.
export class ResendWaiters {
  private waiting: Array<(reconnected: boolean) => void> = []

  wait(): Promise<boolean> {
    return new Promise((resolve) => {
      const settle = (reconnected: boolean) => {
        clearTimeout(timer)
        this.waiting = this.waiting.filter((entry) => entry !== settle)
        resolve(reconnected)
      }
      const timer = setTimeout(() => settle(false), RESEND_RECONNECT_DEADLINE_MS)
      this.waiting.push(settle)
    })
  }

  release(reconnected: boolean) {
    for (const settle of [...this.waiting]) settle(reconnected)
  }
}

// A failed send whose connection was lost is sent once more, with the same
// arguments, after the next open if `stillOwned` still holds. Any other failure,
// and a second failure, is thrown as it came.
export async function sendWithResend<T>(
  send: () => Promise<T>,
  waiters: ResendWaiters,
  stillOwned: () => boolean,
): Promise<T> {
  try {
    return await send()
  } catch (error) {
    if (!(error instanceof MessageDeliveryError) || !error.connectionLost) throw error
    if (!(await waiters.wait()) || !stillOwned()) throw error
    return await send()
  }
}
