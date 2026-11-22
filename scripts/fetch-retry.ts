// Astro fetches every remote Markdown image at build time (dimension probing +
// optimization). On machines where DNS goes through Clash TUN's fake-IP range
// (198.18.0.0/15), dozens of concurrent TLS handshakes to the image hosts
// intermittently fail with ECONNRESET ("Client network socket disconnected
// before secure TLS connection was established").
//
// This module wraps globalThis.fetch for image URLs: at most 6 in flight,
// with a few retries on network errors and 429/5xx. Imported from
// astro.config.ts so `astro check`, `astro build` and `astro dev` all get it.

type FetchInput = Parameters<typeof fetch>[0]
type FetchInit = Parameters<typeof fetch>[1]

const realFetch = globalThis.fetch
const IMAGE_URL_RE = /\.(?:jpe?g|png|webp|gif|svg|avif)(?:\?|$)/i
const MAX_CONCURRENT = 6
const MAX_ATTEMPTS = 5

let active = 0
const waiting: Array<() => void> = []

function acquire(): Promise<void> {
  if (active < MAX_CONCURRENT) {
    active++
    return Promise.resolve()
  }
  return new Promise((resolve) => {
    waiting.push(() => {
      active++
      resolve()
    })
  })
}

function release(): void {
  active--
  const next = waiting.shift()
  next?.()
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

globalThis.fetch = async function fetchWithRetry(
  input: FetchInput,
  init?: FetchInit,
): Promise<Response> {
  if (typeof Request !== 'undefined' && input instanceof Request) {
    return realFetch(input, init)
  }

  const url = typeof input === 'string'
    ? input
    : input instanceof URL
      ? input.href
      : input.url

  if (!url || !IMAGE_URL_RE.test(url)) {
    return realFetch(input, init)
  }

  await acquire()
  try {
    for (let attempt = 0; ; attempt++) {
      try {
        const response = await realFetch(input, init)
        if (
          (response.status === 429 || response.status >= 500)
          && attempt < MAX_ATTEMPTS - 1
        ) {
          await sleep(300 * 2 ** attempt)
          continue
        }
        return response
      }
      catch (error) {
        if (attempt >= MAX_ATTEMPTS - 1) {
          throw error
        }
        await sleep(300 * 2 ** attempt)
      }
    }
  }
  finally {
    release()
  }
}
