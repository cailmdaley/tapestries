// Shared UI utilities
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import 'katex/dist/katex.min.css'
import type { PDFDocumentLoadingTask, PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist'
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

type PdfJsModule = typeof import('pdfjs-dist')
let pdfJsPromise: Promise<PdfJsModule> | null = null
async function loadPdfJs(): Promise<PdfJsModule> {
  if (!pdfJsPromise) {
    pdfJsPromise = (async () => {
      const pdfjs = await import('pdfjs-dist')
      pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc
      return pdfjs
    })()
  }
  return pdfJsPromise
}

const PDF_CACHE_LIMIT = 24
const IMAGE_WARM_CACHE_LIMIT = 96

const pdfDocCache = new Map<string, PDFDocumentProxy>()
const pdfLoadingCache = new Map<string, PDFDocumentLoadingTask>()
const pdfPageCache = new Map<string, PDFPageProxy>()
const pdfAspectRatioCache = new Map<string, number>()
const pdfCacheLru = new Map<string, true>()

const imageWarmCache = new Map<string, Promise<void>>()
const imageWarmLru = new Map<string, true>()

function touchLru(lru: Map<string, true>, key: string): void {
  if (lru.has(key)) lru.delete(key)
  lru.set(key, true)
}

function enforceLruLimit(
  lru: Map<string, true>,
  maxSize: number,
  onEvict: (key: string) => void,
): void {
  while (lru.size > maxSize) {
    const oldestKey = lru.keys().next().value as string | undefined
    if (!oldestKey) return
    lru.delete(oldestKey)
    onEvict(oldestKey)
  }
}

function evictPdfUrl(url: string): void {
  const loadingTask = pdfLoadingCache.get(url)
  if (loadingTask) {
    pdfLoadingCache.delete(url)
    try {
      loadingTask.destroy()
    } catch {}
  }

  const page = pdfPageCache.get(url)
  if (page) {
    pdfPageCache.delete(url)
    try {
      page.cleanup()
    } catch {}
  }

  const doc = pdfDocCache.get(url)
  if (doc) {
    pdfDocCache.delete(url)
    try {
      doc.cleanup()
    } catch {}
    void doc.destroy().catch(() => {})
  }

  pdfAspectRatioCache.delete(url)
  pdfCacheLru.delete(url)
}

function touchPdfCache(url: string): void {
  touchLru(pdfCacheLru, url)
  enforceLruLimit(pdfCacheLru, PDF_CACHE_LIMIT, evictPdfUrl)
}

function getCachedPdfAspectRatio(url: string): number | undefined {
  const ratio = pdfAspectRatioCache.get(url)
  if (ratio && ratio > 0) {
    touchPdfCache(url)
  }
  return ratio
}

function setCachedPdfAspectRatio(url: string, ratio: number): void {
  if (!Number.isFinite(ratio) || ratio <= 0) return
  pdfAspectRatioCache.set(url, ratio)
  touchPdfCache(url)
}

function touchImageWarmCache(url: string): void {
  touchLru(imageWarmLru, url)
  enforceLruLimit(imageWarmLru, IMAGE_WARM_CACHE_LIMIT, (oldestKey) => {
    imageWarmCache.delete(oldestKey)
  })
}

/**
 * Render all pages of a PDF into a scrollable container.
 * Returns the container immediately; pages render asynchronously.
 */
export async function renderPdfAllPages(
  url: string,
  container: HTMLElement,
): Promise<void> {
  const pdfjs = await loadPdfJs()
  let doc = pdfDocCache.get(url)
  if (!doc) {
    let loadingTask = pdfLoadingCache.get(url)
    if (!loadingTask) {
      loadingTask = pdfjs.getDocument(url)
      pdfLoadingCache.set(url, loadingTask)
    }
    try {
      doc = await loadingTask.promise
    } finally {
      if (pdfLoadingCache.get(url) === loadingTask) {
        pdfLoadingCache.delete(url)
      }
    }
    pdfDocCache.set(url, doc)
  }
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const containerWidth = container.clientWidth || 800
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i)
    const unscaledViewport = page.getViewport({ scale: 1 })
    const scale = (containerWidth * dpr) / unscaledViewport.width
    const viewport = page.getViewport({ scale })
    const canvas = document.createElement('canvas')
    canvas.width = viewport.width
    canvas.height = viewport.height
    canvas.style.width = '100%'
    canvas.style.display = 'block'
    container.appendChild(canvas)
    const ctx = canvas.getContext('2d')!
    await page.render({ canvas, canvasContext: ctx, viewport }).promise
  }
}

export function clearArtifactMediaCaches(): void {
  for (const url of Array.from(pdfDocCache.keys())) {
    evictPdfUrl(url)
  }
  for (const url of Array.from(pdfLoadingCache.keys())) {
    evictPdfUrl(url)
  }
  for (const url of Array.from(pdfPageCache.keys())) {
    evictPdfUrl(url)
  }

  pdfDocCache.clear()
  pdfLoadingCache.clear()
  pdfPageCache.clear()
  pdfAspectRatioCache.clear()
  pdfCacheLru.clear()

  imageWarmCache.clear()
  imageWarmLru.clear()
}

export function getArtifactMediaCacheStats(): {
  pdfDocuments: number
  pdfLoadingTasks: number
  pdfPages: number
  pdfAspectRatios: number
  warmedImages: number
} {
  return {
    pdfDocuments: pdfDocCache.size,
    pdfLoadingTasks: pdfLoadingCache.size,
    pdfPages: pdfPageCache.size,
    pdfAspectRatios: pdfAspectRatioCache.size,
    warmedImages: imageWarmCache.size,
  }
}

// Configure marked for safe rendering with KaTeX math support
marked.setOptions({
  gfm: true,        // GitHub Flavored Markdown
  breaks: true,     // Convert \n to <br>
})

// $..$ for inline math, $$...$$ for display math
marked.use(markedKatex({
  throwOnError: false,
  output: 'html',
  nonStandard: true, // allow $...$ after punctuation like hyphen (pseudo-$C_\ell$)
}))

// Custom renderer for code blocks to integrate with Prism
const renderer = new marked.Renderer()
renderer.code = ({ text, lang }: { text: string; lang?: string }) => {
  const language = lang || 'plaintext'
  // Prism will highlight after DOM insertion
  const escapedCode = escapeHtml(text)
  return `<pre class="md-code-block language-${language}"><code class="language-${language}">${escapedCode}</code></pre>`
}

renderer.codespan = ({ text }: { text: string }) => {
  return `<code class="md-inline-code">${escapeHtml(text)}</code>`
}

renderer.link = ({ href, text }: { href: string; text: string }) => {
  return `<a href="${escapeHtml(href)}" class="md-link" target="_blank" rel="noopener">${text}</a>`
}

// Strip KaTeX HTML from image alt text (marked-katex-extension processes $ inside alt)
renderer.image = ({ href, text: alt }: { href: string; text?: string }) => {
  const cleanAlt = (alt || '').replace(/<[^>]*>/g, '')
  return `<img src="${escapeHtml(href)}" alt="${escapeHtml(cleanAlt)}" loading="lazy" />`
}

marked.use({ renderer })

// GFM del rule matches ~text~ (single tilde) as well as ~~text~~ (double).
// Tilde is common as an approximation sign (~2 days), so escape lone tildes
// before parsing — but only outside code spans and fenced code blocks.
marked.use({
  hooks: {
    preprocess(src: string): string {
      // Split on code regions (fenced blocks or backtick spans) and only
      // process the non-code segments.
      const CODE_REGION = /(```[\s\S]*?```|`[^`]*`)/g
      const parts = src.split(CODE_REGION)
      return parts.map((part, i) => {
        // Odd indices are the captured code regions — leave untouched
        if (i % 2 === 1) return part
        // Even indices are plain text — escape lone tildes
        return part.replace(/~+/g, (m) => {
          const pairs = Math.floor(m.length / 2)
          const rem = m.length % 2
          return '~~'.repeat(pairs) + (rem ? '&#126;' : '')
        })
      }).join('')
    },
  },
})

/**
 * Escape HTML to prevent XSS
 */
export function escapeHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

const API_BASE = `http://${window.location.hostname}:4004`

interface RenderMarkdownOptions {
  /** Base directory for resolving relative image paths (e.g. city path) */
  basePath?: string
  /** Origin ID for remote file access */
  originId?: string
}

/**
 * Render markdown to HTML with syntax highlighting.
 * Relative image paths are proxied through /file-content?raw=true when basePath is provided.
 */
export function renderMarkdown(text: string, opts?: RenderMarkdownOptions): string {
  try {
    // Use a per-call renderer to handle image path resolution
    if (opts?.basePath) {
      const localRenderer = new marked.Renderer()
      // Inherit code/codespan/link from the global renderer
      localRenderer.code = renderer.code
      localRenderer.codespan = renderer.codespan
      localRenderer.link = renderer.link
      localRenderer.image = ({ href, text: alt }: { href: string; text?: string }) => {
        let src = href
        // Resolve relative paths through the file-content API
        if (!/^https?:\/\//.test(href) && !/^data:/.test(href)) {
          const fullPath = href.startsWith('/') ? href : `${opts.basePath}/${href}`
          src = `${API_BASE}/file-content?path=${encodeURIComponent(fullPath)}&raw=true`
          if (opts.originId && opts.originId !== 'local') {
            src += `&originId=${encodeURIComponent(opts.originId)}`
          }
        }
        const cleanAlt = (alt || '').replace(/<[^>]*>/g, '')
        return `<img src="${escapeHtml(src)}" alt="${escapeHtml(cleanAlt)}" loading="lazy" />`
      }
      return marked.parse(text, { renderer: localRenderer }) as string
    }
    return marked.parse(text) as string
  } catch (e) {
    console.error('Markdown render error:', e)
    return escapeHtml(text)
  }
}

/**
 * Resolve config[key] / config.x.y / config.yaml: x.y references in rendered markdown.
 * Appends " = value" annotations to matching inline code elements.
 */
export function interpolateConfig(container: HTMLElement, config: Record<string, string>): void {
  container.querySelectorAll<HTMLElement>('p code, td code, li code, h1 code, h2 code, h3 code').forEach(code => {
    const text = code.textContent?.trim() || ''
    const key = text
      .replace(/^config\.yaml:\s*/, '')
      .replace(/^config\[["']?/, '').replace(/["']?\]$/, '')
      .replace(/["']?\]\[["']?/g, '.')
      .replace(/^config\./, '')
    const value = config[key]
    if (value !== undefined) {
      const display = value.length > 60 ? value.slice(0, 57) + '...' : value
      code.textContent = display
      code.classList.add('config-resolved')
      code.title = key  // hover shows original key
    }
  })
}

// Matches inline code that looks like a file path, e.g. server/src/index.ts or ./foo/bar.py:42
// Also accepts :L42 or :L42-55 (GitHub-style line range) — both colon-digit and colon-L-digit forms work.
const INLINE_PATH_RE = /^(?:\.{0,2}\/)?[\w.\-/]+\/[\w.\-]+\.[a-zA-Z]{1,10}(?::L?\d+(?:-\d+)?)?$|^\.\/[\w.\-/]+\.[a-zA-Z]{1,10}(?::L?\d+(?:-\d+)?)?$/

/**
 * Find inline <code> elements whose text looks like a file path and make them clickable.
 * `openFile(path, line)` is called with the resolved relative path and optional line number.
 */
export function attachInlinePathListeners(
  container: HTMLElement,
  openFile: (path: string, line?: number) => void,
): void {
  container.querySelectorAll<HTMLElement>('code.md-inline-code').forEach(code => {
    const text = code.textContent?.trim() || ''
    const m = text.match(/^(.*?)(?::L?(\d+)(?:-\d+)?)?$/)
    if (!m || !INLINE_PATH_RE.test(text)) return
    const path = m[1]
    const line = m[2] ? parseInt(m[2], 10) : undefined
    code.style.cursor = 'pointer'
    code.title = line ? `Open ${path} at line ${line}` : `Open ${path}`
    code.addEventListener('click', (e) => {
      e.preventDefault()
      e.stopPropagation()
      openFile(path, line)
    })
  })
}

/**
 * Apply Prism syntax highlighting to code blocks in a container
 * Call after inserting markdown HTML into the DOM
 */
export function highlightCodeBlocks(container: HTMLElement): void {
  if (typeof window !== 'undefined' && (window as unknown as { Prism?: { highlightAllUnder: (el: HTMLElement) => void } }).Prism) {
    (window as unknown as { Prism: { highlightAllUnder: (el: HTMLElement) => void } }).Prism.highlightAllUnder(container)
  }
}

/**
 * Format a timestamp as relative time (e.g., "5m ago", "2h ago")
 * Returns empty string for invalid timestamps
 */
export function formatTimeAgo(timestamp: number): string {
  // Handle NaN/invalid timestamps (e.g., from malformed date strings)
  if (!Number.isFinite(timestamp)) return ''

  const diffMs = Date.now() - timestamp
  const diffMins = Math.floor(diffMs / 60000)

  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`

  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours}h ago`

  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return `${diffDays}d ago`

  return new Date(timestamp).toLocaleDateString()
}

/** Staleness → color map, shared between TapestryView and FileViewerModal. */
export const STALENESS_COLORS: Record<string, string> = {
  'fresh': '#5A7B7B',
  'stale': '#A87070',
  'no-evidence': '#7A7368',
}

/**
 * Format an ISO date string as "20 Feb 2026".
 */
export function formatFiberDate(iso: string | null | undefined): string {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
  } catch { return iso }
}

/**
 * Render an artifact gallery with ← → navigation.
 * Returns { html, attach(container), detach() }.
 * - `html`: the `.tapestry-artifact-viewer` HTML string to inject
 * - `attach(container)`: binds click + keyboard nav inside container
 * - `detach()`: removes the keyboard listener
 */
export function renderArtifactGallery(
  artifacts: Record<string, string>,
  buildUrl: (path: string) => string,
): { html: string; attach: (container: HTMLElement) => void; detach: () => void } {
  const entries = Object.entries(artifacts)
  if (entries.length === 0) return { html: '', attach: () => {}, detach: () => {} }

  const isPdfArtifact = (path: string) => /\.pdf(?:$|[?#])/i.test(path)
  let pdfRenderNonce = 0
  let pdfResizeObserver: ResizeObserver | null = null
  let observedWidth = 0
  let resizeRaf = 0

  const clearPdfResizeObserver = () => {
    if (resizeRaf) {
      cancelAnimationFrame(resizeRaf)
      resizeRaf = 0
    }
    if (!pdfResizeObserver) return
    pdfResizeObserver.disconnect()
    pdfResizeObserver = null
  }

  const getPdfPage = async (url: string): Promise<PDFPageProxy | null> => {
    try {
      const cachedPage = pdfPageCache.get(url)
      if (cachedPage) {
        touchPdfCache(url)
        return cachedPage
      }

      let pdfDoc = pdfDocCache.get(url)
      if (pdfDoc) touchPdfCache(url)
      if (!pdfDoc) {
        let loadingTask = pdfLoadingCache.get(url)
        if (!loadingTask) {
          const pdfjs = await loadPdfJs()
          loadingTask = pdfjs.getDocument(url)
          pdfLoadingCache.set(url, loadingTask)
        }
        touchPdfCache(url)
        try {
          pdfDoc = await loadingTask.promise
        } finally {
          if (pdfLoadingCache.get(url) === loadingTask) {
            pdfLoadingCache.delete(url)
          }
        }
        pdfDocCache.set(url, pdfDoc)
        touchPdfCache(url)
      }

      const page = await pdfDoc.getPage(1)
      pdfPageCache.set(url, page)
      touchPdfCache(url)
      return page
    } catch {
      evictPdfUrl(url)
      return null
    }
  }

  const warmPdf = async (path: string): Promise<number | null> => {
    const url = buildUrl(path)
    const cached = getCachedPdfAspectRatio(url)
    if (cached && cached > 0) return cached
    const page = await getPdfPage(url)
    if (!page) return null
    const viewport = page.getViewport({ scale: 1 })
    const ratio = viewport.width / viewport.height
    if (Number.isFinite(ratio) && ratio > 0) {
      setCachedPdfAspectRatio(url, ratio)
      return ratio
    }
    return null
  }

  const warmImage = async (path: string): Promise<void> => {
    const url = buildUrl(path)
    let warm = imageWarmCache.get(url)
    if (warm) touchImageWarmCache(url)
    if (!warm) {
      warm = new Promise<void>((resolve) => {
        const img = new Image()
        let settled = false
        const finish = () => {
          if (settled) return
          settled = true
          img.onload = null
          img.onerror = null
          resolve()
        }
        img.onload = finish
        img.onerror = finish
        img.src = url
        if ('decode' in img) {
          void img.decode().then(finish).catch(finish)
        }
      })
      imageWarmCache.set(url, warm)
      touchImageWarmCache(url)
    }
    await warm
  }

  const warmArtifact = async (path: string): Promise<void> => {
    if (isPdfArtifact(path)) {
      await warmPdf(path)
      return
    }
    await warmImage(path)
  }

  const setPdfAspectFromCache = (media: HTMLElement, path: string) => {
    const ratio = getCachedPdfAspectRatio(buildUrl(path))
    if (ratio && ratio > 0) media.style.setProperty('--pdf-aspect-ratio', `${ratio}`)
  }

  const renderPdfPreview = async (container: HTMLElement, path: string): Promise<void> => {
    if (!isPdfArtifact(path)) return
    const media = container.querySelector('.tapestry-artifact .artifact-media.pdf') as HTMLElement | null
    if (!media || media.dataset.artifactPath !== path) return
    const canvas = media.querySelector('canvas') as HTMLCanvasElement | null
    if (!canvas) return

    const renderId = ++pdfRenderNonce
    const url = buildUrl(path)
    setPdfAspectFromCache(media, path)
    const page = await getPdfPage(url)
    if (!page) {
      if (media.isConnected && media.dataset.artifactPath === path) {
        media.classList.remove('loading')
      }
      return
    }
    if (!media.isConnected || media.dataset.artifactPath !== path || renderId !== pdfRenderNonce) return

    const baseViewport = page.getViewport({ scale: 1 })
    const ratio = baseViewport.width / baseViewport.height
    if (Number.isFinite(ratio) && ratio > 0) {
      setCachedPdfAspectRatio(url, ratio)
      media.style.setProperty('--pdf-aspect-ratio', `${ratio}`)
    }

    let lastWidth = 0
    let rendering = false
    let pending = false

    const draw = async (force = false) => {
      if (!media.isConnected || media.dataset.artifactPath !== path || renderId !== pdfRenderNonce) return
      const width = Math.max(1, Math.floor(media.clientWidth))
      if (!force && width === lastWidth) return
      if (rendering) { pending = true; return }
      rendering = true

      const base = page.getViewport({ scale: 1 })
      const scale = width / base.width
      const viewport = page.getViewport({ scale })
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const context = canvas.getContext('2d')
      if (!context) {
        rendering = false
        return
      }

      try {
        canvas.width = Math.max(1, Math.floor(viewport.width * dpr))
        canvas.height = Math.max(1, Math.floor(viewport.height * dpr))
        canvas.style.height = `${viewport.height}px`

        context.setTransform(1, 0, 0, 1, 0, 0)
        context.clearRect(0, 0, canvas.width, canvas.height)
        context.imageSmoothingEnabled = true
        context.setTransform(dpr, 0, 0, dpr, 0, 0)

        await page.render({ canvas, canvasContext: context, viewport }).promise
        lastWidth = width
        if (media.isConnected && media.dataset.artifactPath === path && renderId === pdfRenderNonce) {
          media.classList.remove('loading')
        }
      } finally {
        rendering = false
        if (pending) {
          pending = false
          void draw()
        }
      }
    }

    await draw(true)
    clearPdfResizeObserver()
    observedWidth = Math.max(1, Math.floor(media.clientWidth))
    pdfResizeObserver = new ResizeObserver((entries) => {
      const nextWidth = Math.max(1, Math.floor(entries[0]?.contentRect.width ?? media.clientWidth))
      if (nextWidth === observedWidth) return
      observedWidth = nextWidth
      if (resizeRaf) cancelAnimationFrame(resizeRaf)
      resizeRaf = requestAnimationFrame(() => {
        resizeRaf = 0
        void draw()
      })
    })
    pdfResizeObserver.observe(media)
  }

  const mediaHtml = (name: string, path: string) => {
    const url = buildUrl(path)
    if (isPdfArtifact(path)) {
      const cachedRatio = getCachedPdfAspectRatio(url)
      const aspectStyle = cachedRatio && cachedRatio > 0 ? ` style="--pdf-aspect-ratio:${cachedRatio}"` : ''
      return `
        <div class="artifact-media pdf loading" data-artifact-path="${escapeHtml(path)}"${aspectStyle}>
          <canvas class="artifact-pdf-canvas" data-artifact-name="${escapeHtml(name)}" data-artifact-type="pdf" aria-label="${escapeHtml(name)} preview"></canvas>
          <button type="button" class="artifact-open-overlay" data-artifact-name="${escapeHtml(name)}" title="Open in lightbox" aria-label="Open ${escapeHtml(name)} in lightbox"></button>
        </div>`
    }
    return `
      <div class="artifact-media image loading" data-artifact-path="${escapeHtml(path)}">
        <img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" data-artifact-name="${escapeHtml(name)}" data-artifact-type="image" loading="lazy" />
      </div>`
  }

  let currentIndex = 0
  const hasMultiple = entries.length > 1

  const [name0, path0] = entries[0]
  const html = `
    <div class="tapestry-artifact-viewer">
      ${hasMultiple ? `<span class="artifact-nav" data-delta="-1">\u2190</span>` : ''}
      <div class="tapestry-artifact">
        <span class="artifact-label">${escapeHtml(name0)}${hasMultiple ? ` (1/${entries.length})` : ''}</span>
        ${mediaHtml(name0, path0)}
      </div>
      ${hasMultiple ? `<span class="artifact-nav" data-delta="1">\u2192</span>` : ''}
    </div>`

  const markImageLoaded = (container: HTMLElement, path: string) => {
    if (isPdfArtifact(path)) return
    const media = container.querySelector('.tapestry-artifact .artifact-media.image') as HTMLElement | null
    if (!media || media.dataset.artifactPath !== path) return
    const img = media.querySelector('img')
    if (!img) return
    const onLoad = () => {
      if (media.isConnected && media.dataset.artifactPath === path) media.classList.remove('loading')
    }
    if ((img as HTMLImageElement).complete) onLoad()
    else img.addEventListener('load', onLoad, { once: true })
    img.addEventListener('error', onLoad, { once: true })
  }

  const warmAdjacent = (index: number) => {
    if (entries.length < 2) return
    const prev = entries[(index - 1 + entries.length) % entries.length]?.[1]
    const next = entries[(index + 1) % entries.length]?.[1]
    if (prev) void warmArtifact(prev)
    if (next) void warmArtifact(next)
  }

  const updateImg = (container: HTMLElement) => {
    const [name, path] = entries[currentIndex]
    const media = container.querySelector('.tapestry-artifact .artifact-media') as HTMLElement | null
    clearPdfResizeObserver()
    if (media) media.outerHTML = mediaHtml(name, path)
    const label = container.querySelector('.artifact-label')
    if (label) label.textContent = `${name}${hasMultiple ? ` (${currentIndex + 1}/${entries.length})` : ''}`
    if (isPdfArtifact(path)) void renderPdfPreview(container, path)
    else markImageLoaded(container, path)
    warmAdjacent(currentIndex)
  }

  let keyHandler: ((e: KeyboardEvent) => void) | null = null

  const attach = (container: HTMLElement) => {
    void warmArtifact(entries[currentIndex][1])
    warmAdjacent(currentIndex)
    const [, initialPath] = entries[currentIndex]
    if (isPdfArtifact(initialPath)) void renderPdfPreview(container, initialPath)
    else markImageLoaded(container, initialPath)

    container.querySelectorAll('.artifact-nav').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation()
        const delta = parseInt((btn as HTMLElement).dataset.delta || '0')
        currentIndex = (currentIndex + delta + entries.length) % entries.length
        updateImg(container)
      })
    })

    if (hasMultiple) {
      keyHandler = (e: KeyboardEvent) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
        e.preventDefault()
        const delta = e.key === 'ArrowLeft' ? -1 : 1
        currentIndex = (currentIndex + delta + entries.length) % entries.length
        updateImg(container)
      }
      document.addEventListener('keydown', keyHandler)
    }
  }

  const detach = () => {
    if (resizeRaf) {
      cancelAnimationFrame(resizeRaf)
      resizeRaf = 0
    }
    clearPdfResizeObserver()
    if (keyHandler) {
      document.removeEventListener('keydown', keyHandler)
      keyHandler = null
    }
  }

  return { html, attach, detach }
}

/**
 * Map fiber status to a compact icon glyph.
 * active = half-filled, closed = filled, open/other = hollow.
 */
export function fiberStatusIcon(status: string): string {
  if (status === 'active') return '◐'
  if (status === 'closed') return '●'
  return '○'
}

/**
 * Show a toast notification
 */
export function showToast(message: string, type: 'success' | 'error' = 'success', duration = 3000): void {
  // Remove existing toasts
  const existing = document.querySelector('.portolan-toast')
  if (existing) existing.remove()

  const toast = document.createElement('div')
  toast.className = 'portolan-toast'
  toast.innerHTML = `
    <span class="toast-icon">${type === 'success' ? '✓' : '✕'}</span>
    <span class="toast-message">${escapeHtml(message)}</span>
  `

  // Inject styles if not present
  if (!document.getElementById('portolan-toast-styles')) {
    const style = document.createElement('style')
    style.id = 'portolan-toast-styles'
    style.textContent = `
      .portolan-toast {
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%) translateY(100px);
        background: #1a1a1a;
        color: #f5f5f0;
        padding: 12px 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'EB Garamond', Garamond, serif;
        font-size: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        z-index: 10000;
        opacity: 0;
        animation: toast-in 0.3s ease forwards;
      }
      .portolan-toast.toast-out {
        animation: toast-out 0.3s ease forwards;
      }
      .portolan-toast .toast-icon {
        font-size: 16px;
        font-weight: bold;
      }
      .portolan-toast.success .toast-icon { color: #c9a959; }
      .portolan-toast.error .toast-icon { color: #d9534f; }
      @keyframes toast-in {
        from { opacity: 0; transform: translateX(-50%) translateY(100px); }
        to { opacity: 1; transform: translateX(-50%) translateY(0); }
      }
      @keyframes toast-out {
        from { opacity: 1; transform: translateX(-50%) translateY(0); }
        to { opacity: 0; transform: translateX(-50%) translateY(100px); }
      }
    `
    document.head.appendChild(style)
  }

  toast.classList.add(type)
  document.body.appendChild(toast)

  setTimeout(() => {
    toast.classList.add('toast-out')
    setTimeout(() => toast.remove(), 300)
  }, duration)
}
