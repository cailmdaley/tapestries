import { TapestryView } from '../ui/TapestryView'
import type { TapestryResponse } from '../ui/TapestryView'

interface StaticRuntime {
  view: TapestryView | null
  fetchAbortController: AbortController | null
  popstateHandler: (() => void) | null
  activeRequestId: number
  disposed: boolean
}

const runtime: StaticRuntime = {
  view: null,
  fetchAbortController: null,
  popstateHandler: null,
  activeRequestId: 0,
  disposed: false,
}

function detachPopstateHandler(): void {
  if (!runtime.popstateHandler) return
  window.removeEventListener('popstate', runtime.popstateHandler)
  runtime.popstateHandler = null
}

function attachPopstateHandler(view: TapestryView): void {
  detachPopstateHandler()
  const handler = () => {
    if (window.location.hash) {
      view.selectFromHash()
    }
  }
  runtime.popstateHandler = handler
  window.addEventListener('popstate', handler)
}

function clearDataRequest(): void {
  if (!runtime.fetchAbortController) return
  runtime.fetchAbortController.abort()
  runtime.fetchAbortController = null
}

function isCurrentRequest(requestId: number): boolean {
  return !runtime.disposed && runtime.activeRequestId === requestId
}

function cleanupRuntime(): void {
  if (runtime.disposed) return
  runtime.disposed = true

  clearDataRequest()
  detachPopstateHandler()
  runtime.view?.dispose()
  runtime.view = null

  window.removeEventListener('beforeunload', cleanupRuntime)
}

function bootstrap(): void {
  // Extract city name from path: /tapestries/pure-eb/ -> "pure-eb"
  const pathSegments = window.location.pathname.split('/').filter(Boolean)
  // On GitHub Pages, first segment is repo name "tapestries", second is city.
  const cityName = pathSegments.length >= 2 ? pathSegments[pathSegments.length - 1] : null

  if (!cityName) {
    showLanding()
    return
  }

  const view = new TapestryView()
  runtime.view = view

  // Data is at /tapestries/data/{city}/tapestry.json - use absolute path from base.
  const basePath = pathSegments.slice(0, -1).join('/')
  const dataUrl = `/${basePath}/data/${cityName}/tapestry.json`
  const requestId = ++runtime.activeRequestId
  const controller = new AbortController()

  clearDataRequest()
  runtime.fetchAbortController = controller

  fetch(dataUrl, { signal: controller.signal })
    .then(r => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
      return r.json()
    })
    .then((data: TapestryResponse) => {
      if (!isCurrentRequest(requestId)) return

      const assetBase = `/${basePath}/data/${cityName}/claims`
      view.showStatic(data, cityName, assetBase)
      attachPopstateHandler(view)
    })
    .catch(err => {
      if (!isCurrentRequest(requestId)) return
      if (err instanceof DOMException && err.name === 'AbortError') return

      console.error('Failed to load tapestry data:', err)
      view.showStatic(
        { nodes: [], links: [], downstream: {}, config: null },
        cityName,
      )
      const dag = document.querySelector('.tapestry-dag')
      if (dag) {
        dag.innerHTML = `<div style="
          position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
          font-family: var(--font-main); color: var(--ui-text-muted); text-align: center;
          line-height: 1.8; font-size: 0.95rem;
        ">
          <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">No tapestry data found</div>
          <div style="font-size: 0.8rem;">
            Run <code style="font-family: var(--font-mono); background: var(--ui-dark-surface);
            padding: 0.15rem 0.4rem; border-radius: 2px;">npx tsx scripts/export-tapestry.ts ${cityName}</code>
            then rebuild.
          </div>
        </div>`
      }
    })
    .finally(() => {
      if (isCurrentRequest(requestId)) {
        runtime.fetchAbortController = null
      }
    })
}

window.addEventListener('beforeunload', cleanupRuntime)
bootstrap()

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    cleanupRuntime()
  })
}

function showLanding() {
  document.body.innerHTML = `
    <div style="
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      min-height: 100vh; font-family: var(--font-main); color: var(--ui-text);
      background: var(--ui-dark); padding: 2rem;
    ">
      <h1 style="font-size: 2rem; font-weight: 400; margin-bottom: 0.3rem; color: var(--ui-gold);">Portolan</h1>
      <p style="font-size: 1rem; color: var(--ui-text-muted); margin-bottom: 2rem; font-style: italic;">Research Tapestries</p>
      <div id="tapestry-list" style="display: flex; flex-direction: column; gap: 0.8rem; min-width: 280px;">
        <p style="color: var(--ui-text-muted); font-style: italic; font-size: 0.9rem;">Loading...</p>
      </div>
    </div>
  `

  // Discover tapestries by fetching the manifest
  fetch('./data/manifest.json')
    .then(r => r.ok ? r.json() : [])
    .catch(() => [])
    .then((tapestries: { name: string; nodeCount?: number; updated?: string }[]) => {
      const list = document.getElementById('tapestry-list')!
      if (tapestries.length === 0) {
        list.innerHTML = '<p style="color: var(--ui-text-muted); font-style: italic;">No tapestries exported yet.</p>'
        return
      }
      list.innerHTML = tapestries.map(t => `
        <a href="./${t.name}/" style="
          display: block; padding: 1rem 1.4rem; background: var(--ui-dark-elevated);
          border: 1px solid var(--ui-border); border-radius: 4px;
          text-decoration: none; color: var(--ui-text); transition: border-color 200ms;
        " onmouseover="this.style.borderColor='var(--ui-teal)'" onmouseout="this.style.borderColor='var(--ui-border)'">
          <div style="font-size: 1.1rem; font-weight: 500;">${t.name}</div>
          ${t.nodeCount ? `<div style="font-size: 0.8rem; color: var(--ui-text-muted); margin-top: 0.2rem;">${t.nodeCount} nodes</div>` : ''}
        </a>
      `).join('')
    })
}
