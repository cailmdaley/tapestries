// PlaygroundViewer.ts - Panel for viewing interactive playgrounds

import type { City } from '../state/types'

interface PlaygroundInfo {
  name: string
  url: string
}

export class PlaygroundViewer {
  private panel: HTMLElement
  private iframe: HTMLIFrameElement
  private closeBtn: HTMLElement
  private title: HTMLElement
  private loadingIndicator: HTMLElement
  private playgroundList: HTMLElement
  private playgrounds: PlaygroundInfo[] = []
  private showRequestId = 0
  private showAbortController: AbortController | null = null
  private hideTimeoutId: number | null = null
  private disposed = false

  // Stored listener ref for HMR-safe cleanup
  private escapeHandler: ((e: KeyboardEvent) => void) | null = null

  constructor() {
    this.panel = this.createPanel()
    this.iframe = this.panel.querySelector('iframe')!
    this.closeBtn = this.panel.querySelector('.close-btn')!
    this.title = this.panel.querySelector('h2')!
    this.loadingIndicator = this.panel.querySelector('.loading-indicator')!
    this.playgroundList = this.panel.querySelector('.playground-list')!

    this.setupEventListeners()
    document.body.appendChild(this.panel)
  }

  private createPanel(): HTMLElement {
    const panel = document.createElement('div')
    panel.className = 'playground-viewer'
    panel.innerHTML = `
      <div class="playground-viewer-header">
        <h2>Playgrounds</h2>
        <div class="playground-list"></div>
        <button class="close-btn">&times;</button>
      </div>
      <div class="loading-indicator">Loading playground...</div>
      <iframe src="about:blank" frameborder="0"></iframe>
    `
    return panel
  }

  private setupEventListeners(): void {
    this.closeBtn.addEventListener('click', () => this.hide())

    // Define escape handler (attached/detached dynamically to avoid HMR stacking)
    this.escapeHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && this.isVisible()) {
        this.hide()
      }
    }
  }

  private beginShowRequest(): { requestId: number; signal: AbortSignal } {
    this.cancelShowRequest()
    this.showRequestId += 1
    this.showAbortController = new AbortController()
    return { requestId: this.showRequestId, signal: this.showAbortController.signal }
  }

  private isCurrentShowRequest(requestId: number): boolean {
    return !this.disposed && requestId === this.showRequestId
  }

  private cancelShowRequest(): void {
    if (this.showAbortController) {
      this.showAbortController.abort()
      this.showAbortController = null
    }
  }

  private clearHideTimeout(): void {
    if (this.hideTimeoutId !== null) {
      window.clearTimeout(this.hideTimeoutId)
      this.hideTimeoutId = null
    }
  }

  async show(city: City): Promise<void> {
    if (this.disposed) {
      return
    }

    this.clearHideTimeout()
    const { requestId, signal } = this.beginShowRequest()

    this.title.textContent = `Playgrounds: ${city.name}`

    // Show loading
    this.loadingIndicator.style.display = 'flex'
    this.loadingIndicator.textContent = 'Loading playgrounds...'
    this.iframe.style.opacity = '0'
    this.playgroundList.innerHTML = ''

    // Attach escape listener and show panel
    if (this.escapeHandler) {
      document.addEventListener('keydown', this.escapeHandler)
    }
    this.panel.classList.add('visible')

    // Fetch list of playgrounds
    try {
      const res = await fetch(`http://${window.location.hostname}:4004/playground-list?cityId=${encodeURIComponent(city.id)}`, { signal })
      const data = await res.json()
      if (!this.isCurrentShowRequest(requestId) || signal.aborted) return

      if (!data.playgrounds || data.playgrounds.length === 0) {
        this.loadingIndicator.textContent = 'No playgrounds found'
        return
      }

      this.playgrounds = data.playgrounds.map((name: string) => ({
        name,
        url: `http://${window.location.hostname}:4004/playground?cityId=${encodeURIComponent(city.id)}&name=${encodeURIComponent(name)}`
      }))

      this.renderPlaygroundList()

      // Load first playground by default
      this.loadPlayground(this.playgrounds[0], requestId)
    } catch (err) {
      if (!this.isCurrentShowRequest(requestId) || signal.aborted) return
      this.loadingIndicator.textContent = 'Failed to load playgrounds'
      console.error('Failed to load playgrounds:', err)
    } finally {
      if (this.isCurrentShowRequest(requestId)) {
        this.showAbortController = null
      }
    }
  }

  private renderPlaygroundList(): void {
    this.playgroundList.innerHTML = this.playgrounds.map((pg, idx) => {
      const displayName = pg.name.replace('.html', '')
      const isActive = idx === 0 ? 'active' : ''
      return `<button class="playground-tab ${isActive}" data-idx="${idx}">${displayName}</button>`
    }).join('')

    // Add click handlers
    this.playgroundList.querySelectorAll('.playground-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-idx')!, 10)
        this.loadPlayground(this.playgrounds[idx], this.showRequestId)

        // Update active state
        this.playgroundList.querySelectorAll('.playground-tab').forEach(b => b.classList.remove('active'))
        btn.classList.add('active')
      })
    })
  }

  private loadPlayground(playground: PlaygroundInfo, requestId: number): void {
    if (!this.isCurrentShowRequest(requestId) || !this.isVisible()) {
      return
    }

    // Show loading
    this.loadingIndicator.style.display = 'flex'
    this.loadingIndicator.textContent = 'Loading playground...'
    this.iframe.style.opacity = '0'

    // Set iframe src
    this.iframe.src = playground.url

    // Hide loading when iframe loads
    this.iframe.onload = () => {
      if (!this.isCurrentShowRequest(requestId) || !this.isVisible()) return
      this.loadingIndicator.style.display = 'none'
      this.iframe.style.opacity = '1'
    }
  }

  hide(): void {
    this.cancelShowRequest()
    this.panel.classList.remove('visible')
    this.iframe.onload = null

    // Detach escape listener
    if (this.escapeHandler) {
      document.removeEventListener('keydown', this.escapeHandler)
    }

    // Clear iframe after animation
    this.clearHideTimeout()
    if (this.disposed) {
      this.iframe.src = 'about:blank'
      this.playgrounds = []
      return
    }

    this.hideTimeoutId = window.setTimeout(() => {
      this.hideTimeoutId = null
      if (!this.isVisible()) {
        this.iframe.src = 'about:blank'
        this.playgrounds = []
      }
    }, 300)
  }

  isVisible(): boolean {
    return this.panel.classList.contains('visible')
  }

  getRuntimeStats(): {
    visible: boolean
    disposed: boolean
    showRequestId: number
    hasActiveShowRequest: boolean
    hasHideTimeout: boolean
    playgroundCount: number
    iframeSrc: string
  } {
    return {
      visible: this.isVisible(),
      disposed: this.disposed,
      showRequestId: this.showRequestId,
      hasActiveShowRequest: this.showAbortController !== null,
      hasHideTimeout: this.hideTimeoutId !== null,
      playgroundCount: this.playgrounds.length,
      iframeSrc: this.iframe.getAttribute('src') || '',
    }
  }

  dispose(): void {
    if (this.disposed) return

    this.disposed = true
    this.hide()
    this.panel.remove()
  }
}
