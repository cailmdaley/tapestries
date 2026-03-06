import type { City, GitStatus, Session } from '../state/types'
import { escapeHtml, fiberStatusIcon } from './utils'
import type { DirectoryEntry, Fiber, SearchResult } from './hud-types'
import type { NewWorkerDialog } from './NewWorkerDialog'

interface FibersResponse {
  type: 'fibers'
  cityId: string
  open: Fiber[]
  recentlyClosed: Fiber[]
}

interface DirectoryListingResponse {
  type: 'directoryListing'
  cityId: string
  path: string
  entries: DirectoryEntry[]
  error?: string
}

type FibersCallback = (response: FibersResponse) => void

type HudTab = 'fibers' | 'files'

export class CityHUD {
  private container: HTMLElement
  private sidebar: HTMLElement
  private headerWidget: HTMLElement
  private fiberList: HTMLElement
  private filesList: HTMLElement
  private searchInput: HTMLInputElement
  private searchClear: HTMLElement
  private searchResultsList: HTMLElement
  private currentCity: City | null = null
  private ws: WebSocket | null = null
  private fibersCallback: FibersCallback | null = null
  private ignoreNextClick = false

  private activeTab: HudTab = 'files'

  // Fiber state
  private openFibers: Fiber[] = []
  private closedFibers: Fiber[] = []

  // File tree state
  private directoryCache = new Map<string, DirectoryEntry[]>()
  private expandedDirs = new Set<string>()
  private loadingDirs = new Set<string>()
  private directoryErrors = new Map<string, string>()
  private inFlightDirectoryRequests = new Set<string>()

  // Worker state
  private cityWorkers: Session[] = []

  // Stored listener refs for HMR-safe cleanup
  private clickOutsideHandler: ((e: MouseEvent) => void) | null = null
  private escapeHandler: ((e: KeyboardEvent) => void) | null = null

  // Callbacks
  private onViewClaims: ((city: City) => void) | null = null
  private onViewPlaygrounds: ((city: City) => void) | null = null
  private onOpenFile: ((fullPath: string, originId: string, cityPath: string, cityId: string, line?: number) => void) | null = null
  private onFocusWorker: ((sessionId: string) => void) | null = null
  private newWorkerDialog: NewWorkerDialog | null = null

  // Search state
  private currentSearchId = 0
  private searchResults: SearchResult[] = []
  private searchQuery = ''

  constructor() {
    this.container = this.createContainer()
    this.sidebar = this.container.querySelector('.hud-sidebar')!
    this.headerWidget = this.container.querySelector('.hud-header')!
    this.fiberList = this.container.querySelector('.hud-fiber-list')!
    this.filesList = this.container.querySelector('.hud-file-tree')!
    this.searchInput = this.container.querySelector('.hud-search-input')!
    this.searchClear = this.container.querySelector('.hud-search-clear')!
    this.searchResultsList = this.container.querySelector('.hud-search-results')!
    this.setupEventHandlers()
    this.setupSearch()
    this.setupDelegatedListeners()
    this.setupTabs()
    document.body.appendChild(this.container)
  }

  private createContainer(): HTMLElement {
    const el = document.createElement('div')
    el.id = 'city-hud'
    el.innerHTML = `
      <div class="hud-sidebar">
        <div class="hud-header">
          <div class="hud-header-row">
            <h2 class="hud-city-name"></h2>
            <div class="hud-header-controls">
              <div class="hud-actions"></div>
              <button class="hud-close" title="Close">&times;</button>
            </div>
          </div>
          <p class="hud-city-path"></p>
          <div class="hud-git-detail-content"></div>
          <div class="hud-header-workers"></div>
        </div>

        <div class="hud-tabbar">
          <button class="hud-tab" data-tab="fibers">Fibers</button>
          <button class="hud-tab active" data-tab="files">Files</button>
        </div>

        <div class="hud-content">
          <ul class="hud-search-results" style="display: none;"></ul>
          <div class="hud-pane hud-pane-fibers">
            <ul class="hud-fiber-list"></ul>
          </div>
          <div class="hud-pane hud-pane-files active">
            <ul class="hud-file-tree"></ul>
          </div>
        </div>

        <div class="hud-search-bar">
          <input type="text" class="hud-search-input" placeholder="Search files &amp; fibers…" />
          <button class="hud-search-clear" style="display: none;">&times;</button>
        </div>
      </div>
    `
    return el
  }

  private setupEventHandlers(): void {
    this.clickOutsideHandler = (e: MouseEvent) => {
      if (this.ignoreNextClick) {
        this.ignoreNextClick = false
        return
      }
      if (!this.container.classList.contains('visible')) return
      const path = e.composedPath()
      if (path.includes(this.container)) return
      const target = e.target as HTMLElement
      const fileViewer = document.querySelector('.file-viewer-modal.visible')
      if (fileViewer?.contains(target)) return
      const fileViewerBackdrop = document.querySelector('.file-viewer-backdrop.visible')
      if (fileViewerBackdrop?.contains(target)) return
      this.hide()
    }

    this.escapeHandler = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || !this.container.classList.contains('visible')) return
      const fileViewer = document.querySelector('.file-viewer-modal.visible')
      if (fileViewer) return
      if (this.searchQuery || this.activeTab === 'fibers' && this.sidebar.classList.contains('searching')) {
        this.clearSearch()
        return
      }
      this.hide()
    }

    this.container.querySelector('.hud-close')?.addEventListener('click', (e) => {
      e.stopPropagation()
      this.hide()
    })
  }

  private attachDocumentListeners(): void {
    if (this.clickOutsideHandler) {
      document.addEventListener('click', this.clickOutsideHandler)
    }
    if (this.escapeHandler) {
      document.addEventListener('keydown', this.escapeHandler)
    }
  }

  private detachDocumentListeners(): void {
    if (this.clickOutsideHandler) {
      document.removeEventListener('click', this.clickOutsideHandler)
    }
    if (this.escapeHandler) {
      document.removeEventListener('keydown', this.escapeHandler)
    }
  }

  private setupTabs(): void {
    const tabBar = this.container.querySelector('.hud-tabbar')!
    tabBar.addEventListener('click', (e) => {
      const tabBtn = (e.target as HTMLElement).closest<HTMLElement>('.hud-tab')
      if (!tabBtn) return
      const tab = tabBtn.dataset.tab as HudTab | undefined
      if (!tab || tab === this.activeTab) return
      this.switchTab(tab)
    })
  }

  private switchTab(tab: HudTab): void {
    this.activeTab = tab

    for (const btn of this.container.querySelectorAll<HTMLElement>('.hud-tab')) {
      btn.classList.toggle('active', btn.dataset.tab === tab)
    }
    for (const pane of this.container.querySelectorAll<HTMLElement>('.hud-pane')) {
      const isActive = pane.classList.contains(`hud-pane-${tab}`)
      pane.classList.toggle('active', isActive)
      pane.style.display = isActive ? '' : 'none'
    }

    this.clearSearch()
    this.searchInput.placeholder = tab === 'fibers' ? 'Search fibers & files…' : 'Search files…'

    if (tab === 'files') {
      this.ensureRootListing()
    }
  }

  private renderGitDetail(status?: GitStatus): void {
    const content = this.headerWidget.querySelector('.hud-git-detail-content')!
    if (!status?.isRepo) {
      content.innerHTML = ''
      return
    }

    const rows: string[] = []

    rows.push(`<div class="hud-gd-row">
      <span class="hud-gd-label">branch</span>
      <span class="hud-gd-value hud-gd-branch">${escapeHtml(status.branch)}</span>
    </div>`)

    if (status.ahead > 0 || status.behind > 0) {
      const parts: string[] = []
      if (status.ahead > 0) parts.push(`<span class="hud-gd-ahead">↑${status.ahead}</span>`)
      if (status.behind > 0) parts.push(`<span class="hud-gd-behind">↓${status.behind}</span>`)
      rows.push(`<div class="hud-gd-row">
        <span class="hud-gd-label">remote</span>
        <span class="hud-gd-value">${parts.join(' ')}</span>
      </div>`)
    }

    const staged = status.staged
    if (staged.added + staged.modified + staged.deleted > 0) {
      const parts: string[] = []
      if (staged.added > 0) parts.push(`+${staged.added}`)
      if (staged.modified > 0) parts.push(`~${staged.modified}`)
      if (staged.deleted > 0) parts.push(`-${staged.deleted}`)
      rows.push(`<div class="hud-gd-row">
        <span class="hud-gd-label">staged</span>
        <span class="hud-gd-value hud-gd-staged">${parts.join(' ')}</span>
      </div>`)
    }

    const unstaged = status.unstaged
    if (unstaged.added + unstaged.modified + unstaged.deleted > 0) {
      const parts: string[] = []
      if (unstaged.added > 0) parts.push(`+${unstaged.added}`)
      if (unstaged.modified > 0) parts.push(`~${unstaged.modified}`)
      if (unstaged.deleted > 0) parts.push(`-${unstaged.deleted}`)
      rows.push(`<div class="hud-gd-row">
        <span class="hud-gd-label">unstaged</span>
        <span class="hud-gd-value hud-gd-unstaged">${parts.join(' ')}</span>
      </div>`)
    }

    if (status.untracked > 0) {
      rows.push(`<div class="hud-gd-row">
        <span class="hud-gd-label">untracked</span>
        <span class="hud-gd-value hud-gd-untracked">${status.untracked} file${status.untracked !== 1 ? 's' : ''}</span>
      </div>`)
    }

    if (status.linesAdded > 0 || status.linesRemoved > 0) {
      const parts: string[] = []
      if (status.linesAdded > 0) parts.push(`<span class="hud-git-add">+${status.linesAdded}</span>`)
      if (status.linesRemoved > 0) parts.push(`<span class="hud-git-rm">−${status.linesRemoved}</span>`)
      rows.push(`<div class="hud-gd-row">
        <span class="hud-gd-label">diff</span>
        <span class="hud-gd-value">${parts.join(' ')}</span>
      </div>`)
    }

    if (status.lastCommitMessage) {
      const timeStr = status.lastCommitTime ? this.relativeTime(status.lastCommitTime) : ''
      const msg = status.lastCommitMessage.length > 48
        ? status.lastCommitMessage.slice(0, 48) + '…'
        : status.lastCommitMessage
      rows.push(`<div class="hud-gd-commit">
        <span class="hud-gd-commit-msg">${escapeHtml(msg)}</span>
        ${timeStr ? `<span class="hud-gd-commit-time">${timeStr}</span>` : ''}
      </div>`)
    }

    content.innerHTML = rows.join('')
  }

  private relativeTime(timestamp: number): string {
    const now = Date.now()
    const diff = now - timestamp
    const minutes = Math.floor(diff / 60000)
    if (minutes < 1) return 'just now'
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }

  private renderActions(city: City): void {
    const buttons: string[] = []

    if (city.hasClaims) {
      buttons.push(`<button class="hud-action-btn hud-action-claims" title="Claims">⚖</button>`)
    }

    if (city.hasPlaygrounds) {
      buttons.push(`<button class="hud-action-btn hud-action-playgrounds" title="Playgrounds">▶</button>`)
    }

    const row = this.headerWidget.querySelector('.hud-actions')!
    row.innerHTML = buttons.join('')

    row.querySelector('.hud-action-claims')?.addEventListener('click', (e) => {
      e.stopPropagation()
      if (this.currentCity) this.onViewClaims?.(this.currentCity)
    })

    row.querySelector('.hud-action-playgrounds')?.addEventListener('click', (e) => {
      e.stopPropagation()
      if (this.currentCity) this.onViewPlaygrounds?.(this.currentCity)
    })
  }

  show(city: City): void {
    if (document.querySelector('.tapestry-view.visible')) return

    this.currentCity = city

    this.headerWidget.querySelector('.hud-city-name')!.textContent = city.name
    this.headerWidget.querySelector('.hud-city-path')!.textContent = city.path
    this.renderGitDetail(city.gitStatus)
    this.renderActions(city)

    this.openFibers = []
    this.closedFibers = []
    this.resetFileTreeState()

    this.activeTab = 'files'
    this.switchTab('files')

    this.ignoreNextClick = true

    this.attachDocumentListeners()
    this.container.classList.add('visible')

    this.requestFibers(city.id)
  }

  hide(): void {
    this.container.classList.remove('visible')
    this.currentCity = null
    this.detachDocumentListeners()
    this.clearSearch()
    this.resetFileTreeState()
  }

  isVisible(): boolean {
    return this.container.classList.contains('visible')
  }

  getRuntimeStats(): {
    visible: boolean
    activeTab: 'fibers' | 'files'
    currentCityId: string | null
    openFibers: number
    closedFibers: number
    searchQueryLength: number
    pendingSearchResults: number
    cityWorkerCount: number
    directoryCacheEntries: number
    expandedDirectoryCount: number
    loadingDirectoryCount: number
    directoryErrorCount: number
    inFlightDirectoryRequestCount: number
  } {
    return {
      visible: this.isVisible(),
      activeTab: this.activeTab,
      currentCityId: this.currentCity?.id ?? null,
      openFibers: this.openFibers.length,
      closedFibers: this.closedFibers.length,
      searchQueryLength: this.searchQuery.length,
      pendingSearchResults: this.searchResults.length,
      cityWorkerCount: this.cityWorkers.length,
      directoryCacheEntries: this.directoryCache.size,
      expandedDirectoryCount: this.expandedDirs.size,
      loadingDirectoryCount: this.loadingDirs.size,
      directoryErrorCount: this.directoryErrors.size,
      inFlightDirectoryRequestCount: this.inFlightDirectoryRequests.size,
    }
  }

  setWebSocket(ws: WebSocket): void {
    this.ws = ws
  }

  setOnViewClaims(callback: (city: City) => void): void {
    this.onViewClaims = callback
  }

  setOnViewPlaygrounds(callback: (city: City) => void): void {
    this.onViewPlaygrounds = callback
  }

  setOnOpenFile(callback: (fullPath: string, originId: string, cityPath: string, cityId: string, line?: number) => void): void {
    this.onOpenFile = callback
  }

  setNewWorkerDialog(dialog: NewWorkerDialog): void {
    this.newWorkerDialog = dialog
  }

  setOnFocusWorker(callback: (sessionId: string) => void): void {
    this.onFocusWorker = callback
  }

  updateWorkers(sessions: Session[]): void {
    if (!this.currentCity || !this.container.classList.contains('visible')) return

    this.cityWorkers = sessions.filter(s => s.cityId === this.currentCity!.id)
    this.renderWorkers()
  }

  handleMessage(message: unknown): boolean {
    const msg = message as { type?: string }
    if (msg.type === 'fibers') {
      const response = message as FibersResponse
      if (this.fibersCallback) {
        this.fibersCallback(response)
        this.fibersCallback = null
      }
      return true
    }
    if (msg.type === 'searchResults') {
      const response = message as { searchId: string; results: SearchResult[]; error?: string }
      this.handleSearchResults(response.searchId, response.results, response.error)
      return true
    }
    if (msg.type === 'directoryListing') {
      this.handleDirectoryListing(message as DirectoryListingResponse)
      return true
    }
    return false
  }

  handleSearchResults(searchId: string, results: SearchResult[], error?: string): void {
    if (!this.searchQuery) return
    const expectedPrefix = `${this.currentCity?.id || ''}-${this.currentSearchId}`
    if (!searchId.startsWith(expectedPrefix)) return
    if (error) {
      console.error('Search error:', error)
      return
    }
    for (const r of results) {
      if (!this.searchResults.some(sr => sr.fullPath === r.fullPath)) {
        this.searchResults.push(r)
      }
    }
    this.renderSearchResults()
  }

  private setupSearch(): void {
    this.searchInput.addEventListener('input', () => {
      this.searchQuery = this.searchInput.value.trim()
      this.searchClear.style.display = this.searchInput.value ? 'block' : 'none'

      if (!this.searchQuery) {
        this.collapseSearch()
      } else if (this.activeTab === 'fibers') {
        this.performSearch()
      }
    })

    this.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && this.searchQuery) {
        this.performSearch()
      }
      if (e.key === 'Escape') {
        e.stopPropagation()
        this.clearSearch()
        this.searchInput.blur()
      }
    })

    this.searchInput.addEventListener('focus', () => {
      this.sidebar.classList.add('search-focused')
    })

    this.searchInput.addEventListener('blur', () => {
      if (!this.searchQuery) {
        this.sidebar.classList.remove('search-focused')
      }
    })

    this.searchClear.addEventListener('click', () => {
      this.clearSearch()
      this.searchInput.focus()
    })
  }

  private setupDelegatedListeners(): void {
    this.fiberList.addEventListener('click', (e) => {
      const handoff = (e.target as HTMLElement).closest<HTMLElement>('.hud-fiber-handoff')
      if (handoff) {
        e.stopPropagation()
        const fiberId = handoff.dataset.fiberId
        if (fiberId && this.currentCity && this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: 'handoff',
            fiberId,
            cityPath: this.currentCity.path,
          }))
        }
        return
      }

      const item = (e.target as HTMLElement).closest<HTMLElement>('.hud-fiber-item')
      if (!item) return
      this.openFiber(item.dataset.fiberId)
    })

    this.searchResultsList.addEventListener('click', (e) => {
      const item = (e.target as HTMLElement).closest<HTMLElement>('.hud-search-item')
      if (!item) return
      if (item.dataset.type === 'file') {
        const line = item.dataset.line ? parseInt(item.dataset.line, 10) : undefined
        this.openFile(item.dataset.path, line)
      } else if (item.dataset.type === 'fiber') {
        this.openFiber(item.dataset.fiberId)
      }
    })

    this.filesList.addEventListener('click', (e) => {
      const row = (e.target as HTMLElement).closest<HTMLElement>('.hud-tree-row')
      if (!row) return
      const path = row.dataset.path
      const type = row.dataset.type
      if (!path || !type) return

      if (type === 'dir') {
        this.toggleDirectory(path)
      } else {
        this.openFile(path)
      }
    })
  }

  private performSearch(): void {
    this.searchResults = []
    this.sidebar.classList.add('searching')
    this.fiberList.style.display = 'none'
    this.filesList.style.display = 'none'
    this.searchResultsList.style.display = 'block'
    this.searchResultsList.innerHTML = '<li class="hud-search-loading">Searching…</li>'

    if (this.currentCity && this.ws?.readyState === WebSocket.OPEN) {
      const searchId = `${this.currentCity.id}-${++this.currentSearchId}`

      this.ws.send(JSON.stringify({
        type: 'searchFiles',
        cityId: this.currentCity.id,
        query: this.searchQuery,
        searchId: `${searchId}-name`,
        mode: 'filename',
      }))
    }

    // Render immediately with local fiber results (files arrive async)
    this.renderSearchResults()
  }

  private clearSearch(): void {
    this.searchQuery = ''
    this.searchInput.value = ''
    this.searchClear.style.display = 'none'
    this.sidebar.classList.remove('search-focused')

    this.collapseSearch()
    if (this.activeTab === 'files') {
      this.renderFileTree()
    }
  }

  private collapseSearch(): void {
    this.sidebar.classList.remove('searching')
    this.searchResults = []
    this.searchResultsList.style.display = 'none'
    if (this.activeTab === 'files') {
      this.filesList.style.display = ''
    } else {
      this.fiberList.style.display = ''
    }
  }

  private filterFibersLocally(query: string): Fiber[] {
    const q = query.toLowerCase()
    return [...this.openFibers, ...this.closedFibers].filter(f =>
      f.title.toLowerCase().includes(q) ||
      f.kind.toLowerCase().includes(q) ||
      f.id.toLowerCase().includes(q) ||
      (f.body?.toLowerCase().includes(q) ?? false) ||
      (f.reason?.toLowerCase().includes(q) ?? false)
    )
  }

  private renderSearchResults(): void {
    const fibers = this.activeTab === 'fibers' ? this.filterFibersLocally(this.searchQuery) : []
    const files = this.activeTab === 'files' ? this.searchResults.slice(0, 20) : []

    if (fibers.length === 0 && files.length === 0) {
      if (this.searchResultsList.querySelector('.hud-search-loading')) return
      this.searchResultsList.innerHTML = '<li class="hud-search-empty">No matches</li>'
      return
    }

    let html = ''

    for (const f of fibers.slice(0, 20)) {
      const kind = f.kind || 'task'
      html += `
        <li class="hud-search-item hud-fiber-item ${kind}" data-type="fiber" data-fiber-id="${f.id}">
          <span class="hud-fiber-status">${fiberStatusIcon(f.status)}</span>
          <span class="hud-fiber-title">${escapeHtml(f.title)}</span>
          <span class="hud-fiber-kind">${kind}</span>
        </li>`
    }

    for (const r of files) {
      const fileName = r.path.split('/').pop() || r.path
      const lineInfo = r.line !== undefined ? `:${r.line}` : ''
      const lineAttr = r.line !== undefined ? ` data-line="${r.line}"` : ''
      html += `
        <li class="hud-search-item hud-fiber-item file" data-type="file" data-path="${escapeHtml(r.fullPath)}"${lineAttr}>
          <span class="hud-search-icon">▹</span>
          <span class="hud-fiber-title mono">${escapeHtml(fileName)}${lineInfo}</span>
        </li>`
    }

    this.searchResultsList.innerHTML = html
  }

  private requestFibers(cityId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.fiberList.innerHTML = '<li class="hud-fiber-empty">No connection</li>'
      return
    }

    this.fiberList.innerHTML = '<li class="hud-fiber-empty hud-fiber-loading">Loading…</li>'

    this.fibersCallback = (response) => {
      if (response.cityId === this.currentCity?.id) {
        this.renderFibers(response.open, response.recentlyClosed)
      }
    }

    this.ws.send(JSON.stringify({ type: 'getFibers', cityId }))
  }

  private renderFibers(open: Fiber[], closed: Fiber[]): void {
    this.openFibers = open
    this.closedFibers = closed

    const all = [...open, ...closed]
    if (all.length === 0) {
      this.fiberList.innerHTML = '<li class="hud-fiber-empty">No fibers</li>'
      return
    }

    this.fiberList.innerHTML = all.map(f => this.renderFiberItem(f)).join('')
  }

  private renderFiberItem(fiber: Fiber): string {
    const kind = fiber.kind || 'task'

    return `
      <li class="hud-fiber-item ${kind}" data-fiber-id="${fiber.id}">
        <span class="hud-fiber-status">${fiberStatusIcon(fiber.status)}</span>
        <span class="hud-fiber-title">${escapeHtml(fiber.title)}</span>
        <span class="hud-fiber-kind">${kind}</span>
        <button class="hud-fiber-handoff" data-fiber-id="${fiber.id}" title="Hand off to worker">↗</button>
      </li>
    `
  }

  private ensureRootListing(): void {
    if (!this.currentCity) return
    const root = this.currentCity.path
    this.expandedDirs.add(root)
    if (!this.directoryCache.has(root) && !this.loadingDirs.has(root)) {
      this.requestDirectoryListing(root)
    }
    this.renderFileTree()
  }

  private resetFileTreeState(): void {
    this.directoryCache.clear()
    this.expandedDirs.clear()
    this.loadingDirs.clear()
    this.directoryErrors.clear()
    this.inFlightDirectoryRequests.clear()
    this.filesList.innerHTML = ''
  }

  private toggleDirectory(path: string): void {
    if (this.expandedDirs.has(path)) {
      this.expandedDirs.delete(path)
      this.renderFileTree()
      return
    }

    this.expandedDirs.add(path)
    if (!this.directoryCache.has(path) && !this.loadingDirs.has(path)) {
      this.requestDirectoryListing(path)
    }
    this.renderFileTree()
  }

  private requestDirectoryListing(path: string): void {
    if (!this.currentCity || !this.ws || this.ws.readyState !== WebSocket.OPEN) return
    if (this.inFlightDirectoryRequests.has(path)) return

    this.inFlightDirectoryRequests.add(path)
    this.loadingDirs.add(path)
    this.directoryErrors.delete(path)
    this.renderFileTree()

    this.ws.send(JSON.stringify({
      type: 'listDirectory',
      cityId: this.currentCity.id,
      path,
    }))
  }

  private handleDirectoryListing(response: DirectoryListingResponse): void {
    if (!this.currentCity || response.cityId !== this.currentCity.id) return

    this.inFlightDirectoryRequests.delete(response.path)
    this.loadingDirs.delete(response.path)

    if (response.error) {
      this.directoryErrors.set(response.path, response.error)
    } else {
      this.directoryErrors.delete(response.path)
      this.directoryCache.set(response.path, response.entries)
    }

    this.renderFileTree()
  }

  private renderFileTree(): void {
    if (!this.currentCity) return

    const root = this.currentCity.path
    const rootEntries = this.directoryCache.get(root)
    const rootLoading = this.loadingDirs.has(root)

    if (!this.expandedDirs.has(root)) {
      this.filesList.innerHTML = '<li class="hud-file-empty">Open Files tab to browse this project.</li>'
      return
    }

    if (!rootEntries && rootLoading) {
      this.filesList.innerHTML = '<li class="hud-file-empty">Loading…</li>'
      return
    }

    if (!rootEntries) {
      this.filesList.innerHTML = '<li class="hud-file-empty">No directory listing available.</li>'
      return
    }

    let html = ''

    for (const entry of rootEntries) {
      html += this.renderTreeNode(root, entry, 0, '')
    }

    this.filesList.innerHTML = html || '<li class="hud-file-empty">No matching entries.</li>'
  }

  private renderTreeNode(parentPath: string, entry: DirectoryEntry, depth: number, filter: string): string {
    const fullPath = `${parentPath.replace(/\/$/, '')}/${entry.name}`
    const isDir = entry.type === 'dir'
    const isExpanded = isDir && this.expandedDirs.has(fullPath)
    const isLoading = isDir && this.loadingDirs.has(fullPath)
    const error = this.directoryErrors.get(fullPath)
    const matches = !filter || entry.name.toLowerCase().includes(filter)

    let html = ''

    if (matches) {
      const arrow = isDir ? (isExpanded ? '▼' : '▶') : '•'
      html += `
        <li class="hud-tree-row" data-path="${escapeHtml(fullPath)}" data-type="${entry.type}" style="padding-left: ${depth * 16 + 8}px">
          <span class="hud-tree-arrow">${arrow}</span>
          <span class="hud-tree-name">${escapeHtml(entry.name)}</span>
        </li>
      `
    }

    if (!isDir || !isExpanded) {
      return html
    }

    if (isLoading) {
      html += `
        <li class="hud-tree-status" style="padding-left: ${(depth + 1) * 16 + 8}px">…</li>
      `
      return html
    }

    if (error) {
      html += `
        <li class="hud-tree-status error" style="padding-left: ${(depth + 1) * 16 + 8}px">couldn't read directory</li>
      `
      return html
    }

    const children = this.directoryCache.get(fullPath) || []
    if (children.length === 0) {
      html += `
        <li class="hud-tree-status" style="padding-left: ${(depth + 1) * 16 + 8}px">empty</li>
      `
      return html
    }

    for (const child of children) {
      html += this.renderTreeNode(fullPath, child, depth + 1, filter)
    }

    return html
  }

  private openFiber(fiberId: string | undefined): void {
    if (!fiberId || !this.currentCity || !this.onOpenFile) return
    const feltPath = `${this.currentCity.path}/.felt/${fiberId}.md`
    this.onOpenFile(feltPath, this.currentCity.originId, this.currentCity.path, this.currentCity.id)
  }

  private openFile(fullPath: string | undefined, line?: number): void {
    if (!fullPath || !this.currentCity || !this.onOpenFile) return
    this.onOpenFile(fullPath, this.currentCity.originId, this.currentCity.path, this.currentCity.id, line)
  }

  private renderWorkers(): void {
    const container = this.headerWidget.querySelector('.hud-header-workers')!

    const parts: string[] = [`<span class="hud-header-workers-label">workers</span>`]

    const chips = this.cityWorkers.map(s => {
      const statusClass = s.status === 'working' ? 'working' : 'idle'
      return `<span class="hud-worker-chip ${statusClass}" data-session-id="${s.id}" title="${escapeHtml(s.name)}">` +
        `<span class="hud-worker-dot ${statusClass}">●</span>${escapeHtml(s.name)}</span>`
    })

    chips.push('<button class="hud-worker-add" title="New Worker">+</button>')

    container.innerHTML = parts.concat(chips).join('')

    for (const chip of container.querySelectorAll<HTMLElement>('.hud-worker-chip')) {
      chip.addEventListener('click', (e) => {
        e.stopPropagation()
        const sessionId = chip.dataset.sessionId
        if (sessionId) this.onFocusWorker?.(sessionId)
      })
    }

    container.querySelector('.hud-worker-add')?.addEventListener('click', (e) => {
      e.stopPropagation()
      if (this.currentCity && this.newWorkerDialog) {
        this.newWorkerDialog.show(this.currentCity.name).then(result => {
          if (!result) return
          if (this.ws?.readyState === WebSocket.OPEN && this.currentCity) {
            this.ws.send(JSON.stringify({
              type: 'newWorker',
              cityPath: this.currentCity.path,
              name: result.name || undefined,
              cli: result.cli || undefined,
              chrome: result.chrome || undefined,
              continue: result.continue || undefined,
            }))
          }
        })
      }
    })
  }

  dispose(): void {
    this.hide()
    this.container.remove()
  }
}
