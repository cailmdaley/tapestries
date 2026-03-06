// Shared annotation side panel used by TapestryView and FileViewerModal.
// Handles list rendering, inline edit, CRUD, clear all, global comment, and footer.

import { escapeHtml, formatTimeAgo, showToast } from './utils'
import { showWorkerPicker, type WorkerInfo } from './WorkerPicker'

const API_BASE = `http://${window.location.hostname}:4004`

// ── Types ──────────────────────────────────────────────────────────────

/** Minimal annotation shape that both ClaimsAnnotation and Annotation satisfy. */
export interface BaseAnnotation {
  id: string
  comment: string
  createdAt: number
}

export interface AnnotationPanelOptions<T extends BaseAnnotation> {
  /** CSS class prefix for the panel container (e.g., 'claims-annotation' or 'annotations'). */
  cssPrefix: 'claims' | 'file-viewer'

  /** Text shown when there are no annotations. */
  emptyMessage: string

  /** Render the preview line above the comment for each annotation item. */
  renderPreview: (ann: T, index: number) => string

  /** Called when the user clicks "go to" on an annotation. Omit to hide the button. */
  onGoto?: (ann: T) => void

  /** Called when the user clicks "promote" on an annotation. Omit to hide the button. */
  onPromote?: (ann: T) => void

  /** Called after any mutation (create, update, delete, clear) so the consumer can reload. */
  onRefresh: () => Promise<void> | void

  /** Build the query string for loading annotations (without leading ?). */
  buildLoadQuery: () => string

  /** Return the workers available for sending annotations. */
  getWorkers: () => WorkerInfo[]

  /** Called when annotations are sent to a worker. */
  onSendToWorker: (annotations: T[], workerId?: string, createNew?: boolean) => Promise<void>

  /** Placeholder text for the global comment textarea. */
  globalCommentPlaceholder?: string

  /** Global comment label (FileViewerModal uses "Overall feedback:"). Omit for no label. */
  globalCommentLabel?: string

  /** Hide the "Send N annotations to worker" footer button. */
  hideFooter?: boolean

  /** Called when annotations are filed as a fiber. Omit to hide the button. */
  onFileAsFiber?: (annotations: T[]) => Promise<void>
}

// ── Class ──────────────────────────────────────────────────────────────

export class AnnotationPanel<T extends BaseAnnotation> {
  private listEl: HTMLElement
  private footerEl: HTMLElement
  private globalInputEl: HTMLElement
  private toggleBtn: HTMLElement
  private panelEl: HTMLElement

  private annotations: T[] = []
  private options: AnnotationPanelOptions<T>

  constructor(panelEl: HTMLElement, options: AnnotationPanelOptions<T>) {
    this.panelEl = panelEl
    this.options = options

    this.listEl = panelEl.querySelector('.ann-panel-list')!
    this.footerEl = panelEl.querySelector('.ann-panel-footer')!
    this.globalInputEl = panelEl.querySelector('.ann-panel-global-input')!
    this.toggleBtn = panelEl.querySelector('.ann-panel-toggle')!

    this.setupEventListeners()
  }

  // ── Public API ─────────────────────────────────────────────────────

  /** Replace the annotation list and re-render. */
  setAnnotations(annotations: T[]): void {
    this.annotations = annotations
    this.renderList()
    this.renderFooter()
  }

  getAnnotations(): T[] {
    return this.annotations
  }

  /** Show the panel (remove hidden class) and auto-expand if collapsed. */
  expand(): void {
    this.panelEl.classList.remove('hidden')
    if (this.panelEl.classList.contains('collapsed')) {
      this.panelEl.classList.remove('collapsed')
      this.toggleBtn.textContent = this.collapseGlyph(false)
    }
  }

  /** Hide the panel entirely. */
  hidePanel(): void {
    this.panelEl.classList.add('hidden')
  }

  /** Reset state (clear annotations, reset global input). */
  reset(): void {
    this.annotations = []
    this.resetGlobalInput()
    this.renderList()
    this.renderFooter()
  }

  /** Get the global comment text. */
  getGlobalComment(): string {
    const textarea = this.globalInputEl.querySelector('textarea') as HTMLTextAreaElement | null
    return textarea?.value.trim() || ''
  }

  /** Reset global input textarea to empty. */
  resetGlobalInput(): void {
    const textarea = this.globalInputEl.querySelector('textarea') as HTMLTextAreaElement | null
    if (textarea) {
      textarea.value = ''
      textarea.style.height = ''
    }
  }

  /** Whether there are annotations or a global comment. */
  hasContent(): boolean {
    return this.annotations.length > 0 || this.getGlobalComment().length > 0
  }

  // ── Static HTML generator ──────────────────────────────────────────

  /** Generate the inner HTML for an annotation panel container. */
  static buildPanelHTML(opts: {
    globalCommentPlaceholder?: string
    globalCommentLabel?: string
    showSaveButton?: boolean
  }): string {
    const placeholder = opts.globalCommentPlaceholder || 'General feedback\u2026'
    const label = opts.globalCommentLabel
      ? `<label>${escapeHtml(opts.globalCommentLabel)}</label>`
      : ''
    const saveBtn = opts.showSaveButton
      ? `<button class="ann-global-save">Save</button>`
      : ''

    return `
      <div class="ann-panel-header">
        <h3>Annotations</h3>
        <div class="ann-panel-header-actions">
          <button class="ann-panel-clear-all" title="Clear all annotations">Clear</button>
          <button class="ann-panel-toggle" title="Toggle panel">\u25C0</button>
        </div>
      </div>
      <div class="ann-panel-list"></div>
      <div class="ann-panel-global-input">
        ${label}
        <textarea placeholder="${escapeHtml(placeholder)}" rows="2"></textarea>
        ${saveBtn}
      </div>
      <div class="ann-panel-footer"></div>
    `
  }

  // ── Internal ───────────────────────────────────────────────────────

  private setupEventListeners(): void {
    // Toggle collapse/expand
    this.toggleBtn.addEventListener('click', () => {
      this.panelEl.classList.toggle('collapsed')
      this.toggleBtn.textContent = this.collapseGlyph(
        this.panelEl.classList.contains('collapsed')
      )
    })

    // Clear all
    this.panelEl.querySelector('.ann-panel-clear-all')?.addEventListener('click', () => {
      if (this.annotations.length === 0) return
      this.clearAll()
    })

    // Auto-resize for global textarea
    const globalTextarea = this.globalInputEl.querySelector('textarea')
    if (globalTextarea) {
      globalTextarea.addEventListener('input', () => {
        this.autoResizeTextarea(globalTextarea)
      })
    }

    // Global save button and enter-to-save are wired by consumers directly,
    // since the save behavior is consumer-specific (e.g., TapestryView creates
    // a new annotation via POST, FileViewerModal just tracks the text in state).
  }

  private collapseGlyph(collapsed: boolean): string {
    return collapsed ? '\u25B6' : '\u25C0'
  }

  private autoResizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = 'auto'
    textarea.style.height = textarea.scrollHeight + 'px'
  }

  // ── List rendering ─────────────────────────────────────────────────

  private renderList(): void {
    if (this.annotations.length === 0) {
      this.listEl.innerHTML = `<div class="ann-panel-empty">${escapeHtml(this.options.emptyMessage)}</div>`
      return
    }

    this.listEl.innerHTML = this.annotations.map((ann, i) => {
      const preview = this.options.renderPreview(ann, i)
      const gotoBtn = this.options.onGoto
        ? `<button class="ann-action-goto" title="Go to">\u2197</button>`
        : ''
      const promoteBtn = this.options.onPromote
        ? `<button class="ann-action-promote" title="Promote to felt">\u2B06</button>`
        : ''

      return `
        <div class="ann-panel-item" data-annotation-id="${escapeHtml(ann.id)}">
          ${preview}
          <div class="ann-comment">${escapeHtml(ann.comment)}</div>
          <div class="ann-meta">
            <span>${formatTimeAgo(ann.createdAt)}</span>
            <div class="ann-actions">
              <button class="ann-action-edit" title="Edit">\u270E</button>
              ${gotoBtn}
              ${promoteBtn}
              <button class="ann-action-delete" title="Delete">\u00D7</button>
            </div>
          </div>
        </div>
      `
    }).join('')

    this.bindItemListeners()
  }

  private bindItemListeners(): void {
    this.listEl.querySelectorAll('.ann-panel-item').forEach((card, i) => {
      const ann = this.annotations[i]

      card.querySelector('.ann-action-edit')?.addEventListener('click', (e) => {
        e.stopPropagation()
        this.startEdit(ann)
      })

      card.querySelector('.ann-action-goto')?.addEventListener('click', (e) => {
        e.stopPropagation()
        this.options.onGoto?.(ann)
      })

      card.querySelector('.ann-action-promote')?.addEventListener('click', (e) => {
        e.stopPropagation()
        this.options.onPromote?.(ann)
      })

      card.querySelector('.ann-action-delete')?.addEventListener('click', (e) => {
        e.stopPropagation()
        this.deleteAnnotation(ann.id)
      })
    })
  }

  // ── Inline edit ────────────────────────────────────────────────────

  private startEdit(ann: T): void {
    const card = this.listEl.querySelector(`[data-annotation-id="${ann.id}"]`)
    if (!card) return

    const commentEl = card.querySelector('.ann-comment')
    if (!commentEl) return

    const original = ann.comment
    commentEl.innerHTML = `
      <textarea class="ann-edit-textarea">${escapeHtml(original)}</textarea>
      <div class="ann-edit-actions">
        <button class="ann-edit-cancel">Cancel</button>
        <button class="ann-edit-save">Save</button>
      </div>
    `

    const textarea = commentEl.querySelector('textarea')!
    const saveBtn = commentEl.querySelector('.ann-edit-save')!
    const cancelBtn = commentEl.querySelector('.ann-edit-cancel')!

    this.autoResizeTextarea(textarea)

    const finishEdit = async (save: boolean): Promise<void> => {
      if (save) {
        const newComment = textarea.value.trim()
        if (newComment && newComment !== original) {
          const response = await this.fetchApi(
            `/annotations/${encodeURIComponent(ann.id)}`,
            {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ comment: newComment }),
            }
          )
          if (response) {
            showToast('Annotation updated', 'success', 2000)
            await this.options.onRefresh()
            return
          }
        }
      }
      commentEl.textContent = original
    }

    saveBtn.addEventListener('click', () => finishEdit(true))
    cancelBtn.addEventListener('click', () => finishEdit(false))
    textarea.addEventListener('keydown', (e: KeyboardEvent) => {
      e.stopPropagation()
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        finishEdit(true)
      }
      if (e.key === 'Escape') {
        finishEdit(false)
      }
    })

    requestAnimationFrame(() => {
      textarea.focus()
      textarea.select()
    })
  }

  // ── CRUD ───────────────────────────────────────────────────────────

  private async deleteAnnotation(id: string): Promise<void> {
    const response = await this.fetchApi(
      `/annotations/${encodeURIComponent(id)}`,
      { method: 'DELETE' }
    )

    if (response) {
      showToast('Annotation deleted', 'success', 2000)
      await this.options.onRefresh()
    }
  }

  private async clearAll(): Promise<void> {
    const results = await Promise.all(
      this.annotations.map(ann =>
        this.fetchApi(
          `/annotations/${encodeURIComponent(ann.id)}`,
          { method: 'DELETE' }
        )
      )
    )

    if (results.some(r => r === null)) {
      showToast('Some annotations failed to delete', 'error')
      return
    }

    showToast('All annotations cleared', 'success', 2000)
    await this.options.onRefresh()
  }

  // ── Footer ─────────────────────────────────────────────────────────

  private renderFooter(): void {
    if (this.options.hideFooter || this.annotations.length === 0) {
      this.footerEl.innerHTML = ''
      return
    }

    const count = this.annotations.length
    const fiberBtn = this.options.onFileAsFiber
      ? `<button class="ann-footer-btn ann-footer-fiber">File as fiber</button>`
      : ''

    this.footerEl.innerHTML = `
      <button class="ann-footer-btn ann-footer-send">Send ${count} annotation${count === 1 ? '' : 's'} to worker</button>
      ${fiberBtn}
    `

    this.footerEl.querySelector('.ann-footer-send')?.addEventListener('click', () => {
      this.showWorkerPickerUI()
    })

    this.footerEl.querySelector('.ann-footer-fiber')?.addEventListener('click', () => {
      this.options.onFileAsFiber?.(this.annotations)
    })
  }

  private showWorkerPickerUI(): void {
    const workers = this.options.getWorkers()

    showWorkerPicker(workers, this.annotations.length, {
      onSelectWorker: (workerId) =>
        this.options.onSendToWorker(this.annotations, workerId),
      onNewWorker: () =>
        this.options.onSendToWorker(this.annotations, undefined, true),
    })
  }

  // ── Fetch helper ───────────────────────────────────────────────────

  private async fetchApi(path: string, init?: RequestInit): Promise<Response | null> {
    try {
      const response = await fetch(`${API_BASE}${path}`, init)
      if (!response.ok) {
        console.error(`API error ${path}:`, await response.text())
        showToast('Request failed', 'error')
        return null
      }
      return response
    } catch (err) {
      console.error(`API error ${path}:`, err)
      showToast('Request failed', 'error')
      return null
    }
  }
}
