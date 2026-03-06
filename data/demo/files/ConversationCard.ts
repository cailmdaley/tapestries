// ConversationCard.ts - Map-pinned conversation card as CSS2DObject

import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import type { Session, ConversationMessage } from '../state/types'
import { escapeHtml, formatTimeAgo, renderMarkdown, highlightCodeBlocks } from './utils'

type FileClickCallback = (fullPath: string, originId: string, workerId: string) => void

interface CardOptions {
  onClose: () => void
  onFileClick?: FileClickCallback
  onDoubleClick?: () => void  // For focusing terminal
  onBringToFront?: () => void  // When card is clicked/focused
  onSwarmDrag?: (dx: number, dz: number) => void  // Move swarm in world space
  initialSize?: { width: number; height: number }  // Saved card size
}

export class ConversationCard {
  readonly object: CSS2DObject
  private element: HTMLElement
  private contentEl: HTMLElement
  private session: Session
  private conversation: ConversationMessage[] = []
  private expandedMessages: Set<string> = new Set()
  private options: CardOptions
  private disposed = false

  // Loading state
  private loadingState: 'loading' | 'loaded' | 'error' = 'loading'
  private errorMessage = ''
  private fetchTimeout: ReturnType<typeof setTimeout> | null = null

  // Clickable tools for file viewer
  private readonly clickableTools = ['Read', 'Write', 'Edit']

  // Card interaction (dragging moves swarm, not card)
  private isDragging = false
  private dragStart = { x: 0, y: 0 }

  // Resize state (only bottom-right resize supported)
  private isResizing = false
  private resizeStart = { x: 0, y: 0, width: 0, height: 0 }

  // Scale state (for combining with drag transform)
  private currentScale = 1

  // Minimize state
  private isMinimized = false

  // Chat input state
  private chatInputEl: HTMLTextAreaElement | null = null
  private isSending = false

  constructor(session: Session, options: CardOptions) {
    this.session = session
    this.options = options

    // Create wrapper and card elements
    // CSS2D controls the wrapper's transform, we control the card's transform separately
    const { wrapper, card } = this.createCardElements()
    this.element = card
    this.contentEl = card.querySelector('.card-content')!
    this.chatInputEl = card.querySelector('.chat-input')

    this.object = new CSS2DObject(wrapper)
    // Position set by ZoneRenderer.openConversationCard()

    this.setupEventListeners()
    this.applyTransform()  // Apply initial offset transform
    this.fetchConversation()
  }

  private createCardElements(): { wrapper: HTMLElement; card: HTMLElement } {
    // Wrapper is what CSS2DRenderer transforms - must not have our transforms
    const wrapper = document.createElement('div')
    wrapper.className = 'conversation-card-wrapper'

    // Card element is what we transform for drag/scale
    const card = document.createElement('div')
    card.className = 'conversation-card'
    card.innerHTML = `
      <div class="card-header">
        <span class="card-title">${escapeHtml(this.session.name)}</span>
        <div class="card-recent-files"></div>
        <button class="card-close" title="Close">&times;</button>
      </div>
      <div class="card-content">
        <div class="card-loading">Loading conversation...</div>
      </div>
      <div class="chat-input-container">
        <textarea class="chat-input" placeholder="Send a message..." rows="1"></textarea>
        <button class="chat-send-btn" title="Send (Enter)">↩</button>
      </div>
      <div class="resize-corner resize-se"></div>
    `

    wrapper.appendChild(card)

    // Apply initial size if provided (from localStorage persistence)
    // Only apply saved sizes if user explicitly made the card larger than CSS defaults
    // CSS defaults: 580px width, 520px height - don't restore smaller sizes
    if (this.options.initialSize) {
      const { width, height } = this.options.initialSize
      if (width > 580) {
        card.style.width = `${width}px`
      }
      if (height > 520 && height !== 600) {  // 600 was old default, don't restore it
        card.style.height = `${height}px`
      }
    }

    return { wrapper, card }
  }

  private setupEventListeners(): void {
    // Close button
    const closeBtn = this.element.querySelector('.card-close')!
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation()
      this.options.onClose()
    })

    // Double-click to focus terminal
    this.element.addEventListener('dblclick', (e) => {
      e.stopPropagation()
      this.options.onDoubleClick?.()
    })

    // Bring to front on any click
    this.element.addEventListener('mousedown', () => {
      this.options.onBringToFront?.()
    })

    // Stop click propagation to prevent map interaction
    this.element.addEventListener('click', (e) => {
      e.stopPropagation()
    })

    // Prevent scroll from bubbling to map
    this.element.addEventListener('wheel', (e) => {
      e.stopPropagation()
    })

    // Drag via header
    const header = this.element.querySelector('.card-header') as HTMLElement
    header.style.cursor = 'move'
    header.addEventListener('mousedown', this.startDrag)

    // Resize via corner handle (bottom-right only)
    const resizeHandle = this.element.querySelector('.resize-se')
    resizeHandle?.addEventListener('mousedown', (e) => this.startResize(e as MouseEvent))

    // Chat input
    if (this.chatInputEl) {
      const sendBtn = this.element.querySelector('.chat-send-btn')!
      sendBtn.addEventListener('click', (e) => {
        e.stopPropagation()
        this.sendMessage()
      })

      // Enter sends, Shift+Enter for newline
      this.chatInputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault()
          this.sendMessage()
        }
      })

      // Auto-grow textarea
      this.chatInputEl.addEventListener('input', () => this.autoGrowTextarea())

      // Prevent card close when clicking in chat area
      this.chatInputEl.addEventListener('click', (e) => e.stopPropagation())
      this.chatInputEl.addEventListener('dblclick', (e) => e.stopPropagation())
    }
  }

  private startDrag = (e: MouseEvent): void => {
    // Don't drag if clicking close button
    if ((e.target as HTMLElement).classList.contains('card-close')) return

    // If minimized, restore on click instead of drag
    if (this.isMinimized) {
      this.restore()
      return
    }

    e.preventDefault()
    e.stopPropagation()
    this.options.onBringToFront?.()
    this.isDragging = true
    this.dragStart = { x: e.clientX, y: e.clientY }

    document.addEventListener('mousemove', this.onDrag)
    document.addEventListener('mouseup', this.stopDrag)
  }

  private onDrag = (e: MouseEvent): void => {
    if (!this.isDragging || !this.options.onSwarmDrag) return

    // Move swarm in world space (convert pixels to world units roughly)
    // Approximate: 100 pixels ≈ 1 world unit at default zoom
    const scale = 0.01
    const dx = (e.clientX - this.dragStart.x) * scale
    const dz = (e.clientY - this.dragStart.y) * scale
    this.dragStart = { x: e.clientX, y: e.clientY }
    this.options.onSwarmDrag(dx, dz)
  }

  private applyTransform(): void {
    // CSS2DRenderer centers wrapper at anchor point. Shift card up by 50% so bottom is at anchor.
    this.element.style.transform = `translateY(-50%) scale(${this.currentScale})`
  }

  private stopDrag = (): void => {
    this.isDragging = false
    document.removeEventListener('mousemove', this.onDrag)
    document.removeEventListener('mouseup', this.stopDrag)
  }

  private startResize = (e: MouseEvent): void => {
    e.preventDefault()
    e.stopPropagation()
    this.isResizing = true
    this.resizeStart = {
      x: e.clientX,
      y: e.clientY,
      width: this.element.offsetWidth,
      height: this.element.offsetHeight,
    }

    document.addEventListener('mousemove', this.onResize)
    document.addEventListener('mouseup', this.stopResize)
  }

  private onResize = (e: MouseEvent): void => {
    if (!this.isResizing) return

    const dx = e.clientX - this.resizeStart.x
    const dy = e.clientY - this.resizeStart.y

    const newWidth = Math.max(300, this.resizeStart.width + dx)
    const newHeight = Math.max(200, this.resizeStart.height + dy)

    this.element.style.width = `${newWidth}px`
    this.element.style.maxHeight = `${newHeight}px`
  }

  private stopResize = (): void => {
    this.isResizing = false
    document.removeEventListener('mousemove', this.onResize)
    document.removeEventListener('mouseup', this.stopResize)
  }

  private autoGrowTextarea(): void {
    const input = this.chatInputEl
    if (!input) return

    const maxHeight = 80
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, maxHeight)}px`
    input.style.overflowY = input.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }

  private async sendMessage(): Promise<void> {
    const input = this.chatInputEl
    if (!input || this.isSending) return

    const message = input.value.trim()
    if (!message) return

    const sendBtn = this.element.querySelector('.chat-send-btn') as HTMLButtonElement
    this.isSending = true
    sendBtn.textContent = '...'
    input.disabled = true

    try {
      const response = await fetch('http://localhost:4004/send-message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: this.session.id, message }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || `HTTP ${response.status}`)
      }

      input.value = ''
      input.style.height = 'auto'
      await this.fetchConversation()
    } catch (error) {
      console.error('Failed to send message:', error)
      const originalPlaceholder = input.placeholder
      input.placeholder = `Error: ${error instanceof Error ? error.message : 'Failed to send'}`
      setTimeout(() => { input.placeholder = originalPlaceholder }, 3000)
    } finally {
      this.isSending = false
      sendBtn.textContent = '↩'
      input.disabled = false
      input.focus()
    }
  }

  private async fetchConversation(): Promise<void> {
    if (this.disposed) return

    this.loadingState = 'loading'
    this.renderContent()

    // Set timeout for loading
    const TIMEOUT_MS = 5000
    this.fetchTimeout = setTimeout(() => {
      if (this.loadingState === 'loading') {
        this.loadingState = 'error'
        this.errorMessage = 'Request timed out'
        this.renderContent()
      }
    }, TIMEOUT_MS)

    try {
      const url = new URL('http://localhost:4004/conversation')
      url.searchParams.set('sessionId', this.session.id)
      url.searchParams.set('tmuxSession', this.session.tmuxSession)
      url.searchParams.set('limit', '100')

      const response = await fetch(url.toString())

      if (this.disposed) return

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = await response.json()
      this.clearFetchTimeout()

      this.conversation = Array.isArray(data.messages) ? data.messages : []
      this.loadingState = 'loaded'
      this.renderContent()
    } catch (error) {
      if (this.disposed) return
      this.clearFetchTimeout()

      this.loadingState = 'error'
      this.errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.renderContent()
    }
  }

  private clearFetchTimeout(): void {
    if (this.fetchTimeout) {
      clearTimeout(this.fetchTimeout)
      this.fetchTimeout = null
    }
  }

  /**
   * Handle WebSocket conversation update.
   * Appends new messages incrementally. PostToolUse hook provides mid-turn
   * updates; Stop provides assistant text at end of turn.
   */
  handleMessage(sessionId: string | undefined, _tmuxSession: string, messages: ConversationMessage[]): void {
    if (this.disposed) return

    // Strict sessionId matching only — no tmux fallback.
    // The tmux fallback caused cross-contamination: after disconnects, empty cards
    // with matching tmux names would accept messages from other sessions.
    // Cards that haven't received hooks yet get their data via fetchConversation().
    if (!sessionId || sessionId !== this.session.id) return

    // Deduplicate by timestamp and toolUseId
    const existingTimestamps = new Set(this.conversation.map(m => m.timestamp))
    const existingToolUseIds = new Set(
      this.conversation.filter(m => m.toolUseId).map(m => m.toolUseId)
    )
    const toAdd = messages.filter(m => {
      if (m.toolUseId && existingToolUseIds.has(m.toolUseId)) return false
      return !existingTimestamps.has(m.timestamp)
    })

    if (toAdd.length > 0) {
      this.conversation.push(...toAdd)
      if (this.conversation.length > 100) {
        this.conversation = this.conversation.slice(-100)
      }
      this.loadingState = 'loaded'
      this.renderContent()
    }
  }

  private renderContent(): void {
    if (this.disposed) return

    switch (this.loadingState) {
      case 'loading':
        this.contentEl.innerHTML = '<div class="card-loading">Loading conversation...</div>'
        return

      case 'error':
        this.contentEl.innerHTML = `
          <div class="card-error">
            <span>${escapeHtml(this.errorMessage)}</span>
            <button class="retry-btn">Retry</button>
          </div>
        `
        this.contentEl.querySelector('.retry-btn')?.addEventListener('click', (e) => {
          e.stopPropagation()
          this.fetchConversation()
        })
        return

      case 'loaded':
        break
    }

    if (this.conversation.length === 0) {
      this.contentEl.innerHTML = '<div class="card-empty">No conversation yet</div>'
      return
    }

    // Show last few exchanges (user + assistant pairs)
    const recentMessages = this.getRecentExchanges(3)
    const groups = this.buildMessageGroups(recentMessages)
    const html = groups.map(group => this.renderGroup(group)).join('')

    this.contentEl.innerHTML = html
    this.attachListeners()
    highlightCodeBlocks(this.contentEl)

    // Update recent files in header
    this.updateRecentFiles()

    // Auto-scroll to bottom to show most recent messages
    this.contentEl.scrollTop = this.contentEl.scrollHeight
  }

  /**
   * Get last N exchanges (user/assistant pairs, including tool groups between them)
   */
  private getRecentExchanges(count: number): ConversationMessage[] {
    // Walk backward, counting user messages as exchange boundaries
    const result: ConversationMessage[] = []
    let exchangeCount = 0

    for (let i = this.conversation.length - 1; i >= 0 && exchangeCount < count; i--) {
      const msg = this.conversation[i]
      result.unshift(msg)
      if (msg.type === 'user') {
        exchangeCount++
      }
    }

    return result
  }

  private buildMessageGroups(messages: ConversationMessage[]): MessageGroup[] {
    // Build tool_use_id -> tool_result map
    const toolResultMap = new Map<string, ConversationMessage>()
    for (const msg of messages) {
      if (msg.type === 'tool_result' && msg.toolUseId) {
        toolResultMap.set(msg.toolUseId, msg)
      }
    }

    const groups: MessageGroup[] = []
    let currentToolGroup: ToolGroup | null = null

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i]

      if (msg.type === 'tool_result') continue // Rendered with tool_use

      if (msg.type === 'user' || msg.type === 'assistant') {
        if (currentToolGroup) {
          groups.push(currentToolGroup)
          currentToolGroup = null
        }
        groups.push({ type: 'message', msg })
      } else {
        // thinking, tool_use, system - add to tool group
        if (!currentToolGroup) {
          currentToolGroup = { type: 'tool_group', items: [], startIndex: i }
        }
        const result = msg.type === 'tool_use' && msg.toolUseId
          ? toolResultMap.get(msg.toolUseId)
          : undefined
        currentToolGroup.items.push({ msg, result })
      }
    }

    if (currentToolGroup) {
      groups.push(currentToolGroup)
    }

    return groups
  }

  private renderGroup(group: MessageGroup): string {
    if (group.type === 'message') {
      return this.renderMessage(group.msg)
    }
    return this.renderToolGroup(group)
  }

  private renderMessage(msg: ConversationMessage): string {
    if (msg.type !== 'user' && msg.type !== 'assistant') return ''

    const timeAgo = formatTimeAgo(new Date(msg.timestamp).getTime())
    const content = renderMarkdown(msg.content)
    const msgClass = msg.type === 'user' ? 'user-msg' : 'assistant-msg'

    return `
      <div class="card-msg ${msgClass}">
        <div class="msg-content markdown-content">${content}</div>
        <span class="msg-time">${timeAgo}</span>
      </div>
    `
  }

  private renderToolGroup(group: ToolGroup): string {
    const { items } = group

    // Single-item groups expand directly (per spec)
    if (items.length === 1) {
      const { msg, result } = items[0]
      return this.renderToolItem(msg, result, msg.timestamp)
    }

    // Multi-item groups get a collapsible header
    const summary = this.buildGroupSummary(items)
    const groupKey = `group-${items[0]?.msg.timestamp || group.startIndex}`
    const isExpanded = this.expandedMessages.has(groupKey)

    const itemsHtml = items.map(({ msg, result }) =>
      this.renderToolItem(msg, result, msg.timestamp)
    ).join('')

    return `
      <div class="card-tool-group ${isExpanded ? 'expanded' : ''}" data-group-key="${groupKey}">
        <div class="tool-group-header">
          <span class="expand-icon">▶</span>
          <span class="tool-group-summary">${items.length} steps (${escapeHtml(summary)})</span>
        </div>
        <div class="tool-group-content">${itemsHtml}</div>
      </div>
    `
  }

  private buildGroupSummary(items: ToolGroup['items']): string {
    const toolCounts = new Map<string, number>()
    let thinkingCount = 0

    for (const { msg } of items) {
      if (msg.type === 'thinking') {
        thinkingCount++
      } else if (msg.type === 'tool_use') {
        const name = msg.toolName || 'Tool'
        toolCounts.set(name, (toolCounts.get(name) || 0) + 1)
      }
    }

    const parts: string[] = []
    if (thinkingCount > 0) parts.push(`${thinkingCount} thinking`)
    for (const [name, count] of toolCounts) {
      parts.push(`${count} ${name}`)
    }
    return parts.join(', ')
  }

  private renderToolItem(msg: ConversationMessage, result?: ConversationMessage, key?: string): string {
    const timeAgo = formatTimeAgo(new Date(msg.timestamp).getTime())
    const msgKey = key || msg.timestamp
    const isExpanded = this.expandedMessages.has(msgKey)

    if (msg.type === 'thinking') {
      const preview = msg.preview || this.truncate(msg.content, 40)
      return `
        <div class="card-tool-item thinking ${isExpanded ? 'expanded' : ''}" data-msg-key="${msgKey}">
          <div class="tool-header">
            <span class="expand-icon">▶</span>
            <span class="tool-preview">${escapeHtml(preview)}</span>
          </div>
          <div class="tool-content">${escapeHtml(msg.content)}</div>
          <span class="msg-time">${timeAgo}</span>
        </div>
      `
    }

    if (msg.type === 'tool_use') {
      const toolName = msg.toolName || 'Tool'
      const summary = this.getToolSummary(msg)
      const fullInput = this.formatToolInput(msg.toolInput)
      const resultOutput = result?.content ?? ''
      const hasDetails = fullInput || resultOutput

      const isClickable = this.isClickableTool(msg)
      const openBtn = isClickable
        ? `<button class="tool-open-btn" data-tool-key="${msgKey}">Open</button>`
        : ''

      return `
        <div class="card-tool-item ${isExpanded ? 'expanded' : ''}" data-msg-key="${msgKey}">
          <div class="tool-header">
            ${hasDetails ? '<span class="expand-icon">▶</span>' : ''}
            <span class="tool-badge">${toolName}</span>
            <span class="tool-summary">${escapeHtml(summary)}</span>
            ${openBtn}
          </div>
          ${hasDetails ? `
          <div class="tool-content">
            ${fullInput ? `<div class="tool-input">${escapeHtml(fullInput)}</div>` : ''}
            ${resultOutput ? `
              <div class="tool-output-label">Output:</div>
              <div class="tool-output">${escapeHtml(resultOutput)}</div>
            ` : ''}
          </div>
          ` : ''}
          <span class="msg-time">${timeAgo}</span>
        </div>
      `
    }

    if (msg.type === 'system') {
      const preview = msg.preview || this.truncate(msg.content, 40)
      const badge = msg.systemType === 'skill' ? 'Skill' : 'System'
      return `
        <div class="card-tool-item system ${isExpanded ? 'expanded' : ''}" data-msg-key="${msgKey}">
          <div class="tool-header">
            <span class="expand-icon">▶</span>
            <span class="tool-badge">${badge}</span>
            <span class="tool-preview">${escapeHtml(preview)}</span>
          </div>
          <div class="tool-content">${escapeHtml(msg.content)}</div>
          <span class="msg-time">${timeAgo}</span>
        </div>
      `
    }

    return ''
  }

  private attachListeners(): void {
    // Tool group headers (multi-item groups)
    this.contentEl.querySelectorAll('[data-group-key]').forEach(el => {
      const header = el.querySelector('.tool-group-header')
      header?.addEventListener('click', (e) => {
        e.stopPropagation()
        const key = (el as HTMLElement).dataset.groupKey!
        this.toggleExpanded(key)
      })
    })

    // Individual expandable items
    this.contentEl.querySelectorAll('[data-msg-key]').forEach(el => {
      el.addEventListener('click', (e) => {
        // Don't expand if clicking open button
        if ((e.target as HTMLElement).classList.contains('tool-open-btn')) return
        e.stopPropagation()
        const key = (el as HTMLElement).dataset.msgKey!
        this.toggleExpanded(key)
      })
    })

    // File open buttons
    this.contentEl.querySelectorAll('[data-tool-key]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation()
        const key = (btn as HTMLElement).dataset.toolKey!
        const msg = this.conversation.find(m => m.timestamp === key)
        if (msg && this.options.onFileClick) {
          const filePath = this.getToolFilePath(msg)
          if (filePath) {
            this.options.onFileClick(filePath, this.session.originId, this.session.id)
          }
        }
      })
    })
  }

  private toggleExpanded(key: string): void {
    if (this.expandedMessages.has(key)) {
      this.expandedMessages.delete(key)
    } else {
      this.expandedMessages.add(key)
    }
    this.renderContent()
  }

  private getToolSummary(msg: ConversationMessage): string {
    const input = msg.toolInput
    if (!input) return msg.content

    const filePath = this.getToolFilePath(msg)
    if (filePath) return filePath
    if (input.command) return this.truncate(String(input.command), 40)
    if (input.pattern) return String(input.pattern)

    return msg.content || msg.toolName || 'tool'
  }

  private isClickableTool(msg: ConversationMessage): boolean {
    if (!msg.toolName || !this.clickableTools.includes(msg.toolName)) return false
    return !!this.getToolFilePath(msg)
  }

  private getToolFilePath(msg: ConversationMessage): string | undefined {
    return msg.toolInput?.file_path ?? msg.toolInput?.path
  }

  private formatToolInput(input: ConversationMessage['toolInput']): string {
    if (!input) return ''
    if (typeof input === 'string') return input
    return Object.entries(input)
      .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
      .join('\n')
  }

  private truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) return text
    return text.slice(0, maxLen) + '...'
  }

  /**
   * Extract and render recent files from conversation tool calls.
   * Shows up to 3 most recently accessed files in the header.
   */
  private updateRecentFiles(): void {
    const container = this.element.querySelector('.card-recent-files')
    if (!container) return

    // Build path -> timestamp map directly (later timestamps overwrite earlier)
    const pathTimestamps = new Map<string, string>()
    for (const msg of this.conversation) {
      if (msg.type !== 'tool_use') continue
      if (!this.clickableTools.includes(msg.toolName || '')) continue

      const filePath = this.getToolFilePath(msg)
      if (filePath) {
        pathTimestamps.set(filePath, msg.timestamp)
      }
    }

    // Sort by timestamp descending, take top 3
    const recentFiles = [...pathTimestamps.entries()]
      .sort((a, b) => b[1].localeCompare(a[1]))
      .slice(0, 3)
      .map(([path]) => path)

    if (recentFiles.length === 0) {
      container.innerHTML = ''
      return
    }

    // Render file chips
    container.innerHTML = recentFiles.map(path => {
      const filename = path.split('/').pop() || path
      const truncated = filename.length > 15 ? filename.slice(0, 12) + '...' : filename
      return `<span class="recent-file-chip" data-path="${escapeHtml(path)}" title="${escapeHtml(path)}">${escapeHtml(truncated)}</span>`
    }).join('')

    // Attach click handlers
    container.querySelectorAll('.recent-file-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation()
        const path = (chip as HTMLElement).dataset.path
        if (path && this.options.onFileClick) {
          this.options.onFileClick(path, this.session.originId, this.session.id)
        }
      })
    })
  }

  /**
   * Update scale based on camera distance (for zoom clamping)
   */
  setScale(scale: number): void {
    // Clamp scale for readability (card base width is 580px)
    const minScale = 0.6   // 580 * 0.6 = 348px at far zoom
    const maxScale = 1.0   // 580 * 1.0 = 580px at close zoom
    this.currentScale = Math.max(minScale, Math.min(maxScale, scale))
    this.applyTransform()
  }

  /**
   * Update viewport clamping (call every frame after CSS2DRenderer updates wrapper position)
   */
  updateViewportClamp(): void {
    this.applyTransform()
  }

  /**
   * Re-fetch conversation from server (called on WebSocket reconnect)
   */
  refetch(): void {
    this.fetchConversation()
  }

  get workerId(): string {
    return this.session.id
  }

  get tmuxSession(): string {
    return this.session.tmuxSession
  }

  /**
   * Get the prefixed tmuxSession for matching WebSocket messages
   * Remote sessions use "originId/tmuxSession" format in conversation cache
   */
  get prefixedTmuxSession(): string {
    return this.session.originId === 'local'
      ? this.session.tmuxSession
      : `${this.session.originId}/${this.session.tmuxSession}`
  }

  /**
   * Set the card's z-index (for layering when overlapping)
   * Stores in data attribute so it can be reapplied after CSS2DRenderer overwrites
   */
  setZIndex(zIndex: number): void {
    const wrapper = this.element.parentElement
    if (wrapper) {
      // Store our z-index in data attribute (CSS2DRenderer won't touch this)
      wrapper.dataset.portolanZIndex = String(zIndex)
      // Apply immediately (will be overwritten by CSS2DRenderer, then reapplied)
      wrapper.style.setProperty('z-index', String(zIndex), 'important')
    }
  }

  /**
   * Get current size
   */
  getSize(): { width: number; height: number } {
    return {
      width: this.element.offsetWidth,
      height: this.element.offsetHeight,
    }
  }

  /**
   * Minimize the card
   */
  minimize(): void {
    this.isMinimized = true
    this.element.classList.add('minimized')
  }

  /**
   * Restore the card from minimized state
   */
  restore(): void {
    this.isMinimized = false
    this.element.classList.remove('minimized')
  }

  /**
   * Check if card is minimized
   */
  get minimized(): boolean {
    return this.isMinimized
  }

  dispose(): void {
    this.disposed = true
    this.clearFetchTimeout()
    // Clean up drag/resize listeners
    document.removeEventListener('mousemove', this.onDrag)
    document.removeEventListener('mouseup', this.stopDrag)
    document.removeEventListener('mousemove', this.onResize)
    document.removeEventListener('mouseup', this.stopResize)
    // Remove wrapper (parent of card element)
    const wrapper = this.element.parentElement
    if (wrapper) {
      wrapper.remove()
    } else {
      this.element.remove()
    }
  }
}

// Types for message grouping
type MessageGroup =
  | { type: 'message'; msg: ConversationMessage }
  | ToolGroup

interface ToolGroup {
  type: 'tool_group'
  items: Array<{ msg: ConversationMessage; result?: ConversationMessage }>
  startIndex: number
}
