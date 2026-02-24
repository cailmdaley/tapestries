// ContextMenu.ts - Right-click context menu for hex grid

export interface MenuItem {
  label: string
  action: () => void
  disabled?: boolean
  danger?: boolean
}

export class ContextMenu {
  private menu: HTMLElement
  private closeHandler: ((e: MouseEvent) => void) | null = null

  constructor() {
    this.menu = this.createMenu()
    document.body.appendChild(this.menu)
  }

  private createMenu(): HTMLElement {
    const menu = document.createElement('div')
    menu.className = 'context-menu'
    menu.style.cssText = `
      position: fixed;
      display: none;
      min-width: 160px;
      background: var(--bg-card, #EDE8E0);
      border: 1px solid var(--border, #8B7355);
      border-radius: 4px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 1000;
      padding: 4px 0;
      font-family: 'EB Garamond', Garamond, serif;
      font-size: 14px;
    `
    return menu
  }

  show(x: number, y: number, items: MenuItem[]): void {
    // Clear previous items
    this.menu.innerHTML = ''

    for (const item of items) {
      const menuItem = document.createElement('div')
      menuItem.className = 'context-menu-item'
      menuItem.textContent = item.label

      const dangerColor = '#A03030'
      const normalColor = item.danger ? dangerColor : 'var(--text-primary, #2E2A26)'
      const hoverBg = item.danger ? dangerColor : 'var(--gold, #9A7B35)'

      menuItem.style.cssText = `
        padding: 8px 16px;
        cursor: ${item.disabled ? 'default' : 'pointer'};
        color: ${item.disabled ? 'var(--text-muted, #7A7368)' : normalColor};
        opacity: ${item.disabled ? '0.5' : '1'};
        transition: background 0.1s;
      `

      if (!item.disabled) {
        menuItem.addEventListener('mouseenter', () => {
          menuItem.style.background = hoverBg
          menuItem.style.color = 'white'
        })
        menuItem.addEventListener('mouseleave', () => {
          menuItem.style.background = ''
          menuItem.style.color = normalColor
        })
        menuItem.addEventListener('click', (e) => {
          e.stopPropagation()
          this.hide()
          item.action()
        })
      }

      this.menu.appendChild(menuItem)
    }

    // Position menu, keeping it on screen
    this.menu.style.display = 'block'
    const menuRect = this.menu.getBoundingClientRect()

    let posX = x
    let posY = y

    if (x + menuRect.width > window.innerWidth) {
      posX = window.innerWidth - menuRect.width - 8
    }
    if (y + menuRect.height > window.innerHeight) {
      posY = window.innerHeight - menuRect.height - 8
    }

    this.menu.style.left = `${posX}px`
    this.menu.style.top = `${posY}px`

    // Add close handler (click outside to close)
    this.removeCloseHandler()
    this.closeHandler = (e: MouseEvent) => {
      if (!this.menu.contains(e.target as Node)) {
        this.hide()
      }
    }
    document.addEventListener('click', this.closeHandler)
  }

  private removeCloseHandler(): void {
    if (this.closeHandler) {
      document.removeEventListener('click', this.closeHandler)
      this.closeHandler = null
    }
  }

  hide(): void {
    this.menu.style.display = 'none'
    this.removeCloseHandler()
  }

  dispose(): void {
    this.hide()
    this.menu.remove()
  }
}
