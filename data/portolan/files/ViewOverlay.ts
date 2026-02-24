// ViewOverlay.ts - Fullscreen overlay for Plots and Plans views (iframe-based)

export class ViewOverlay {
  private overlay: HTMLElement
  private iframe: HTMLIFrameElement
  private loadingIndicator: HTMLElement

  constructor() {
    this.overlay = this.createOverlay()
    this.iframe = this.overlay.querySelector('iframe')!
    this.loadingIndicator = this.overlay.querySelector('.loading-indicator')!
    document.body.appendChild(this.overlay)
  }

  private createOverlay(): HTMLElement {
    const overlay = document.createElement('div')
    overlay.className = 'view-overlay'
    overlay.innerHTML = `
      <div class="view-overlay-frame">
        <div class="loading-indicator">Loading...</div>
        <iframe src="about:blank" frameborder="0"></iframe>
      </div>
    `
    return overlay
  }

  show(url: string): void {
    // Show loading indicator
    this.loadingIndicator.style.display = 'flex'
    this.iframe.style.opacity = '0'

    // Set iframe src
    this.iframe.src = url

    // Hide loading when iframe loads
    this.iframe.onload = () => {
      this.loadingIndicator.style.display = 'none'
      this.iframe.style.opacity = '1'
    }

    // Show overlay with animation
    this.overlay.classList.add('visible')
  }

  hide(): void {
    this.overlay.classList.remove('visible')

    // Clear iframe after animation
    setTimeout(() => {
      if (!this.overlay.classList.contains('visible')) {
        this.iframe.src = 'about:blank'
      }
    }, 300)
  }

  isVisible(): boolean {
    return this.overlay.classList.contains('visible')
  }

  dispose(): void {
    this.hide()
    this.overlay.remove()
  }
}
