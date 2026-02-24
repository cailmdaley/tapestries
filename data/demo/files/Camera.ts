// Camera.ts - Orthographic camera with pan/zoom controls (isometric-style)

import { OrthographicCamera, Vector3, Raycaster, Plane, Vector2 } from 'three'
import type { CartesianCoord } from '../state/types'

// Safari-specific gesture event (pinch-to-zoom)
interface GestureEvent extends Event {
  scale: number
  rotation: number
}

export class Camera {
  camera: OrthographicCamera
  private canvas: HTMLCanvasElement
  private eventTarget: HTMLElement  // May be overlay for Safari compatibility

  // Camera state - target point we're looking at
  private target = new Vector3(0, 0, 0)
  private zoom = 30  // View half-width in world units (smaller = more zoomed in)
  private angle = Math.PI / 4  // 45° from horizontal (matches sprite perspective)
  private rotation = 0  // 0° around Y axis (straight-on view)

  // Zoom limits (smaller = more zoomed in)
  private minZoom = 2
  private maxZoom = 15

  // Pan state
  private isDragging = false
  private wasDrag = false // Set true if mouse moved significantly during drag
  private dragStart = { x: 0, y: 0 }
  private dragAnchor: CartesianCoord | null = null // World point to keep under mouse
  private readonly dragThreshold = 5 // pixels

  // Raycasting for accurate screen-to-world conversion
  private raycaster = new Raycaster()
  private groundPlane = new Plane(new Vector3(0, 1, 0), 0) // Y=0 plane

  // Stored listeners for cleanup (HMR)
  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null
  private mouseUpHandler: (() => void) | null = null
  private keyDownHandler: ((e: KeyboardEvent) => void) | null = null
  // EventTarget (canvas) listeners
  private mouseDownHandler: ((e: MouseEvent) => void) | null = null
  private clickHandler: (() => void) | null = null
  private wheelHandler: ((e: WheelEvent) => void) | null = null
  private gestureStartHandler: ((e: Event) => void) | null = null
  private gestureChangeHandler: ((e: Event) => void) | null = null
  private gestureEndHandler: ((e: Event) => void) | null = null

  constructor(canvas: HTMLCanvasElement, eventTarget?: HTMLElement) {
    this.canvas = canvas
    this.eventTarget = eventTarget || canvas

    // Create orthographic camera (isometric-style, no perspective distortion)
    const aspect = canvas.clientWidth / canvas.clientHeight
    this.camera = new OrthographicCamera(
      -this.zoom * aspect, this.zoom * aspect,  // left, right
      this.zoom, -this.zoom,                     // top, bottom
      0.1, 1000                                  // near, far
    )

    this.setupControls()
    this.updateCamera()
  }

  private setupControls(): void {
    // Mouse drag for pan (sieve behavior)
    this.mouseDownHandler = (e: MouseEvent) => {
      if (e.button === 0) { // Left click
        this.isDragging = true
        this.wasDrag = false
        this.dragStart = { x: e.clientX, y: e.clientY }
        // Store the world point under the mouse — this stays fixed during drag
        this.dragAnchor = this.screenToWorld(e.clientX, e.clientY)
      }
    }
    this.eventTarget.addEventListener('mousedown', this.mouseDownHandler)

    this.mouseMoveHandler = (e: MouseEvent) => {
      if (!this.isDragging || !this.dragAnchor) return

      // Check if we've moved enough from start to count as a drag
      const totalDx = e.clientX - this.dragStart.x
      const totalDy = e.clientY - this.dragStart.y
      if (Math.abs(totalDx) > this.dragThreshold || Math.abs(totalDy) > this.dragThreshold) {
        this.wasDrag = true
      }

      // Sieve: keep the anchor world point under the mouse
      // 1. Where does the mouse currently point in world space?
      const currentWorld = this.screenToWorld(e.clientX, e.clientY)
      // 2. Move target so anchor stays under mouse
      this.target.x += this.dragAnchor.x - currentWorld.x
      this.target.z += this.dragAnchor.z - currentWorld.z
      this.updateCamera()
    }
    window.addEventListener('mousemove', this.mouseMoveHandler)

    this.mouseUpHandler = () => {
      this.isDragging = false
      this.dragAnchor = null
      // wasDrag is kept until click handler checks it
    }
    window.addEventListener('mouseup', this.mouseUpHandler)

    // Reset wasDrag after click has had a chance to check it
    this.clickHandler = () => {
      // Use setTimeout to reset after current click event fully processes
      setTimeout(() => { this.wasDrag = false }, 0)
    }
    this.eventTarget.addEventListener('click', this.clickHandler)

    // Scroll wheel for zoom
    this.wheelHandler = (e: WheelEvent) => {
      // Don't capture wheel events over panels - let them scroll
      const target = e.target as HTMLElement
      if (target.closest('#city-panel')) {
        return // Let panel handle scroll
      }
      e.preventDefault()
      e.stopPropagation()
      const delta = e.deltaY > 0 ? 1.02 : 0.98  // Gentle zoom
      this.zoomBy(delta)
    }
    this.eventTarget.addEventListener('wheel', this.wheelHandler, { passive: false })

    // Safari pinch-to-zoom (gesture events)
    let lastScale = 1
    this.gestureStartHandler = (e: Event) => {
      e.preventDefault()
      lastScale = 1
    }
    this.eventTarget.addEventListener('gesturestart', this.gestureStartHandler)

    this.gestureChangeHandler = (e: Event) => {
      e.preventDefault()
      const ge = e as GestureEvent
      const scaleDelta = ge.scale / lastScale
      lastScale = ge.scale
      // Invert: scale > 1 means pinch out = zoom in = smaller distance
      this.zoomBy(1 / scaleDelta)
    }
    this.eventTarget.addEventListener('gesturechange', this.gestureChangeHandler)

    this.gestureEndHandler = (e: Event) => {
      e.preventDefault()
    }
    this.eventTarget.addEventListener('gestureend', this.gestureEndHandler)

    // Arrow keys for navigation
    this.keyDownHandler = (e: KeyboardEvent) => {
      // Don't intercept arrow keys when typing in inputs/textareas
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) return

      const step = 1.5
      switch (e.key) {
        case 'ArrowUp':
          this.pan(0, step)
          e.preventDefault()
          break
        case 'ArrowDown':
          this.pan(0, -step)
          e.preventDefault()
          break
        case 'ArrowLeft':
          this.pan(-step, 0)
          e.preventDefault()
          break
        case 'ArrowRight':
          this.pan(step, 0)
          e.preventDefault()
          break
      }
    }
    window.addEventListener('keydown', this.keyDownHandler)
  }

  /**
   * Clean up event listeners (call before recreating Camera during HMR)
   */
  dispose(): void {
    // Window listeners
    if (this.mouseMoveHandler) {
      window.removeEventListener('mousemove', this.mouseMoveHandler)
    }
    if (this.mouseUpHandler) {
      window.removeEventListener('mouseup', this.mouseUpHandler)
    }
    if (this.keyDownHandler) {
      window.removeEventListener('keydown', this.keyDownHandler)
    }
    // EventTarget (canvas) listeners
    if (this.mouseDownHandler) {
      this.eventTarget.removeEventListener('mousedown', this.mouseDownHandler)
    }
    if (this.clickHandler) {
      this.eventTarget.removeEventListener('click', this.clickHandler)
    }
    if (this.wheelHandler) {
      this.eventTarget.removeEventListener('wheel', this.wheelHandler)
    }
    if (this.gestureStartHandler) {
      this.eventTarget.removeEventListener('gesturestart', this.gestureStartHandler)
    }
    if (this.gestureChangeHandler) {
      this.eventTarget.removeEventListener('gesturechange', this.gestureChangeHandler)
    }
    if (this.gestureEndHandler) {
      this.eventTarget.removeEventListener('gestureend', this.gestureEndHandler)
    }
  }

  pan(screenDx: number, screenDy: number): void {
    // Camera looks from position toward target.
    // For 45° rotation: camera is at (+X, +Z) relative to target, looking toward (-X, -Z)
    //
    // Screen right → camera's right vector in world XZ
    // Screen up → camera's forward vector in world XZ (toward what you're looking at)
    //
    // With rotation = π/4:
    //   right  = (cos(rotation), -sin(rotation)) in XZ = (0.707, -0.707)
    //   forward = (-sin(rotation), -cos(rotation)) in XZ = (-0.707, -0.707)
    //
    // But we want sieve: drag right = target moves right = view shifts left
    // So screen movement directly moves target in camera-relative directions.

    const cosR = Math.cos(this.rotation)
    const sinR = Math.sin(this.rotation)

    // Right vector (screen X → world XZ)
    const rightX = cosR
    const rightZ = -sinR

    // Forward vector (screen Y → world XZ, "into" the screen)
    const forwardX = -sinR
    const forwardZ = -cosR

    // Move target: positive screenDx = drag right = target moves right
    this.target.x += screenDx * rightX + screenDy * forwardX
    this.target.z += screenDx * rightZ + screenDy * forwardZ
    this.updateCamera()
  }

  zoomBy(factor: number): void {
    this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * factor))
    this.updateCamera()
  }

  focusOn(pos: CartesianCoord): void {
    this.target.x = pos.x
    this.target.z = pos.z
    this.updateCamera()
  }

  /**
   * Set zoom to a specific level (clamped to min/max)
   */
  setZoom(zoom: number): void {
    this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, zoom))
    this.updateCamera()
  }

  /**
   * Focus on position and zoom to a specific level
   * @param screenOffsetY - Optional vertical offset (0-1, where 0.5 = center, 0 = top, 1 = bottom)
   */
  focusAndZoom(pos: CartesianCoord, zoom: number, screenOffsetY = 0.5): void {
    this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, zoom))

    // Calculate offset to position the city at the desired screen Y position
    // For 45° isometric view, shifting target in -Z moves focus point down on screen
    const verticalOffset = (screenOffsetY - 0.5) * this.zoom * 2

    this.target.x = pos.x
    this.target.z = pos.z - verticalOffset
    this.updateCamera()
  }

  private updateCamera(): void {
    // Update orthographic bounds based on zoom
    const aspect = this.canvas.clientWidth / this.canvas.clientHeight
    this.camera.left = -this.zoom * aspect
    this.camera.right = this.zoom * aspect
    this.camera.top = this.zoom
    this.camera.bottom = -this.zoom

    // Position camera looking at target from angle
    // Distance is arbitrary for ortho, just needs to be far enough
    const distance = 100
    const y = distance * Math.sin(this.angle)
    const horizontal = distance * Math.cos(this.angle)
    const x = this.target.x + horizontal * Math.sin(this.rotation)
    const z = this.target.z + horizontal * Math.cos(this.rotation)

    this.camera.position.set(x, y, z)
    this.camera.lookAt(this.target)

    this.camera.updateProjectionMatrix()
  }

  /**
   * Convert screen coordinates to world coordinates (on Y=0 plane)
   * For orthographic camera, this is a simple linear mapping
   */
  screenToWorld(screenX: number, screenY: number): CartesianCoord {
    const rect = this.canvas.getBoundingClientRect()

    // Normalize to -1 to 1 (NDC)
    const ndc = new Vector2(
      ((screenX - rect.left) / rect.width) * 2 - 1,
      -((screenY - rect.top) / rect.height) * 2 + 1
    )

    // Use raycaster to find intersection with ground plane (Y=0)
    this.raycaster.setFromCamera(ndc, this.camera)
    const intersection = new Vector3()
    this.raycaster.ray.intersectPlane(this.groundPlane, intersection)

    return { x: intersection.x, z: intersection.z }
  }

  /**
   * Handle window resize
   */
  resize(): void {
    this.updateCamera()  // Ortho bounds updated in updateCamera
  }

  /**
   * Check if user just completed a drag (to prevent click events during pan)
   */
  get dragging(): boolean {
    return this.wasDrag
  }

  /**
   * Get current zoom level for zoom-aware label scaling
   */
  get cameraDistance(): number {
    return this.zoom  // Higher = more zoomed out
  }

  /**
   * Get camera target (center of view) for distance-based label fading
   */
  get cameraCenter(): { x: number; z: number } {
    return { x: this.target.x, z: this.target.z }
  }
}
