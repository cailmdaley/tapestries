// ZoneRenderer.ts - Renders hexes with Cartographic Warmth styling

import {
  Scene,
  Mesh,
  MeshStandardMaterial,
  MeshBasicMaterial,
  Shape,
  ExtrudeGeometry,
  Group,
  PlaneGeometry,
  DoubleSide,
  CanvasTexture,
  LineLoop,
  BufferGeometry,
  LineBasicMaterial,
  Vector3,
  Object3D,
  Material,
} from 'three'
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import { HexGrid } from './HexGrid'
import { createVellumPlane } from './VellumShader'
import { createRhumbLines } from './RhumbLines'
import { CitySpritesManager } from './CitySpritesManager'
import { WorkerSwarm } from './WorkerSwarm'
import type { City, Session, HexCoord, ConversationMessage } from '../state/types'
import { PALETTE } from '../state/types'
import { ConversationCard } from '../ui/ConversationCard'

interface Activity {
  tool: string
  summary?: string
  timestamp: number
}

interface HexMeshData {
  group: Group
  hex: HexCoord
  type: 'city' | 'worker' | 'empty'
  entityId?: string
  entityName?: string  // Worker name for tooltip
  tmuxSession?: string  // For workers - to route activity events
  status?: 'idle' | 'working'  // Worker status for swarm activity
  activityMesh?: Mesh  // Activity ground decal
  labelObject?: CSS2DObject  // HTML label (CSS2D for OpenType features)
  workerLabels?: CSS2DObject[]  // Worker labels clustered on city sprite
  cityName?: string  // For toggling label text
  workerCount?: number  // For toggling label text
}

export class ZoneRenderer {
  private scene: Scene
  private hexGrid: HexGrid
  private hexMeshes: Map<string, HexMeshData> = new Map()
  private groundPlane: Mesh | null = null
  private rhumbLinesGroup: Group | null = null
  private selectionRing: Group | null = null

  // Hex geometry settings
  private readonly hexHeight = 0.15
  private readonly planeSize = 500  // World units (large for ~infinite appearance)

  // City sprites manager (nano-banana generated city plans)
  private citySprites: CitySpritesManager

  // Worker swarms (murmuration particles replacing ship sprites)
  private workerSwarms: Map<string, WorkerSwarm> = new Map()  // workerId -> swarm

  // Camera rotation (45° = π/4) - must match Camera.ts
  private readonly cameraRotation = Math.PI / 4

  // Callback for worker label clicks (since CSS2D labels need direct handlers)
  private onWorkerClick: ((workerId: string, tmuxSession: string) => void) | null = null
  private onWorkerDblClick: ((workerId: string, tmuxSession: string) => void) | null = null

  // Callback for city label clicks (needed for remote cities without sprites)
  private onCityLabelClick: ((cityId: string) => void) | null = null

  // Label drag state (for dragging swarm via label when card is closed)
  private labelDrag: {
    workerId: string
    startX: number
    startY: number
    moved: boolean  // Track if mouse moved (to distinguish click from drag)
    labelEl: HTMLElement  // Store element for cursor reset
  } | null = null

  // Track city positions for rhumb line avoidance
  private lastCityPositions: string = ''

  // Track city/worker state signatures for diffing (avoid unnecessary re-renders)
  private lastCitySignatures: Map<string, string> = new Map()

  // Store current state for sprite reload re-renders
  private currentCities: Map<string, City> = new Map()
  private currentWorkersByCity: Map<string, Session[]> = new Map()

  // Animation optimization: cache last values to skip redundant work
  private lastCameraDistance: number = -1
  private lastFontSizes: { city: number; worker: number } = { city: -1, worker: -1 }
  private lastAnimateTime: number = 0  // For delta time calculation

  // Conversation cards - map-pinned worker conversations
  private conversationCards: Map<string, ConversationCard> = new Map()  // workerId -> card
  private onCardFileClick: ((fullPath: string, originId: string, workerId: string) => void) | null = null
  private topZIndex = 1000  // Track highest z-index for bringing cards to front (must be >> CSS2DRenderer's depth values)
  private lastFocusedCardId: string | null = null  // Most recently focused card

  constructor(scene: Scene, hexGrid: HexGrid) {
    this.scene = scene
    this.hexGrid = hexGrid
    this.citySprites = new CitySpritesManager()
    this.createGroundPlane()

    // Re-render city when its custom sprite finishes loading
    this.citySprites.onSpriteLoaded((cityId) => {
      const city = this.currentCities.get(cityId)
      if (city) {
        const workers = this.currentWorkersByCity.get(cityId) || []
        this.renderCity(city, workers)
      }
    })
  }

  /**
   * Set callback for worker label clicks
   */
  setWorkerClickHandler(onClick: (workerId: string, tmuxSession: string) => void): void {
    this.onWorkerClick = onClick
  }

  /**
   * Set callback for worker label double-clicks
   */
  setWorkerDblClickHandler(onDblClick: (workerId: string, tmuxSession: string) => void): void {
    this.onWorkerDblClick = onDblClick
  }

  /**
   * Set callback for city label clicks (needed for remote cities without sprites)
   */
  setCityLabelClickHandler(onClick: (cityId: string) => void): void {
    this.onCityLabelClick = onClick
  }

  /**
   * Get CSS class for worker label based on status
   */
  private workerLabelClass(status: 'idle' | 'working'): string {
    return status === 'working' ? 'worker-label working' : 'worker-label'
  }

  /**
   * Setup drag handlers for a worker label (allows dragging swarm when card is closed)
   */
  private setupLabelDrag(labelEl: HTMLElement, workerId: string): void {
    labelEl.addEventListener('mousedown', (e) => {
      e.preventDefault()
      e.stopPropagation()

      this.labelDrag = {
        workerId,
        startX: e.clientX,
        startY: e.clientY,
        moved: false,
        labelEl,
      }

      // Set custom grabbing cursor (bird for everything)
      const birdCursor = 'var(--cursor-bird)'
      labelEl.style.cursor = birdCursor
      document.body.style.cursor = birdCursor

      document.addEventListener('mousemove', this.onLabelDrag)
      document.addEventListener('mouseup', this.stopLabelDrag)
    })
  }

  /**
   * Setup click handlers for a worker label (click opens card, dblclick focuses terminal)
   */
  private setupLabelClickHandlers(labelEl: HTMLElement, workerId: string, tmuxSession: string): void {
    labelEl.addEventListener('click', (e) => {
      e.stopPropagation()
      // Don't trigger click if we just finished dragging
      if (this.labelDrag?.workerId === workerId && this.labelDrag.moved) {
        return
      }
      if (this.onWorkerClick) this.onWorkerClick(workerId, tmuxSession)
    })
    labelEl.addEventListener('dblclick', (e) => {
      e.stopPropagation()
      if (this.onWorkerDblClick) this.onWorkerDblClick(workerId, tmuxSession)
    })
  }

  private onLabelDrag = (e: MouseEvent): void => {
    if (!this.labelDrag) return

    const dx = e.clientX - this.labelDrag.startX
    const dy = e.clientY - this.labelDrag.startY

    // Only count as "moved" if dragged more than a few pixels (prevents accidental drags)
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      this.labelDrag.moved = true
    }

    if (this.labelDrag.moved && this.screenToWorldConverter) {
      // Convert screen positions to world positions for accurate movement
      const worldStart = this.screenToWorldConverter(this.labelDrag.startX, this.labelDrag.startY)
      const worldNow = this.screenToWorldConverter(e.clientX, e.clientY)

      const worldDx = worldNow.x - worldStart.x
      const worldDz = worldNow.z - worldStart.z

      // Reset drag start to current position for incremental movement
      this.labelDrag.startX = e.clientX
      this.labelDrag.startY = e.clientY

      // Move the swarm
      const swarm = this.workerSwarms.get(this.labelDrag.workerId)
      if (swarm) {
        swarm.userOffset.x += worldDx
        swarm.userOffset.z += worldDz
        swarm.group.position.x += worldDx
        swarm.group.position.z += worldDz
      }
    }
  }

  // Screen-to-world converter function (set from main.ts with camera access)
  private screenToWorldConverter: ((x: number, y: number) => { x: number; z: number }) | null = null

  /**
   * Set the screen-to-world conversion function (from camera)
   */
  setScreenToWorldConverter(converter: (x: number, y: number) => { x: number; z: number }): void {
    this.screenToWorldConverter = converter
  }

  private stopLabelDrag = (): void => {
    // Reset cursors
    if (this.labelDrag?.labelEl) {
      this.labelDrag.labelEl.style.cursor = 'grab'
    }
    document.body.style.cursor = ''

    // Keep labelDrag briefly so click handler can check if we moved
    const drag = this.labelDrag
    setTimeout(() => {
      if (this.labelDrag === drag) {
        this.labelDrag = null
      }
    }, 10)

    document.removeEventListener('mousemove', this.onLabelDrag)
    document.removeEventListener('mouseup', this.stopLabelDrag)
  }

  /**
   * Start dragging a swarm by workerId (called from main.ts on swarm mousedown)
   * Returns true if swarm found and drag started
   */
  startSwarmDrag(workerId: string, screenX: number, screenY: number): boolean {
    const swarm = this.workerSwarms.get(workerId)
    if (!swarm) return false

    // Find the label element to update its cursor
    const labelEl = swarm.getLabel()?.element as HTMLElement | undefined

    this.labelDrag = {
      workerId,
      startX: screenX,
      startY: screenY,
      moved: false,
      labelEl: labelEl || document.createElement('div'),  // Dummy if no label
    }

    const birdCursor = 'var(--cursor-bird)'
    document.body.style.cursor = birdCursor
    if (labelEl) labelEl.style.cursor = birdCursor

    document.addEventListener('mousemove', this.onLabelDrag)
    document.addEventListener('mouseup', this.stopLabelDrag)

    return true
  }

  /**
   * Check if currently dragging a swarm
   */
  get isDraggingSwarm(): boolean {
    return this.labelDrag !== null && this.labelDrag.moved
  }

  /**
   * Convert hex-aligned offset to world XZ coordinates.
   * For elements rotated 60° to match hex orientation.
   * +X = right along hex axis, +Y = up along hex axis
   */
  private hexToWorld(hexX: number, hexY: number): { x: number; z: number } {
    const angle = this.cameraRotation + Math.PI / 3  // 45° + 60° = 105°
    const cos = Math.cos(angle)
    const sin = Math.sin(angle)
    return {
      x: hexX * cos - hexY * sin,
      z: -hexX * sin - hexY * cos,
    }
  }

  /**
   * Get or create a worker swarm, loading saved offset if new
   */
  private getOrCreateSwarm(workerId: string, tmuxSession: string): WorkerSwarm {
    const existing = this.workerSwarms.get(workerId)
    if (existing) return existing

    const swarm = new WorkerSwarm(workerId, tmuxSession)
    this.workerSwarms.set(workerId, swarm)

    const savedOffset = this.loadSwarmOffset(workerId)
    if (savedOffset) {
      swarm.userOffset.x = savedOffset.x
      swarm.userOffset.z = savedOffset.z
    }

    return swarm
  }

  /**
   * Dispose all Three.js resources in an object tree.
   * Prevents memory leaks by releasing GPU resources.
   * Skips textures marked as managed (owned by sprite managers).
   */
  private disposeObject(obj: Object3D): void {
    obj.traverse((child) => {
      // Dispose geometry for any object that has it
      if ('geometry' in child && child.geometry) {
        (child.geometry as BufferGeometry).dispose()
      }

      // Dispose materials
      if ('material' in child && child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material]
        for (const mat of materials) {
          mat.dispose()
          // Only dispose unmanaged textures (CanvasTexture from activity decals, etc.)
          const basicMat = mat as MeshBasicMaterial
          if (basicMat.map && !basicMat.map.userData?.managed) {
            basicMat.map.dispose()
          }
        }
      }
    })
  }

  private createGroundPlane(): void {
    // Vellum background - aged parchment with procedural shader
    this.groundPlane = createVellumPlane(this.planeSize, this.planeSize)
    this.groundPlane.position.y = -0.05  // Just below hex level
    this.scene.add(this.groundPlane)

    // Initial rhumb lines (will be regenerated when cities are known)
    this.updateRhumbLines([])
  }

  private updateRhumbLines(cityPositions: { x: number; z: number }[]): void {
    // Remove and dispose existing rhumb lines
    if (this.rhumbLinesGroup) {
      this.disposeObject(this.rhumbLinesGroup)
      this.scene.remove(this.rhumbLinesGroup)
    }

    // Create new rhumb lines avoiding city positions
    this.rhumbLinesGroup = createRhumbLines({
      seed: 42,
      primaryRoses: 4,
      primaryDirections: 16,
      primaryOpacity: 0.20,
      mapRadius: this.planeSize,
      clusterRadius: 25,
      avoidPositions: cityPositions,
    })
    this.scene.add(this.rhumbLinesGroup)
  }

  private createHexShape(scale = 1): Shape {
    const r = this.hexGrid.hexRadius * scale
    const shape = new Shape()

    // Pointy-top hexagon (matches axialToCartesian spacing)
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2
      const x = r * Math.cos(angle)
      const y = r * Math.sin(angle)
      if (i === 0) {
        shape.moveTo(x, y)
      } else {
        shape.lineTo(x, y)
      }
    }
    shape.closePath()

    return shape
  }

  private createHexMesh(
    color: number,
    scale = 1,
    height = this.hexHeight
  ): Mesh {
    const shape = this.createHexShape(scale)
    const geometry = new ExtrudeGeometry(shape, {
      depth: height,
      bevelEnabled: true,
      bevelThickness: 0.02,
      bevelSize: 0.02,
      bevelSegments: 2,
    })

    const material = new MeshStandardMaterial({
      color,
      roughness: 0.8,
      metalness: 0.1,
    })

    const mesh = new Mesh(geometry, material)
    mesh.rotation.x = -Math.PI / 2
    mesh.castShadow = true
    mesh.receiveShadow = true

    return mesh
  }

  /**
   * Create a hex edge outline (just the border, no fill)
   */
  private createHexEdge(scale = 1, opacity = 0.15): LineLoop {
    const r = this.hexGrid.hexRadius * scale
    const points: Vector3[] = []

    // Pointy-top hexagon
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2
      points.push(new Vector3(r * Math.cos(angle), 0, r * Math.sin(angle)))
    }

    const geometry = new BufferGeometry().setFromPoints(points)
    const material = new LineBasicMaterial({
      color: 0x6B5B4B,  // Warm brown
      transparent: true,
      opacity,
    })

    return new LineLoop(geometry, material)
  }

  /**
   * Add hex grid overlay around a city (3-hex radius)
   */
  private addCityHexGrid(group: Group, centerHex: HexCoord): void {
    const radius = 4  // 4-hex radius around city
    const hexes = this.hexGrid.getHexesInRadius(centerHex, radius)

    for (const hex of hexes) {
      const relPos = this.hexGrid.axialToCartesian(hex)
      const centerPos = this.hexGrid.axialToCartesian(centerHex)

      const edge = this.createHexEdge(0.98, 0.10)
      edge.position.set(relPos.x - centerPos.x, 0.05, relPos.z - centerPos.z)  // Above sprite
      group.add(edge)
    }
  }

  renderCity(city: City, workers: Session[] = []): void {
    const key = this.hexGrid.hexKey(city.hex)

    // Remove existing mesh at this position
    this.removeHex(key)

    const group = new Group()
    const pos = this.hexGrid.axialToCartesian(city.hex)

    // Light hex grid around city for spatial reference
    this.addCityHexGrid(group, city.hex)

    // City sprite - nano-banana generated city plan, lying flat on vellum
    const texture = this.citySprites.getSprite(city)

    if (texture) {
      // Use a flat plane mesh instead of billboard sprite
      // With 1 hex = 1 world unit, sprite covers 3 hex radius (7 hexes across)
      const spriteSize = 6.5  // World units diameter (3 hex radius)
      const geometry = new PlaneGeometry(spriteSize, spriteSize)
      const material = new MeshBasicMaterial({
        map: texture,
        transparent: true,
        side: DoubleSide,
        depthWrite: false,  // Prevent z-fighting with vellum
      })
      const cityMesh = new Mesh(geometry, material)
      cityMesh.rotation.x = -Math.PI / 2  // Lie flat on XZ plane
      cityMesh.position.y = 0.02  // Just above vellum
      group.add(cityMesh)
    } else {
      // Fallback: small marker while sprites load
      const fallbackMesh = this.createHexMesh(PALETTE.cityHex, 0.3, 0.05)
      group.add(fallbackMesh)
    }

    // City label - above the sprite
    const labelDiv = document.createElement('div')
    labelDiv.className = 'city-label'
    labelDiv.textContent = city.name
    labelDiv.style.cursor = 'pointer'

    // Click handler for city label (needed for remote cities without sprites)
    const cityId = city.id
    labelDiv.addEventListener('click', (e) => {
      e.stopPropagation()
      if (this.onCityLabelClick) this.onCityLabelClick(cityId)
    })

    const labelObject = new CSS2DObject(labelDiv)
    labelObject.position.set(0, 1.5, 0)  // Above center
    group.add(labelObject)

    // Workers as particle swarms positioned around the northern arc of the city
    // Camera is at +Z looking toward -Z, so "north" (above on screen) is -Z direction
    const workerLabels: CSS2DObject[] = []
    const swarmRadius = 3.0  // Distance from city center
    const arcStart = -Math.PI * 0.75  // Start at -135° (left-back)
    const arcEnd = -Math.PI * 0.25    // End at -45° (right-back)

    workers.forEach((worker, i) => {
      // Distribute swarms along the northern arc (above city on screen)
      const arcSpan = arcEnd - arcStart
      const angle = workers.length === 1
        ? -Math.PI * 0.5  // Single worker at center-back (directly away from camera = top)
        : arcStart + (arcSpan * i / (workers.length - 1))

      const swarmX = Math.cos(angle) * swarmRadius
      const swarmZ = Math.sin(angle) * swarmRadius

      const swarm = this.getOrCreateSwarm(worker.id, worker.tmuxSession)

      // Position swarm relative to city (including any user offset)
      swarm.group.position.set(
        swarmX + swarm.userOffset.x,
        0,
        swarmZ + swarm.userOffset.z
      )
      swarm.setActivity(worker.status === 'working' ? 1 : 0)
      group.add(swarm.group)

      // Worker label attached to swarm (moves with swarm)
      const workerDiv = document.createElement('div')
      workerDiv.className = this.workerLabelClass(worker.status)
      workerDiv.textContent = worker.name
      workerDiv.dataset.workerId = worker.id
      workerDiv.dataset.tmuxSession = worker.tmuxSession
      workerDiv.style.cursor = 'grab'

      // Drag to move swarm (via label)
      this.setupLabelDrag(workerDiv, worker.id)
      this.setupLabelClickHandlers(workerDiv, worker.id, worker.tmuxSession)

      const workerLabelObj = new CSS2DObject(workerDiv)
      // Attach label to swarm (positioned relative to swarm, at y=0.45 above particles)
      swarm.setLabel(workerLabelObj)
      workerLabels.push(workerLabelObj)
    })

    group.position.set(pos.x, 0, pos.z)
    this.scene.add(group)

    this.hexMeshes.set(key, {
      group, hex: city.hex, type: 'city', entityId: city.id,
      labelObject, workerLabels,
      cityName: city.name, workerCount: workers.length
    })
  }

  /**
   * Render an orphan worker (no city) as a particle swarm
   * These are workers that exist but aren't associated with any city
   */
  renderOrphanWorker(session: Session): void {
    if (!session.hex) return

    const key = this.hexGrid.hexKey(session.hex)

    // Check if worker already exists - just update status
    const existing = this.hexMeshes.get(key)
    if (existing && existing.type === 'worker' && existing.entityId === session.id) {
      // Update swarm activity and label class for status change
      if (existing.status !== session.status) {
        if (existing.labelObject) {
          existing.labelObject.element.className = this.workerLabelClass(session.status)
        }
        // Update swarm activity
        const swarm = this.workerSwarms.get(session.id)
        swarm?.setActivity(session.status === 'working' ? 1 : 0)
        existing.status = session.status
      }
      return
    }

    // Remove existing mesh at this position
    this.removeHex(key)

    const group = new Group()
    const pos = this.hexGrid.axialToCartesian(session.hex)

    const swarm = this.getOrCreateSwarm(session.id, session.tmuxSession)
    // Apply user offset to swarm position (orphan workers position relative to hex center)
    swarm.group.position.set(swarm.userOffset.x, 0, swarm.userOffset.z)
    swarm.setActivity(session.status === 'working' ? 1 : 0)
    group.add(swarm.group)

    // Worker label with drag capability
    const labelDiv = document.createElement('div')
    labelDiv.className = this.workerLabelClass(session.status)
    labelDiv.textContent = session.name
    labelDiv.dataset.workerId = session.id
    labelDiv.dataset.tmuxSession = session.tmuxSession
    labelDiv.style.cursor = 'move'

    // Drag to move swarm
    this.setupLabelDrag(labelDiv, session.id)
    this.setupLabelClickHandlers(labelDiv, session.id, session.tmuxSession)

    const labelObject = new CSS2DObject(labelDiv)
    labelObject.position.y = 0.6  // Above swarm
    group.add(labelObject)

    group.position.set(pos.x, 0, pos.z)
    this.scene.add(group)

    this.hexMeshes.set(key, {
      group,
      hex: session.hex,
      type: 'worker',
      entityId: session.id,
      entityName: session.name,
      tmuxSession: session.tmuxSession,
      status: session.status,
      labelObject,
    })
  }

  private removeHex(key: string): void {
    const data = this.hexMeshes.get(key)
    if (!data) return

    // Clean up CSS2D label DOM elements
    data.labelObject?.element.remove()
    data.workerLabels?.forEach(label => label.element.remove())

    // Remove swarms from group before disposing (they may be reused)
    // Swarm groups have userData.workerId set
    const swarmsToPreserve: Group[] = []
    data.group.traverse((child) => {
      if (child.userData?.workerId && child.parent === data.group) {
        swarmsToPreserve.push(child as Group)
      }
    })
    for (const swarmGroup of swarmsToPreserve) {
      data.group.remove(swarmGroup)
    }

    // Dispose Three.js resources before removing from scene
    this.disposeObject(data.group)
    this.scene.remove(data.group)
    this.hexMeshes.delete(key)
  }

  /**
   * Build a signature string for a city+workers state.
   * Used for diffing to avoid unnecessary re-renders.
   */
  private buildCitySignature(city: City, workers: Session[]): string {
    const workerSigs = workers
      .map(w => `${w.id}:${w.status}:${w.name}`)
      .sort()
      .join(',')
    return `${city.name}|${city.hex.q},${city.hex.r}|${city.fiberCount}|${workerSigs}`
  }

  updateState(cities: City[], sessions: Session[]): void {
    // Track what should exist
    const expectedKeys = new Set<string>()
    const newSignatures = new Map<string, string>()

    // Group workers by city
    const workersByCity = new Map<string, Session[]>()
    const orphanWorkers: Session[] = []

    for (const session of sessions) {
      if (session.cityId) {
        const existing = workersByCity.get(session.cityId) || []
        existing.push(session)
        workersByCity.set(session.cityId, existing)
      } else if (session.hex) {
        // Worker without a city - render as standalone marker
        orphanWorkers.push(session)
      }
    }

    // Store current state for sprite reload re-renders
    this.currentCities.clear()
    for (const city of cities) {
      this.currentCities.set(city.id, city)
    }
    this.currentWorkersByCity = workersByCity

    // Check if city positions changed - regenerate rhumb lines if so
    const cityPositions = cities.map(c => {
      const pos = this.hexGrid.axialToCartesian(c.hex)
      return { x: pos.x, z: pos.z }
    })
    const positionsKey = JSON.stringify(cityPositions.map(p => `${p.x.toFixed(1)},${p.z.toFixed(1)}`))
    if (positionsKey !== this.lastCityPositions) {
      this.lastCityPositions = positionsKey
      this.updateRhumbLines(cityPositions)
    }

    // Render cities with their workers (only if changed)
    for (const city of cities) {
      const key = this.hexGrid.hexKey(city.hex)
      expectedKeys.add(key)
      const cityWorkers = workersByCity.get(city.id) || []

      // Build signature and check if re-render needed
      const signature = this.buildCitySignature(city, cityWorkers)
      newSignatures.set(key, signature)

      if (this.lastCitySignatures.get(key) !== signature) {
        this.renderCity(city, cityWorkers)
      }
    }

    // Render orphan workers (no city) as small markers
    // (renderOrphanWorker already has internal diffing)
    for (const session of orphanWorkers) {
      if (session.hex) {
        const key = this.hexGrid.hexKey(session.hex)
        expectedKeys.add(key)
        this.renderOrphanWorker(session)
      }
    }

    // Remove hexes that no longer exist (except empty background hexes)
    for (const [key, data] of this.hexMeshes) {
      if (!expectedKeys.has(key) && data.type !== 'empty') {
        this.removeHex(key)
      }
    }

    // Close conversation cards for workers that no longer exist
    const currentWorkerIds = new Set(sessions.map(s => s.id))
    for (const workerId of this.conversationCards.keys()) {
      if (!currentWorkerIds.has(workerId)) {
        this.closeConversationCard(workerId)
      }
    }

    // Dispose swarms for workers that no longer exist
    for (const workerId of this.workerSwarms.keys()) {
      if (!currentWorkerIds.has(workerId)) {
        const swarm = this.workerSwarms.get(workerId)
        if (swarm) {
          swarm.dispose()
          this.workerSwarms.delete(workerId)
        }
      }
    }

    // Update signature cache (removes old, adds new)
    this.lastCitySignatures = newSignatures
  }

  getHexAtPosition(x: number, z: number): HexCoord {
    return this.hexGrid.cartesianToHex(x, z)
  }

  /**
   * Find entity at a hex position
   */
  getEntityAtHex(hex: HexCoord): { type: 'city' | 'worker' | 'empty'; entityId?: string; entityName?: string } | null {
    const key = this.hexGrid.hexKey(hex)
    const data = this.hexMeshes.get(key)
    if (!data) return null
    return { type: data.type, entityId: data.entityId, entityName: data.entityName }
  }

  /**
   * Find nearest city within sprite radius of a world position
   * City sprites are ~6 world units, so use radius of 3
   */
  getCityAtWorldPos(worldX: number, worldZ: number): { entityId: string } | null {
    const spriteRadius = 3.25  // 3 hex radius (matches sprite)
    let nearest: { entityId: string; dist: number } | null = null

    for (const [, data] of this.hexMeshes) {
      if (data.type === 'city' && data.entityId) {
        const pos = this.hexGrid.axialToCartesian(data.hex)
        const dist = Math.sqrt((pos.x - worldX) ** 2 + (pos.z - worldZ) ** 2)
        if (dist <= spriteRadius && (!nearest || dist < nearest.dist)) {
          nearest = { entityId: data.entityId, dist }
        }
      }
    }

    return nearest ? { entityId: nearest.entityId } : null
  }

  /**
   * Find worker swarm at world position
   * Uses swarm hit test for click detection
   */
  getWorkerAtWorldPos(worldX: number, worldZ: number): { workerId: string; tmuxSession: string } | null {
    let nearestDist = Infinity
    let nearestWorker: { workerId: string; tmuxSession: string } | null = null

    for (const [, data] of this.hexMeshes) {
      if (data.type === 'city') {
        const cityPos = this.hexGrid.axialToCartesian(data.hex)
        // Check all swarm groups within the city
        data.group.traverse((child) => {
          if (child.userData?.workerId) {
            // Swarm position in world space
            const swarmWorldX = cityPos.x + child.position.x
            const swarmWorldZ = cityPos.z + child.position.z
            const dist = Math.sqrt((swarmWorldX - worldX) ** 2 + (swarmWorldZ - worldZ) ** 2)
            const hitRadius = 0.8  // Swarm hit radius
            if (dist <= hitRadius && dist < nearestDist) {
              nearestDist = dist
              nearestWorker = {
                workerId: child.userData.workerId as string,
                tmuxSession: child.userData.tmuxSession as string,
              }
            }
          }
        })
      }
    }

    return nearestWorker
  }

  /**
   * Create a ring shape (hex with hex hole)
   */
  private createRingShape(outerScale: number, innerScale: number): Shape {
    const outerR = this.hexGrid.hexRadius * outerScale
    const innerR = this.hexGrid.hexRadius * innerScale

    // Outer hex (pointy-top to match grid)
    const shape = new Shape()
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2
      const x = outerR * Math.cos(angle)
      const y = outerR * Math.sin(angle)
      if (i === 0) {
        shape.moveTo(x, y)
      } else {
        shape.lineTo(x, y)
      }
    }
    shape.closePath()

    // Inner hex hole (counter-clockwise for hole, pointy-top)
    const hole = new Shape()
    for (let i = 5; i >= 0; i--) {
      const angle = (Math.PI / 3) * i - Math.PI / 2
      const x = innerR * Math.cos(angle)
      const y = innerR * Math.sin(angle)
      if (i === 5) {
        hole.moveTo(x, y)
      } else {
        hole.lineTo(x, y)
      }
    }
    hole.closePath()
    shape.holes.push(hole)

    return shape
  }

  /**
   * Set selection highlight on a hex
   */
  setSelection(hex: HexCoord | null): void {
    // Remove and dispose existing selection ring
    if (this.selectionRing) {
      this.disposeObject(this.selectionRing)
      this.scene.remove(this.selectionRing)
      this.selectionRing = null
    }

    if (!hex) return

    const group = new Group()
    const pos = this.hexGrid.axialToCartesian(hex)

    // Create ring (gold highlight - Porch Morning)
    const ringShape = this.createRingShape(1.12, 0.92)
    const ringGeometry = new ExtrudeGeometry(ringShape, {
      depth: 0.08,
      bevelEnabled: false,
    })
    const ringMaterial = new MeshStandardMaterial({
      color: PALETTE.selection,
      roughness: 0.6,
      metalness: 0.2,
      transparent: true,
      opacity: 0.8,
    })
    const ringMesh = new Mesh(ringGeometry, ringMaterial)
    ringMesh.rotation.x = -Math.PI / 2
    ringMesh.position.y = 0.25 // Above other hexes
    group.add(ringMesh)

    group.position.set(pos.x, 0, pos.z)
    this.scene.add(group)
    this.selectionRing = group
  }

  /**
   * Animate and update zoom-based label visibility
   */
  animate(cameraDistance?: number): void {
    // Calculate delta time for swarm animation
    const now = performance.now()
    const deltaTime = this.lastAnimateTime === 0 ? 1 / 60 : Math.min((now - this.lastAnimateTime) / 1000, 0.1)
    this.lastAnimateTime = now

    // Scale labels and cards: only update when camera distance actually changed
    if (cameraDistance !== undefined && cameraDistance !== this.lastCameraDistance) {
      this.lastCameraDistance = cameraDistance

      const scale = this.calculateCardScale(cameraDistance)
      const cityFontSize = Math.round(28 * scale)   // City base: 28px
      const workerFontSize = Math.round(18 * scale) // Worker base: 18px

      // Only update DOM if font sizes actually changed
      if (cityFontSize !== this.lastFontSizes.city || workerFontSize !== this.lastFontSizes.worker) {
        this.lastFontSizes = { city: cityFontSize, worker: workerFontSize }

        for (const [, data] of this.hexMeshes) {
          if (data.type === 'city') {
            if (data.labelObject) {
              const label = data.labelObject.element as HTMLElement
              label.style.fontSize = `${cityFontSize}px`
            }
            data.workerLabels?.forEach(w => {
              const el = w.element as HTMLElement
              el.style.fontSize = `${workerFontSize}px`
            })
          }
        }
      }

      // Scale conversation cards with zoom
      for (const card of this.conversationCards.values()) {
        card.setScale(scale)
      }
    }

    // Update card viewport clamping every frame (wrapper position changes with panning)
    for (const card of this.conversationCards.values()) {
      card.updateViewportClamp()
    }

    // Update all worker swarms (they handle their own animation)
    for (const swarm of this.workerSwarms.values()) {
      if (cameraDistance !== undefined) {
        swarm.setCameraDistance(cameraDistance)
      }
      swarm.update(deltaTime)
    }
  }

  /**
   * Create activity ground decal showing recent tool calls
   * Returns a flat Mesh that lies on the hex surface
   */
  private createActivityDecal(activities: Activity[]): Mesh {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!

    const width = 256
    const height = 144  // 1.8x taller to fill hex
    const fontSize = 13

    canvas.width = width * 2
    canvas.height = height * 2
    ctx.scale(2, 2)

    // Semi-transparent dark background
    ctx.fillStyle = 'rgba(26, 24, 22, 0.7)'
    ctx.roundRect(0, 0, width, height, 4)
    ctx.fill()

    if (activities.length > 0) {
      const displayActivities = activities.slice(0, 3)  // Show 3 activities
      const lineHeight = 24
      const startY = 18
      const centerX = width / 2

      displayActivities.forEach((activity, i) => {
        const y = startY + i * lineHeight
        const opacity = 1 - i * 0.25  // Fade older entries

        // Build full text line
        let text = activity.tool
        if (activity.summary) {
          const summaryText = activity.summary.length > 18
            ? activity.summary.slice(0, 15) + '...'
            : activity.summary
          text += ` ${summaryText}`
        }

        // Draw centered
        ctx.font = `bold ${fontSize}px 'JetBrains Mono', monospace`
        ctx.textAlign = 'center'
        ctx.fillStyle = `rgba(201, 162, 39, ${opacity})`
        ctx.fillText(text, centerX, y)
      })
    }

    const texture = new CanvasTexture(canvas)
    const worldWidth = 1.0
    const worldHeight = worldWidth * (height / width)  // Maintain aspect ratio
    const geometry = new PlaneGeometry(worldWidth, worldHeight)
    const material = new MeshBasicMaterial({
      map: texture,
      transparent: true,
      side: DoubleSide,
      depthWrite: false,
    })

    const mesh = new Mesh(geometry, material)
    mesh.rotation.x = -Math.PI / 2  // Lie flat
    mesh.rotation.z = Math.PI / 3   // 60° rotation
    return mesh
  }

  /**
   * Update activity display for a worker by tmux session
   */
  updateWorkerActivity(tmuxSession: string, activities: Activity[]): void {
    const workerData = Array.from(this.hexMeshes.values()).find(
      data => data.type === 'worker' && data.tmuxSession === tmuxSession && data.activityMesh
    )
    if (!workerData) return

    // Dispose old decal resources before removing
    this.disposeObject(workerData.activityMesh!)
    workerData.group.remove(workerData.activityMesh!)

    // Create new decal with updated activities
    const newMesh = this.createActivityDecal(activities)
    newMesh.position.y = this.hexHeight + 0.08  // Just above hex surface
    const activityOffset = this.hexToWorld(-0.023, -0.03)
    newMesh.position.x = activityOffset.x
    newMesh.position.z = activityOffset.z
    workerData.group.add(newMesh)
    workerData.activityMesh = newMesh
  }

  /**
   * Debug utility: count scene objects for memory verification
   * Call from browser console: window.zoneRenderer?.debugResourceCounts()
   */
  debugResourceCounts(): { meshes: number; geometries: number; materials: number; textures: number; hexes: number } {
    let meshes = 0
    let geometries = 0
    let materials = 0
    const textureSet = new Set<number>()

    this.scene.traverse((obj) => {
      if (obj instanceof Mesh) {
        meshes++
        if (obj.geometry) geometries++
        if (obj.material instanceof Material) {
          materials++
          const mat = obj.material as MeshBasicMaterial
          if (mat.map) textureSet.add(mat.map.id)
        } else if (Array.isArray(obj.material)) {
          materials += obj.material.length
          obj.material.forEach(m => {
            const mat = m as MeshBasicMaterial
            if (mat.map) textureSet.add(mat.map.id)
          })
        }
      }
    })

    const counts = {
      meshes,
      geometries,
      materials,
      textures: textureSet.size,
      hexes: this.hexMeshes.size
    }
    console.table(counts)
    return counts
  }

  // ═══════════════════════════════════════════════════════════
  // CONVERSATION CARDS - Map-pinned worker conversations
  // ═══════════════════════════════════════════════════════════

  /**
   * Set callback for file clicks in conversation cards
   */
  setCardFileClickHandler(handler: (fullPath: string, originId: string, workerId: string) => void): void {
    this.onCardFileClick = handler
  }

  /**
   * Open a conversation card for a worker
   * @param session The worker session to show conversation for
   * @returns The created card, or existing card if already open
   */
  async openConversationCard(session: Session): Promise<ConversationCard | null> {
    // Check if card already exists - bring to front if so
    const existing = this.conversationCards.get(session.id)
    if (existing) {
      this.bringCardToFront(session.id)
      return existing
    }

    // Find the worker's swarm - card attaches to it
    const swarm = this.workerSwarms.get(session.id)
    if (!swarm) {
      console.warn(`Cannot find swarm for worker ${session.id}`)
      return null
    }

    // Ensure card states are loaded from server (cached after first call)
    await this.ensureWorkerStatesLoaded()

    // Load saved position if available
    const savedState = this.loadCardState(session.id)

    // Create card with saved size
    const card = new ConversationCard(session, {
      onClose: () => this.closeConversationCard(session.id),
      onFileClick: this.onCardFileClick ?? undefined,
      onDoubleClick: () => {
        if (this.onWorkerDblClick) {
          this.onWorkerDblClick(session.id, session.tmuxSession)
        }
      },
      onBringToFront: () => this.bringCardToFront(session.id),
      onSwarmDrag: (dx: number, dz: number) => this.moveSwarm(session.id, dx, dz),
      initialSize: savedState?.size,
    })

    // Position card just above swarm - card bottom will be at this anchor point
    card.object.position.set(0, 0.7, 0)

    // Hide label - card header takes its place
    swarm.hideLabel()

    // Add to swarm group so it moves with swarm
    swarm.addChild(card.object)
    this.conversationCards.set(session.id, card)

    // Set initial z-index and track as most recent
    this.topZIndex++
    card.setZIndex(this.topZIndex)
    this.lastFocusedCardId = session.id

    // Apply current scale
    if (this.lastCameraDistance > 0) {
      const scale = this.calculateCardScale(this.lastCameraDistance)
      card.setScale(scale)
    }

    return card
  }

  /**
   * Get the world position of a worker's swarm (for camera focus)
   */
  getSwarmWorldPosition(workerId: string): { x: number, z: number } | null {
    const swarm = this.workerSwarms.get(workerId)
    if (!swarm) return null
    // Swarm group is parented to city group — need world position
    swarm.group.updateWorldMatrix(true, false)
    const pos = new Vector3()
    swarm.group.getWorldPosition(pos)
    return { x: pos.x, z: pos.z }
  }

  /**
   * Move a worker's swarm (and everything attached: label, card) by offset
   */
  private moveSwarm(workerId: string, dx: number, dz: number): void {
    const swarm = this.workerSwarms.get(workerId)
    if (!swarm) return

    // Update user offset (persisted between renders)
    swarm.userOffset.x += dx
    swarm.userOffset.z += dz

    // Update swarm group position
    swarm.group.position.x += dx
    swarm.group.position.z += dz

    // Persist the offset
    this.saveSwarmOffset(workerId, { x: swarm.userOffset.x, z: swarm.userOffset.z })
  }

  /**
   * Bring a card to front (highest z-index)
   */
  private bringCardToFront(workerId: string): void {
    const card = this.conversationCards.get(workerId)
    if (!card) return

    this.topZIndex++
    card.setZIndex(this.topZIndex)
    this.lastFocusedCardId = workerId
  }

  /**
   * Reapply card z-indexes after CSS2DRenderer (which overwrites them on each render)
   * Call this AFTER labelRenderer.render() in the animation loop
   */
  reapplyCardZIndexes(): void {
    for (const card of this.conversationCards.values()) {
      const wrapper = card.object.element
      const savedZIndex = wrapper.dataset.portolanZIndex || '1000'
      wrapper.style.setProperty('z-index', savedZIndex, 'important')
    }
  }

  /**
   * Close the most recently focused card
   * @returns true if a card was closed, false if no cards are open
   */
  closeMostRecentCard(): boolean {
    // Try focused card first, then fall back to any card
    const cardId = this.lastFocusedCardId && this.conversationCards.has(this.lastFocusedCardId)
      ? this.lastFocusedCardId
      : this.conversationCards.keys().next().value

    if (cardId) {
      this.closeConversationCard(cardId)
      return true
    }
    return false
  }

  /**
   * Minimize the most recently focused card
   * @returns true if a card was minimized, false if no cards are open
   */
  minimizeMostRecentCard(): boolean {
    // Try last focused card first
    const focusedCard = this.lastFocusedCardId
      ? this.conversationCards.get(this.lastFocusedCardId)
      : null

    if (focusedCard && !focusedCard.minimized) {
      focusedCard.minimize()
      return true
    }

    // Fall back to any non-minimized card
    for (const card of this.conversationCards.values()) {
      if (!card.minimized) {
        card.minimize()
        return true
      }
    }

    return false
  }

  /**
   * Close a conversation card
   */
  closeConversationCard(workerId: string): void {
    const card = this.conversationCards.get(workerId)
    if (!card) return

    // Save card state before closing
    this.saveCardState(workerId, card)

    // Remove from swarm group (card is child of swarm, not scene)
    const swarm = this.workerSwarms.get(workerId)
    if (swarm) {
      swarm.removeChild(card.object)
      // Show the label again
      swarm.showLabel()
    } else {
      // Fallback: remove from scene if swarm not found
      this.scene.remove(card.object)
    }

    card.dispose()
    this.conversationCards.delete(workerId)

    // Clear lastFocusedCardId if this was the focused card
    if (this.lastFocusedCardId === workerId) {
      this.lastFocusedCardId = null
    }
  }

  /**
   * Close all conversation cards
   */
  closeAllConversationCards(): void {
    for (const workerId of this.conversationCards.keys()) {
      this.closeConversationCard(workerId)
    }
  }

  /**
   * Check if a conversation card is open for a worker
   */
  hasConversationCard(workerId: string): boolean {
    return this.conversationCards.has(workerId)
  }

  /**
   * Handle WebSocket conversation update for cards
   * The tmuxSession from server is prefixed for remote sessions: "originId/tmuxSession"
   */
  handleConversationMessage(sessionId: string | undefined, tmuxSession: string, messages: ConversationMessage[]): void {
    for (const card of this.conversationCards.values()) {
      card.handleMessage(sessionId, tmuxSession, messages)
    }
  }

  /**
   * Re-fetch conversations for all open cards (called on WebSocket reconnect)
   */
  refetchAllConversations(): void {
    for (const card of this.conversationCards.values()) {
      card.refetch()
    }
  }

  /** Scale factor for camera distance (below threshold: 1.0, above: shrinks proportionally) */
  private readonly SCALE_THRESHOLD = 8  // Cards reach full size at this distance

  private calculateCardScale(cameraDistance: number): number {
    if (cameraDistance <= this.SCALE_THRESHOLD) return 1.0
    return this.SCALE_THRESHOLD / cameraDistance
  }

  // ═══════════════════════════════════════════════════════════
  // CARD STATE PERSISTENCE (server-side via ~/.portolan/card-states.json)
  // ═══════════════════════════════════════════════════════════

  // Cache loaded states to avoid repeated fetches
  private workerStateCache: Map<string, { size?: { width: number; height: number }; swarmOffset?: { x: number; z: number } }> = new Map()
  private workerStateCacheLoaded = false

  /**
   * Save card size to server
   */
  private saveCardState(workerId: string, card: ConversationCard): void {
    const size = card.getSize()
    const existing = this.workerStateCache.get(workerId)

    // Update cache immediately (preserve swarmOffset)
    this.workerStateCache.set(workerId, { ...existing, size })

    // Fire and forget - don't await
    fetch(`http://localhost:4004/card-state/${encodeURIComponent(workerId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ size }),
    }).catch(() => {
      // Ignore save errors
    })
  }

  /**
   * Save swarm position to server
   */
  private saveSwarmOffset(workerId: string, swarmOffset: { x: number; z: number }): void {
    const existing = this.workerStateCache.get(workerId)

    // Update cache immediately (preserve size)
    this.workerStateCache.set(workerId, { ...existing, swarmOffset })

    // Fire and forget - don't await
    fetch(`http://localhost:4004/card-state/${encodeURIComponent(workerId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ swarmOffset }),
    }).catch(() => {
      // Ignore save errors
    })
  }

  /**
   * Load saved card state from cache
   */
  private loadCardState(workerId: string): { size?: { width: number; height: number } } | null {
    return this.workerStateCache.get(workerId) ?? null
  }

  /**
   * Load saved swarm offset from cache
   */
  private loadSwarmOffset(workerId: string): { x: number; z: number } | null {
    return this.workerStateCache.get(workerId)?.swarmOffset ?? null
  }

  /**
   * Preload all worker states from server into cache.
   * Called early to have swarm positions ready when rendering.
   */
  async ensureWorkerStatesLoaded(): Promise<void> {
    if (this.workerStateCacheLoaded) return

    try {
      const response = await fetch('http://localhost:4004/card-states')
      if (response.ok) {
        const data = await response.json()
        for (const state of data.states || []) {
          this.workerStateCache.set(state.workerId, {
            size: state.size,
            swarmOffset: state.swarmOffset,
          })
        }
      }
    } catch {
      // Ignore load errors - will use defaults
    }
    this.workerStateCacheLoaded = true
  }

  /**
   * Dispose all resources (call before recreating during HMR)
   */
  dispose(): void {
    // Close all conversation cards
    this.closeAllConversationCards()

    // Remove and dispose all hex meshes
    for (const key of this.hexMeshes.keys()) {
      this.removeHex(key)
    }

    // Dispose scene objects
    const sceneObjects: Array<{ ref: Mesh | Group | null; clear: () => void }> = [
      { ref: this.groundPlane, clear: () => { this.groundPlane = null } },
      { ref: this.rhumbLinesGroup, clear: () => { this.rhumbLinesGroup = null } },
      { ref: this.selectionRing, clear: () => { this.selectionRing = null } },
    ]
    for (const { ref, clear } of sceneObjects) {
      if (ref) {
        this.disposeObject(ref)
        this.scene.remove(ref)
        clear()
      }
    }

    // Dispose sprite managers and swarms
    this.citySprites.dispose()
    for (const swarm of this.workerSwarms.values()) {
      swarm.dispose()
    }
    this.workerSwarms.clear()
  }
}
