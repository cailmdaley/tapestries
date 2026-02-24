// main.ts - Bootstrap and render loop

import {
  Scene,
  WebGLRenderer,
  AmbientLight,
  DirectionalLight,
  Color,
} from 'three'
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import { HexGrid } from './render/HexGrid'
import { ZoneRenderer } from './render/ZoneRenderer'
import { Camera } from './render/Camera'
import { CityHUD } from './ui/CityHUD'
import { FileViewerModal } from './ui/FileViewerModal'
import { ContextMenu } from './ui/ContextMenu'
// GlobalView type for potential future view switching
type GlobalView = 'map' | 'plots' | 'plans'
import { ViewOverlay } from './ui/ViewOverlay'
import { TabbedPlansView } from './ui/TabbedPlansView'
import { TapestryView } from './ui/TapestryView'
import { PlaygroundViewer } from './ui/PlaygroundViewer'
import { NewWorkerDialog } from './ui/NewWorkerDialog'
import type { Activity, City, Session, ServerCity, ServerSession, ServerOrigin, HexCoord } from './state/types'
import { PALETTE, normalizeCity, normalizeSession } from './state/types'

// Get canvas
const canvas = document.getElementById('canvas') as HTMLCanvasElement

// Context menu: double-click (universal) + right-click (Chrome/Firefox)
// Safari doesn't reliably fire contextmenu on canvas elements
// Named handler for HMR cleanup
const contextMenuHandler = (e: MouseEvent) => {
  const target = e.target as HTMLElement
  // Handle context menu on canvas or any label (city/worker)
  if (target === canvas || target.closest('.label-container')) {
    e.preventDefault()
    handleContextMenu(e.clientX, e.clientY)
  }
}
document.addEventListener('contextmenu', contextMenuHandler)

// Alias for event handlers
const canvasOverlay = canvas

// Setup renderer
const renderer = new WebGLRenderer({
  canvas,
  antialias: true,
})
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setClearColor(new Color(PALETTE.bgPrimary))
renderer.shadowMap.enabled = true

// CSS2D renderer for HTML labels (small caps, petite caps)
const labelRenderer = new CSS2DRenderer()
labelRenderer.setSize(window.innerWidth, window.innerHeight)
labelRenderer.domElement.style.position = 'absolute'
labelRenderer.domElement.style.top = '0'
labelRenderer.domElement.style.left = '0'
labelRenderer.domElement.style.pointerEvents = 'none'
labelRenderer.domElement.style.userSelect = 'none'
labelRenderer.domElement.classList.add('label-container')
document.body.appendChild(labelRenderer.domElement)

// Setup scene
const scene = new Scene()

// Lighting - warm desert sun
const ambientLight = new AmbientLight(0xffeedd, 0.6)
scene.add(ambientLight)

const directionalLight = new DirectionalLight(0xfff5e1, 0.8)
directionalLight.position.set(10, 20, 5)
directionalLight.castShadow = true
directionalLight.shadow.mapSize.width = 2048
directionalLight.shadow.mapSize.height = 2048
directionalLight.shadow.camera.near = 1
directionalLight.shadow.camera.far = 50
directionalLight.shadow.camera.left = -20
directionalLight.shadow.camera.right = 20
directionalLight.shadow.camera.top = 20
directionalLight.shadow.camera.bottom = -20
scene.add(directionalLight)

// Setup hex grid (size 50 = radius of 50 hexes, hexRadius 1.0)
// hexRadius = 1/sqrt(3) makes adjacent hex centers exactly 1 world unit apart
const hexGrid = new HexGrid(100, 1 / Math.sqrt(3))

// Setup camera (use overlay for events - Safari compatibility)
const camera = new Camera(canvas, canvasOverlay)

// Setup zone renderer
const zoneRenderer = new ZoneRenderer(scene, hexGrid)

// Load worker states early so swarm positions are available when first rendering
zoneRenderer.ensureWorkerStatesLoaded()

// Expose for debugging:
//   window.zoneRenderer.debugResourceCounts() - scene traversal counts
//   window.debugWebGL() - WebGL resource counts from renderer.info
;(window as unknown as { zoneRenderer: typeof zoneRenderer }).zoneRenderer = zoneRenderer
;(window as unknown as { debugWebGL: () => void }).debugWebGL = () => {
  const info = renderer.info
  console.table({
    'Geometries (GPU)': info.memory.geometries,
    'Textures (GPU)': info.memory.textures,
    'Draw calls': info.render.calls,
    'Triangles': info.render.triangles,
    'Points': info.render.points,
    'Lines': info.render.lines,
  })
}

// Wire up worker label click handlers (CSS2D labels need direct handlers)
// Opens map-pinned conversation card instead of side panel
zoneRenderer.setWorkerClickHandler((workerId, _tmuxSession) => {
  const session = sessions.find(s => s.id === workerId)
  if (session) {
    zoneRenderer.openConversationCard(session)
  }
})

zoneRenderer.setWorkerDblClickHandler((workerId, _tmuxSession) => {
  focusKittyTab(workerId)
})

// Wire up city label click handler (needed for remote cities without sprites)
// Uses handleCityClick defined below (after cityPanel initialization)
zoneRenderer.setCityLabelClickHandler((cityId) => {
  const city = cities.find(c => c.id === cityId)
  if (city) handleCityClick(city)
})

// Provide camera's screen-to-world conversion for accurate drag
zoneRenderer.setScreenToWorldConverter((x, y) => camera.screenToWorld(x, y))

// Setup city HUD (corner-anchored widgets, replaces CityPanel)
const cityPanel = new CityHUD()

// Shared handler for city clicks (used by sprite click and label click)
function handleCityClick(city: City): void {
  selectedHex = city.hex

  // Focus on city and zoom to detail level
  const pos = hexGrid.axialToCartesian(city.hex)
  camera.focusAndZoom(pos, 6, 0.95)

  if (city.isDormant && city.originId !== 'local') {
    activateRemoteCity(city)
  } else {
    cityPanel.show(city)
    cityPanel.updateWorkers(sessions)
  }
}

// Setup file viewer modal
const fileViewerModal = new FileViewerModal()

// Wire up file click from conversation cards to file viewer
zoneRenderer.setCardFileClickHandler((fullPath, originId, workerId) => {
  const city = findBestMatchingCity(originId, fullPath)
  fileViewerModal.show(fullPath, originId, workerId, undefined, city?.path, city?.id)
})

// Wire up worker lookup for send-to-worker feature
fileViewerModal.setOnGetWorkers(async (originId: string, path: string) => {
  const city = findBestMatchingCity(originId, path)
  if (!city) return []

  // Return workers (sessions) assigned to this city
  return sessions
    .filter(s => s.cityId === city.id && s.originId === originId)
    .map(s => ({ id: s.id, name: s.name, tmuxSession: s.tmuxSession }))
})

// Wire up file search click from city panel to file viewer
cityPanel.setOnOpenFile((fullPath, originId, cityPath, cityId, line) => {
  fileViewerModal.show(fullPath, originId, undefined, undefined, cityPath, cityId, line)
})

// Setup context menu
const contextMenu = new ContextMenu()

// Setup new worker dialog
const newWorkerDialog = new NewWorkerDialog()

// Wire up new worker dialog to city panel
cityPanel.setNewWorkerDialog(newWorkerDialog)

// Wire up worker click → open conversation card on map
cityPanel.setOnFocusWorker((sessionId) => {
  const session = sessions.find(s => s.id === sessionId)
  if (session) {
    zoneRenderer.openConversationCard(session)
  }
})

// Setup view switching
const viewOverlay = new ViewOverlay()
const tabbedPlansView = new TabbedPlansView()

// View URLs from environment or defaults
const PLOT_SERVER_URL = 'http://localhost:8873'  // Plot server gallery

// Track current view state (used in handleViewChange)
// @ts-expect-error Tracked for potential state persistence
let currentView: GlobalView = 'map'

// @ts-expect-error Kept for future view switching via keyboard/API
function handleViewChange(view: GlobalView): void {
  currentView = view

  if (view === 'map') {
    // Return to hex grid
    viewOverlay.hide()
    tabbedPlansView.hide()
    canvas.style.display = 'block'
  } else if (view === 'plots') {
    // Show plot server gallery
    canvas.style.display = 'none'
    tabbedPlansView.hide()
    viewOverlay.show(PLOT_SERVER_URL)
  } else if (view === 'plans') {
    // Show tabbed plannotator view
    canvas.style.display = 'none'
    viewOverlay.hide()
    tabbedPlansView.show(origins)
  }
}

// ViewSwitcher removed — view stays on 'map' for now

// Tapestry view — native DAG visualization for fibers
const tapestryView = new TapestryView()

// Wire up View Claims button — uses native TapestryView
cityPanel.setOnViewClaims((city) => {
  tapestryView.show(city)
})

// Wire up worker lookup for tapestry view
function getCityWorkers(city: City): { id: string; name: string; tmuxSession: string }[] {
  return sessions
    .filter(s => s.cityId === city.id && s.originId === city.originId)
    .map(s => ({ id: s.id, name: s.name, tmuxSession: s.tmuxSession }))
}
tapestryView.setOnGetWorkers(getCityWorkers)

// Wire up file navigation from tapestry view — open files in file viewer
tapestryView.setOnOpenFile((filePath, city, line) => {
  fileViewerModal.show(filePath, city.originId, undefined, undefined, city.path, city.id, line)
})

// Setup playground viewer
const playgroundViewer = new PlaygroundViewer()

// Wire up View Playgrounds button
cityPanel.setOnViewPlaygrounds((city) => {
  playgroundViewer.show(city)
})


// State
let cities: City[] = []
let sessions: Session[] = []
let origins: ServerOrigin[] = []
let ws: WebSocket | null = null
let wsCleanedUp = false  // Prevent reconnect on HMR cleanup
let hasReceivedInitialState = false  // Track first state for initial camera focus
// @ts-expect-error Tracked for potential state persistence
let selectedHex: { q: number; r: number } | null = null

// Move mode: when set, next click will move this city to that hex
let movingCityId: string | null = null

// Activity stream per tmux session
const activityBySession = new Map<string, Activity[]>()
const MAX_ACTIVITIES_PER_SESSION = 10

// Handle incoming activity event
function handleActivityEvent(activity: { tmuxSession: string; tool: string; summary?: string; fullPath?: string; timestamp: number }): void {
  // Store activity
  let activities = activityBySession.get(activity.tmuxSession)
  if (!activities) {
    activities = []
    activityBySession.set(activity.tmuxSession, activities)
  }
  activities.unshift({
    tool: activity.tool,
    summary: activity.summary,
    fullPath: activity.fullPath,
    timestamp: activity.timestamp,
  })
  if (activities.length > MAX_ACTIVITIES_PER_SESSION) {
    activities.pop()
  }

  // Update ZoneRenderer worker marker activity
  zoneRenderer.updateWorkerActivity(activity.tmuxSession, activities)

  // Update HUD worker list (activity text may have changed)
  cityPanel.updateWorkers(sessions)
}

// Connect to server
function connectWebSocket(): void {
  const wsUrl = `ws://${window.location.hostname}:4004`
  ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    console.log('Connected to portolan server')
    cityPanel.setWebSocket(ws!)
    // Re-fetch conversations for all open cards (recovers from missed messages during disconnect)
    zoneRenderer.refetchAllConversations()
  }

  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data)
      // Route to panels that handle specific message types
      if (cityPanel.handleMessage(message)) return

      // Route conversation updates to open cards (by sessionId + tmuxSession)
      if (message.type === 'conversation' && message.messages) {
        zoneRenderer.handleConversationMessage(message.sessionId, message.tmuxSession, message.messages)
      }

      handleMessage(message)
    } catch (e) {
      console.error('Failed to parse message:', e)
    }
  }

  ws.onclose = () => {
    if (wsCleanedUp) return  // Don't reconnect on HMR cleanup
    console.log('Disconnected from server, reconnecting...')
    setTimeout(connectWebSocket, 2000)
  }

  ws.onerror = (e) => {
    console.error('WebSocket error:', e)
  }
}

interface ServerState {
  cities: ServerCity[]
  sessions: ServerSession[]
  origins?: ServerOrigin[]
  activities?: Record<string, Activity[]>
}

interface ConfirmUnpinMessage {
  type: 'confirmUnpin'
  cityId: string
  cityName: string
  sessionCount: number
}

interface CityPinnedMessage {
  type: 'cityPinned'
  city: ServerCity
}

interface CityUnpinnedMessage {
  type: 'cityUnpinned'
  cityId: string
}

interface CityMovedMessage {
  type: 'cityMoved'
  cityId: string
  newPosition: { q: number; r: number }
}

interface ErrorMessage {
  type: 'error'
  message: string
}

interface ActivityMessage {
  type: 'activity'
  activity: {
    tmuxSession: string
    tool: string
    summary?: string
    fullPath?: string
    timestamp: number
  }
}

type ServerMessage = ServerState | ConfirmUnpinMessage | CityPinnedMessage | CityUnpinnedMessage | CityMovedMessage | ErrorMessage | ActivityMessage

function handleMessage(message: ServerMessage): void {
  // Handle persistence-related messages
  if ('type' in message) {
    if (message.type === 'confirmUnpin') {
      // Show confirmation dialog
      const msg = message as ConfirmUnpinMessage
      const confirmed = window.confirm(
        `City "${msg.cityName}" has ${msg.sessionCount} active session(s).\n\n` +
        `The city will remain visible while sessions are active.\n` +
        `Remove persistence anyway?`
      )
      if (confirmed && ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'confirmUnpinCity', cityId: msg.cityId }))
      }
      return
    }

    if (message.type === 'cityPinned') {
      return
    }

    if (message.type === 'cityUnpinned') {
      return
    }

    if (message.type === 'cityMoved') {
      return
    }

    if (message.type === 'error') {
      const errorMsg = (message as ErrorMessage).message
      console.error('[Frontend] Server error:', errorMsg)
      // Show user-facing errors in an alert
      alert(errorMsg)
      return
    }

    if (message.type === 'activity') {
      const actMsg = message as ActivityMessage
      handleActivityEvent(actMsg.activity)
      return
    }
  }

  // Server sends state directly: { cities: [...], sessions: [...], origins: [...], activities: {...} }
  const state = message as ServerState
  if (state.cities && state.sessions) {
    cities = state.cities.map(normalizeCity)
    sessions = state.sessions.map(normalizeSession)
    if (state.origins) {
      origins = state.origins
      // Update tabbed plans view if visible (handles new agents connecting)
      tabbedPlansView.update(origins)
    }
    // Populate activity history from state (backfill on connect)
    if (state.activities) {
      for (const [tmuxSession, acts] of Object.entries(state.activities)) {
        activityBySession.set(tmuxSession, acts)
        zoneRenderer.updateWorkerActivity(tmuxSession, acts)
      }
    }
    // Clean up activities for sessions that no longer exist
    const currentTmuxSessions = new Set(sessions.map(s => s.tmuxSession))
    for (const tmuxSession of activityBySession.keys()) {
      if (!currentTmuxSessions.has(tmuxSession)) {
        activityBySession.delete(tmuxSession)
      }
    }
    zoneRenderer.updateState(cities, sessions)

    // Update HUD worker list if visible
    cityPanel.updateWorkers(sessions)

    // On first state, focus camera on most recently active city
    if (!hasReceivedInitialState && cities.length > 0) {
      hasReceivedInitialState = true

      // Find city with most recent session activity
      let mostRecentCity: City | null = null
      let mostRecentTime = 0

      for (const session of sessions) {
        if (session.cityId && session.lastActivity > mostRecentTime) {
          const city = cities.find(c => c.id === session.cityId)
          if (city) {
            mostRecentCity = city
            mostRecentTime = session.lastActivity
          }
        }
      }

      // Fall back to first city if no sessions
      const targetCity = mostRecentCity || cities[0]
      console.log('[InitialFocus]', mostRecentCity ? `Most recent: ${targetCity.name}` : `Fallback: ${targetCity.name}`,
        sessions.length, 'sessions,', sessions.filter(s => s.cityId).length, 'with cityId')
      const pos = hexGrid.axialToCartesian(targetCity.hex)
      camera.focusAndZoom(pos, 6, 0.95)

      // Restore tapestry from URL — if ?city= is set, auto-open the tapestry
      const urlCityId = new URLSearchParams(window.location.search).get('city')
      if (urlCityId) {
        const urlCity = cities.find(c => c.id === urlCityId)
        if (urlCity) tapestryView.show(urlCity)
      }
    }
  }
}

// Swarm drag handling (intercept mousedown on swarms before camera pan)
canvasOverlay.addEventListener('mousedown', (e) => {
  // Only handle left button
  if (e.button !== 0) return

  const worldPos = camera.screenToWorld(e.clientX, e.clientY)
  const workerHit = zoneRenderer.getWorkerAtWorldPos(worldPos.x, worldPos.z)

  if (workerHit) {
    // Start swarm drag - this prevents camera from panning
    if (zoneRenderer.startSwarmDrag(workerHit.workerId, e.clientX, e.clientY)) {
      e.stopPropagation()  // Prevent camera from starting its pan
    }
  }
}, true)  // Use capture phase to run before camera's handler

// Click handling
canvasOverlay.addEventListener('click', (e) => {
  // Ignore clicks that were drags (including swarm drags)
  if (camera.dragging || zoneRenderer.isDraggingSwarm) return

  const worldPos = camera.screenToWorld(e.clientX, e.clientY)
  const hex = hexGrid.cartesianToHex(worldPos.x, worldPos.z)

  // Handle move mode: clicking a hex moves the city there
  if (movingCityId) {
    moveCity(movingCityId, hex)
    movingCityId = null
    document.body.style.cursor = ''
    return
  }

  // First check for worker swarm at world position
  const workerHit = zoneRenderer.getWorkerAtWorldPos(worldPos.x, worldPos.z)
  if (workerHit) {
    const session = sessions.find(s => s.id === workerHit.workerId)
    if (session) {
      selectedHex = session.hex || hex
      // Open map-pinned conversation card
      zoneRenderer.openConversationCard(session)
      return
    }
  }

  // Then check for city at world position (radius-based, for large sprites)
  const cityHit = zoneRenderer.getCityAtWorldPos(worldPos.x, worldPos.z)
  if (cityHit) {
    const city = cities.find(c => c.id === cityHit.entityId)
    if (city) {
      handleCityClick(city)
      return
    }
  }

  // Fall back to hex-based entity lookup (for workers)
  const entity = zoneRenderer.getEntityAtHex(hex)

  if (entity?.type === 'worker' && entity.entityId) {
    selectedHex = hex
    const session = sessions.find(s => s.id === entity.entityId)
    if (session) {
      // Open map-pinned conversation card
      zoneRenderer.openConversationCard(session)
    }
  } else {
    // Empty tile: close most recent card, or clear selection if none
    const closed = zoneRenderer.closeMostRecentCard()
    if (!closed) {
      selectedHex = null
    }
  }
})

// Double-click for primary actions (focus terminal / new worker)
canvasOverlay.addEventListener('dblclick', (e) => {
  if (camera.dragging) return

  const worldPos = camera.screenToWorld(e.clientX, e.clientY)
  const hex = hexGrid.cartesianToHex(worldPos.x, worldPos.z)

  // First check for worker swarm (double-click focuses terminal)
  const workerHit = zoneRenderer.getWorkerAtWorldPos(worldPos.x, worldPos.z)
  if (workerHit) {
    focusKittyTab(workerHit.workerId)
    return
  }

  // Then check for city at world position (radius-based)
  const cityHit = zoneRenderer.getCityAtWorldPos(worldPos.x, worldPos.z)
  if (cityHit) {
    const city = cities.find(c => c.id === cityHit.entityId)
    if (city) {
      promptNewWorker(city)
      return
    }
  }

  // Fall back to hex-based lookup
  const entity = zoneRenderer.getEntityAtHex(hex)

  if (entity?.type === 'worker' && entity.entityId) {
    focusKittyTab(entity.entityId)
  } else {
    // Double-click empty tile → new worker if near city
    const nearestCity = findNearestCity(hex)
    if (nearestCity && hexGrid.distance(hex, nearestCity.hex) <= 3) {
      promptNewWorker(nearestCity)
    }
  }
})

// Force Touch (Mac trackpad) for context menu
// Track mouse position for force touch (event doesn't include coordinates)
let forceMouseX = 0
let forceMouseY = 0
let forceTouchFired = false

canvasOverlay.addEventListener('mousemove', (e) => {
  forceMouseX = e.clientX
  forceMouseY = e.clientY

  // Custom cursor based on what's under the mouse
  if (camera.dragging || movingCityId) return  // Don't change cursor while dragging

  const worldPos = camera.screenToWorld(e.clientX, e.clientY)

  // Check for worker swarm first (smaller hit area, more specific)
  const workerHit = zoneRenderer.getWorkerAtWorldPos(worldPos.x, worldPos.z)
  if (workerHit) {
    canvas.style.cursor = 'grab'
    return
  }

  if (zoneRenderer.getCityAtWorldPos(worldPos.x, worldPos.z)) {
    canvas.style.cursor = 'var(--cursor-bird)'
    return
  }

  canvas.style.cursor = ''
})

// Claim gesture to prevent system Quick Look
canvasOverlay.addEventListener('webkitmouseforcewillbegin', (e) => {
  e.preventDefault()
})

// Force click: deep-press a city opens claims/playground, otherwise context menu
canvasOverlay.addEventListener('webkitmouseforcedown', () => {
  if (camera.dragging) return
  forceTouchFired = true

  // Hit-test for city with deep content
  const worldPos = camera.screenToWorld(forceMouseX, forceMouseY)
  const cityHit = zoneRenderer.getCityAtWorldPos(worldPos.x, worldPos.z)
  if (cityHit) {
    const city = cities.find(c => c.id === cityHit.entityId)
    if (city && (city.hasClaims || city.hasPlaygrounds)) {
      handleCityClick(city)
      if (city.hasClaims) {
        tapestryView.show(city)
      } else {
        playgroundViewer.show(city)
      }
      return
    }
  }

  handleContextMenu(forceMouseX, forceMouseY)
})

// Suppress click after force touch (force touch fires normal click on release)
// Named handler for HMR cleanup
const forceClickCaptureHandler = (e: MouseEvent) => {
  if (forceTouchFired) {
    e.stopPropagation()
    forceTouchFired = false
  }
}
document.addEventListener('click', forceClickCaptureHandler, true) // capture phase to intercept before other handlers

// Right-click context menu handler
function handleContextMenu(clientX: number, clientY: number) {
  // Ignore if dragging
  if (camera.dragging) return

  const worldPos = camera.screenToWorld(clientX, clientY)
  const hex = hexGrid.cartesianToHex(worldPos.x, worldPos.z)

  // First check for city at world position (radius-based)
  const cityHit = zoneRenderer.getCityAtWorldPos(worldPos.x, worldPos.z)
  if (cityHit) {
    const city = cities.find(c => c.id === cityHit.entityId)
    if (city) {
      contextMenu.show(clientX, clientY, [
        {
          label: 'New Worker',
          action: () => promptNewWorker(city),
        },
        {
          label: 'Move City',
          action: () => startMoveCity(city.id),
        },
        {
          label: 'Remove City',
          action: () => unpinCity(city.id),
          danger: true,
        },
      ])
      return
    }
  }

  // Fall back to hex-based lookup
  const entity = zoneRenderer.getEntityAtHex(hex)

  if (entity?.type === 'worker' && entity.entityId) {
    // Worker right-click: show retire option
    const session = sessions.find(s => s.id === entity.entityId)
    if (session) {
      contextMenu.show(clientX, clientY, [
        {
          label: 'Focus Tab',
          action: () => focusKittyTab(session.id),
        },
        {
          label: 'Retire Worker',
          action: () => killWorker(session.id),
          danger: true,
        },
      ])
    }
  } else {
    // Check distance to nearest city
    const nearestCity = findNearestCity(hex)

    if (nearestCity && hexGrid.distance(hex, nearestCity.hex) <= 3) {
      // Within 3 tiles of a city: offer new worker
      contextMenu.show(clientX, clientY, [
        {
          label: `New Worker (${nearestCity.name})`,
          action: () => promptNewWorker(nearestCity),
        },
      ])
    } else {
      // Far from any city: offer new city
      contextMenu.show(clientX, clientY, [
        {
          label: 'Add City Here',
          action: () => promptAddCity(hex),
        },
      ])
    }
  }
}



// Find nearest city to a hex
function findNearestCity(hex: HexCoord): City | null {
  if (cities.length === 0) return null

  let nearest: City | null = null
  let minDist = Infinity

  for (const city of cities) {
    const dist = hexGrid.distance(hex, city.hex)
    if (dist < minDist) {
      minDist = dist
      nearest = city
    }
  }

  return nearest
}

// Find city with longest matching path for a file (most specific match)
function findBestMatchingCity(originId: string, filePath: string): City | null {
  let best: City | null = null
  for (const city of cities) {
    if (city.originId !== originId) continue
    if (!filePath.startsWith(city.path)) continue
    if (!best || city.path.length > best.path.length) {
      best = city
    }
  }
  return best
}

// Prompt for new worker name and create it
async function promptNewWorker(city: City): Promise<void> {
  const result = await newWorkerDialog.show(city.name)
  if (!result) return  // Cancelled

  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'newWorker',
      cityPath: city.path,
      name: result.name || undefined,  // undefined if empty (will auto-generate)
      cli: result.cli || undefined,  // 'claude' | 'codex'
      chrome: result.chrome || undefined,  // only send if true
      continue: result.continue || undefined,  // only send if true
    }))
  }
}

// Pin a new city at the given hex
function promptAddCity(hex: HexCoord): void {
  const path = window.prompt('Enter the full path for the new city:')
  if (!path) return

  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'pinCity',
      path: path.trim(),
      position: { q: hex.q, r: hex.r },
    }))
  } else {
    console.error('WebSocket not ready, readyState:', ws?.readyState)
  }
}

// Unpin a city
function unpinCity(cityId: string): void {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'unpinCity',
      cityId,
    }))
  }
}

// Start moving a city (enter move mode)
function startMoveCity(cityId: string): void {
  movingCityId = cityId
  document.body.style.cursor = 'crosshair'
}

// Move a city to a new hex position
function moveCity(cityId: string, hex: HexCoord): void {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'moveCity',
      cityId,
      newPosition: { q: hex.q, r: hex.r },
    }))
  }
}

// Kill a worker (tmux session)
function killWorker(sessionId: string): void {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'killWorker',
      sessionId,
    }))
  }
}

// Focus Kitty tab
function focusKittyTab(sessionId: string): void {
  // Send focus request to server
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'focus', sessionId }))
  }
}

// Activate dormant remote city (SSH + start agent)
async function activateRemoteCity(city: City): Promise<void> {

  try {
    const response = await fetch(`http://localhost:4004/activate-city?cityId=${city.id}`, {
      method: 'POST',
    })

    const result = await response.json()

    if (response.ok) {
      // Agent will connect and sessions will appear via WebSocket
      // Show panel while waiting
      cityPanel.show(city)
    } else {
      console.error(`[Activate] Failed: ${result.error}`)
      alert(`Failed to activate remote city: ${result.error}`)
    }
  } catch (err) {
    console.error('[Activate] Network error:', err)
    alert(`Failed to connect to server`)
  }
}

// Escape key: cancel move mode or close most recent card
function handleEscapeKey(e: KeyboardEvent): void {
  if (e.key !== 'Escape') return

  if (movingCityId) {
    movingCityId = null
    document.body.style.cursor = ''
    return
  }

  zoneRenderer.closeMostRecentCard()
}
window.addEventListener('keydown', handleEscapeKey)

// Cycling keyboard shortcuts
// Cmd+Alt+9/0: cycle workers, Cmd+Ctrl+9/0: cycle cities
let workerCycleIndex = -1
let cityCycleIndex = -1

function handleCycleKeys(e: KeyboardEvent): void {
  if (!e.metaKey) return
  if (e.code !== 'Digit9' && e.code !== 'Digit0') return

  const direction = e.code === 'Digit0' ? 1 : -1

  if (e.altKey && !e.ctrlKey) {
    // Cmd+Alt+9/0: cycle workers
    e.preventDefault()
    if (sessions.length === 0) return
    workerCycleIndex = (workerCycleIndex + direction + sessions.length) % sessions.length
    const session = sessions[workerCycleIndex]
    zoneRenderer.openConversationCard(session)
    // Focus camera on worker's swarm
    const swarmPos = zoneRenderer.getSwarmWorldPosition(session.id)
    if (swarmPos) camera.focusAndZoom(swarmPos, 6, 0.95)
  } else if (e.ctrlKey && !e.altKey) {
    // Cmd+Ctrl+9/0: cycle cities (only those with active workers)
    e.preventDefault()
    const activeCities = cities.filter(c => sessions.some(s => s.cityId === c.id))
    if (activeCities.length === 0) return
    cityCycleIndex = (cityCycleIndex + direction + activeCities.length) % activeCities.length
    handleCityClick(activeCities[cityCycleIndex])
  }
}
window.addEventListener('keydown', handleCycleKeys)

// Window resize handler
function resizeHandler(): void {
  renderer.setSize(window.innerWidth, window.innerHeight)
  labelRenderer.setSize(window.innerWidth, window.innerHeight)
  camera.resize()
}
window.addEventListener('resize', resizeHandler)

// Render loop
function animate(): void {
  requestAnimationFrame(animate)

  // Animate (breathing pulse, label visibility, distance fading)
  zoneRenderer.animate(camera.cameraDistance)

  renderer.render(scene, camera.camera)
  labelRenderer.render(scene, camera.camera)

  // Reapply card z-indexes after CSS2DRenderer (which overwrites them based on depth)
  zoneRenderer.reapplyCardZIndexes()
}

// Start
connectWebSocket()
animate()

// Add some mock data for testing when server is not available
setTimeout(() => {
  if (cities.length === 0) {
    const mockCities: City[] = [
      { id: '1', name: 'hexarchy-v2', path: '/projects/hexarchy-v2', hex: { q: 0, r: 0 }, fiberCount: 3, hasClaims: false, hasPlaygrounds: true, isDormant: false, originId: 'local' },
      { id: '2', name: 'loom', path: '/projects/loom', hex: { q: 2, r: -1 }, fiberCount: 7, hasClaims: true, hasPlaygrounds: false, isDormant: false, originId: 'local' },
      { id: '3', name: 'pure-eb', path: '/projects/pure-eb', hex: { q: -2, r: 1 }, fiberCount: 0, hasClaims: true, hasPlaygrounds: false, isDormant: true, originId: 'remote-candide' },
    ]
    const mockSessions: Session[] = [
      { id: 's1', name: 'claude-0', tmuxSession: 'mock-0', cityId: '1', hex: { q: 1, r: 0 }, status: 'working', originId: 'local', lastActivity: Date.now() },
      { id: 's2', name: 'claude-1', tmuxSession: 'mock-1', cityId: '1', hex: { q: 0, r: 1 }, status: 'idle', originId: 'local', lastActivity: Date.now() },
      { id: 's3', name: 'claude-2', tmuxSession: 'mock-2', cityId: '2', hex: { q: 3, r: -1 }, status: 'idle', originId: 'local', lastActivity: Date.now() },
    ]
    cities = mockCities
    sessions = mockSessions
    zoneRenderer.updateState(cities, sessions)
  }
}, 1000)

// HMR cleanup
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    // Close WebSocket and prevent reconnection attempts
    wsCleanedUp = true
    ws?.close()

    // Remove all event listeners (named handlers for clean unsubscribe)
    document.removeEventListener('contextmenu', contextMenuHandler)
    document.removeEventListener('click', forceClickCaptureHandler, true)
    window.removeEventListener('keydown', handleEscapeKey)
    window.removeEventListener('keydown', handleCycleKeys)
    window.removeEventListener('resize', resizeHandler)

    // Dispose UI panels (removes DOM and detaches document listeners)
    cityPanel.dispose()
    fileViewerModal.dispose()
    contextMenu.dispose()
    newWorkerDialog.dispose()
    viewOverlay.dispose()
    tabbedPlansView.dispose()
    tapestryView.dispose()
    playgroundViewer.dispose()

    // Dispose renderer components in reverse initialization order
    camera.dispose()
    zoneRenderer.dispose()
    renderer.dispose()

    // Clean up DOM elements
    labelRenderer.domElement.remove()
  })
}
