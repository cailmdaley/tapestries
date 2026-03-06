// Types for portolan

export interface HexCoord {
  q: number
  r: number
}

export interface CartesianCoord {
  x: number
  z: number
}

// Worker activity event
export interface Activity {
  tool: string
  summary?: string
  fullPath?: string   // Full file path for Read/Write/Edit
  timestamp: number
  eventType?: 'tool' | 'user_prompt'  // Distinguish tool calls from user prompts
  prompt?: string     // Full prompt text for user_prompt events
}

// Conversation message from Claude transcript
export interface ConversationMessage {
  type: 'user' | 'assistant' | 'thinking' | 'tool_use' | 'tool_result' | 'system'
  content: string
  timestamp: string
  toolName?: string       // For tool_use
  toolInput?: any         // For tool_use
  toolUseId?: string      // For linking tool_use to tool_result
  preview?: string        // For thinking blocks
  systemType?: 'skill' | 'reminder'  // For system messages
}

// Git status for a repository
export interface GitStatus {
  branch: string
  ahead: number
  behind: number
  staged: { added: number; modified: number; deleted: number }
  unstaged: { added: number; modified: number; deleted: number }
  untracked: number
  totalFiles: number
  linesAdded: number
  linesRemoved: number
  lastCommitTime: number | null
  lastCommitMessage: string | null
  isRepo: boolean
  lastChecked: number
}

// Server types (raw from WebSocket)
export interface ServerCity {
  id: string
  name: string
  path: string
  position: HexCoord
  fiberCount?: number
  hasClaims?: boolean  // Has claims directory (workflow/config or results/claims)
  hasPlaygrounds?: boolean  // Has .portolan/playgrounds/ with HTML files
  isDormant?: boolean  // No active sessions (persisted city with no workers)
  gitStatus?: GitStatus  // Git repository status
  originId: string  // 'local' | 'remote-{hostname}'
}

export interface ServerSession {
  id: string
  name: string
  tmuxSession: string
  cwd: string
  cityId?: string | null
  workerHex?: HexCoord
  cli?: string  // 'claude' | 'codex'
  status: 'idle' | 'working' | 'offline'
  createdAt: number
  lastActivity: number
  originId: string  // 'local' | 'remote-{hostname}'
}

export interface ServerOrigin {
  id: string
  name: string
  type: 'local' | 'remote'
  sshHost?: string
  plannotatorPort?: number
  position: HexCoord
  connectedAt: number
  lastSeen: number
}

// Frontend types (normalized for rendering)
export interface City {
  id: string
  name: string
  path: string
  hex: HexCoord
  fiberCount: number
  hasClaims: boolean
  hasPlaygrounds: boolean
  isDormant: boolean
  gitStatus?: GitStatus
  originId: string
}

export interface Session {
  id: string
  name: string
  tmuxSession: string
  cityId: string | null
  hex: HexCoord | null
  cli?: string  // 'claude' | 'codex'
  status: 'idle' | 'working'
  originId: string  // 'local' | 'remote-{hostname}'
  lastActivity: number
}

// Transform server city to frontend city
export function normalizeCity(city: ServerCity): City {
  return {
    id: city.id,
    name: city.name,
    path: city.path,
    hex: city.position,
    fiberCount: city.fiberCount ?? 0,
    hasClaims: city.hasClaims ?? false,
    hasPlaygrounds: city.hasPlaygrounds ?? false,
    isDormant: city.isDormant ?? false,
    gitStatus: city.gitStatus,
    originId: city.originId,
  }
}

// Transform server session to frontend session
export function normalizeSession(session: ServerSession): Session {
  return {
    id: session.id,
    name: session.name,
    tmuxSession: session.tmuxSession,
    cityId: session.cityId ?? null,
    hex: session.workerHex ?? null,
    // Map 'offline' to 'idle' for rendering (offline sessions shouldn't appear anyway)
    status: session.status === 'offline' ? 'idle' : session.status,
    originId: session.originId,
    cli: session.cli,
    lastActivity: session.lastActivity,
  }
}

// Porch Morning palette (hex colors used in Three.js)
export const PALETTE = {
  bgPrimary: 0xc8b8a8,     // Ground plane
  border: 0xa89888,        // Hex borders
  cityHex: 0x9a7b35,       // Gold — active cities
  cityDormant: 0x8a8070,   // Muted — dormant cities
  workerIdle: 0x7a7368,    // Muted — idle workers
  workerActive: 0x5a7b7b,  // Teal — working
  selection: 0xc4a86a,     // Gold highlight ring
} as const

