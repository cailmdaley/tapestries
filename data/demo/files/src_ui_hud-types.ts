// Shared types for CityHUD

export interface Fiber {
  id: string
  title: string
  status: string
  kind: string
  priority: number
  createdAt: string
  body?: string
  reason?: string
}

export interface SearchResult {
  path: string
  fullPath: string
  line?: number
  match?: string
}

export interface DirectoryEntry {
  name: string
  type: 'file' | 'dir'
}
