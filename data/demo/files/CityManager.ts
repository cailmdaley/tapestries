/**
 * CityManager - Manage cities (both session-derived and persisted)
 *
 * Cities can be:
 * - Session-derived: exist while ≥1 session has that cwd
 * - Persisted (pinned): survive beyond sessions, stored in ~/.portolan/cities.json
 *
 * Persisted cities form the base layer. Session activity overlays
 * fiber counts and workers but doesn't change position or existence.
 *
 * Each city has:
 * - A filesystem path (unique per origin)
 * - An originId (local or remote-{hostname})
 * - A hex grid position (q, r) — offset by origin's compass position
 * - A display name
 */
import { resolve, basename } from 'path';
import { randomUUID } from 'crypto';
import { existsSync, readdirSync } from 'fs';
import type { GitStatus } from './GitStatusManager.js';

// ============================================================================
// Types
// ============================================================================

export interface City {
  id: string;
  path: string;         // absolute filesystem path
  name: string;         // display name
  position: { q: number; r: number };
  fiberCount?: number;  // injected by FiberReader
  hasClaims?: boolean;  // has workflow/config or results/claims
  hasPlaygrounds?: boolean;  // has .portolan/playgrounds/ with files
  gitStatus?: GitStatus;  // injected by GitStatusManager
  createdAt?: number;
  originId: string;     // 'local' | 'remote-{hostname}'
}

export interface SessionInfo {
  cwd: string;
  originId: string;
}

interface OriginPosition {
  q: number;
  r: number;
}

// ============================================================================
// CityManager
// ============================================================================

export class CityManager {
  // In-memory: key → city (key = `${normalizedOrigin}:${path}`)
  private citiesByKey = new Map<string, City>();

  // Track which cities are pinned (won't be deleted when sessions leave)
  private pinnedCityIds = new Set<string>();

  // Track occupied worker hexes per city: Map<cityId, Set<hexKey>>
  private occupiedWorkerHexes = new Map<string, Set<string>>();

  // Track origin positions: Map<originId, position>
  private originPositions = new Map<string, OriginPosition>();

  // Track sshHost for each origin: Map<originId, sshHost>
  // Used to normalize keys so different login nodes share cities
  private originSshHosts = new Map<string, string>();

  constructor() {
    // Local origin is always at center
    this.originPositions.set('local', { q: 0, r: 0 });
  }

  /**
   * Set the sshHost for an origin (used for key normalization)
   * e.g., "remote-login05.leonardo.local" → "cineca-login05"
   */
  setOriginSshHost(originId: string, sshHost: string): void {
    // Extract base sshHost (e.g., "cineca-login05" → "cineca")
    const baseSshHost = sshHost.replace(/-login\d+$/, '');
    this.originSshHosts.set(originId, baseSshHost);
  }

  /**
   * Get the base sshHost for an origin (for key normalization)
   */
  private getBaseSshHost(originId: string): string | undefined {
    return this.originSshHosts.get(originId);
  }

  /**
   * Set the position for an origin (for remote origins)
   */
  setOriginPosition(originId: string, position: OriginPosition): void {
    this.originPositions.set(originId, position);
  }

  /**
   * Get the position for an origin
   */
  getOriginPosition(originId: string): OriginPosition {
    return this.originPositions.get(originId) || { q: 0, r: 0 };
  }

  /**
   * Add a persisted (pinned) city. Called on startup with cities from CityPersistence.
   * Position and ID come from persistence, not auto-assigned.
   */
  addPinnedCity(
    id: string,
    path: string,
    name: string,
    position: { q: number; r: number },
    originId: string
  ): City {
    const resolvedPath = resolve(path);
    const key = this.makeKey(originId, resolvedPath);

    // If city already exists at this path, update it to be pinned
    const existing = this.citiesByKey.get(key);
    if (existing) {
      // Update to persisted values
      existing.id = id;
      existing.position = position;
      existing.name = name;
      this.pinnedCityIds.add(id);
      return existing;
    }

    // Create new pinned city
    const city: City = {
      id,
      path: resolvedPath,
      name,
      position,
      originId,
    };

    this.citiesByKey.set(key, city);
    this.pinnedCityIds.add(id);
    return city;
  }

  /**
   * Pin an existing session-derived city or create a new pinned city
   * Returns the city (for CityPersistence to save)
   */
  pinCity(
    path: string,
    position: { q: number; r: number },
    originId: string = 'local',
    name?: string
  ): City {
    const resolvedPath = resolve(path);
    const key = this.makeKey(originId, resolvedPath);
    const existing = this.citiesByKey.get(key);

    if (existing) {
      // Pin the existing city, update position
      existing.position = position;
      if (name) existing.name = name;
      this.pinnedCityIds.add(existing.id);
      return existing;
    }

    // Create new pinned city
    const city: City = {
      id: randomUUID(),
      path: resolvedPath,
      name: name || basename(resolvedPath),
      position,
      originId,
    };

    this.citiesByKey.set(key, city);
    this.pinnedCityIds.add(city.id);
    return city;
  }

  /**
   * Unpin a city. If it has no sessions, it will be removed.
   * Returns session count for the city (for warning user).
   */
  unpinCity(cityId: string): { removed: boolean; sessionCount: number } {
    this.pinnedCityIds.delete(cityId);

    // Find the city
    for (const [key, city] of this.citiesByKey) {
      if (city.id === cityId) {
        // City will be removed by next updateFromSessions if no sessions
        // For now, just return that it's unpinned
        return { removed: false, sessionCount: 0 };
      }
    }

    return { removed: false, sessionCount: 0 };
  }

  /**
   * Check if a city is pinned
   */
  isPinned(cityId: string): boolean {
    return this.pinnedCityIds.has(cityId);
  }

  /**
   * Move a city to a new position.
   * Only pinned cities can be moved.
   * Returns the city or null if not found.
   */
  moveCity(cityId: string, newPosition: { q: number; r: number }): City | null {
    const city = this.getCityById(cityId);
    if (!city) return null;

    city.position = newPosition;
    return city;
  }

  /**
   * Get city by ID
   */
  getCityById(cityId: string): City | null {
    for (const city of this.citiesByKey.values()) {
      if (city.id === cityId) {
        return city;
      }
    }
    return null;
  }

  /**
   * Make city key from originId and path.
   * For remote origins with a known sshHost, normalizes the key so different
   * login nodes (e.g., login05, login07) share the same city.
   */
  private makeKey(originId: string, path: string): string {
    const resolvedPath = resolve(path);

    // For remote origins, use base sshHost if known (e.g., "cineca" instead of full hostname)
    // This allows cities to be shared across login nodes on the same HPC system
    if (originId !== 'local') {
      const baseSshHost = this.getBaseSshHost(originId);
      if (baseSshHost) {
        return `remote-${baseSshHost}:${resolvedPath}`;
      }
    }

    return `${originId}:${resolvedPath}`;
  }

  /**
   * Derive cities from a list of sessions.
   * Session-derived cities that no longer have sessions are removed.
   * Pinned cities are preserved regardless of session activity.
   * New session cwds get cities created (using persisted position if pinned).
   * Returns the current set of cities.
   */
  updateFromSessions(sessions: SessionInfo[]): City[] {
    // Build set of active keys
    const activeKeys = new Set<string>();
    for (const session of sessions) {
      const key = this.makeKey(session.originId, session.cwd);
      activeKeys.add(key);
    }

    // Remove cities that no longer have sessions (but keep pinned cities)
    // Collect keys to delete (avoid modifying map during iteration)
    const keysToDelete: string[] = [];
    for (const [key, city] of this.citiesByKey) {
      if (!activeKeys.has(key) && !this.pinnedCityIds.has(city.id)) {
        keysToDelete.push(key);
      }
    }
    for (const key of keysToDelete) {
      const city = this.citiesByKey.get(key);
      if (city) {
        this.citiesByKey.delete(key);
        this.occupiedWorkerHexes.delete(city.id);
      }
    }

    // Add cities for new session cwds, or update originId if accessing from different login node
    for (const session of sessions) {
      const key = this.makeKey(session.originId, session.cwd);
      const existing = this.citiesByKey.get(key);
      if (existing) {
        // Update originId if session is from a different login node (but same normalized key)
        // This keeps the city associated with the currently active agent
        if (existing.originId !== session.originId) {
          existing.originId = session.originId;
        }
      } else {
        const originPos = this.getOriginPosition(session.originId);
        const city: City = {
          id: randomUUID(),
          path: resolve(session.cwd),
          name: basename(session.cwd),
          position: this.autoAssignPosition(originPos, session.originId),
          originId: session.originId,
        };
        this.citiesByKey.set(key, city);
      }
    }

    return this.getCities();
  }

  /**
   * Get all cities, sorted by name
   */
  getCities(): City[] {
    return [...this.citiesByKey.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Find the city for a given path and origin (exact match)
   */
  findCityForPath(cwd: string, originId: string = 'local'): City | null {
    const key = this.makeKey(originId, cwd);
    return this.citiesByKey.get(key) || null;
  }

  /**
   * Assign a worker hex position for a session in a city.
   * Returns the next available hex in spiral order from city center.
   * First ring (distance 1) = workers 1-6
   * Second ring (distance 2) = workers 7-18
   */
  assignWorkerHex(cityId: string): { q: number; r: number } {
    // Ensure we have a set for this city
    if (!this.occupiedWorkerHexes.has(cityId)) {
      this.occupiedWorkerHexes.set(cityId, new Set());
    }

    const occupied = this.occupiedWorkerHexes.get(cityId)!;

    // Spiral outward from city center to find first unoccupied hex
    // Start at ring 1 (workers around the city center)
    for (let ring = 1; ring < 10; ring++) {
      const positions = this.hexRing(0, 0, ring);
      for (const pos of positions) {
        const key = `${pos.q},${pos.r}`;
        if (!occupied.has(key)) {
          // Mark as occupied
          occupied.add(key);
          return pos;
        }
      }
    }

    // Fallback (should never reach here with reasonable worker counts)
    return { q: 1, r: 0 };
  }

  /**
   * Release a worker hex position when a session ends or moves.
   */
  releaseWorkerHex(cityId: string, hex: { q: number; r: number }): void {
    if (!this.occupiedWorkerHexes.has(cityId)) {
      return;
    }

    const occupied = this.occupiedWorkerHexes.get(cityId)!;
    const key = `${hex.q},${hex.r}`;
    occupied.delete(key);
  }

  /**
   * Auto-assign a hex position by spiraling outward from origin center
   * Enforces minimum 4-tile spacing between city centers within the same origin
   */
  private autoAssignPosition(originPos: OriginPosition, originId: string): { q: number; r: number } {
    // Get cities for this origin only (cities from other origins don't constrain positioning)
    const originCities = this.getCities().filter(c => c.originId === originId);

    // Try origin center first (always valid for first city in this origin)
    const center = { q: originPos.q, r: originPos.r };
    if (this.isValidCityPosition(center, originCities, 4)) {
      return center;
    }

    // Spiral outward from origin center, checking both occupancy and minimum spacing
    for (let ring = 1; ring < 100; ring++) {
      const positions = this.hexRing(originPos.q, originPos.r, ring);
      for (const pos of positions) {
        if (this.isValidCityPosition(pos, originCities, 4)) {
          return pos;
        }
      }
    }

    // Fallback (should never reach here)
    return center;
  }

  /**
   * Calculate hex distance between two positions
   * Uses axial coordinate system: distance = (|q1-q2| + |q1+r1-q2-r2| + |r1-r2|) / 2
   */
  private hexDistance(a: { q: number; r: number }, b: { q: number; r: number }): number {
    return (
      Math.abs(a.q - b.q) +
      Math.abs(a.q + a.r - b.q - b.r) +
      Math.abs(a.r - b.r)
    ) / 2;
  }

  /**
   * Generate all hex positions in a ring around (centerQ, centerR)
   * Uses cube coordinate system
   */
  private hexRing(centerQ: number, centerR: number, radius: number): Array<{ q: number; r: number }> {
    if (radius === 0) {
      return [{ q: centerQ, r: centerR }];
    }

    const positions: Array<{ q: number; r: number }> = [];

    // Hex directions in axial (q, r) coordinates
    const directions = [
      { q: 1, r: 0 },   // E
      { q: 1, r: -1 },  // NE
      { q: 0, r: -1 },  // NW
      { q: -1, r: 0 },  // W
      { q: -1, r: 1 },  // SW
      { q: 0, r: 1 },   // SE
    ];

    // Start at radius steps in one direction
    let q = centerQ - radius;
    let r = centerR + radius;

    // Walk around the ring
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < radius; j++) {
        positions.push({ q, r });
        q += directions[i].q;
        r += directions[i].r;
      }
    }

    return positions;
  }

  /**
   * Check if a position is at least minDistance tiles from all existing cities
   */
  private isValidCityPosition(position: { q: number; r: number }, cities: City[], minDistance: number): boolean {
    for (const city of cities) {
      if (this.hexDistance(position, city.position) < minDistance) {
        return false;
      }
    }
    return true;
  }

  /**
   * Detect if a city has claims (workflow/config or results/claims directories)
   * Only works for local cities.
   */
  detectClaims(city: City): boolean {
    // Only detect for local cities (remote would need agent support)
    if (city.originId !== 'local') {
      return false;
    }

    const hasWorkflowConfig = existsSync(resolve(city.path, 'workflow/config'));
    const hasResultsClaims = existsSync(resolve(city.path, 'results/claims'));
    const hasFelt = existsSync(resolve(city.path, '.felt'));
    return hasWorkflowConfig || hasResultsClaims || hasFelt;
  }

  /**
   * Update hasClaims for local cities only.
   * Remote cities get hasClaims from agent data, so we don't overwrite.
   */
  updateClaimsStatus(): void {
    for (const city of this.citiesByKey.values()) {
      if (city.originId === 'local') {
        city.hasClaims = this.detectClaims(city);
      }
      // Remote cities: hasClaims is set by handleAgentSessionsUpdate
    }
  }

  /**
   * Detect if a city has playgrounds (.portolan/playgrounds/ with .html files)
   * Only works for local cities.
   */
  detectPlaygrounds(city: City): boolean {
    if (city.originId !== 'local') {
      return false;
    }

    const playgroundsDir = resolve(city.path, '.portolan/playgrounds');
    if (!existsSync(playgroundsDir)) {
      return false;
    }

    try {
      const files = readdirSync(playgroundsDir);
      return files.some(f => f.endsWith('.html'));
    } catch {
      return false;
    }
  }

  /**
   * Update hasPlaygrounds for local cities only.
   * Remote cities get hasPlaygrounds from agent data.
   */
  updatePlaygroundsStatus(): void {
    for (const city of this.citiesByKey.values()) {
      if (city.originId === 'local') {
        city.hasPlaygrounds = this.detectPlaygrounds(city);
      }
      // Remote cities: hasPlaygrounds is set by handleAgentSessionsUpdate
    }
  }
}
