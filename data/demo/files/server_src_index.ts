/**
 * Portolan Server
 *
 * Wires together all managers and serves state to browser via WebSocket.
 * HTTP endpoints handled by HttpApi, terminal commands by KittyIntegration,
 * message routing by MessageRouter.
 */

import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { exec, execFile, spawn, ChildProcess } from 'child_process';
import { promisify } from 'util';
import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync } from 'fs';
import { readdir, stat } from 'fs/promises';
import { homedir } from 'os';
import { join, relative, resolve } from 'path';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

// Track active file searches for cancellation
const activeSearches = new Map<string, ChildProcess>();

// Check which search tools are available (cached)
let hasFd: boolean | null = null;
let hasRg: boolean | null = null;

async function checkSearchTools(): Promise<void> {
  if (hasFd === null) {
    try {
      await execAsync('which fd');
      hasFd = true;
    } catch {
      hasFd = false;
    }
  }
  if (hasRg === null) {
    try {
      await execAsync('which rg');
      hasRg = true;
    } catch {
      hasRg = false;
    }
  }
}

// Initialize on startup
checkSearchTools();

import { SessionTracker, Session } from './SessionTracker.js';
import { CityManager, City, SessionInfo } from './CityManager.js';
import { OriginManager, Origin } from './OriginManager.js';
import { CityPersistence } from './CityPersistence.js';
import { AnnotationPersistence } from './AnnotationPersistence.js';
import { GitStatusManager, GitStatus } from './GitStatusManager.js';
import { RecentFileTracker } from './RecentFileTracker.js';
import { countOpenFibers, getOpenFibers, getRecentlyClosed } from './FiberReader.js';
import { EventWatcher, type ActivityEvent } from './EventWatcher.js';
import { HttpApi } from './HttpApi.js';
import { KittyIntegration, expandHome, shellEscape } from './KittyIntegration.js';
import { MessageRouter, AgentSessionsUpdateMessage, AgentActivityMessage } from './MessageRouter.js';
import { RemoteWorkingSessionTracker } from './RemoteWorkingSessionTracker.js';
import { reconcilePreviousLocalSessions } from './PreviousSessionReconciler.js';

// ============================================================================
// Types
// ============================================================================

interface StateUpdate {
  cities: City[];
  sessions: Session[];
  origins?: Origin[];
  activities?: Record<string, ActivityEvent[]>;  // activitySessionKey -> recent activities
}

// ============================================================================
// Constants
// ============================================================================

const PORT = process.env.VITEST ? 4099 : 4004;
const FIBER_REFRESH_INTERVAL = 10000; // 10 seconds
const LOCAL_ORIGIN_ID = 'local';
let fiberRefreshIntervalHandle: NodeJS.Timeout | null = null;
let remoteWorkingTimeoutIntervalHandle: NodeJS.Timeout | null = null;

function getActivitySessionKey(originId: string, tmuxSession: string): string {
  return `${originId}:${tmuxSession}`;
}

// ============================================================================
// Initialization
// ============================================================================

const cityManager = new CityManager();
const cityPersistence = new CityPersistence();
const annotationPersistence = new AnnotationPersistence();
const sessionTracker = new SessionTracker();
const originManager = new OriginManager();
const eventWatcher = new EventWatcher();
const gitStatusManager = new GitStatusManager();
const recentFileTracker = new RecentFileTracker();

// Load persisted cities into CityManager
const persistedCities = cityPersistence.load();

// Load persisted annotations
annotationPersistence.load();
for (const pc of persistedCities) {
  // Set sshHost first so city keys are normalized correctly
  if (pc.sshHost && pc.originId !== 'local') {
    cityManager.setOriginSshHost(pc.originId, pc.sshHost);
  }
  cityManager.addPinnedCity(pc.id, pc.path, pc.name, pc.position, pc.originId);
}

// Track remote sessions: Map<originId, Map<tmuxSession, Session>>
const remoteSessions = new Map<string, Map<string, Session>>();

// Track remote git statuses: Map<"originId:path", GitStatus>
const remoteGitStatuses = new Map<string, GitStatus>();

// Track remote activities: Map<activitySessionKey, ActivityEvent[]>
const remoteActivities = new Map<string, ActivityEvent[]>();
const MAX_REMOTE_ACTIVITIES = 50;

// Track remote working timeout ownership by explicit origin/session keys.
const remoteWorkingSessions = new RemoteWorkingSessionTracker();
const REMOTE_WORKING_TIMEOUT = 30_000; // 30 seconds, same as EventWatcher

// Activity persistence
const activityPersistencePath = join(homedir(), '.portolan', 'remote-activities.json');

function loadActivityPersistence(): void {
  if (!existsSync(activityPersistencePath)) return;
  try {
    const content = readFileSync(activityPersistencePath, 'utf-8');
    const data = JSON.parse(content) as { version: 1; activities: Record<string, ActivityEvent[]> };
    if (data.version === 1 && data.activities) {
      for (const [activitySessionKey, acts] of Object.entries(data.activities)) {
        remoteActivities.set(activitySessionKey, acts);
      }
      console.log(`[Activity] Loaded ${remoteActivities.size} remote session activities`);
    }
  } catch (error) {
    console.error('[Activity] Failed to load persistence:', error);
  }
}

function saveActivityPersistence(): void {
  const dataDir = join(homedir(), '.portolan');
  if (!existsSync(dataDir)) {
    mkdirSync(dataDir, { recursive: true });
  }
  const activities: Record<string, ActivityEvent[]> = {};
  for (const [activitySessionKey, acts] of remoteActivities.entries()) {
    activities[activitySessionKey] = acts;
  }
  const data = { version: 1 as const, activities };
  const tmpPath = activityPersistencePath + '.tmp';
  try {
    writeFileSync(tmpPath, JSON.stringify(data), 'utf-8');
    renameSync(tmpPath, activityPersistencePath);
  } catch (error) {
    console.error('[Activity] Failed to save persistence:', error);
  }
}

// Load activity on startup
loadActivityPersistence();

// Track connected browser clients
const clients: Set<WebSocket> = new Set();

// Track last broadcast state for fiber count comparison
let lastBroadcastState: StateUpdate | null = null;

// Track previous sessions to detect removals
let previousSessions = new Map<string, Session>();

// ============================================================================
// Lookup Adapters (for extracted modules)
// ============================================================================

const sessionLookup = {
  findSession(sessionId: string): Session | undefined {
    const local = sessionTracker.getSessions().find(s => s.id === sessionId);
    if (local) return local;
    for (const originSessions of remoteSessions.values()) {
      for (const session of originSessions.values()) {
        if (session.id === sessionId) return session;
      }
    }
    return undefined;
  },
  getAllSessions(): Session[] {
    const all: Session[] = [...sessionTracker.getSessions()];
    for (const originSessions of remoteSessions.values()) {
      all.push(...originSessions.values());
    }
    return all;
  },
};

const cityLookup = {
  findCityByPath(path: string): City | undefined {
    return cityManager.getCities().find(c => c.path === path);
  },
  getSshHost(city: City): string | undefined {
    const origin = originManager.getOrigin(city.originId);
    const persistedCity = cityPersistence.getCityById(city.id);
    return origin?.sshHost || persistedCity?.sshHost || city.originId.replace('remote-', '');
  },
};

// ============================================================================
// Extracted Modules
// ============================================================================

const httpApi = new HttpApi(cityManager, originManager, cityPersistence);
httpApi.setAnnotationPersistence(annotationPersistence);
httpApi.setSessionLookup(sessionLookup);
httpApi.setRecentFileTracker(recentFileTracker);
httpApi.setRuntimeDiagnosticsProvider(() => {
  const localSessionCount = sessionTracker.getSessions().length;
  const remoteSessionCount = getRemoteSessionCount();
  const connectedRemoteOrigins = originManager
    .getOrigins()
    .filter((origin) => origin.type === 'remote' && originManager.isOriginConnected(origin.id))
    .length;

  return {
    sessions: {
      local: localSessionCount,
      remote: remoteSessionCount,
      total: localSessionCount + remoteSessionCount,
      previousSessionRecords: previousSessions.size,
    },
    websocket: {
      browserClients: clients.size,
      connectedRemoteOrigins,
    },
    maps: {
      activeSearches: activeSearches.size,
      remoteSessionOrigins: remoteSessions.size,
      remoteGitStatuses: remoteGitStatuses.size,
      remoteActivities: remoteActivities.size,
      remoteActivityEvents: getRemoteActivityEventCount(),
    },
    intervals: {
      fiberRefreshActive: fiberRefreshIntervalHandle !== null,
      remoteWorkingTimeoutActive: remoteWorkingTimeoutIntervalHandle !== null,
    },
    eventWatcher: eventWatcher.getStats(),
    remoteWorkingSessions: remoteWorkingSessions.getStats(),
    recentFiles: {
      sessionCount: recentFileTracker.getSessionCount(),
      entryCount: recentFileTracker.getTotalEntryCount(),
    },
  };
});
const kitty = new KittyIntegration(sessionLookup, originManager, cityLookup);

// Callback for creating new workers (used by send-annotations endpoint)
httpApi.setOnCreateNewWorker(async (cityPath: string, originId: string) => {
  const city = cityManager.getCities().find(c => c.path === cityPath || cityPath.startsWith(c.path + '/'));
  const isRemote = originId !== 'local' && !!originId;

  // Get SSH host and display name for remote cities
  const sshHost = isRemote && city ? cityLookup.getSshHost(city) : undefined;
  const originDisplayName = city?.originId.replace('remote-', '');

  return kitty.createWorker(cityPath, { sshHost, originDisplayName });
});

// Callback for focusing sessions in Kitty (used by send-annotations endpoint)
httpApi.setOnFocusSession((sessionId: string) => {
  kitty.focusSession(sessionId);
  kitty.activateKitty();
});

// ============================================================================
// State Management
// ============================================================================

function getAllSessions(): Session[] {
  return sessionLookup.getAllSessions();
}

function getRemoteSessionCount(): number {
  let total = 0;
  for (const sessionsByOrigin of remoteSessions.values()) {
    total += sessionsByOrigin.size;
  }
  return total;
}

function getRemoteActivityEventCount(): number {
  let total = 0;
  for (const activities of remoteActivities.values()) {
    total += activities.length;
  }
  return total;
}

/**
 * Assign a session to a city, handling hex allocation and cleanup.
 * Releases previous hex if session is moving between cities.
 */
function assignSessionToCity(session: Session, city: City): void {
  const previousCityId = session.cityId;
  if (previousCityId !== city.id) {
    if (previousCityId && session.workerHex) {
      cityManager.releaseWorkerHex(previousCityId, session.workerHex);
    }
    session.cityId = city.id;
    session.workerHex = cityManager.assignWorkerHex(city.id);
  } else if (!session.workerHex) {
    session.workerHex = cityManager.assignWorkerHex(city.id);
  }
}

async function buildState(): Promise<StateUpdate> {
  const sessions = getAllSessions();
  const cities = cityManager.getCities();

  cityManager.updateClaimsStatus();
  cityManager.updatePlaygroundsStatus();

  const activeCityIds = new Set(sessions.filter(s => s.cityId).map(s => s.cityId));

  const citiesWithFibers = await Promise.all(
    cities.map(async (city) => {
      let gitStatus: GitStatus | undefined;
      if (city.originId === 'local') {
        gitStatus = gitStatusManager.getStatus(city.path) ?? undefined;
      } else {
        const remoteKey = `${city.originId}:${city.path}`;
        gitStatus = remoteGitStatuses.get(remoteKey);
      }

      return {
        ...city,
        fiberCount: city.originId === 'local' ? await countOpenFibers(city.path) : 0,
        hasClaims: city.hasClaims ?? false,
        hasPlaygrounds: city.hasPlaygrounds ?? false,
        isDormant: !activeCityIds.has(city.id),
        gitStatus,
      };
    })
  );

  const cityMap = new Map(cities.map((c) => [c.id, c]));

  const sessionsWithAbsoluteHex = sessions.map((session) => {
    if (session.workerHex && session.cityId) {
      const city = cityMap.get(session.cityId);
      if (city) {
        return {
          ...session,
          workerHex: {
            q: city.position.q + session.workerHex.q,
            r: city.position.r + session.workerHex.r,
          },
        };
      }
    }
    return session;
  });

  // Collect recent activities for each session
  const activities: Record<string, ActivityEvent[]> = {};
  for (const session of sessions) {
    const activitySessionKey = getActivitySessionKey(session.originId, session.tmuxSession);
    const sessionActivities = session.originId === LOCAL_ORIGIN_ID
      ? eventWatcher.getRecentActivities(session.tmuxSession)
      : remoteActivities.get(activitySessionKey) || [];
    if (sessionActivities.length > 0) {
      activities[activitySessionKey] = sessionActivities;
    }
  }

  return {
    cities: citiesWithFibers,
    sessions: sessionsWithAbsoluteHex,
    origins: originManager.getOrigins(),
    activities,
  };
}

function broadcast(state: StateUpdate): void {
  const message = JSON.stringify(state);
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  }
  lastBroadcastState = state;
}

function broadcastActivity(activity: ActivityEvent, originId: string): void {
  const message = JSON.stringify({
    type: 'activity',
    activity: {
      ...activity,
      originId,
      activitySessionKey: getActivitySessionKey(originId, activity.tmuxSession),
    },
  });
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  }
}

function fiberCountsChanged(oldState: StateUpdate | null, newState: StateUpdate): boolean {
  if (!oldState) return true;
  if (oldState.cities.length !== newState.cities.length) return true;
  for (const newCity of newState.cities) {
    const oldCity = oldState.cities.find((c) => c.id === newCity.id);
    if (!oldCity || oldCity.fiberCount !== newCity.fiberCount) return true;
  }
  return false;
}

async function refreshFiberCounts(): Promise<void> {
  const state = await buildState();
  if (fiberCountsChanged(lastBroadcastState, state)) {
    console.log('Fiber counts changed, broadcasting update');
    broadcast(state);
  }
}

function rebuildCities(): void {
  const allSessions = getAllSessions();
  const sessionInfos: SessionInfo[] = allSessions
    .filter(s => s.cwd)
    .map(s => ({ cwd: s.cwd, originId: s.originId }));
  cityManager.updateFromSessions(sessionInfos);

  for (const city of cityManager.getCities()) {
    if (!cityManager.isPinned(city.id)) {
      const sshHost = city.originId !== 'local'
        ? originManager.getOrigin(city.originId)?.sshHost
        : undefined;
      cityPersistence.pin(city.path, city.position, city.originId, city.name, sshHost);
      cityManager.addPinnedCity(city.id, city.path, city.name, city.position, city.originId);
    } else if (city.originId !== 'local') {
      const origin = originManager.getOrigin(city.originId);
      const persistedCity = cityPersistence.getCityById(city.id);
      if (origin?.sshHost && persistedCity && !persistedCity.sshHost) {
        cityPersistence.pin(city.path, city.position, city.originId, city.name, origin.sshHost);
      }
    }

    if (city.originId === 'local') {
      gitStatusManager.track(city.path);
    }
  }
}

// ============================================================================
// Session Change Handler (Local Sessions)
// ============================================================================

sessionTracker.onSessionsChange((localSessions) => {
  eventWatcher.reconcileActiveSessions(localSessions.map(session => session.tmuxSession));

  const removedLocalSessions = reconcilePreviousLocalSessions(
    previousSessions,
    localSessions,
    LOCAL_ORIGIN_ID,
  );

  for (const removedSession of removedLocalSessions) {
    if (removedSession.cityId && removedSession.workerHex) {
      cityManager.releaseWorkerHex(removedSession.cityId, removedSession.workerHex);
    }
    recentFileTracker.removeSession(removedSession.id);
  }

  rebuildCities();

  for (const session of localSessions) {
    if (!session.cwd) continue;
    const city = cityManager.findCityForPath(session.cwd, session.originId);
    if (!city) continue;

    assignSessionToCity(session, city);
  }

  buildState().then(broadcast);
});
// ============================================================================
// Remote Session Handling
// ============================================================================

function pruneStaleRemoteActivities(): boolean {
  const activeKeys = new Set<string>();
  for (const [originId, sessionsByOrigin] of remoteSessions.entries()) {
    for (const tmuxSession of sessionsByOrigin.keys()) {
      activeKeys.add(getActivitySessionKey(originId, tmuxSession));
    }
  }

  let changed = false;
  for (const activitySessionKey of remoteActivities.keys()) {
    if (!activeKeys.has(activitySessionKey)) {
      remoteActivities.delete(activitySessionKey);
      changed = true;
    }
  }
  return changed;
}

/**
 * Remove all side-channel state that is owned by a remote session.
 * Returns true when activity persistence should be rewritten.
 */
function cleanupRemoteSessionData(originId: string, session: Session): boolean {
  let activityChanged = false;

  remoteWorkingSessions.clear(originId, session.tmuxSession);

  if (remoteActivities.delete(getActivitySessionKey(originId, session.tmuxSession))) {
    activityChanged = true;
  }

  return activityChanged;
}

function pruneRemoteGitStatus(originId: string, activeCwds: Set<string>): void {
  const prefix = `${originId}:`;
  for (const [key] of remoteGitStatuses.entries()) {
    if (!key.startsWith(prefix)) continue;
    const cwd = key.slice(prefix.length);
    if (!activeCwds.has(cwd)) {
      remoteGitStatuses.delete(key);
    }
  }
}

function handleAgentSessionsUpdate(
  originId: string,
  agentSessions: AgentSessionsUpdateMessage['payload']['sessions']
): void {
  if (!remoteSessions.has(originId)) {
    remoteSessions.set(originId, new Map());
  }
  const originSessionsMap = remoteSessions.get(originId)!;
  const updatedTmuxSessions = new Set(agentSessions.map(s => s.tmuxSession));
  const removedSessions: Session[] = [];

  for (const [tmuxSession, session] of originSessionsMap) {
    if (!updatedTmuxSessions.has(tmuxSession)) {
      if (session.cityId && session.workerHex) {
        cityManager.releaseWorkerHex(session.cityId, session.workerHex);
      }
      originSessionsMap.delete(tmuxSession);
      previousSessions.delete(session.id);
      removedSessions.push(session);
      console.log(`Remote session removed: ${session.name} from ${originId}`);
    }
  }

  let activityChanged = false;
  for (const removed of removedSessions) {
    recentFileTracker.removeSession(removed.id);
    activityChanged = cleanupRemoteSessionData(originId, removed) || activityChanged;
  }
  activityChanged = pruneStaleRemoteActivities() || activityChanged;
  if (activityChanged) {
    saveActivityPersistence();
  }

  for (const agentSession of agentSessions) {
    const status = agentSession.status || 'idle';
    const existing = originSessionsMap.get(agentSession.tmuxSession);
    if (existing) {
      existing.cwd = agentSession.cwd;
      existing.status = status;
      existing.lastActivity = Date.now();
    } else {
      const session: Session = {
        id: `remote-${originId}-${agentSession.tmuxSession}`,
        name: agentSession.name,
        tmuxSession: agentSession.tmuxSession,
        cwd: agentSession.cwd,
        status,
        createdAt: Date.now(),
        lastActivity: Date.now(),
        originId,
      };
      originSessionsMap.set(agentSession.tmuxSession, session);
      console.log(`Remote session discovered: ${session.name} from ${originId}`);
    }

    // Keep timeout ownership aligned with authoritative agent status.
    remoteWorkingSessions.reconcile(originId, agentSession.tmuxSession, status);
  }

  rebuildCities();

  const claimsByCwd = new Map<string, boolean>();
  const playgroundsByCwd = new Map<string, boolean>();
  for (const agentSession of agentSessions) {
    if (agentSession.hasClaims !== undefined) {
      claimsByCwd.set(agentSession.cwd, agentSession.hasClaims);
    }
    if (agentSession.hasPlaygrounds !== undefined) {
      playgroundsByCwd.set(agentSession.cwd, agentSession.hasPlaygrounds);
    }
  }

  for (const agentSession of agentSessions) {
    if (agentSession.gitStatus) {
      const remoteKey = `${originId}:${agentSession.cwd}`;
      remoteGitStatuses.set(remoteKey, agentSession.gitStatus);
    }
  }
  pruneRemoteGitStatus(originId, new Set(agentSessions.map(s => s.cwd)));

  for (const session of originSessionsMap.values()) {
    if (!session.cwd) continue;
    const city = cityManager.findCityForPath(session.cwd, session.originId);
    if (!city) continue;

    const hasClaims = claimsByCwd.get(session.cwd);
    if (hasClaims !== undefined) {
      city.hasClaims = hasClaims;
    }

    const hasPlaygrounds = playgroundsByCwd.get(session.cwd);
    if (hasPlaygrounds !== undefined) {
      city.hasPlaygrounds = hasPlaygrounds;
    }

    assignSessionToCity(session, city);

    previousSessions.set(session.id, session);
  }

  buildState().then(broadcast);
}

function reconnectTunnel(sshHost: string): void {
  console.log(`Reconnecting SSH tunnel to ${sshHost}...`);
  execFile('ssh', ['-fN', sshHost], (err) => {
    if (err) console.error(`SSH tunnel reconnect to ${sshHost} failed:`, err.message);
    else console.log(`SSH tunnel to ${sshHost} re-established`);
  });
}

function handleAgentDisconnect(originId: string, sshHost?: string): void {
  const originSessionsMap = remoteSessions.get(originId);
  if (!originSessionsMap) return;

  if (!originManager.isOriginConnected(originId)) {
    let activityChanged = false;

    for (const [tmuxSession, session] of originSessionsMap) {
      if (session.cityId && session.workerHex) {
        cityManager.releaseWorkerHex(session.cityId, session.workerHex);
      }
      originSessionsMap.delete(tmuxSession);
      previousSessions.delete(session.id);
      recentFileTracker.removeSession(session.id);
      activityChanged = cleanupRemoteSessionData(originId, session) || activityChanged;
    }
    activityChanged = pruneStaleRemoteActivities() || activityChanged;
    if (activityChanged) {
      saveActivityPersistence();
    }

    // Clean up git statuses for this origin
    pruneRemoteGitStatus(originId, new Set<string>());
    remoteWorkingSessions.clearOrigin(originId);

    remoteSessions.delete(originId);
    rebuildCities();
    buildState().then(broadcast);

    if (sshHost) reconnectTunnel(sshHost);
  }
}

// ============================================================================
// Message Handlers
// ============================================================================

async function handleGetFibers(ws: WebSocket, cityId: string): Promise<void> {
  const city = cityManager.getCityById(cityId);
  if (!city) {
    ws.send(JSON.stringify({ type: 'fibers', cityId, open: [], recentlyClosed: [] }));
    return;
  }

  try {
    if (city.originId === 'local') {
      const [open, recentlyClosed] = await Promise.all([
        getOpenFibers(city.path),
        getRecentlyClosed(city.path, 5),
      ]);
      ws.send(JSON.stringify({ type: 'fibers', cityId, open, recentlyClosed }));
    } else {
      const origin = originManager.getOrigin(city.originId);
      if (!origin?.sshHost) {
        ws.send(JSON.stringify({ type: 'fibers', cityId, open: [], recentlyClosed: [] }));
        return;
      }
      const [open, recentlyClosed] = await Promise.all([
        getRemoteFibers(origin.sshHost, city.path, 'open'),
        getRemoteFibers(origin.sshHost, city.path, 'closed'),
      ]);
      ws.send(JSON.stringify({ type: 'fibers', cityId, open, recentlyClosed }));
    }
  } catch (error) {
    console.error('Failed to get fibers:', error);
    ws.send(JSON.stringify({ type: 'fibers', cityId, open: [], recentlyClosed: [] }));
  }
}

async function getRemoteFibers(
  sshHost: string,
  cityPath: string,
  status: 'open' | 'closed'
): Promise<Array<{ id: string; title: string; kind: string; status: string; body?: string; outcome?: string }>> {
  const escapedPath = shellEscape(cityPath);
  const statusFlag = status === 'open' ? '-s open' : '-s closed';
  const recentFlag = status === 'closed' ? '--recent 5' : '';

  try {
    const { stdout } = await execFileAsync(
      'ssh', [sshHost, `cd ${escapedPath} && felt ls ${statusFlag} ${recentFlag} --json --body 2>/dev/null || echo '[]'`],
      { timeout: 10000 }
    );
    const fibers = JSON.parse(stdout.trim() || '[]');
    return fibers.map((f: any) => ({
      id: f.id,
      title: f.title,
      kind: f.kind || 'task',
      status: f.status || status,
      body: f.body || undefined,
      outcome: f.outcome || f.close_reason || undefined,
    }));
  } catch (error) {
    console.error(`Failed to get remote fibers from ${sshHost}:${cityPath}:`, error);
    return [];
  }
}

// ============================================================================
// File Search
// ============================================================================

interface SearchResult {
  path: string;        // Relative path from city root
  fullPath: string;    // Full path for opening
  line?: number;       // Line number (for content search)
  match?: string;      // Matching line content (for content search)
}

interface DirectoryEntry {
  name: string;
  type: 'file' | 'dir';
}

/**
 * Parse search output lines into SearchResult array.
 * Used by both searchLocal and searchRemote.
 */
function parseSearchResults(
  stdout: string,
  cityPath: string,
  mode: 'filename' | 'content'
): SearchResult[] {
  const results: SearchResult[] = [];
  const lines = stdout.trim().split('\n').filter(Boolean).slice(0, 50);

  for (const line of lines) {
    if (mode === 'filename') {
      const relativePath = line.startsWith('./') ? line.slice(2) : line;
      results.push({
        path: relativePath,
        fullPath: `${cityPath}/${relativePath}`,
      });
    } else {
      // rg/grep format: file:line:content or ./file:line:content
      const match = line.match(/^(?:\.\/)?([^:]+):(\d+):(.*)$/);
      if (match) {
        results.push({
          path: match[1],
          fullPath: `${cityPath}/${match[1]}`,
          line: parseInt(match[2], 10),
          match: match[3].trim().slice(0, 100),
        });
      }
    }
  }

  return results;
}

function handleSearchFiles(
  ws: WebSocket,
  cityId: string,
  query: string,
  searchId: string,
  mode: 'filename' | 'content' = 'filename'
): void {
  // Cancel any previous search with the same searchId base (same search session)
  // searchId format: "cityId-counter-name" or "cityId-counter-content"
  // Extract base: everything before the last hyphen (name/content suffix)
  const searchBase = searchId.replace(/-(?:name|content)$/, '');
  const searchKeyPrefix = `${cityId}:${searchBase}`;
  for (const [key, proc] of activeSearches) {
    // Only cancel searches from a previous search session, not parallel name/content searches
    if (key.startsWith(`${cityId}:`) && !key.startsWith(searchKeyPrefix)) {
      proc.kill();
      activeSearches.delete(key);
    }
  }

  const city = cityManager.getCityById(cityId);
  if (!city) {
    ws.send(JSON.stringify({ type: 'searchResults', searchId, results: [], error: 'City not found' }));
    return;
  }

  if (!query.trim()) {
    ws.send(JSON.stringify({ type: 'searchResults', searchId, results: [] }));
    return;
  }

  const searchKey = `${cityId}:${searchId}`;

  if (city.originId === 'local') {
    searchLocal(ws, city.path, query, searchId, searchKey, mode);
  } else {
    const origin = originManager.getOrigin(city.originId);
    const persistedCity = cityPersistence.getCityById(city.id);
    const sshHost = origin?.sshHost || persistedCity?.sshHost;
    if (!sshHost) {
      ws.send(JSON.stringify({ type: 'searchResults', searchId, results: [], error: 'No SSH host for remote city' }));
      return;
    }
    searchRemote(ws, sshHost, city.path, query, searchId, searchKey, mode);
  }
}

function searchLocal(
  ws: WebSocket,
  cityPath: string,
  query: string,
  searchId: string,
  searchKey: string,
  mode: 'filename' | 'content'
): void {
  let proc: ChildProcess;

  // Escape query for use in shell/regex (basic escaping)
  const safeQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  if (mode === 'filename') {
    if (hasFd) {
      // Use fd (fast, --no-ignore to include gitignored files like build/)
      proc = spawn('fd', [
        '--type', 'f',
        '--follow',
        '--full-path',
        '--hidden',
        '--no-ignore',
        '--exclude', '.git',
        '--exclude', '.felt',
        '--exclude', 'node_modules',
        '--exclude', '__pycache__',
        '--color', 'never',
        query
      ], { cwd: cityPath });
    } else {
      // Fallback: find + grep
      const cmd = `find -L . \\( -name '.git' -o -name '.felt' -o -name 'node_modules' -o -name '__pycache__' \\) -prune -o -type f -print 2>/dev/null | grep -i '${safeQuery}' | head -50`;
      proc = spawn('sh', ['-c', cmd], { cwd: cityPath });
    }
  } else {
    if (hasRg) {
      // Use rg (fast, --no-ignore to include gitignored files like build/)
      proc = spawn('rg', [
        '--line-number',
        '--no-heading',
        '--color', 'never',
        '--max-count', '1',
        '--follow',
        '--no-ignore',
        '--glob', '!.git',
        '--glob', '!node_modules',
        '--glob', '!__pycache__',
        query
      ], { cwd: cityPath });
    } else {
      // Fallback: grep -r
      const cmd = `grep -Rn --include='*' -I '${safeQuery}' . --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=__pycache__ 2>/dev/null | head -50`;
      proc = spawn('sh', ['-c', cmd], { cwd: cityPath });
    }
  }

  activeSearches.set(searchKey, proc);

  let stdout = '';
  let timedOut = false;
  proc.stdout?.on('data', (data) => {
    stdout += data.toString();
  });

  // Timeout: kill after 10 seconds and return partial results
  const timeout = setTimeout(() => {
    timedOut = true;
    proc.kill('SIGTERM');
    console.log(`[Search] Timeout for ${searchKey}, returning partial results`);
  }, 10000);

  proc.on('close', () => {
    clearTimeout(timeout);
    activeSearches.delete(searchKey);
    const results = parseSearchResults(stdout, cityPath, mode);
    ws.send(JSON.stringify({ type: 'searchResults', searchId, results, timedOut }));
  });

  proc.on('error', (error) => {
    clearTimeout(timeout);
    activeSearches.delete(searchKey);
    console.error('Search error:', error);
    ws.send(JSON.stringify({ type: 'searchResults', searchId, results: [], error: error.message }));
  });
}

function searchRemote(
  ws: WebSocket,
  sshHost: string,
  cityPath: string,
  query: string,
  searchId: string,
  searchKey: string,
  mode: 'filename' | 'content'
): void {
  const escapedPath = shellEscape(cityPath);
  const escapedQuery = shellEscape(query);
  // Escape for shell and regex (for fallback grep)
  const safeQuery = query.replace(/[.*+?^${}()|[\]\\'"]/g, '\\$&');

  // Try fd/rg first, fall back to find/grep
  // Remote machines may or may not have fd/rg installed
  // --no-ignore to include gitignored files (build/, dist/, etc.)
  let remoteCmd: string;
  if (mode === 'filename') {
    // Try fd, fall back to find+grep
    remoteCmd = `(fd --type f --follow --full-path --hidden --no-ignore --exclude .git --exclude .felt --exclude node_modules --exclude __pycache__ --color never ${escapedQuery} 2>/dev/null || find -L . \\( -name '.git' -o -name '.felt' -o -name 'node_modules' -o -name '__pycache__' \\) -prune -o -type f -print 2>/dev/null | grep -i '${safeQuery}') | head -50`;
  } else {
    // Try rg, fall back to grep
    remoteCmd = `(rg --line-number --no-heading --color never --max-count 1 --follow --no-ignore --glob '!.git' --glob '!node_modules' --glob '!__pycache__' ${escapedQuery} 2>/dev/null || grep -Rn --include='*' -I '${safeQuery}' . --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=__pycache__ 2>/dev/null) | head -50`;
  }

  const remoteScript = `cd ${escapedPath} && ${remoteCmd}`;

  const proc = spawn('ssh', [sshHost, remoteScript]);
  activeSearches.set(searchKey, proc);

  let stdout = '';
  let timedOut = false;
  proc.stdout?.on('data', (data) => {
    stdout += data.toString();
  });

  // Timeout: kill after 15 seconds for remote (slower)
  const timeout = setTimeout(() => {
    timedOut = true;
    proc.kill('SIGTERM');
    console.log(`[Search] Remote timeout for ${searchKey}, returning partial results`);
  }, 15000);

  proc.on('close', () => {
    clearTimeout(timeout);
    activeSearches.delete(searchKey);
    const results = parseSearchResults(stdout, cityPath, mode);
    ws.send(JSON.stringify({ type: 'searchResults', searchId, results, timedOut }));
  });

  proc.on('error', (error) => {
    clearTimeout(timeout);
    activeSearches.delete(searchKey);
    ws.send(JSON.stringify({ type: 'searchResults', searchId, results: [], error: error.message }));
  });
}

// ============================================================================
// Directory Listing (Files tab)
// ============================================================================

const NON_GIT_SKIP = new Set(['.git', 'node_modules', '__pycache__', '.DS_Store']);
const gitRepoCache = new Map<string, boolean>();

function sortDirectoryEntries(entries: DirectoryEntry[]): DirectoryEntry[] {
  return entries.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
    return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
  });
}

function parseLsName(raw: string): { name: string; type: 'file' | 'dir' } | null {
  const trimmed = raw.trim();
  if (!trimmed || trimmed === '.' || trimmed === '..') return null;
  const last = trimmed.charAt(trimmed.length - 1);
  const isDir = last === '/';
  const name = trimmed.replace(/[\\/@*|=]+$/, '');
  if (!name || NON_GIT_SKIP.has(name)) return null;
  return { name, type: isDir ? 'dir' : 'file' };
}

async function isGitRepo(cityPath: string): Promise<boolean> {
  const cached = gitRepoCache.get(cityPath);
  if (cached !== undefined) return cached;
  try {
    const { stdout } = await execFileAsync('git', ['-C', cityPath, 'rev-parse', '--is-inside-work-tree'], { timeout: 3000 });
    const result = stdout.trim() === 'true';
    gitRepoCache.set(cityPath, result);
    return result;
  } catch {
    gitRepoCache.set(cityPath, false);
    return false;
  }
}

async function isIgnoredByGit(cityPath: string, relPath: string): Promise<boolean> {
  try {
    await execFileAsync('git', ['-C', cityPath, 'check-ignore', '-q', relPath], { timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}

async function readPhysicalDirectoryEntries(targetPath: string): Promise<DirectoryEntry[]> {
  const dirents = await readdir(targetPath, { withFileTypes: true });
  const entries: DirectoryEntry[] = [];

  for (const dirent of dirents) {
    if (NON_GIT_SKIP.has(dirent.name)) continue;
    if (dirent.isDirectory()) {
      entries.push({ name: dirent.name, type: 'dir' });
      continue;
    }
    if (dirent.isFile()) {
      entries.push({ name: dirent.name, type: 'file' });
      continue;
    }
    if (dirent.isSymbolicLink()) {
      try {
        const target = await stat(join(targetPath, dirent.name));
        entries.push({ name: dirent.name, type: target.isDirectory() ? 'dir' : 'file' });
      } catch {
        // Broken symlink: keep as file so it can still be inspected.
        entries.push({ name: dirent.name, type: 'file' });
      }
    }
  }

  return entries;
}

async function listLocalDirectory(cityPath: string, targetPath: string): Promise<DirectoryEntry[]> {
  const safeRoot = resolve(cityPath);
  const safeTarget = resolve(targetPath);
  const relTarget = relative(safeRoot, safeTarget);
  if (relTarget.startsWith('..') || relTarget.includes('/../')) {
    throw new Error('Path is outside city root');
  }

  const physicalEntries = await readPhysicalDirectoryEntries(safeTarget);

  if (await isGitRepo(cityPath)) {
    const visible = await Promise.all(
      physicalEntries.map(async (entry) => {
        const relPath = relTarget ? `${relTarget}/${entry.name}` : entry.name;
        return await isIgnoredByGit(cityPath, relPath) ? null : entry;
      })
    );
    return sortDirectoryEntries(visible.filter((entry): entry is DirectoryEntry => Boolean(entry)));
  }

  return sortDirectoryEntries(physicalEntries);
}

async function listRemoteDirectory(sshHost: string, targetPath: string): Promise<DirectoryEntry[]> {
  const escapedPath = shellEscape(targetPath);
  const fdScript = `cd ${escapedPath} && ((fd --follow --max-depth 1 --type d --color never . | sed 's|^\\./||;s|$|/' && fd --follow --max-depth 1 --type f --type l --color never . | sed 's|^\\./||') 2>/dev/null || true)`;
  const lsScript = `cd ${escapedPath} && ls -1AF 2>/dev/null`;

  const parseEntries = (stdout: string): DirectoryEntry[] => {
    const entriesByName = new Map<string, DirectoryEntry>();
    for (const rawLine of stdout.split('\n')) {
      const line = rawLine.trim().replace(/^\.\//, '');
      if (!line || line === '.') continue;

      let parsed: { name: string; type: 'file' | 'dir' } | null = null;
      if (line.endsWith('/')) {
        parsed = parseLsName(line);
      } else {
        const plainName = line.replace(/[\\/@*|=]+$/, '');
        if (plainName && !NON_GIT_SKIP.has(plainName)) {
          parsed = { name: plainName, type: 'file' };
        }
      }

      if (!parsed) continue;
      const existing = entriesByName.get(parsed.name);
      if (!existing || existing.type === 'file' && parsed.type === 'dir') {
        entriesByName.set(parsed.name, parsed);
      }
    }
    return sortDirectoryEntries([...entriesByName.values()]);
  };

  const { stdout: fdStdout } = await execFileAsync('ssh', [sshHost, fdScript], { timeout: 15000, maxBuffer: 5 * 1024 * 1024 });
  const fdEntries = parseEntries(fdStdout);
  if (fdEntries.length > 0) return fdEntries;

  const { stdout: lsStdout } = await execFileAsync('ssh', [sshHost, lsScript], { timeout: 15000, maxBuffer: 5 * 1024 * 1024 });
  return parseEntries(lsStdout);
}

async function handleListDirectory(ws: WebSocket, cityId: string, path: string): Promise<void> {
  const city = cityManager.getCityById(cityId);
  if (!city) {
    ws.send(JSON.stringify({ type: 'directoryListing', cityId, path, entries: [], error: 'City not found' }));
    return;
  }

  try {
    const safePath = resolve(path);
    const safeCity = resolve(city.path);
    if (!(safePath === safeCity || safePath.startsWith(`${safeCity}/`))) {
      throw new Error('Path is outside city root');
    }

    let entries: DirectoryEntry[] = [];
    if (city.originId === 'local') {
      entries = await listLocalDirectory(city.path, safePath);
    } else {
      const origin = originManager.getOrigin(city.originId);
      const persistedCity = cityPersistence.getCityById(city.id);
      const sshHost = origin?.sshHost || persistedCity?.sshHost;
      if (!sshHost) {
        throw new Error('No SSH host for remote city');
      }
      entries = await listRemoteDirectory(sshHost, safePath);
    }

    ws.send(JSON.stringify({ type: 'directoryListing', cityId, path: safePath, entries }));
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Could not read directory';
    ws.send(JSON.stringify({ type: 'directoryListing', cityId, path, entries: [], error: msg }));
  }
}

function handlePinCity(
  ws: WebSocket,
  path: string,
  position: { q: number; r: number },
  name?: string
): void {
  try {
    const expandedPath = expandHome(path);
    const city = cityManager.pinCity(expandedPath, position, 'local', name);
    cityPersistence.pin(expandedPath, position, 'local', name || city.name);
    console.log(`City pinned: ${city.name} at (${position.q}, ${position.r})`);
    buildState().then(broadcast);
    ws.send(JSON.stringify({ type: 'cityPinned', city }));
  } catch (error) {
    console.error('Failed to pin city:', error);
    ws.send(JSON.stringify({ type: 'error', message: 'Failed to pin city' }));
  }
}

function handleUnpinCity(ws: WebSocket, cityId: string): void {
  try {
    const allSessions = getAllSessions();
    const city = cityManager.getCityById(cityId);
    if (!city) {
      ws.send(JSON.stringify({ type: 'error', message: 'City not found' }));
      return;
    }

    const sessionCount = allSessions.filter(
      (s) => s.cwd && s.cwd === city.path && s.originId === city.originId
    ).length;

    if (sessionCount > 0) {
      ws.send(JSON.stringify({
        type: 'confirmUnpin',
        cityId,
        cityName: city.name,
        sessionCount,
      }));
      return;
    }

    performUnpin(ws, cityId);
  } catch (error) {
    console.error('Failed to unpin city:', error);
    ws.send(JSON.stringify({ type: 'error', message: 'Failed to unpin city' }));
  }
}

function performUnpin(ws: WebSocket, cityId: string): void {
  const city = cityManager.getCityById(cityId);
  if (!city) return;

  cityManager.unpinCity(cityId);
  cityPersistence.unpin(cityId);
  console.log(`City unpinned: ${city.name}`);

  rebuildCities();
  buildState().then(broadcast);
  ws.send(JSON.stringify({ type: 'cityUnpinned', cityId }));
}

function handleMoveCity(
  ws: WebSocket,
  cityId: string,
  newPosition: { q: number; r: number }
): void {
  try {
    const city = cityManager.getCityById(cityId);
    if (!city) {
      ws.send(JSON.stringify({ type: 'error', message: 'City not found' }));
      return;
    }

    if (!cityManager.isPinned(cityId)) {
      ws.send(JSON.stringify({ type: 'error', message: 'Can only move pinned cities' }));
      return;
    }

    cityManager.moveCity(cityId, newPosition);
    cityPersistence.updatePosition(cityId, newPosition);
    console.log(`City moved: ${city.name} to (${newPosition.q}, ${newPosition.r})`);

    buildState().then(broadcast);
    ws.send(JSON.stringify({ type: 'cityMoved', cityId, newPosition }));
  } catch (error) {
    console.error('Failed to move city:', error);
    ws.send(JSON.stringify({ type: 'error', message: 'Failed to move city' }));
  }
}

// ============================================================================
// Message Router Setup
// ============================================================================

const messageRouter = new MessageRouter({
  onFocus: (sessionId) => kitty.focusSession(sessionId),
  onGetFibers: handleGetFibers,
  onHandoff: (fiberId, cityPath) => kitty.handoff(fiberId, cityPath),
  onNewWorker: (ws, cityPath, name, chrome, continueSession, cli) => kitty.newWorker(ws, cityPath, name, chrome, continueSession, cli),
  onPinCity: handlePinCity,
  onUnpinCity: handleUnpinCity,
  onConfirmUnpin: performUnpin,
  onKillWorker: (sessionId) => kitty.killWorker(sessionId),
  onSearchFiles: handleSearchFiles,
  onMoveCity: handleMoveCity,
  onListDirectory: (ws, cityId, path) => {
    handleListDirectory(ws, cityId, path);
  },
});

// ============================================================================
// HTTP Server
// ============================================================================

const server = createServer(async (req, res) => {
  const handled = await httpApi.handleRequest(req, res);
  if (handled) return;

  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('Portolan server running\n');
});

// ============================================================================
// WebSocket Server
// ============================================================================

const wss = new WebSocketServer({ server });

wss.on('connection', async (ws, req) => {
  const url = new URL(req.url || '', `http://${req.headers.host}`);
  const isAgent = url.searchParams.get('agent') === 'true';
  const originName = url.searchParams.get('origin');
  const sshHost = url.searchParams.get('sshHost') || undefined;
  const plannotatorPortParam = url.searchParams.get('plannotatorPort');
  const plannotatorPort = plannotatorPortParam ? parseInt(plannotatorPortParam, 10) : undefined;

  if (isAgent && originName) {
    // Agent connection
    const origin = originManager.registerAgent(originName, ws, sshHost, plannotatorPort);
    cityManager.setOriginPosition(origin.id, origin.position);
    // Track sshHost for city key normalization (so different login nodes share cities)
    if (sshHost) {
      cityManager.setOriginSshHost(origin.id, sshHost);
    }

    ws.send(JSON.stringify({
      type: 'connected',
      payload: { originId: origin.id, position: origin.position },
    }));

    console.log(`Agent connected: ${originName} (${origin.id})`);

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        if (message.type === 'agent_sessions_update') {
          handleAgentSessionsUpdate(origin.id, message.payload.sessions);
        } else if (message.type === 'agent_activity') {
          const activity = (message as AgentActivityMessage).activity;

          // Only track file operations (Read, Write, Edit)
          if (!['Read', 'Write', 'Edit'].includes(activity.tool)) {
            return;
          }

          // Store remote activity with deduplication
          const activitySessionKey = getActivitySessionKey(origin.id, activity.tmuxSession);
          let activities = remoteActivities.get(activitySessionKey);
          if (!activities) {
            activities = [];
            remoteActivities.set(activitySessionKey, activities);
          }

          // Deduplicate: skip if same tool+fullPath within last 2 seconds
          const recent = activities[0];
          if (recent &&
              recent.tool === activity.tool &&
              recent.fullPath === activity.fullPath &&
              Math.abs(recent.timestamp - activity.timestamp) < 2000) {
            return; // Skip duplicate
          }

          activities.unshift(activity);
          if (activities.length > MAX_REMOTE_ACTIVITIES) {
            activities.pop();
          }
          saveActivityPersistence();

          // Update remote session status to 'working'
          const sessionMap = remoteSessions.get(origin.id);
          const session = sessionMap?.get(activity.tmuxSession);
          const now = Date.now();
          if (session && session.status !== 'working') {
            session.status = 'working';
            session.lastActivity = now;
            remoteWorkingSessions.touch(origin.id, activity.tmuxSession, now);
            // Trigger rebuild to broadcast the status change
            sessionTracker['notifyChange']();
          } else if (session) {
            // Just update the timestamp
            session.lastActivity = now;
            remoteWorkingSessions.touch(origin.id, activity.tmuxSession, now);
          }

          broadcastActivity(activity, origin.id);
        }
      } catch (error) {
        console.error('Failed to handle agent message:', error);
      }
    });

    ws.on('close', () => {
      const disconnectedOrigin = originManager.handleDisconnect(ws);
      if (disconnectedOrigin) {
        handleAgentDisconnect(disconnectedOrigin.id, disconnectedOrigin.sshHost);
        broadcast({ ...lastBroadcastState!, origins: originManager.getOrigins() });
      }
      console.log(`Agent disconnected: ${originName}`);
    });

    ws.on('error', (error) => console.error('Agent WebSocket error:', error));
  } else {
    // Browser client
    clients.add(ws);
    console.log('Browser client connected');

    const state = await buildState();
    ws.send(JSON.stringify(state));

    ws.on('message', (data) => messageRouter.routeClientMessage(ws, data.toString()));
    ws.on('close', () => {
      clients.delete(ws);
      console.log('Browser client disconnected');
    });
    ws.on('error', (error) => console.error('WebSocket error:', error));
  }
});

// ============================================================================
// Startup
// ============================================================================

// Set local plannotator port from environment
const localPlannotatorPort = process.env.PLANNOTATOR_PORT ? parseInt(process.env.PLANNOTATOR_PORT, 10) : undefined;
if (localPlannotatorPort) {
  originManager.setLocalPlannotatorPort(localPlannotatorPort);
  console.log(`Local plannotator port: ${localPlannotatorPort}`);
}

sessionTracker.start(2000);

eventWatcher.setSessionTracker(sessionTracker);
eventWatcher.onActivity((activity) => {
  console.log('[Activity]', activity.tmuxSession, activity.tool, activity.summary || '');

  broadcastActivity(activity, LOCAL_ORIGIN_ID);
});
eventWatcher.start();

gitStatusManager.setUpdateHandler(({ path, status }) => {
  console.log(`[Git] ${path}: ${status.branch} +${status.linesAdded}/-${status.linesRemoved}`);
  buildState().then(broadcast);
});
gitStatusManager.start();

fiberRefreshIntervalHandle = setInterval(refreshFiberCounts, FIBER_REFRESH_INTERVAL);

// Check for remote session working timeouts
remoteWorkingTimeoutIntervalHandle = setInterval(() => {
  const now = Date.now();
  let changed = false;
  const cutoff = now - REMOTE_WORKING_TIMEOUT;
  const expired = remoteWorkingSessions.consumeExpired(cutoff);
  for (const { originId, tmuxSession } of expired) {
    const sessionMap = remoteSessions.get(originId);
    const session = sessionMap?.get(tmuxSession);
    if (session && session.status === 'working') {
      session.status = 'idle';
      changed = true;
    }
  }
  if (changed) {
    sessionTracker['notifyChange']();
  }
}, 5000);

server.on('error', (err: NodeJS.ErrnoException) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`Port ${PORT} already in use - another instance is running`);
  } else {
    throw err;
  }
});

server.listen(PORT, () => {
  console.log(`Portolan server running on port ${PORT}`);
  console.log(`WebSocket: ws://localhost:${PORT}`);
});

let shuttingDown = false;

function stopBackgroundTimers(): void {
  if (fiberRefreshIntervalHandle) {
    clearInterval(fiberRefreshIntervalHandle);
    fiberRefreshIntervalHandle = null;
  }
  if (remoteWorkingTimeoutIntervalHandle) {
    clearInterval(remoteWorkingTimeoutIntervalHandle);
    remoteWorkingTimeoutIntervalHandle = null;
  }
}

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('\nShutting down...');
  stopBackgroundTimers();
  sessionTracker.stop();
  gitStatusManager.stop();
  eventWatcher.stop();
  wss.close();
  server.close(() => {
    process.exit(0);
  });
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
