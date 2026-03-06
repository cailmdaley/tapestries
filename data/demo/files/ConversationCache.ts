/**
 * ConversationCache - In-memory conversation cache with disk persistence
 *
 * Receives conversation updates via hooks (not polling).
 * Stores messages in memory, persists to disk every 30s + on shutdown.
 *
 * Architecture:
 *   Hook fires → POST /hook/message → ConversationCache.addMessages()
 *                                   → WebSocket broadcast → UI
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync } from 'fs';
import { homedir } from 'os';
import { join, dirname } from 'path';

export interface CachedMessage {
  type: 'user' | 'assistant' | 'thinking' | 'tool_use' | 'tool_result' | 'system';
  content: string;
  timestamp: string;
  // For tool_use
  toolName?: string;
  toolInput?: Record<string, unknown>;
  toolUseId?: string;
  // For thinking
  preview?: string;
  // For system messages
  systemType?: 'skill' | 'reminder';
}

interface SessionCache {
  messages: CachedMessage[];
  tmuxSession: string;
  cwd: string;
  lastUpdate: number;
}

interface PersistedData {
  version: 1;
  sessions: Record<string, SessionCache>;
}

type MessageCallback = (sessionId: string, tmuxSession: string, messages: CachedMessage[]) => void;

export class ConversationCache {
  private sessions: Map<string, SessionCache> = new Map();
  private persistPath: string;
  private persistInterval: ReturnType<typeof setInterval> | null = null;
  private maxMessagesPerSession = 100;  // Keep last 100 messages per session
  private maxSessions = 50;  // Cap total sessions to prevent unbounded growth
  private sessionMaxAge = 7 * 24 * 60 * 60 * 1000;  // 7 days in ms
  private messageCallback: MessageCallback | null = null;
  private lastEventBySession: Map<string, number> = new Map();
  private persistDebounce: ReturnType<typeof setTimeout> | null = null;

  constructor(persistPath?: string) {
    this.persistPath = persistPath ?? join(homedir(), '.portolan', 'conversations.json');
  }

  /**
   * Set callback for new messages (used to broadcast via WebSocket)
   */
  onMessage(callback: MessageCallback): void {
    this.messageCallback = callback;
  }

  /**
   * Start persistence timer
   */
  start(): void {
    // Restore from disk
    this.restore();

    // Persist every 30 seconds
    this.persistInterval = setInterval(() => {
      this.persist();
    }, 30_000);

    console.log(`[ConversationCache] Started (${this.sessions.size} sessions restored)`);
  }

  /**
   * Stop and persist
   */
  stop(): void {
    if (this.persistInterval) {
      clearInterval(this.persistInterval);
      this.persistInterval = null;
    }
    this.persist();
    console.log('[ConversationCache] Stopped');
  }

  /**
   * Debounced persist — writes within 2s of last message, prevents data loss on restart
   */
  private schedulePersist(): void {
    if (this.persistDebounce) clearTimeout(this.persistDebounce);
    this.persistDebounce = setTimeout(() => {
      this.persistDebounce = null;
      this.persist();
    }, 2000);
  }

  /**
   * Add messages from a hook event
   */
  addMessages(
    sessionId: string,
    tmuxSession: string,
    cwd: string,
    messages: CachedMessage[]
  ): void {
    if (messages.length === 0) return;

    // Skip invalid sessionIds
    if (!sessionId || sessionId === 'undefined') {
      console.warn(`[ConversationCache] Skipping invalid sessionId for ${tmuxSession}`);
      return;
    }

    let cache = this.sessions.get(sessionId);
    if (!cache) {
      cache = {
        messages: [],
        tmuxSession,
        cwd,
        lastUpdate: Date.now(),
      };
      this.sessions.set(sessionId, cache);
    }

    // Update metadata
    cache.tmuxSession = tmuxSession;
    cache.cwd = cwd;
    cache.lastUpdate = Date.now();

    // Deduplicate by exact timestamp AND by toolUseId.
    // Millisecond precision in timestamps distinguishes blocks within the same second.
    // toolUseId dedup handles PostToolUse → Stop overlap (different timestamps, same tool call).
    // Also dedup within the incoming batch (transcript scan + payload can overlap).
    const existingTimestamps = new Set(cache.messages.map(m => m.timestamp));
    const existingToolUseIds = new Set(
      cache.messages.filter(m => m.toolUseId).map(m => m.toolUseId)
    );
    const newMessages = messages.filter(m => {
      if (m.toolUseId) {
        if (existingToolUseIds.has(m.toolUseId)) return false;
        existingToolUseIds.add(m.toolUseId);  // prevent within-batch duplicates
      }
      if (existingTimestamps.has(m.timestamp)) return false;
      existingTimestamps.add(m.timestamp);  // prevent within-batch duplicates
      return true;
    });

    if (newMessages.length === 0) return;

    // Add new messages and sort chronologically.
    // Messages arrive out of order: PostToolUse uses wall-clock timestamps,
    // transcript scans use original timestamps with ms precision.
    cache.messages.push(...newMessages);
    cache.messages.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

    // Trim to max
    if (cache.messages.length > this.maxMessagesPerSession) {
      cache.messages = cache.messages.slice(-this.maxMessagesPerSession);
    }

    // Update last event time
    this.lastEventBySession.set(sessionId, Date.now());

    // Notify callback
    if (this.messageCallback) {
      this.messageCallback(sessionId, tmuxSession, newMessages);
    }

    // Persist immediately — dev server restarts (tsx watch) lose unpersisted data
    this.schedulePersist();

    console.log(`[ConversationCache] ${sessionId}: +${newMessages.length} messages (${cache.messages.length} total)`);
  }

  /**
   * Get messages for a session
   */
  getMessages(sessionId: string, limit?: number): CachedMessage[] {
    const messages = this.sessions.get(sessionId)?.messages ?? [];
    return limit ? messages.slice(-limit) : messages;
  }

  /**
   * Get messages for a session by tmux session name (fallback for lookup)
   * Aggregates messages from recent Claude sessions in this tmux session
   */
  getMessagesByTmux(tmuxSession: string, limit?: number): CachedMessage[] {
    const allMessages: CachedMessage[] = [];
    for (const cache of this.sessions.values()) {
      if (cache.tmuxSession === tmuxSession) {
        allMessages.push(...cache.messages);
      }
    }

    if (allMessages.length === 0) return [];

    // Sort and deduplicate by exact timestamp
    allMessages.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    const seen = new Set<string>();
    const deduped = allMessages.filter(m => {
      if (seen.has(m.timestamp)) return false;
      seen.add(m.timestamp);
      return true;
    });

    return limit ? deduped.slice(-limit) : deduped;
  }

  /**
   * Get all session IDs with their last event times (for /hook/health)
   */
  getHealthInfo(): Record<string, { lastEvent: number; messageCount: number; tmuxSession: string }> {
    const info: Record<string, { lastEvent: number; messageCount: number; tmuxSession: string }> = {};
    for (const [sessionId, cache] of this.sessions) {
      info[sessionId] = {
        lastEvent: this.lastEventBySession.get(sessionId) ?? cache.lastUpdate,
        messageCount: cache.messages.length,
        tmuxSession: cache.tmuxSession,
      };
    }
    return info;
  }

  /**
   * Delete a session from both maps
   */
  private deleteSession(sessionId: string): void {
    this.sessions.delete(sessionId);
    this.lastEventBySession.delete(sessionId);
  }

  /**
   * Clean up old/excess sessions to prevent unbounded memory growth
   */
  private cleanup(): void {
    const now = Date.now();

    // Delete sessions older than max age
    const expiredIds = [...this.sessions.entries()]
      .filter(([_, cache]) => now - cache.lastUpdate > this.sessionMaxAge)
      .map(([sessionId]) => sessionId);

    for (const sessionId of expiredIds) {
      this.deleteSession(sessionId);
    }

    // If still over limit, remove oldest sessions
    let excessRemoved = 0;
    if (this.sessions.size > this.maxSessions) {
      const sortedByAge = [...this.sessions.entries()]
        .sort((a, b) => a[1].lastUpdate - b[1].lastUpdate);

      excessRemoved = this.sessions.size - this.maxSessions;
      for (let i = 0; i < excessRemoved; i++) {
        this.deleteSession(sortedByAge[i][0]);
      }
    }

    if (expiredIds.length > 0 || excessRemoved > 0) {
      console.log(`[ConversationCache] Cleanup: removed ${expiredIds.length} expired + ${excessRemoved} excess sessions, ${this.sessions.size} remain`);
    }
  }

  /**
   * Persist to disk
   */
  persist(): void {
    // Clean up before persisting
    this.cleanup();

    const dir = dirname(this.persistPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    const data: PersistedData = {
      version: 1,
      sessions: {},
    };

    // Only persist sessions with messages
    for (const [sessionId, cache] of this.sessions) {
      if (cache.messages.length > 0) {
        // Keep only last 10 messages per session in persistence (lighter footprint)
        data.sessions[sessionId] = {
          ...cache,
          messages: cache.messages.slice(-10),
        };
      }
    }

    const tmpPath = this.persistPath + '.tmp';
    try {
      writeFileSync(tmpPath, JSON.stringify(data, null, 2));
      renameSync(tmpPath, this.persistPath);
    } catch (error) {
      console.error('[ConversationCache] Failed to persist:', error);
    }
  }

  /**
   * Restore from disk
   */
  restore(): void {
    if (!existsSync(this.persistPath)) return;

    try {
      const content = readFileSync(this.persistPath, 'utf-8');
      const data = JSON.parse(content) as PersistedData;

      if (data.version !== 1) return;

      for (const [sessionId, cache] of Object.entries(data.sessions)) {
        // Skip invalid sessionIds from old data
        if (!sessionId || sessionId === 'undefined') continue;

        // Dedup and sort messages on restore (handles corruption from earlier bugs)
        const seenTimestamps = new Set<string>();
        const seenToolUseIds = new Set<string>();
        cache.messages = cache.messages.filter(m => {
          if (m.toolUseId) {
            if (seenToolUseIds.has(m.toolUseId)) return false;
            seenToolUseIds.add(m.toolUseId);
          }
          if (seenTimestamps.has(m.timestamp)) return false;
          seenTimestamps.add(m.timestamp);
          return true;
        });
        cache.messages.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

        this.sessions.set(sessionId, cache);
        this.lastEventBySession.set(sessionId, cache.lastUpdate);
      }
    } catch (error) {
      console.error('[ConversationCache] Failed to restore:', error);
    }
  }

  /**
   * Clear cache for a session (e.g., when session ends)
   */
  clearSession(sessionId: string): void {
    this.deleteSession(sessionId);
  }
}
