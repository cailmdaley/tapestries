/**
 * SessionTracker - Discovers and tracks tmux sessions running Claude
 *
 * Polls tmux every N seconds to discover sessions running Claude.
 * This is the source of truth for sessions - derived from tmux state.
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import { isCliProcess, pgrepPattern, detectCli } from './cli-provider.js';

const execAsync = promisify(exec);

export interface Session {
  id: string;
  name: string;
  tmuxSession: string;
  cwd: string;
  cityId?: string | null;
  workerHex?: { q: number; r: number };
  cli?: string;  // 'claude' | 'codex' — which CLI this session runs
  status: 'idle' | 'working' | 'offline';
  createdAt: number;
  lastActivity: number;
  originId: string;  // 'local' | 'remote-{hostname}'
}

type SessionsChangeCallback = (sessions: Session[]) => void;

export class SessionTracker {
  private sessions: Map<string, Session> = new Map();
  private pollInterval: NodeJS.Timeout | null = null;
  private changeCallbacks: SessionsChangeCallback[] = [];

  /**
   * Start polling tmux for sessions
   */
  start(intervalMs: number = 2000): void {
    if (this.pollInterval) {
      return;
    }

    // Initial discovery
    this.refresh();

    // Poll on interval
    this.pollInterval = setInterval(() => {
      this.refresh();
    }, intervalMs);
  }

  /**
   * Stop polling
   */
  stop(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  /**
   * Register callback for session changes
   */
  onSessionsChange(callback: SessionsChangeCallback): void {
    this.changeCallbacks.push(callback);
  }

  /**
   * Get all sessions
   */
  getSessions(): Session[] {
    return Array.from(this.sessions.values());
  }

  /**
   * Update session status (called by EventWatcher)
   */
  updateStatus(tmuxSession: string, status: 'idle' | 'working'): boolean {
    const session = this.sessions.get(tmuxSession);
    if (session && session.status !== status) {
      session.status = status;
      if (status === 'working') {
        session.lastActivity = Date.now();
      }
      this.notifyChange();
      return true;
    }
    return false;
  }

  /**
   * Discover sessions from tmux
   */
  private async refresh(): Promise<void> {
    try {
      const discovered = await this.discoverSessions();
      const discoveredMap = new Map<string, { tmuxSession: string; cwd: string; cli?: string }>();

      for (const { tmuxSession, cwd, cli } of discovered) {
        discoveredMap.set(tmuxSession, { tmuxSession, cwd, cli });
      }

      let changed = false;

      // Remove sessions that no longer exist
      for (const [tmuxSession, session] of this.sessions) {
        if (!discoveredMap.has(tmuxSession)) {
          this.sessions.delete(tmuxSession);
          changed = true;
        }
      }

      // Add or update sessions
      for (const { tmuxSession, cwd, cli } of discovered) {
        const existing = this.sessions.get(tmuxSession);

        if (existing) {
          // Update cwd if changed
          if (existing.cwd !== cwd) {
            existing.cwd = cwd;
            existing.cityId = null; // Will be reassigned by index.ts
            changed = true;
          }
          // Update cli if detected
          if (cli && existing.cli !== cli) {
            existing.cli = cli;
            changed = true;
          }
          // Ensure not offline
          if (existing.status === 'offline') {
            existing.status = 'idle';
            changed = true;
          }
        } else {
          // New session - starts idle, EventWatcher will update to working
          const session: Session = {
            id: this.generateId(tmuxSession),
            name: this.truncateName(tmuxSession),
            tmuxSession,
            cwd,
            cli,
            status: 'idle',
            createdAt: Date.now(),
            lastActivity: Date.now(),
            originId: 'local',
          };
          this.sessions.set(tmuxSession, session);
          changed = true;
        }
      }

      // Notify listeners if changed
      if (changed) {
        this.notifyChange();
      }
    } catch (error) {
      console.error('Session refresh failed:', error);
    }
  }

  /**
   * Discover all tmux sessions running Claude
   */
  private async discoverSessions(): Promise<Array<{ tmuxSession: string; cwd: string; cli?: string }>> {
    try {
      // Get all tmux panes with session name, cwd, and pane PID
      const { stdout } = await execAsync(
        'tmux list-panes -a -F "#{session_name}\t#{pane_current_path}\t#{pane_pid}"'
      );

      const lines = stdout.trim().split('\n').filter(Boolean);
      const paneData: Array<{ tmuxSession: string; cwd: string; panePid: string }> = [];

      for (const line of lines) {
        const [tmuxSession, cwd, panePid] = line.split('\t');
        if (panePid) {
          paneData.push({ tmuxSession, cwd: cwd || process.cwd(), panePid });
        }
      }

      if (paneData.length === 0) {
        return [];
      }

      // Check which panes have a CLI running (claude or codex)
      const cliSessions: Array<{ tmuxSession: string; cwd: string; cli?: string }> = [];

      // NOTE: Detection logic duplicated in server/agent.js — keep in sync
      for (const { tmuxSession, cwd, panePid } of paneData) {
        try {
          // Check if pane process ITSELF is a CLI (when zsh -c execs into claude/codex)
          // Use both comm (macOS) and full args (Linux, where codex runs as "node .../codex")
          const { stdout: paneComm } = await execAsync(
            `ps -o comm= -p ${panePid} 2>/dev/null || true`
          );
          const paneCommTrimmed = paneComm.trim();
          if (isCliProcess(paneCommTrimmed)) {
            cliSessions.push({ tmuxSession, cwd, cli: detectCli(paneCommTrimmed) });
            continue;
          }
          // Check full args for cases like "node .../bin/codex" on Linux
          const { stdout: paneArgs } = await execAsync(
            `ps -o args= -p ${panePid} 2>/dev/null || true`
          );
          if (paneArgs.trim() && isCliProcess(paneArgs.trim())) {
            cliSessions.push({ tmuxSession, cwd, cli: detectCli(paneArgs.trim()) });
            continue;
          }

          // BFS through descendants (up to depth 4) to handle deep chains
          // e.g. bash → python3 → MainThread → codex (ralph-launched sessions)
          let matchedChildPid: string | undefined;
          let frontier = [panePid];
          outer: for (let depth = 0; depth < 4; depth++) {
            const nextFrontier: string[] = [];
            for (const pid of frontier) {
              const { stdout: childPidsOut } = await execAsync(
                `pgrep -P ${pid} 2>/dev/null || true`
              );
              for (const candidatePid of childPidsOut.trim().split('\n').filter(Boolean)) {
                try {
                  const { stdout: childComm } = await execAsync(
                    `ps -o comm= -p ${candidatePid} 2>/dev/null || true`
                  );
                  if (isCliProcess(childComm.trim())) { matchedChildPid = candidatePid; break outer; }
                  const { stdout: childArgs } = await execAsync(
                    `ps -o args= -p ${candidatePid} 2>/dev/null || true`
                  );
                  if (isCliProcess(childArgs.trim())) { matchedChildPid = candidatePid; break outer; }
                  nextFrontier.push(candidatePid);
                } catch { /* skip */ }
              }
            }
            frontier = nextFrontier;
          }

          if (matchedChildPid) {
            // Determine which CLI by checking args
            let cli: string | undefined;
            try {
              const { stdout: childArgs } = await execAsync(
                `ps -o args= -p ${matchedChildPid} 2>/dev/null || true`
              );
              cli = detectCli(childArgs.trim()) ?? detectCli(
                (await execAsync(`ps -o comm= -p ${matchedChildPid} 2>/dev/null || true`)).stdout.trim()
              );
            } catch { /* fall through with undefined cli */ }
            cliSessions.push({ tmuxSession, cwd, cli });
          }
        } catch {
          // Ignore errors from pgrep
        }
      }

      return cliSessions;
    } catch {
      // tmux not running or error
      return [];
    }
  }

  /**
   * Truncate long session names for display
   * ralph-global-views-map-plots-plans-3374f8fb → ralph-glob…
   */
  private truncateName(tmuxSession: string): string {
    const MAX_LENGTH = 10;
    if (tmuxSession.length <= MAX_LENGTH) {
      return tmuxSession;
    }
    return `${tmuxSession.slice(0, MAX_LENGTH)}…`;
  }

  /**
   * Generate stable session ID from tmux session name
   */
  private generateId(tmuxSession: string): string {
    // Use simple hash of tmux session name for stable ID
    let hash = 0;
    for (let i = 0; i < tmuxSession.length; i++) {
      const char = tmuxSession.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return `session-${Math.abs(hash).toString(36)}`;
  }

  /**
   * Notify listeners of session changes
   */
  private notifyChange(): void {
    const sessions = this.getSessions();
    for (const callback of this.changeCallbacks) {
      callback(sessions);
    }
  }
}
