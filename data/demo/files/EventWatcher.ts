/**
 * EventWatcher - Watches portolan events file to track session activity
 *
 * Reads from ~/.portolan/data/events.jsonl (written by portolan-hook.sh)
 * and updates session status based on events like:
 *   - user_prompt_submit, pre_tool_use → 'working'
 *   - stop, session_end → 'idle'
 */

import { readFileSync, writeFileSync, statSync, existsSync, watch } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import type { SessionTracker } from './SessionTracker.js';
import { extractActivityDetails } from './activityUtils.js';

interface PortolanEvent {
  id: string;
  timestamp: number;
  type: string;
  sessionId: string;
  cwd: string;
  tmuxSession: string;
  tool?: string;
  toolInput?: Record<string, unknown>;  // Used to extract summary/fullPath
  prompt?: string;                       // User prompt from UserPromptSubmit
}

export interface ActivityEvent {
  tmuxSession: string;
  tool: string;
  summary?: string;
  fullPath?: string;                     // Full file path for Read/Write/Edit
  timestamp: number;
  eventType?: 'tool' | 'user_prompt';    // Distinguish tool calls from user prompts
  prompt?: string;                       // User prompt text for user_prompt events
  sessionId?: string;                    // Claude transcript session UUID
}

type StatusChangeCallback = (tmuxSession: string, status: 'idle' | 'working') => void;
type ActivityCallback = (activity: ActivityEvent) => void;

export class EventWatcher {
  private eventsFile: string;
  private lastFileSize: number = 0;
  private lastCharPosition: number = 0;  // Track character position, not bytes
  private pollInterval: NodeJS.Timeout | null = null;
  private watcher: ReturnType<typeof watch> | null = null;
  private changeCallback: StatusChangeCallback | null = null;
  private activityCallback: ActivityCallback | null = null;
  private sessionTracker: SessionTracker | null = null;

  // Working timeout: if no activity for 30s, assume idle
  private workingTimeout = 30_000;
  private lastActivityBySession: Map<string, number> = new Map();
  private timeoutCheckInterval: NodeJS.Timeout | null = null;

  // Recent activities per tmux session (for initial state)
  private recentActivities: Map<string, ActivityEvent[]> = new Map();
  private maxActivitiesPerSession = 50;

  constructor(eventsFile?: string) {
    this.eventsFile = eventsFile ?? join(homedir(), '.portolan', 'data', 'events.jsonl');
  }

  /**
   * Set callback for status changes (alternative to setSessionTracker)
   */
  onStatusChange(callback: StatusChangeCallback): void {
    this.changeCallback = callback;
  }

  /**
   * Set callback for activity events
   */
  onActivity(callback: ActivityCallback): void {
    this.activityCallback = callback;
  }

  /**
   * Set session tracker to update directly
   */
  setSessionTracker(tracker: SessionTracker): void {
    this.sessionTracker = tracker;
  }

  /**
   * Get recent activities for a tmux session
   */
  getRecentActivities(tmuxSession: string): ActivityEvent[] {
    return this.recentActivities.get(tmuxSession) || [];
  }

  /**
   * Start watching the events file
   */
  start(): void {
    if (!existsSync(this.eventsFile)) {
      console.log(`EventWatcher: Events file not found: ${this.eventsFile}`);
      console.log('EventWatcher: Status detection disabled. Install portolan-hook.sh to enable.');
      return;
    }

    // Truncate if too large (keep last 10k lines)
    this.truncateIfNeeded(10000);

    // Get initial file size and character position
    try {
      const stats = statSync(this.eventsFile);
      this.lastFileSize = stats.size;
      // Read file to get character length (differs from byte length for UTF-8)
      const content = readFileSync(this.eventsFile, 'utf-8');
      this.lastCharPosition = content.length;
    } catch {
      this.lastFileSize = 0;
      this.lastCharPosition = 0;
    }

    // Process recent events to get initial state
    this.processRecentEvents();

    // Watch for file changes
    try {
      this.watcher = watch(this.eventsFile, { persistent: false }, (eventType) => {
        if (eventType === 'change') {
          this.processNewEvents();
        }
      });
    } catch (err) {
      console.error('EventWatcher: Failed to watch events file:', err);
    }

    // Also poll periodically (fallback for systems where watch is unreliable)
    this.pollInterval = setInterval(() => {
      this.processNewEvents();
    }, 1000);

    // Check for working timeout
    this.timeoutCheckInterval = setInterval(() => {
      this.checkWorkingTimeouts();
    }, 5000);

    console.log(`EventWatcher: Watching ${this.eventsFile}`);
  }

  /**
   * Truncate events file if it exceeds maxLines
   */
  private truncateIfNeeded(maxLines: number): void {
    try {
      const content = readFileSync(this.eventsFile, 'utf-8');
      const lines = content.trim().split('\n');
      if (lines.length > maxLines) {
        const kept = lines.slice(-maxLines);
        writeFileSync(this.eventsFile, kept.join('\n') + '\n');
        console.log(`EventWatcher: Truncated ${lines.length} → ${maxLines} lines`);
      }
    } catch {
      // Ignore errors
    }
  }

  /**
   * Stop watching
   */
  stop(): void {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    if (this.timeoutCheckInterval) {
      clearInterval(this.timeoutCheckInterval);
      this.timeoutCheckInterval = null;
    }
  }

  /**
   * Process recent events (last 1000 lines) for initial state
   */
  private processRecentEvents(): void {
    try {
      const content = readFileSync(this.eventsFile, 'utf-8');
      const lines = content.trim().split('\n').slice(-1000);

      // Track most recent status per tmux session
      const latestStatus: Map<string, { status: 'idle' | 'working'; timestamp: number }> = new Map();

      // Also collect recent activities per session (in chronological order first)
      const activitiesBySession: Map<string, ActivityEvent[]> = new Map();

      for (const line of lines) {
        if (!line) continue;
        try {
          const event = JSON.parse(line) as PortolanEvent;
          if (!event.tmuxSession) continue;

          const status = this.eventToStatus(event.type);
          if (status && event.timestamp > (latestStatus.get(event.tmuxSession)?.timestamp ?? 0)) {
            latestStatus.set(event.tmuxSession, { status, timestamp: event.timestamp });
          }

          // Collect pre_tool_use events for activity backfill
          if (event.type === 'pre_tool_use' && event.tool) {
            const details = extractActivityDetails(event.tool, event.toolInput);
            const activity: ActivityEvent = {
              tmuxSession: event.tmuxSession,
              tool: event.tool,
              summary: details?.summary,
              fullPath: details?.fullPath,
              timestamp: event.timestamp,
              eventType: 'tool',
              sessionId: event.sessionId,
            };
            let acts = activitiesBySession.get(event.tmuxSession);
            if (!acts) {
              acts = [];
              activitiesBySession.set(event.tmuxSession, acts);
            }
            acts.push(activity);
          }

          // Collect user_prompt_submit events for conversation backfill
          if (event.type === 'user_prompt_submit' && event.prompt) {
            const activity: ActivityEvent = {
              tmuxSession: event.tmuxSession,
              tool: 'UserPrompt',  // Pseudo-tool for display
              summary: event.prompt.length > 100 ? event.prompt.slice(0, 100) + '...' : event.prompt,
              timestamp: event.timestamp,
              eventType: 'user_prompt',
              prompt: event.prompt,
              sessionId: event.sessionId,
            };
            let acts = activitiesBySession.get(event.tmuxSession);
            if (!acts) {
              acts = [];
              activitiesBySession.set(event.tmuxSession, acts);
            }
            acts.push(activity);
          }
        } catch {
          // Skip malformed lines
        }
      }

      // Store activities (reverse to get newest first, keep max)
      for (const [tmuxSession, acts] of activitiesBySession) {
        const recent = acts.slice(-this.maxActivitiesPerSession).reverse();
        this.recentActivities.set(tmuxSession, recent);
      }

      // Apply recent status (only if within timeout window)
      const now = Date.now();
      for (const [tmuxSession, { status, timestamp }] of latestStatus) {
        // Only apply 'working' status if recent (within timeout)
        if (status === 'working' && now - timestamp > this.workingTimeout) {
          this.updateStatus(tmuxSession, 'idle');
        } else {
          this.updateStatus(tmuxSession, status);
          if (status === 'working') {
            this.lastActivityBySession.set(tmuxSession, timestamp);
          }
        }
      }
    } catch (err) {
      console.error('EventWatcher: Failed to process recent events:', err);
    }
  }

  /**
   * Process new events since last read
   */
  private processNewEvents(): void {
    try {
      const stats = statSync(this.eventsFile);
      if (stats.size <= this.lastFileSize) {
        return; // No new data
      }

      // Read entire file and slice by character position (not bytes)
      const content = readFileSync(this.eventsFile, 'utf-8');
      if (content.length <= this.lastCharPosition) {
        this.lastFileSize = stats.size;
        return;
      }

      const newContent = content.slice(this.lastCharPosition);
      this.lastCharPosition = content.length;
      this.lastFileSize = stats.size;

      const lines = newContent.trim().split('\n');
      for (const line of lines) {
        if (!line) continue;
        try {
          const event = JSON.parse(line) as PortolanEvent;
          this.processEvent(event);
        } catch (err) {
          console.log(`EventWatcher: Parse error on line: ${line.substring(0, 50)}... Error: ${err}`);
        }
      }
    } catch {
      // File may have been truncated or rotated
      try {
        const stats = statSync(this.eventsFile);
        this.lastFileSize = stats.size;
      } catch {
        this.lastFileSize = 0;
      }
    }
  }

  /**
   * Process a single event
   */
  private processEvent(event: PortolanEvent): void {
    if (!event.tmuxSession) return;

    const status = this.eventToStatus(event.type);
    if (status) {
      this.updateStatus(event.tmuxSession, status);
      if (status === 'working') {
        this.lastActivityBySession.set(event.tmuxSession, Date.now());
      }
    }

    // Emit activity event for pre_tool_use (has tool info, fires immediately)
    if (event.type === 'pre_tool_use' && event.tool) {
      const details = extractActivityDetails(event.tool, event.toolInput);
      const activity: ActivityEvent = {
        tmuxSession: event.tmuxSession,
        tool: event.tool,
        summary: details?.summary,
        fullPath: details?.fullPath,
        timestamp: event.timestamp,
        eventType: 'tool',
        sessionId: event.sessionId,
      };

      // Store in recent activities
      this.storeActivity(event.tmuxSession, activity);

      // Emit callback
      if (this.activityCallback) {
        this.activityCallback(activity);
      }
    }

    // Emit activity event for user_prompt_submit (user message)
    if (event.type === 'user_prompt_submit' && event.prompt) {
      const activity: ActivityEvent = {
        tmuxSession: event.tmuxSession,
        tool: 'UserPrompt',
        summary: event.prompt.length > 100 ? event.prompt.slice(0, 100) + '...' : event.prompt,
        timestamp: event.timestamp,
        eventType: 'user_prompt',
        prompt: event.prompt,
        sessionId: event.sessionId,
      };

      // Store in recent activities
      this.storeActivity(event.tmuxSession, activity);

      // Emit callback
      if (this.activityCallback) {
        this.activityCallback(activity);
      }
    }
  }

  /**
   * Store activity, keeping only recent ones and preventing duplicates
   */
  private storeActivity(tmuxSession: string, activity: ActivityEvent): void {
    let activities = this.recentActivities.get(tmuxSession);
    if (!activities) {
      activities = [];
      this.recentActivities.set(tmuxSession, activities);
    }

    // Deduplicate: skip if same tool+fullPath within last 2 seconds
    const recent = activities[0];
    if (recent &&
        recent.tool === activity.tool &&
        recent.fullPath === activity.fullPath &&
        Math.abs(recent.timestamp - activity.timestamp) < 2000) {
      return; // Skip duplicate
    }

    activities.unshift(activity); // Add to front
    if (activities.length > this.maxActivitiesPerSession) {
      activities.pop(); // Remove oldest
    }
  }

  /**
   * Map event type to status
   */
  private eventToStatus(eventType: string): 'idle' | 'working' | null {
    switch (eventType) {
      case 'user_prompt_submit':
      case 'pre_tool_use':
        return 'working';
      case 'stop':
      case 'subagent_stop':
      case 'session_end':
        return 'idle';
      default:
        return null; // Don't change status for unknown events
    }
  }

  /**
   * Check for sessions that have been 'working' too long without activity
   */
  private checkWorkingTimeouts(): void {
    const now = Date.now();
    for (const [tmuxSession, lastActivity] of this.lastActivityBySession) {
      if (now - lastActivity > this.workingTimeout) {
        this.updateStatus(tmuxSession, 'idle');
        this.lastActivityBySession.delete(tmuxSession);
      }
    }
  }

  /**
   * Update status via callback or tracker
   */
  private updateStatus(tmuxSession: string, status: 'idle' | 'working'): void {
    if (this.sessionTracker) {
      this.sessionTracker.updateStatus(tmuxSession, status);
    }
    if (this.changeCallback) {
      this.changeCallback(tmuxSession, status);
    }
  }
}
