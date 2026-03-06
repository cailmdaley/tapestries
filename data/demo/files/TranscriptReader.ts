/**
 * TranscriptReader - Reads full conversation from Claude session transcripts
 *
 * Claude Code stores full conversation transcripts in:
 *   ~/.claude/projects/{escaped-cwd}/{session-id}.jsonl
 *
 * This module:
 * 1. Maps tmux session cwd → Claude project directory
 * 2. Finds the most recent transcript file
 * 3. Parses conversation messages (user, assistant, thinking, tool_use)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';
import { exec } from 'child_process';
import { promisify } from 'util';
import { isCliProcess, pgrepPattern } from './cli-provider.js';

const execAsync = promisify(exec);

export interface ConversationMessage {
  type: 'user' | 'assistant' | 'thinking' | 'tool_use' | 'tool_result' | 'system';
  content: string;
  timestamp: string;
  // For tool_use
  toolName?: string;
  toolInput?: any;
  toolUseId?: string;  // For linking tool_use to tool_result
  // For thinking
  preview?: string;
  // For system messages (skill content, reminders)
  systemType?: 'skill' | 'reminder';
}

export interface TranscriptInfo {
  sessionId: string;
  projectPath: string;
  transcriptPath: string;
  messages: ConversationMessage[];
  lastModified: number;
}

export class TranscriptReader {
  private claudeProjectsDir: string;
  private cache: Map<string, { info: TranscriptInfo; mtime: number }> = new Map();
  private cacheTimeout = 2000; // Refresh every 2 seconds
  // Track which transcript file each session is using: sessionId -> transcriptPath
  private sessionTranscriptMap: Map<string, string> = new Map();
  private mappingsFile: string;

  constructor() {
    this.claudeProjectsDir = path.join(os.homedir(), '.claude', 'projects');
    this.mappingsFile = path.join(os.homedir(), '.portolan', 'transcript-mappings.json');
    this.loadMappings();
  }

  /**
   * Load persisted session→transcript mappings from disk
   */
  private loadMappings(): void {
    try {
      if (fs.existsSync(this.mappingsFile)) {
        const data = JSON.parse(fs.readFileSync(this.mappingsFile, 'utf-8'));
        // Filter out mappings for transcript files that no longer exist
        for (const [sessionId, transcriptPath] of Object.entries(data)) {
          if (typeof transcriptPath === 'string' && fs.existsSync(transcriptPath)) {
            this.sessionTranscriptMap.set(sessionId, transcriptPath);
          }
        }
        console.log(`[Transcript] Loaded ${this.sessionTranscriptMap.size} session mappings`);
      }
    } catch (err) {
      console.log('[Transcript] No existing mappings file or failed to load');
    }
  }

  /**
   * Save session→transcript mappings to disk
   */
  private saveMappings(): void {
    try {
      const dir = path.dirname(this.mappingsFile);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      const data = Object.fromEntries(this.sessionTranscriptMap);
      fs.writeFileSync(this.mappingsFile, JSON.stringify(data, null, 2));
    } catch (err) {
      console.error('[Transcript] Failed to save mappings:', err);
    }
  }

  /**
   * Associate a session with a specific transcript file
   */
  setSessionTranscript(sessionId: string, transcriptPath: string): void {
    this.sessionTranscriptMap.set(sessionId, transcriptPath);
    this.saveMappings();
  }

  /**
   * Get the transcript file associated with a session
   */
  getSessionTranscript(sessionId: string): string | undefined {
    return this.sessionTranscriptMap.get(sessionId);
  }

  /**
   * Check if a transcript file is stale (not modified within threshold)
   */
  isTranscriptStale(transcriptPath: string, thresholdMs: number): boolean {
    try {
      const stats = fs.statSync(transcriptPath);
      return Date.now() - stats.mtimeMs > thresholdMs;
    } catch {
      return true; // If we can't stat it, consider it stale
    }
  }

  /**
   * Get all session-to-transcript mappings (for debugging)
   */
  getAllSessionMappings(): Map<string, string> {
    return new Map(this.sessionTranscriptMap);
  }

  /**
   * Detect the transcript for a tmux session by checking the Claude process's open files
   * Claude keeps the task directory open, which contains the session UUID
   */
  async detectTranscriptFromTmux(tmuxSession: string, cwd: string): Promise<string | null> {
    try {
      // Get the pane PID for this tmux session
      const { stdout: paneInfo } = await execAsync(
        `tmux list-panes -t "${tmuxSession}" -F "#{pane_pid}" 2>/dev/null`
      );
      const panePid = paneInfo.trim().split('\n')[0];
      if (!panePid) return null;

      // Find the CLI process PID (claude or codex — either pane process or a child)
      let claudePid: string | null = null;

      // Check if pane process is a CLI
      const { stdout: paneComm } = await execAsync(
        `ps -o comm= -p ${panePid} 2>/dev/null || true`
      );
      if (isCliProcess(paneComm.trim())) {
        claudePid = panePid;
      } else {
        // Check full args (codex on Linux runs as "node .../bin/codex")
        const { stdout: paneArgs } = await execAsync(
          `ps -o args= -p ${panePid} 2>/dev/null || true`
        );
        if (paneArgs.trim() && isCliProcess(paneArgs.trim())) {
          claudePid = panePid;
        } else {
          // Check children by name first, then by args
          const { stdout: pgrepOut } = await execAsync(
            `pgrep -P ${panePid} -x ${pgrepPattern()} 2>/dev/null || true`
          );
          claudePid = pgrepOut.trim().split('\n')[0] || null;

          if (!claudePid) {
            const { stdout: childPids } = await execAsync(
              `pgrep -P ${panePid} 2>/dev/null || true`
            );
            for (const candidatePid of childPids.trim().split('\n').filter(Boolean)) {
              try {
                const { stdout: childArgs } = await execAsync(
                  `ps -o args= -p ${candidatePid} 2>/dev/null || true`
                );
                if (isCliProcess(childArgs.trim())) {
                  claudePid = candidatePid;
                  break;
                }
              } catch { /* skip */ }
            }
          }
        }
      }

      if (!claudePid) return null;

      // Use lsof to find which task directory Claude has open
      const { stdout: lsofOut } = await execAsync(
        `lsof -p ${claudePid} 2>/dev/null | grep '/.claude/tasks/' || true`
      );

      // Extract the UUID from the task directory path
      const match = lsofOut.match(/\.claude\/tasks\/([a-f0-9-]{36})/);
      if (!match) return null;

      const sessionUuid = match[1];

      // Find the corresponding transcript file
      const escapedPath = this.escapePathForClaude(cwd);
      const projectDir = path.join(this.claudeProjectsDir, escapedPath);
      const transcriptPath = path.join(projectDir, `${sessionUuid}.jsonl`);

      if (fs.existsSync(transcriptPath)) {
        console.log(`[Transcript] Detected transcript for ${tmuxSession} via lsof: ${sessionUuid}`);
        return transcriptPath;
      }

      return null;
    } catch (err) {
      // lsof or tmux command failed
      return null;
    }
  }

  /**
   * List all transcript files for a project directory, sorted by modification time (most recent first)
   */
  listTranscripts(cwd: string): Array<{ path: string; mtime: number; sessionId: string }> {
    const escapedPath = this.escapePathForClaude(cwd);
    const projectDir = path.join(this.claudeProjectsDir, escapedPath);

    try {
      const files = fs.readdirSync(projectDir)
        .filter(f => f.endsWith('.jsonl'))
        .map(f => {
          const fullPath = path.join(projectDir, f);
          return {
            path: fullPath,
            mtime: fs.statSync(fullPath).mtimeMs,
            sessionId: f.replace('.jsonl', '')
          };
        })
        .sort((a, b) => b.mtime - a.mtime);

      return files;
    } catch {
      return [];
    }
  }

  /**
   * Detect which transcript file is currently active based on recent modifications
   * Returns the transcript that was modified in the last N seconds
   */
  detectActiveTranscript(cwd: string, withinMs: number = 5000): string | null {
    const transcripts = this.listTranscripts(cwd);
    const now = Date.now();

    for (const t of transcripts) {
      if (now - t.mtime < withinMs) {
        return t.path;
      }
    }

    // No recently modified transcript, return the most recent one
    return transcripts.length > 0 ? transcripts[0].path : null;
  }

  /**
   * Convert a cwd path to Claude's escaped project directory name
   * e.g., /Users/foo/bar → -Users-foo-bar
   */
  private escapePathForClaude(cwd: string): string {
    // Claude Code escapes both / and _ to - in project directory names
    return cwd.replace(/[/_]/g, '-');
  }

  /**
   * Find the most recent transcript file for a project
   */
  private findLatestTranscript(projectDir: string): string | null {
    try {
      const files = fs.readdirSync(projectDir)
        .filter(f => f.endsWith('.jsonl'))
        .map(f => ({
          name: f,
          path: path.join(projectDir, f),
          mtime: fs.statSync(path.join(projectDir, f)).mtimeMs
        }))
        .sort((a, b) => b.mtime - a.mtime);

      return files.length > 0 ? files[0].path : null;
    } catch {
      return null;
    }
  }

  /**
   * Parse a transcript JSONL file and extract conversation messages
   */
  private async parseTranscript(transcriptPath: string): Promise<ConversationMessage[]> {
    const messages: ConversationMessage[] = [];

    try {
      const fileStream = fs.createReadStream(transcriptPath);
      const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
      });

      for await (const line of rl) {
        try {
          const event = JSON.parse(line);
          const parsed = this.parseEvent(event);
          if (parsed.length > 0) {
            messages.push(...parsed);
          }
        } catch {
          // Skip malformed lines
        }
      }
    } catch {
      // File read error
    }

    return messages;
  }

  /**
   * Parse a single transcript event into conversation messages
   */
  private parseEvent(event: any): ConversationMessage[] {
    const messages: ConversationMessage[] = [];
    const timestamp = event.timestamp || new Date().toISOString();

    // Handle meta messages (skill content, system reminders) as system type
    if (event.isMeta && event.type === 'user') {
      const messageContent = event.message?.content;
      if (Array.isArray(messageContent)) {
        for (const block of messageContent) {
          if (block.type === 'text' && block.text) {
            // Detect skill content vs other meta content
            const isSkill = block.text.startsWith('Base directory for this skill:') ||
                            block.text.includes('# /') && block.text.includes('---');
            messages.push({
              type: 'system',
              content: block.text,
              timestamp,
              systemType: isSkill ? 'skill' : 'reminder',
              preview: block.text.slice(0, 80),
            });
          }
        }
      }
      return messages;
    }

    if (event.type === 'user') {
      // User message - may contain text and/or tool_result blocks
      const messageContent = event.message?.content;

      if (Array.isArray(messageContent)) {
        // Multi-block message: extract tool_results and text separately
        for (const block of messageContent) {
          if (block.type === 'tool_result') {
            // Extract tool result content
            let resultContent = '';
            if (typeof block.content === 'string') {
              resultContent = block.content;
            } else if (Array.isArray(block.content)) {
              resultContent = block.content
                .filter((c: any) => c.type === 'text')
                .map((c: any) => c.text)
                .join('\n');
            }
            messages.push({
              type: 'tool_result',
              content: resultContent,
              toolUseId: block.tool_use_id,
              timestamp
            });
          } else if (block.type === 'text' && block.text) {
            // Regular user text
            messages.push({
              type: 'user',
              content: block.text,
              timestamp
            });
          }
        }
      } else if (typeof messageContent === 'string') {
        // Simple string message
        messages.push({
          type: 'user',
          content: messageContent,
          timestamp
        });
      }
    } else if (event.type === 'assistant') {
      // Assistant message - may contain multiple content blocks
      const contentBlocks = event.message?.content;
      if (Array.isArray(contentBlocks)) {
        for (const block of contentBlocks) {
          if (block.type === 'thinking') {
            messages.push({
              type: 'thinking',
              content: block.thinking || '',
              preview: (block.thinking || '').slice(0, 100),
              timestamp
            });
          } else if (block.type === 'text') {
            messages.push({
              type: 'assistant',
              content: block.text || '',
              timestamp
            });
          } else if (block.type === 'tool_use') {
            messages.push({
              type: 'tool_use',
              content: block.name || 'tool',
              toolName: block.name,
              toolInput: block.input,
              toolUseId: block.id,
              timestamp
            });
          }
        }
      }
    }

    return messages;
  }

  /**
   * Extract user message content (handles string or array format)
   */
  private extractUserContent(message: any): string | null {
    if (!message) return null;

    if (typeof message.content === 'string') {
      return message.content;
    }

    if (Array.isArray(message.content)) {
      // Find text content, skip tool_result blocks
      for (const block of message.content) {
        if (block.type === 'text') {
          return block.text;
        }
      }
    }

    return null;
  }

  /**
   * Get conversation for a tmux session's working directory
   */
  async getConversation(cwd: string): Promise<TranscriptInfo | null> {
    const escapedPath = this.escapePathForClaude(cwd);
    const projectDir = path.join(this.claudeProjectsDir, escapedPath);

    // Check cache
    const cached = this.cache.get(cwd);
    if (cached && Date.now() - cached.mtime < this.cacheTimeout) {
      return cached.info;
    }

    // Find latest transcript
    const transcriptPath = this.findLatestTranscript(projectDir);
    if (!transcriptPath) {
      return null;
    }

    // Check if file was modified since last cache
    const stats = fs.statSync(transcriptPath);
    if (cached && cached.info.transcriptPath === transcriptPath && stats.mtimeMs === cached.info.lastModified) {
      return cached.info;
    }

    // Parse transcript
    const messages = await this.parseTranscript(transcriptPath);

    const info: TranscriptInfo = {
      sessionId: path.basename(transcriptPath, '.jsonl'),
      projectPath: cwd,
      transcriptPath,
      messages,
      lastModified: stats.mtimeMs
    };

    this.cache.set(cwd, { info, mtime: Date.now() });
    return info;
  }

  /**
   * Get recent messages (last N) for a conversation
   * If sessionId is provided and has a mapped transcript, use that specific file
   */
  async getRecentMessages(cwd: string, limit: number = 50, sessionId?: string): Promise<ConversationMessage[]> {
    // Check if we have a specific transcript for this session
    if (sessionId) {
      const mappedTranscript = this.sessionTranscriptMap.get(sessionId);
      if (mappedTranscript && fs.existsSync(mappedTranscript)) {
        const messages = await this.parseTranscript(mappedTranscript);
        return messages.slice(-limit);
      }
    }

    // Fall back to the most recent transcript for the cwd
    const transcript = await this.getConversation(cwd);
    if (!transcript) {
      return [];
    }
    return transcript.messages.slice(-limit);
  }

  /**
   * Read messages from a specific transcript file path
   */
  async getMessagesFromFile(transcriptPath: string, limit: number = 50): Promise<ConversationMessage[]> {
    if (!fs.existsSync(transcriptPath)) {
      return [];
    }
    const messages = await this.parseTranscript(transcriptPath);
    return messages.slice(-limit);
  }
}
