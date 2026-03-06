#!/usr/bin/env node
/**
 * Portolan Remote Agent
 *
 * Lightweight agent that runs on remote machines to bridge Claude Code sessions
 * to a local portolan server via SSH tunnel.
 *
 * Usage:
 *   portolan-agent connect [host:port] [--ssh-host=name]
 *   portolan-agent status
 *
 * The agent:
 * 1. Discovers local tmux sessions running Claude Code
 * 2. Connects to portolan server (default localhost:4004, tunneled from local)
 * 3. Registers with the hostname as origin
 * 4. Sends periodic session updates
 *
 * SSH tunnel setup (on local machine's ~/.ssh/config):
 *   Host remotebox
 *     RemoteForward 4004 127.0.0.1:4004
 */

import WebSocket from 'ws';
import { exec } from 'child_process';
import { hostname, homedir } from 'os';
import { promisify } from 'util';
import { existsSync, readFileSync, readdirSync } from 'fs';
import { resolve, join } from 'path';

const execAsync = promisify(exec);

// ============================================================================
// Configuration
// ============================================================================
const ORIGIN_NAME = process.env.PORTOLAN_ORIGIN || hostname();
const DEFAULT_SERVER = 'localhost:4004';
const RECONNECT_INTERVAL = 5000;
const POLL_INTERVAL = 5000;  // Session discovery interval
const EVENTS_FILE = join(homedir(), '.portolan', 'data', 'events.jsonl');
const DEBUG = process.env.PORTOLAN_DEBUG === 'true';
const PLANNOTATOR_PORT = process.env.PLANNOTATOR_PORT ? parseInt(process.env.PLANNOTATOR_PORT, 10) : null;

// CLI provider detection — detect both claude and codex simultaneously
const ALL_CLI_PROCESS_NAMES = ['claude', 'codex'];
function isCliProcess(comm) { return ALL_CLI_PROCESS_NAMES.some(n => comm.includes(n)); }

// ============================================================================
// State
// ============================================================================
let ws = null;
let reconnectTimeout = null;
let connected = false;
let pollInterval = null;
let eventsWatchInterval = null;
let lastEventsCharPosition = 0;

// ============================================================================
// Logging
// ============================================================================
function log(...args) {
    console.log(`[portolan-agent ${new Date().toISOString()}]`, ...args);
}

function debug(...args) {
    if (DEBUG) {
        console.log(`[DEBUG]`, ...args);
    }
}

// ============================================================================
// Session Discovery
// ============================================================================

/**
 * Check if a directory has claims (workflow/config or results/claims)
 */
function detectClaims(cwd) {
    const hasWorkflowConfig = existsSync(resolve(cwd, 'workflow/config'));
    const hasResultsClaims = existsSync(resolve(cwd, 'results/claims'));
    return hasWorkflowConfig || hasResultsClaims;
}

/**
 * Check if a directory has playgrounds (.portolan/playgrounds/ with .html files)
 */
function detectPlaygrounds(cwd) {
    const playgroundsDir = resolve(cwd, '.portolan/playgrounds');
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
 * Get git status for a directory
 * Returns null if not a git repo or on error
 */
async function getGitStatus(directory) {
    const TIMEOUT = 5000;

    try {
        // Check if it's a git repo
        await execAsync('git rev-parse --git-dir', { cwd: directory, timeout: TIMEOUT });
    } catch {
        // Not a git repo
        return null;
    }

    const status = {
        branch: '',
        ahead: 0,
        behind: 0,
        staged: { added: 0, modified: 0, deleted: 0 },
        unstaged: { added: 0, modified: 0, deleted: 0 },
        untracked: 0,
        totalFiles: 0,
        linesAdded: 0,
        linesRemoved: 0,
        lastCommitTime: null,
        lastCommitMessage: null,
        isRepo: true,
        lastChecked: Date.now(),
    };

    // Run all git commands in parallel
    const [branchResult, statusResult, diffStagedResult, diffUnstagedResult, logResult] =
        await Promise.all([
            execAsync('git rev-parse --abbrev-ref HEAD', { cwd: directory, timeout: TIMEOUT }).catch(() => ({ stdout: '' })),
            execAsync('git status --porcelain', { cwd: directory, timeout: TIMEOUT }).catch(() => ({ stdout: '' })),
            execAsync('git diff --cached --shortstat', { cwd: directory, timeout: TIMEOUT }).catch(() => ({ stdout: '' })),
            execAsync('git diff --shortstat', { cwd: directory, timeout: TIMEOUT }).catch(() => ({ stdout: '' })),
            execAsync('git log -1 --format=%ct|||%s', { cwd: directory, timeout: TIMEOUT }).catch(() => ({ stdout: '' })),
        ]);

    // Parse branch
    status.branch = branchResult.stdout.trim();

    // Parse ahead/behind
    try {
        const abResult = await execAsync(
            'git rev-list --left-right --count @{upstream}...HEAD',
            { cwd: directory, timeout: TIMEOUT }
        );
        const [behind, ahead] = abResult.stdout.trim().split(/\s+/).map(Number);
        status.ahead = ahead || 0;
        status.behind = behind || 0;
    } catch {
        // No upstream configured
    }

    // Parse status --porcelain
    const statusLines = statusResult.stdout.trim().split('\n').filter(Boolean);
    for (const line of statusLines) {
        const staged = line[0];
        const unstaged = line[1];

        // Staged changes
        if (staged === 'A') status.staged.added++;
        else if (staged === 'M') status.staged.modified++;
        else if (staged === 'D') status.staged.deleted++;

        // Unstaged changes
        if (unstaged === 'M') status.unstaged.modified++;
        else if (unstaged === 'D') status.unstaged.deleted++;

        // Untracked
        if (staged === '?' && unstaged === '?') status.untracked++;
    }

    // Parse diff stats
    const parseDiffStat = (output) => {
        const match = output.match(/(\d+) insertion.*?(\d+) deletion/i);
        if (match) {
            return { added: parseInt(match[1], 10), removed: parseInt(match[2], 10) };
        }
        const addMatch = output.match(/(\d+) insertion/i);
        const delMatch = output.match(/(\d+) deletion/i);
        return {
            added: addMatch ? parseInt(addMatch[1], 10) : 0,
            removed: delMatch ? parseInt(delMatch[1], 10) : 0,
        };
    };

    const stagedDiff = parseDiffStat(diffStagedResult.stdout);
    const unstagedDiff = parseDiffStat(diffUnstagedResult.stdout);

    status.linesAdded = stagedDiff.added + unstagedDiff.added;
    status.linesRemoved = stagedDiff.removed + unstagedDiff.removed;

    // Total files
    status.totalFiles =
        status.staged.added +
        status.staged.modified +
        status.staged.deleted +
        status.unstaged.modified +
        status.unstaged.deleted +
        status.untracked;

    // Parse last commit
    if (logResult.stdout) {
        const [timestamp, message] = logResult.stdout.trim().split('|||');
        status.lastCommitTime = parseInt(timestamp, 10) || null;
        status.lastCommitMessage = message || null;
    }

    return status;
}

/**
 * Discover tmux sessions running Claude Code
 */
async function discoverSessions() {
    try {
        // Get all tmux panes with their session name, cwd, and pane PID
        const { stdout } = await execAsync(
            'tmux list-panes -a -F "#{session_name}\t#{pane_current_path}\t#{pane_pid}"'
        );

        const lines = stdout.trim().split('\n').filter(Boolean);
        const paneData = [];

        for (const line of lines) {
            const [tmuxSession, cwd, panePid] = line.split('\t');
            // Skip the agent's own tmux session
            if (panePid && tmuxSession !== 'portolan-agent') {
                paneData.push({ tmuxSession, cwd: cwd || process.cwd(), panePid });
            }
        }

        if (paneData.length === 0) {
            return [];
        }

        // Check which panes have a CLI running (claude or codex, as process itself or child)
        // NOTE: Detection logic duplicated in src/SessionTracker.ts — keep in sync
        const claudeSessionsBasic = [];

        for (const { tmuxSession, cwd, panePid } of paneData) {
            try {
                // Check if pane process ITSELF is a CLI (when zsh -c execs into claude/codex)
                const { stdout: paneComm } = await execAsync(
                    `ps -o comm= -p ${panePid} 2>/dev/null || true`,
                    { timeout: 2000 }
                );
                if (isCliProcess(paneComm.trim())) {
                    claudeSessionsBasic.push({ tmuxSession, cwd });
                    continue;
                }
                // Check full args for cases like "node .../bin/codex" on Linux
                const { stdout: paneArgs } = await execAsync(
                    `ps -o args= -p ${panePid} 2>/dev/null || true`,
                    { timeout: 2000 }
                );
                if (paneArgs.trim() && isCliProcess(paneArgs.trim())) {
                    claudeSessionsBasic.push({ tmuxSession, cwd });
                    continue;
                }

                // BFS through descendants (up to depth 4) to handle deep chains
                // e.g. bash → python3 → MainThread → codex (ralph-launched sessions)
                let found = false;
                let frontier = [panePid];
                for (let depth = 0; depth < 4 && !found; depth++) {
                    const nextFrontier = [];
                    for (const pid of frontier) {
                        const { stdout: childPidsOut } = await execAsync(
                            `pgrep -P ${pid} 2>/dev/null || true`,
                            { timeout: 2000 }
                        );
                        for (const candidatePid of childPidsOut.trim().split('\n').filter(Boolean)) {
                            try {
                                const { stdout: childComm } = await execAsync(
                                    `ps -o comm= -p ${candidatePid} 2>/dev/null || true`,
                                    { timeout: 2000 }
                                );
                                if (isCliProcess(childComm.trim())) { found = true; break; }
                                const { stdout: childArgs } = await execAsync(
                                    `ps -o args= -p ${candidatePid} 2>/dev/null || true`,
                                    { timeout: 2000 }
                                );
                                if (isCliProcess(childArgs.trim())) { found = true; break; }
                                nextFrontier.push(candidatePid);
                            } catch { /* skip */ }
                        }
                        if (found) break;
                    }
                    frontier = nextFrontier;
                }
                if (found) {
                    claudeSessionsBasic.push({ tmuxSession, cwd });
                    continue;
                }
            } catch {
                // Ignore errors from pgrep
            }
        }

        // Fetch git status for each session (in parallel)
        const claudeSessions = await Promise.all(
            claudeSessionsBasic.map(async ({ tmuxSession, cwd }) => {
                const gitStatus = await getGitStatus(cwd);
                return {
                    name: tmuxSession,
                    tmuxSession,
                    cwd,
                    status: 'idle',
                    hasClaims: detectClaims(cwd),
                    hasPlaygrounds: detectPlaygrounds(cwd),
                    gitStatus,  // May be null if not a git repo
                };
            })
        );

        return claudeSessions;
    } catch {
        // tmux not running or error
        debug('tmux discovery failed (tmux not running?)');
        return [];
    }
}

/**
 * Poll sessions and send update to server
 */
async function pollSessions() {
    const sessions = await discoverSessions();

    if (connected && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'agent_sessions_update',
            payload: { sessions },
        }));
        debug(`Sent ${sessions.length} sessions to server: ${sessions.map(s => s.tmuxSession).join(', ')}`);
    }
}

// ============================================================================
// Event Watching (for activity stream)
// ============================================================================

/**
 * Tools to track in activity feed (file operations only, no Bash/Grep/Glob/Task)
 */
const TRACKED_TOOLS = new Set(['Read', 'Write', 'Edit']);

/**
 * Extract short summary from tool input
 *
 * NOTE: Keep in sync with server/src/activityUtils.ts (source of truth).
 * This is duplicated here because agent.js runs standalone on remote machines.
 */
function extractSummary(tool, input) {
    if (!input) return undefined;

    // Only track file operations
    if (!TRACKED_TOOLS.has(tool)) return undefined;

    if (input.file_path) {
        const parts = String(input.file_path).split('/');
        const filename = parts.pop();
        const parent = parts.pop();
        if (parent && filename) {
            const display = `${parent}/${filename}`;
            return display.length > 35 ? `…${display.slice(-34)}` : display;
        }
        return filename;
    }
    return undefined;
}

/**
 * Extract detailed activity information including full paths
 *
 * NOTE: Keep in sync with server/src/activityUtils.ts (source of truth).
 */
function extractActivityDetails(tool, input) {
    if (!input) return undefined;

    const summary = extractSummary(tool, input);
    if (!summary) return undefined;

    const details = { summary };

    // Include full path for file operations (already filtered by extractSummary)
    if (input.file_path) {
        details.fullPath = String(input.file_path);
    }

    return details;
}

/**
 * Start watching the events file for activity
 */
function startEventsWatcher() {
    if (!existsSync(EVENTS_FILE)) {
        log(`Events file not found: ${EVENTS_FILE}`);
        log('Activity tracking disabled. Install portolan-hook.sh to enable.');
        return;
    }

    // Initialize position to end of file
    try {
        const content = readFileSync(EVENTS_FILE, 'utf-8');
        lastEventsCharPosition = content.length;
    } catch {
        lastEventsCharPosition = 0;
    }

    // Poll for new events
    eventsWatchInterval = setInterval(processNewEvents, 1000);
    log(`Watching events file: ${EVENTS_FILE}`);
}

/**
 * Process new events since last read
 */
function processNewEvents() {
    if (!existsSync(EVENTS_FILE)) return;

    try {
        const content = readFileSync(EVENTS_FILE, 'utf-8');
        if (content.length <= lastEventsCharPosition) {
            return; // No new data
        }

        const newContent = content.slice(lastEventsCharPosition);
        lastEventsCharPosition = content.length;

        const lines = newContent.trim().split('\n');
        for (const line of lines) {
            if (!line) continue;
            try {
                const event = JSON.parse(line);
                processEvent(event);
            } catch {
                // Skip malformed lines
            }
        }
    } catch (err) {
        debug(`Event read error: ${err.message}`);
    }
}

/**
 * Process a single event
 */
function processEvent(event) {
    if (!event.tmuxSession) return;

    // Only send activity for pre_tool_use events (has tool info)
    if (event.type === 'pre_tool_use' && event.tool) {
        const details = extractActivityDetails(event.tool, event.toolInput);
        const activity = {
            tmuxSession: event.tmuxSession,
            tool: event.tool,
            summary: details?.summary,
            fullPath: details?.fullPath,
            timestamp: event.timestamp,
        };

        // Send to server
        if (connected && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'agent_activity',
                activity,
            }));
            debug(`Activity: ${activity.tool} ${activity.summary || ''}`);
        }
    }
}

// ============================================================================
// WebSocket Connection
// ============================================================================

/**
 * Build specific node SSH alias from base sshHost and hostname.
 * e.g., sshHost="cineca", hostname="login05.leonardo.local" → "cineca-login05"
 */
function buildSpecificSshHost(baseSshHost) {
    if (!baseSshHost) return null;

    const nodeMatch = ORIGIN_NAME.match(/^(login\d+)\./);
    if (nodeMatch) {
        // Don't double-append if baseSshHost already ends with this login node
        if (baseSshHost.endsWith(`-${nodeMatch[1]}`)) {
            log(`SSH host already specific: ${baseSshHost}`);
            return baseSshHost;
        }
        const specificHost = `${baseSshHost}-${nodeMatch[1]}`;
        log(`Using specific node SSH host: ${specificHost} (from ${ORIGIN_NAME})`);
        return specificHost;
    }
    return baseSshHost;
}

/**
 * Connect to portolan server
 */
function connect(serverUrl, sshHost) {
    if (ws) {
        ws.close();
    }

    // Build specific node SSH alias for multi-node HPC systems
    const specificSshHost = buildSpecificSshHost(sshHost);

    // Build URL with query params
    let url = `ws://${serverUrl}?agent=true&origin=${encodeURIComponent(ORIGIN_NAME)}`;
    if (specificSshHost) {
        url += `&sshHost=${encodeURIComponent(specificSshHost)}`;
    }
    if (PLANNOTATOR_PORT) {
        url += `&plannotatorPort=${PLANNOTATOR_PORT}`;
    }

    log(`Connecting to ${url}...`);

    ws = new WebSocket(url);

    ws.on('open', () => {
        connected = true;
        log(`Connected to portolan server at ${serverUrl}`);

        // Send initial sessions
        pollSessions();
    });

    ws.on('message', (data) => {
        try {
            const message = JSON.parse(data.toString());
            handleMessage(message);
        } catch (e) {
            debug(`Failed to parse message: ${e}`);
        }
    });

    ws.on('close', () => {
        connected = false;
        log('Disconnected from server');
        scheduleReconnect(serverUrl, sshHost);
    });

    ws.on('error', (error) => {
        debug(`WebSocket error: ${error.message}`);
    });
}

/**
 * Handle incoming messages from server
 */
function handleMessage(message) {
    switch (message.type) {
        case 'connected':
            log(`Registered with server (origin: ${message.payload?.originId})`);
            break;

        default:
            debug(`Unhandled message type: ${message.type}`);
    }
}

/**
 * Schedule reconnection
 */
function scheduleReconnect(serverUrl, sshHost) {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
    }

    log(`Reconnecting in ${RECONNECT_INTERVAL / 1000}s...`);
    reconnectTimeout = setTimeout(() => {
        connect(serverUrl, sshHost);
    }, RECONNECT_INTERVAL);
}

// ============================================================================
// CLI
// ============================================================================
function printUsage() {
    console.log(`
Portolan Remote Agent

Usage:
  portolan-agent connect [host:port] [--ssh-host=name]
  portolan-agent status
  portolan-agent help

Options:
  host:port       Server address (default: localhost:4004)
  --ssh-host=X    SSH config name for this host (default: hostname)

Environment:
  PORTOLAN_ORIGIN     Origin name (default: hostname)
  PORTOLAN_DEBUG      Enable debug logging (true/false)

SSH tunnel setup (in local ~/.ssh/config):
  Host yourserver
    RemoteForward 4004 127.0.0.1:4004

Example:
  # On remote machine (with SSH tunnel active)
  portolan-agent connect

  # With custom SSH host name (for focus commands)
  portolan-agent connect --ssh-host=myserver
`);
}

async function main() {
    const args = process.argv.slice(2);
    const command = args[0];

    // Parse --ssh-host flag
    let sshHost = null;
    const sshHostArg = args.find(a => a.startsWith('--ssh-host='));
    if (sshHostArg) {
        sshHost = sshHostArg.split('=')[1];
    }

    // Get server URL (first positional arg that doesn't start with --)
    const positionalArgs = args.filter(a => !a.startsWith('--'));

    switch (command) {
        case 'connect': {
            const serverUrl = positionalArgs[1] || DEFAULT_SERVER;
            log(`Starting portolan-agent (origin: ${ORIGIN_NAME})`);

            // Start session polling
            pollInterval = setInterval(pollSessions, POLL_INTERVAL);
            await pollSessions();

            // Start events watcher (for activity stream)
            startEventsWatcher();

            // Connect to server
            connect(serverUrl, sshHost);

            // Keep alive
            process.on('SIGINT', () => {
                log('Shutting down...');
                if (pollInterval) clearInterval(pollInterval);
                if (eventsWatchInterval) clearInterval(eventsWatchInterval);
                if (ws) ws.close();
                process.exit(0);
            });
            break;
        }

        case 'status': {
            const sessions = await discoverSessions();
            if (sessions.length === 0) {
                console.log('No Claude Code sessions found');
            } else {
                console.log(`Found ${sessions.length} Claude Code session(s):\n`);
                for (const session of sessions) {
                    console.log(`  ${session.tmuxSession}`);
                    console.log(`    cwd: ${session.cwd}`);
                    console.log();
                }
            }
            break;
        }

        case 'help':
        case '--help':
        case '-h':
            printUsage();
            break;

        default:
            if (command) {
                console.error(`Unknown command: ${command}\n`);
            }
            printUsage();
            process.exit(command ? 1 : 0);
    }
}

main().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
});
