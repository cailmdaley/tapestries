/**
 * Activity utilities - shared between EventWatcher and agent.js
 *
 * NOTE: This is the source of truth. When updating extractSummary or
 * extractActivityDetails, also update the copy in agent.js (which can't
 * import this directly since it runs standalone on remote machines).
 */

/**
 * Detailed activity information for file viewer support
 */
export interface ActivityDetails {
  summary: string;           // Short display text (filename, command, pattern)
  fullPath?: string;         // Full file path for Read/Write/Edit
}

/**
 * Tools to track in activity feed (file operations only, no Bash/Grep/Glob/Task)
 */
const TRACKED_TOOLS = new Set(['Read', 'Write', 'Edit']);

/**
 * Extract a short summary from tool input for activity display
 * Only tracks file operations (Read, Write, Edit)
 */
function extractSummary(tool: string, input?: Record<string, unknown>): string | undefined {
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
 * Extract detailed activity information including full paths for file viewer
 */
export function extractActivityDetails(tool: string, input?: Record<string, unknown>): ActivityDetails | undefined {
  if (!input) return undefined;

  const summary = extractSummary(tool, input);
  if (!summary) return undefined;

  const details: ActivityDetails = { summary };

  // Include full path (already filtered to file operations by extractSummary)
  if (input.file_path) {
    details.fullPath = String(input.file_path);
  }

  return details;
}
