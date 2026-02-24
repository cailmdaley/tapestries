/**
 * MessageRouter - WebSocket message dispatch
 *
 * Routes incoming WebSocket messages to appropriate handlers.
 * Separates the "what type of message is this" logic from
 * the actual message handling.
 */

import { WebSocket } from 'ws';
import type { GitStatus } from './GitStatusManager.js';

// ============================================================================
// Message Types
// ============================================================================

export interface FocusMessage {
  type: 'focus';
  sessionId: string;
}

export interface AgentSessionsUpdateMessage {
  type: 'agent_sessions_update';
  payload: {
    sessions: Array<{
      id?: string;
      name: string;
      tmuxSession: string;
      cwd: string;
      status?: 'idle' | 'working' | 'offline';
      hasClaims?: boolean;
      hasPlaygrounds?: boolean;
      gitStatus?: GitStatus;
    }>;
  };
}

export interface AgentActivityMessage {
  type: 'agent_activity';
  activity: {
    tmuxSession: string;
    tool: string;
    summary?: string;
    fullPath?: string;                     // Full file path for Read/Write/Edit
    timestamp: number;
  };
}

export interface AgentConversationMessage {
  type: 'agent_conversation';
  payload: {
    sessionId: string;
    tmuxSession: string;
    cwd: string;
    messages: Array<{
      type: 'user' | 'assistant' | 'thinking' | 'tool_use' | 'tool_result';
      content: string;
      timestamp: string;
      toolName?: string;
      toolInput?: any;
      preview?: string;
    }>;
  };
}

export interface GetFibersMessage {
  type: 'getFibers';
  cityId: string;
}

export interface HandoffMessage {
  type: 'handoff';
  fiberId: string;
  cityPath: string;
}

export interface NewWorkerMessage {
  type: 'newWorker';
  cityPath: string;
  name?: string;
  chrome?: boolean;
  continue?: boolean;
  cli?: string;
}

export interface PinCityMessage {
  type: 'pinCity';
  path: string;
  position: { q: number; r: number };
  name?: string;
}

export interface UnpinCityMessage {
  type: 'unpinCity';
  cityId: string;
}

export interface ConfirmUnpinMessage {
  type: 'confirmUnpinCity';
  cityId: string;
}

export interface KillWorkerMessage {
  type: 'killWorker';
  sessionId: string;
}

export interface SearchFilesMessage {
  type: 'searchFiles';
  cityId: string;
  query: string;
  searchId: string;  // For cancellation
  mode?: 'filename' | 'content';  // Default: filename
}

export interface MoveCityMessage {
  type: 'moveCity';
  cityId: string;
  newPosition: { q: number; r: number };
}

export type ClientMessage =
  | FocusMessage
  | AgentSessionsUpdateMessage
  | GetFibersMessage
  | HandoffMessage
  | NewWorkerMessage
  | PinCityMessage
  | UnpinCityMessage
  | ConfirmUnpinMessage
  | KillWorkerMessage
  | SearchFilesMessage
  | MoveCityMessage;

// ============================================================================
// Handler Interface
// ============================================================================

export interface MessageHandlers {
  onFocus(sessionId: string): void;
  onGetFibers(ws: WebSocket, cityId: string): Promise<void>;
  onHandoff(fiberId: string, cityPath: string): void | Promise<void>;
  onNewWorker(ws: WebSocket, cityPath: string, name?: string, chrome?: boolean, continueSession?: boolean, cli?: string): void;
  onPinCity(ws: WebSocket, path: string, position: { q: number; r: number }, name?: string): void;
  onUnpinCity(ws: WebSocket, cityId: string): void;
  onConfirmUnpin(ws: WebSocket, cityId: string): void;
  onKillWorker(sessionId: string): void;
  onSearchFiles(ws: WebSocket, cityId: string, query: string, searchId: string, mode?: 'filename' | 'content'): void;
  onMoveCity(ws: WebSocket, cityId: string, newPosition: { q: number; r: number }): void;
}

// ============================================================================
// MessageRouter
// ============================================================================

export class MessageRouter {
  private handlers: MessageHandlers;

  constructor(handlers: MessageHandlers) {
    this.handlers = handlers;
  }

  /**
   * Route a message from a browser client to the appropriate handler
   */
  routeClientMessage(ws: WebSocket, data: string): void {
    try {
      const message = JSON.parse(data) as ClientMessage;
      console.log('[MessageRouter] Received:', message.type);

      switch (message.type) {
        case 'focus':
          this.handlers.onFocus(message.sessionId);
          break;

        case 'getFibers':
          this.handlers.onGetFibers(ws, message.cityId);
          break;

        case 'handoff':
          this.handlers.onHandoff(message.fiberId, message.cityPath);
          break;

        case 'newWorker':
          this.handlers.onNewWorker(ws, message.cityPath, message.name, message.chrome, message.continue, message.cli);
          break;

        case 'pinCity':
          this.handlers.onPinCity(ws, message.path, message.position, message.name);
          break;

        case 'unpinCity':
          this.handlers.onUnpinCity(ws, message.cityId);
          break;

        case 'confirmUnpinCity':
          this.handlers.onConfirmUnpin(ws, message.cityId);
          break;

        case 'killWorker':
          this.handlers.onKillWorker(message.sessionId);
          break;

        case 'searchFiles':
          this.handlers.onSearchFiles(ws, message.cityId, message.query, message.searchId, message.mode);
          break;

        case 'moveCity':
          this.handlers.onMoveCity(ws, message.cityId, message.newPosition);
          break;

        default:
          console.warn('[MessageRouter] Unknown message type:', (message as any).type);
      }
    } catch (error) {
      console.error('[MessageRouter] Failed to handle message:', error);
    }
  }
}
