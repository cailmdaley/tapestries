/**
 * HttpApi - HTTP request handlers
 *
 * Handles non-WebSocket HTTP endpoints:
 * - Tapestry DAG (fibers, evidence, staleness)
 * - Evidence artifact serving
 * - Annotation CRUD
 * - City activation (start remote agent)
 */

import { IncomingMessage, ServerResponse } from 'http';
import { URL } from 'url';
import { exec, execFile, execFileSync, spawn, execSync } from 'child_process';
import { readFile, writeFile } from 'fs/promises';
import { promisify } from 'util';
import { extname } from 'path';
import type { City } from './CityManager.js';
import type { Origin } from './OriginManager.js';
import type { AnnotationPersistence, Annotation } from './AnnotationPersistence.js';
import type { Session } from './SessionTracker.js';
import type { TranscriptReader } from './TranscriptReader.js';
import type { ConversationCache, CachedMessage } from './ConversationCache.js';
import type { CardStatePersistence } from './CardStatePersistence.js';
import { shellEscape } from './KittyIntegration.js';
import { getAllFibers, type Fiber } from './FiberReader.js';
import { readEvidence, readEvidenceBatch, getSpecName, computeStaleness, type Evidence } from './EvidenceReader.js';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

/** Shared MIME type map for binary/asset serving (tapestry assets, file-content) */
const MIME_TYPES: Record<string, string> = {
  'png': 'image/png',
  'jpg': 'image/jpeg',
  'jpeg': 'image/jpeg',
  'gif': 'image/gif',
  'svg': 'image/svg+xml',
  'webp': 'image/webp',
  'ico': 'image/x-icon',
  'pdf': 'application/pdf',
  'otf': 'font/otf',
  'ttf': 'font/ttf',
  'woff': 'font/woff',
  'woff2': 'font/woff2',
  'css': 'text/css',
  'js': 'application/javascript',
};

// ============================================================================
// Types
// ============================================================================

interface CityLookup {
  getCityById(cityId: string): City | null;
}

interface OriginLookup {
  getOrigin(originId: string): Origin | null | undefined;
}

interface PersistenceLookup {
  getCityById(cityId: string): { sshHost?: string } | null;
}

interface SessionLookup {
  findSession(sessionId: string): Session | undefined;
  getAllSessions(): Session[];
}

// ============================================================================
// HttpApi
// ============================================================================

// Remote conversation lookup callback
export type RemoteConversationLookup = (sessionId: string) => any[] | undefined;

export class HttpApi {
  private cityLookup: CityLookup;
  private originLookup: OriginLookup;
  private persistenceLookup: PersistenceLookup;
  private annotationPersistence: AnnotationPersistence | null = null;
  private sessionLookup: SessionLookup | null = null;
  private transcriptReader: TranscriptReader | null = null;
  private remoteConversationLookup: RemoteConversationLookup | null = null;
  private conversationCache: ConversationCache | null = null;
  private cardStatePersistence: CardStatePersistence | null = null;

  constructor(
    cityLookup: CityLookup,
    originLookup: OriginLookup,
    persistenceLookup: PersistenceLookup
  ) {
    this.cityLookup = cityLookup;
    this.originLookup = originLookup;
    this.persistenceLookup = persistenceLookup;
  }

  /**
   * Set annotation persistence instance
   */
  setAnnotationPersistence(persistence: AnnotationPersistence): void {
    this.annotationPersistence = persistence;
  }

  /**
   * Set session lookup instance
   */
  setSessionLookup(lookup: SessionLookup): void {
    this.sessionLookup = lookup;
  }

  /**
   * Set transcript reader for conversation history
   */
  setTranscriptReader(reader: TranscriptReader): void {
    this.transcriptReader = reader;
  }

  /**
   * Set remote conversation lookup for remote worker transcripts
   */
  setRemoteConversationLookup(lookup: RemoteConversationLookup): void {
    this.remoteConversationLookup = lookup;
  }

  /**
   * Set conversation cache for hook-based conversation updates
   */
  setConversationCache(cache: ConversationCache): void {
    this.conversationCache = cache;
  }

  /**
   * Set card state persistence for conversation card positions
   */
  setCardStatePersistence(persistence: CardStatePersistence): void {
    this.cardStatePersistence = persistence;
  }

  /**
   * Handle HTTP request - returns true if handled, false to fall through
   */
  async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
    const url = new URL(req.url || '/', `http://${req.headers.host}`);

    // Handle CORS preflight for all HTTP methods
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '86400',
      });
      res.end();
      return true;
    }

    if (url.pathname === '/tapestry') {
      await this.handleTapestry(url, res);
      return true;
    }

    if (url.pathname.startsWith('/tapestry-asset/')) {
      await this.handleTapestryAsset(url, res);
      return true;
    }

    if (req.method === 'POST' && url.pathname === '/activate-city') {
      await this.handleActivateCity(url, res);
      return true;
    }

    if (url.pathname === '/file-content') {
      await this.handleFileContent(url, res);
      return true;
    }

    if (req.method === 'POST' && url.pathname === '/save-file') {
      await this.handleSaveFile(req, res);
      return true;
    }

    // Annotation endpoints
    if (url.pathname === '/annotations' && req.method === 'GET') {
      await this.handleGetAnnotations(url, res);
      return true;
    }

    if (url.pathname === '/recent-annotations' && req.method === 'GET') {
      await this.handleRecentAnnotations(url, res);
      return true;
    }

    if (url.pathname === '/annotations' && req.method === 'POST') {
      await this.handleCreateAnnotation(req, res);
      return true;
    }

    if (url.pathname.match(/^\/annotations\/[^/]+$/) && req.method === 'PUT') {
      const id = url.pathname.split('/')[2];
      await this.handleUpdateAnnotation(id, req, res);
      return true;
    }

    if (url.pathname.match(/^\/annotations\/[^/]+$/) && req.method === 'DELETE') {
      const id = url.pathname.split('/')[2];
      await this.handleDeleteAnnotation(id, res);
      return true;
    }

    if (url.pathname === '/send-annotations' && req.method === 'POST') {
      await this.handleSendAnnotations(req, res);
      return true;
    }

    if (url.pathname === '/send-message' && req.method === 'POST') {
      await this.handleSendMessage(req, res);
      return true;
    }

    if (url.pathname === '/file-as-fiber' && req.method === 'POST') {
      await this.handleFileAsFiber(req, res);
      return true;
    }

    if (url.pathname === '/promote-to-felt' && req.method === 'POST') {
      await this.handlePromoteToFelt(req, res);
      return true;
    }

    if (url.pathname === '/playground-list') {
      await this.handlePlaygroundList(url, res);
      return true;
    }

    if (url.pathname === '/playground') {
      await this.handlePlayground(url, res);
      return true;
    }

    if (url.pathname === '/conversation') {
      await this.handleConversation(url, res);
      return true;
    }

    if (url.pathname === '/debug-transcripts') {
      await this.handleDebugTranscripts(res);
      return true;
    }

    // Hook endpoints for conversation capture
    if (req.method === 'POST' && url.pathname === '/hook/message') {
      await this.handleHookMessage(req, res);
      return true;
    }

    if (url.pathname === '/hook/health') {
      await this.handleHookHealth(res);
      return true;
    }

    // Card state persistence endpoints
    if (url.pathname === '/card-states' && req.method === 'GET') {
      await this.handleGetCardStates(res);
      return true;
    }

    if (url.pathname.match(/^\/card-state\/[^/]+$/) && req.method === 'GET') {
      const workerId = decodeURIComponent(url.pathname.split('/')[2]);
      await this.handleGetCardState(workerId, res);
      return true;
    }

    if (url.pathname.match(/^\/card-state\/[^/]+$/) && req.method === 'PUT') {
      const workerId = decodeURIComponent(url.pathname.split('/')[2]);
      await this.handleSaveCardState(workerId, req, res);
      return true;
    }

    if (url.pathname.match(/^\/card-state\/[^/]+$/) && req.method === 'DELETE') {
      const workerId = decodeURIComponent(url.pathname.split('/')[2]);
      await this.handleDeleteCardState(workerId, res);
      return true;
    }

    return false;
  }

  /**
   * Get SSH host for a city (from origin or persistence)
   */
  private getSshHost(city: City): string {
    const origin = this.originLookup.getOrigin(city.originId);
    const persistedCity = this.persistenceLookup.getCityById(city.id);
    return origin?.sshHost || persistedCity?.sshHost || city.originId.replace('remote-', '');
  }

  // ============================================================================
  // Tapestry Endpoint
  // ============================================================================

  /**
   * GET /tapestry?cityId=xxx
   *
   * Returns the full DAG for TapestryView: fibers with tapestry: tags,
   * dependency edges, evidence summary per fiber, staleness flags.
   */
  private async handleTapestry(url: URL, res: ServerResponse): Promise<void> {
    const cityId = url.searchParams.get('cityId');
    if (!cityId) {
      this.sendJsonError(res, 400, 'Missing cityId parameter');
      return;
    }

    const city = this.cityLookup.getCityById(cityId);
    if (!city) {
      this.sendJsonError(res, 404, 'City not found');
      return;
    }

    const sshHost = city.originId !== 'local' ? this.getSshHost(city) : undefined;

    try {
      // Single read: get all fibers, then partition into tapestry vs non-tapestry
      const allFibers = await this.getAllCityFibers(city.path, sshHost);
      const ruleFibers = allFibers.filter(f => f.tags?.some(t => t.startsWith('tapestry:') || t.startsWith('rule:')));
      const fiberIds = new Set(ruleFibers.map(f => f.id));

      // Build spec name map: fiberId → specName
      const fiberSpecMap = new Map<string, string>();
      for (const fiber of ruleFibers) {
        const specName = getSpecName(fiber.tags || []);
        if (specName) {
          fiberSpecMap.set(fiber.id, specName);
        }
      }

      // Read evidence for each unique specName
      const uniqueSpecNames = Array.from(new Set(fiberSpecMap.values()));
      let evidenceMap: Map<string, Evidence | null>;
      if (sshHost && uniqueSpecNames.length > 0) {
        // Single SSH call for all specs — avoids connection exhaustion
        evidenceMap = await readEvidenceBatch(city.path, uniqueSpecNames, sshHost);
      } else {
        evidenceMap = new Map<string, Evidence | null>();
        await Promise.all(
          uniqueSpecNames.map(async (specName) => {
            const ev = await readEvidence(city.path, specName);
            evidenceMap.set(specName, ev);
          })
        );
      }

      // Build response nodes
      const nodes = ruleFibers.map(fiber => {
        const specName = fiberSpecMap.get(fiber.id);
        const evidence = specName ? evidenceMap.get(specName) : null;
        const deps = (fiber.dependsOn || []).filter(d => fiberIds.has(d));
        const staleness = computeStaleness(fiber.id, deps, evidenceMap, fiberSpecMap);

        return {
          id: fiber.id,
          title: fiber.title,
          kind: fiber.kind,
          status: fiber.status,
          body: fiber.body,
          outcome: fiber.outcome || null,
          tags: fiber.tags || [],
          createdAt: fiber.createdAt || null,
          closedAt: fiber.closedAt || null,
          dependsOn: deps,
          specName: specName || null,
          staleness,
          evidence: evidence ? {
            metrics: evidence.metrics,
            artifacts: evidence.artifacts,
            mtime: evidence.mtime,
            generated: evidence.generated ?? null,
          } : null,
        };
      });

      // Build edges
      const links: Array<{ source: string; target: string }> = [];
      for (const fiber of ruleFibers) {
        for (const dep of (fiber.dependsOn || [])) {
          if (fiberIds.has(dep)) {
            links.push({ source: dep, target: fiber.id });
          }
        }
      }

      // Downstream concerns: all fibers that depend on rule fibers
      const downstreamMap: Record<string, Array<{ id: string; title: string; status: string; kind: string }>> = {};
      for (const fiber of allFibers) {
        for (const dep of (fiber.dependsOn || [])) {
          if (fiberIds.has(dep)) {
            if (!downstreamMap[dep]) downstreamMap[dep] = [];
            downstreamMap[dep].push({
              id: fiber.id,
              title: fiber.title,
              status: fiber.status,
              kind: fiber.kind,
            });
          }
        }
      }

      // Read project config (workflow/config/config.yaml) if it exists
      const config = await this.readCityConfig(city.path, sshHost);

      // All fibers (for sidebar listing)
      const fibers = allFibers.map(f => ({
        id: f.id,
        title: f.title,
        status: f.status,
        kind: f.kind,
        tags: f.tags,
        body: f.body,
        outcome: f.outcome || null,
        createdAt: f.createdAt || null,
        closedAt: f.closedAt || null,
        dependsOn: f.dependsOn || [],
      }));

      this.sendJsonSuccess(res, {
        nodes,
        links,
        downstream: downstreamMap,
        config,
        fibers,
      });
    } catch (error: any) {
      console.error('Failed to build tapestry:', error);
      this.sendJsonError(res, 500, 'Failed to build tapestry: ' + error.message);
    }
  }

  /**
   * Get all fibers for a city (local or remote).
   * Remote cities use `felt ls --json --body` via SSH.
   */
  private async getAllCityFibers(cityPath: string, sshHost?: string): Promise<Fiber[]> {
    if (!sshHost) {
      return getAllFibers(cityPath);
    }

    const cmd = `cd ${shellEscape(cityPath)} && felt ls -s all --json --body 2>/dev/null || echo '[]'`;
    const { stdout } = await execFileAsync(
      'ssh', [sshHost, cmd],
      { maxBuffer: 10 * 1024 * 1024, timeout: 30000 },
    );

    const raw = JSON.parse(stdout.trim() || '[]');
    return raw.map((f: any): Fiber => ({
      id: f.id,
      title: f.title || f.id,
      status: f.status || 'open',
      kind: f.kind || 'task',
      priority: f.priority || 2,
      createdAt: f.created_at || '',
      closedAt: f.closed_at,
      outcome: f.outcome || f.close_reason,
      body: f.body,
      // Normalize comma-separated tags: "claim, tapestry:foo" → ["claim", "tapestry:foo"]
      tags: f.tags?.flatMap((t: string) => t.includes(',') ? t.split(',').map((s: string) => s.trim()).filter(Boolean) : [t]),
      dependsOn: f.depends_on?.map((d: any) => typeof d === 'string' ? d : d.id),
    }));
  }

  /**
   * Read and flatten project config (workflow/config/config.yaml).
   * Returns a flat Record<string, string> of dotted key paths to values,
   * or null if no config exists.
   */
  private async readCityConfig(
    cityPath: string,
    sshHost?: string,
  ): Promise<Record<string, string> | null> {
    // Try candidate config locations in priority order
    const candidates = [
      `${cityPath}/config/config.yaml`,
      `${cityPath}/workflow/config/config.yaml`,
    ];

    try {
      let content: string = '';
      if (sshHost) {
        const tryPaths = candidates.map(p => `cat ${shellEscape(p)} 2>/dev/null`).join(' || ');
        const { stdout } = await execFileAsync(
          'ssh', [sshHost, `${tryPaths} || echo ''`],
          { maxBuffer: 1024 * 1024, timeout: 10000 },
        );
        content = stdout.trim();
      } else {
        const { readFile } = await import('fs/promises');
        for (const p of candidates) {
          try { content = await readFile(p, 'utf-8'); break; } catch { /* try next */ }
        }
      }

      if (!content) return null;

      const { parse } = await import('yaml');
      const data = parse(content);
      if (!data || typeof data !== 'object') return null;

      // Flatten to dotted paths
      const flat: Record<string, string> = {};
      const walk = (obj: unknown, prefix: string) => {
        if (obj === null || obj === undefined) return;
        if (Array.isArray(obj)) {
          flat[prefix] = JSON.stringify(obj);
          return;
        }
        if (typeof obj === 'object') {
          for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
            walk(v, prefix ? `${prefix}.${k}` : k);
          }
          return;
        }
        flat[prefix] = String(obj);
      };
      walk(data, '');
      return flat;
    } catch {
      return null;
    }
  }

  /**
   * Serve evidence artifacts (plots, images) from results/claims/{specName}/
   * GET /tapestry-asset/{specName}/{filename}?cityId=xxx
   */
  private async handleTapestryAsset(url: URL, res: ServerResponse): Promise<void> {
    const cityId = url.searchParams.get('cityId');
    const rawPath = url.pathname.replace('/tapestry-asset/', '');
    const parts = rawPath.split('/');

    if (!cityId || parts.length < 2) {
      res.writeHead(400, { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' });
      res.end('Missing cityId or invalid asset path');
      return;
    }

    let assetPath: string;
    try {
      assetPath = decodeURIComponent(parts.join('/'));
    } catch {
      res.writeHead(400, { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' });
      res.end('Invalid asset path');
      return;
    }

    await this.serveTapestryAsset(cityId, assetPath, res);
  }

  // ── Shared asset serving ──────────────────────────────────────────

  /**
   * Serve a tapestry asset file (plot, image, etc.) from results/claims/.
   * Validates path, resolves city, reads file locally or via SSH.
   */
  private async serveTapestryAsset(cityId: string, assetPath: string, res: ServerResponse): Promise<void> {
    // Security: prevent directory traversal and shell injection
    if (assetPath.includes('..') || /[`$"\\]/.test(assetPath)) {
      res.writeHead(400, { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' });
      res.end('Invalid asset path');
      return;
    }

    const city = this.cityLookup.getCityById(cityId);
    if (!city) {
      res.writeHead(404, { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' });
      res.end('City not found');
      return;
    }

    const fullPath = `${city.path}/results/claims/${assetPath}`;
    const ext = assetPath.split('.').pop()?.toLowerCase();
    const contentType = MIME_TYPES[ext || ''] || 'application/octet-stream';

    try {
      let data: Buffer;
      if (city.originId === 'local') {
        data = await readFile(fullPath);
      } else {
        const sshHost = this.getSshHost(city);
        // encoding: 'buffer' returns { stdout: Buffer } at runtime, but
        // promisify(execFile) types don't express this overload
        const result = await execFileAsync(
          'ssh', [sshHost, `cat ${shellEscape(fullPath)}`],
          { maxBuffer: 10 * 1024 * 1024, encoding: 'buffer' as BufferEncoding },
        );
        data = Buffer.from(result.stdout as unknown as Buffer);
      }

      res.writeHead(200, {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache',
      });
      res.end(data);
    } catch (error) {
      res.writeHead(404, { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' });
      res.end(`Asset not found: ${assetPath}`);
    }
  }

  /**
   * File content endpoint for activity viewer
   * GET /file-content?path=/full/path/to/file&originId=optional&binary=true
   */
  private async handleFileContent(url: URL, res: ServerResponse): Promise<void> {
    const filePath = url.searchParams.get('path');
    const originId = url.searchParams.get('originId');
    const binary = url.searchParams.get('binary') === 'true';
    const raw = url.searchParams.get('raw') === 'true';

    if (!filePath) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing path parameter' }));
      return;
    }

    // Security: prevent directory traversal
    if (filePath.includes('..')) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid path' }));
      return;
    }

    const ext = extname(filePath).toLowerCase().slice(1);

    // Handle raw binary response (for <img src="..."> tags)
    if (raw && this.isBinaryExtension(ext)) {
      await this.handleRawBinaryContent(filePath, originId, ext, res);
      return;
    }

    // Handle binary files (images, PDFs) - returns JSON with data URL
    if (binary && this.isBinaryExtension(ext)) {
      await this.handleBinaryContent(filePath, originId, ext, res);
      return;
    }

    try {
      let content: string;

      if (!originId || originId === 'local') {
        // Local file: read directly
        content = await readFile(filePath, 'utf-8');
      } else {
        // Remote file: fetch via SSH
        const origin = this.originLookup.getOrigin(originId);
        if (!origin?.sshHost) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Origin not found or not connected' }));
          return;
        }

        const { stdout } = await execFileAsync(
          'ssh', [origin.sshHost, `cat ${shellEscape(filePath)}`],
          { maxBuffer: 10 * 1024 * 1024, timeout: 10000 }
        );
        content = stdout;
      }

      // Detect language from extension
      const language = this.extToLanguage(ext);

      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(JSON.stringify({ content, language, path: filePath }));
    } catch (error: any) {
      console.error('Failed to fetch file content:', error.message);
      const statusCode = error.code === 'ENOENT' ? 404 : 500;
      res.writeHead(statusCode, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.code === 'ENOENT' ? 'File not found' : 'Failed to read file' }));
    }
  }

  /** Binary extensions: subset of MIME_TYPES used for base64 data URL serving */
  private static readonly BINARY_EXTENSIONS = new Set([
    'png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'pdf',
  ]);

  private isBinaryExtension(ext: string): boolean {
    return HttpApi.BINARY_EXTENSIONS.has(ext);
  }

  /**
   * Handle binary file content (images, PDFs) - returns base64 data URL
   */
  private async handleBinaryContent(
    filePath: string,
    originId: string | null,
    ext: string,
    res: ServerResponse
  ): Promise<void> {
    const mimeType = MIME_TYPES[ext] || 'application/octet-stream';
    const fileType = ext === 'pdf' ? 'pdf' : 'image';
    // PDFs need larger buffer/timeout
    const maxBuffer = ext === 'pdf' ? 50 * 1024 * 1024 : 10 * 1024 * 1024;
    const timeout = ext === 'pdf' ? 60000 : 30000;

    try {
      let data: Buffer;

      if (!originId || originId === 'local') {
        data = await readFile(filePath);
      } else {
        const origin = this.originLookup.getOrigin(originId);
        if (!origin?.sshHost) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Origin not found or not connected' }));
          return;
        }

        const { stdout } = await execFileAsync(
          'ssh', [origin.sshHost, `base64 ${shellEscape(filePath)}`],
          { maxBuffer, timeout }
        );
        data = Buffer.from(stdout.replace(/\s/g, ''), 'base64');
      }

      const dataUrl = `data:${mimeType};base64,${data.toString('base64')}`;

      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(JSON.stringify({ type: fileType, url: dataUrl, path: filePath }));
    } catch (error: any) {
      console.error(`Failed to fetch ${fileType} content:`, error.message);
      const statusCode = error.code === 'ENOENT' ? 404 : 500;
      res.writeHead(statusCode, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: error.code === 'ENOENT' ? `${fileType} not found` : `Failed to read ${fileType}`
      }));
    }
  }

  /**
   * Serve raw binary content with proper Content-Type (for direct <img src="..."> use)
   */
  private async handleRawBinaryContent(
    filePath: string,
    originId: string | null,
    ext: string,
    res: ServerResponse
  ): Promise<void> {
    const mimeType = MIME_TYPES[ext] || 'application/octet-stream';
    const maxBuffer = ext === 'pdf' ? 50 * 1024 * 1024 : 10 * 1024 * 1024;
    const timeout = ext === 'pdf' ? 60000 : 30000;

    try {
      let data: Buffer;

      if (!originId || originId === 'local') {
        data = await readFile(filePath);
      } else {
        const origin = this.originLookup.getOrigin(originId);
        if (!origin?.sshHost) {
          res.writeHead(404, { 'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*' });
          res.end('Origin not found');
          return;
        }

        const { stdout } = await execFileAsync(
          'ssh', [origin.sshHost, `base64 ${shellEscape(filePath)}`],
          { maxBuffer, timeout }
        );
        data = Buffer.from(stdout.replace(/\s/g, ''), 'base64');
      }

      res.writeHead(200, {
        'Content-Type': mimeType,
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'public, max-age=3600',
        'Content-Length': data.length,
      });
      res.end(data);
    } catch (error: any) {
      console.error('Failed to serve raw binary:', error.message, error.stderr || '');
      res.writeHead(error.code === 'ENOENT' ? 404 : 500, {
        'Content-Type': 'text/plain',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(error.message || 'Not found');
    }
  }

  /**
   * Save file content endpoint
   * POST /save-file
   * Body: { path: string, content: string, originId?: string }
   */
  private async handleSaveFile(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const data = await this.parseJsonBody<{ path?: string; content?: string; originId?: string }>(req, res);
    if (!data) return;

    const { path: filePath, content, originId } = data;

    if (!filePath || content === undefined) {
      this.sendJsonError(res, 400, 'Missing path or content');
      return;
    }

    // Security: prevent directory traversal
    if (filePath.includes('..')) {
      this.sendJsonError(res, 400, 'Invalid path');
      return;
    }

    try {
      if (!originId || originId === 'local') {
        await writeFile(filePath, content, 'utf-8');
      } else {
        const origin = this.originLookup.getOrigin(originId);
        if (!origin?.sshHost) {
          this.sendJsonError(res, 404, 'Origin not found or not connected');
          return;
        }

        await this.writeRemoteFile(origin.sshHost, filePath, content);
      }

      this.sendJsonSuccess(res, { success: true, path: filePath });
    } catch (error: any) {
      console.error('Failed to save file:', error.message);
      this.sendJsonError(res, 500, 'Failed to save file: ' + error.message);
    }
  }

  /**
   * Write file to remote host via SSH stdin pipe
   */
  private writeRemoteFile(sshHost: string, filePath: string, content: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const ssh = spawn('ssh', [sshHost, `cat > ${shellEscape(filePath)}`], {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stderr = '';
      ssh.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      ssh.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(stderr || `SSH exited with code ${code}`));
        }
      });

      ssh.on('error', reject);

      // Write content to stdin and close
      ssh.stdin.write(content);
      ssh.stdin.end();
    });
  }

  /**
   * Map file extension to Prism.js language identifier
   */
  private extToLanguage(ext: string): string {
    const mapping: Record<string, string> = {
      'ts': 'typescript',
      'tsx': 'tsx',
      'js': 'javascript',
      'jsx': 'jsx',
      'json': 'json',
      'md': 'markdown',
      'py': 'python',
      'rs': 'rust',
      'go': 'go',
      'sh': 'bash',
      'bash': 'bash',
      'zsh': 'bash',
      'css': 'css',
      'scss': 'scss',
      'html': 'html',
      'xml': 'xml',
      'yaml': 'yaml',
      'yml': 'yaml',
      'toml': 'toml',
      'sql': 'sql',
      'c': 'c',
      'cpp': 'cpp',
      'h': 'c',
      'hpp': 'cpp',
      'java': 'java',
      'rb': 'ruby',
      'php': 'php',
      'swift': 'swift',
      'kt': 'kotlin',
      'lua': 'lua',
      'vim': 'vim',
      'dockerfile': 'docker',
      'makefile': 'makefile',
      'mk': 'makefile',
    };
    return mapping[ext] || 'plaintext';
  }

  /**
   * Activate dormant remote city endpoint
   * POST /activate-city?cityId=xxx
   */
  private async handleActivateCity(url: URL, res: ServerResponse): Promise<void> {
    console.log(`[Activate] Received request for cityId=${url.searchParams.get('cityId')}`);
    const cityId = url.searchParams.get('cityId');
    if (!cityId) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing cityId parameter' }));
      return;
    }

    const city = this.cityLookup.getCityById(cityId);
    if (!city) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'City not found' }));
      return;
    }

    // Only activate remote cities
    if (city.originId === 'local') {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Cannot activate local city - start a session manually' }));
      return;
    }

    const sshHost = this.getSshHost(city);

    try {
      // Check if agent is already running on this host
      // Use -T to disable TTY allocation (avoids "Pseudo-terminal will not be allocated" warnings)
      const { stdout: checkOutput } = await execFileAsync(
        'ssh', ['-T', sshHost, 'tmux has-session -t portolan-agent 2>/dev/null && echo running || echo stopped'],
        { timeout: 10000 }
      );

      if (checkOutput.trim() === 'running') {
        res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
        res.end(JSON.stringify({ status: 'already_running', message: 'Agent already running on ' + sshHost }));
        return;
      }

      // Start the agent via SSH
      // Use -T to disable TTY allocation, bash -l to get login shell with nvm/node in PATH
      console.log(`[Activate] Starting portolan-agent on ${sshHost}...`);
      await execFileAsync(
        'ssh', ['-T', sshHost, `tmux new-session -d -s portolan-agent "bash -l -c \\"node ~/bin/portolan-agent.js connect --ssh-host=${sshHost}\\""`],
        { timeout: 30000 }
      );

      res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      res.end(JSON.stringify({ status: 'started', message: 'Agent started on ' + sshHost }));
    } catch (error: any) {
      console.error(`[Activate] Failed to start agent on ${sshHost}:`, error.message);
      res.writeHead(500, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      res.end(JSON.stringify({ error: 'Failed to start agent: ' + error.message }));
    }
  }

  // ============================================================================
  // Annotation Endpoints
  // ============================================================================

  /**
   * Get recently annotated files
   * GET /recent-annotations?originId=optional&limit=10
   */
  private async handleRecentAnnotations(url: URL, res: ServerResponse): Promise<void> {
    if (!this.annotationPersistence) {
      this.sendJsonError(res, 500, 'Annotation persistence not initialized');
      return;
    }

    const originId = url.searchParams.get('originId') || undefined;
    const limit = parseInt(url.searchParams.get('limit') || '10', 10);

    const recentFiles = this.annotationPersistence.getRecentFiles(originId, limit);
    this.sendJsonSuccess(res, { files: recentFiles });
  }

  /**
   * Get annotations for a file
   * GET /annotations?path=/path/to/file&originId=optional
   */
  private async handleGetAnnotations(url: URL, res: ServerResponse): Promise<void> {
    if (!this.annotationPersistence) {
      this.sendJsonError(res, 500, 'Annotation persistence not initialized');
      return;
    }

    const filePath = url.searchParams.get('path');
    const claimId = url.searchParams.get('claimId');
    const allClaims = url.searchParams.get('claims') === 'true';
    const originId = url.searchParams.get('originId') || 'local';

    if (!filePath && !claimId && !allClaims) {
      this.sendJsonError(res, 400, 'Missing path, claimId, or claims parameter');
      return;
    }

    let annotations: Annotation[];
    if (allClaims) {
      annotations = this.annotationPersistence.getAllClaims();
    } else if (claimId) {
      annotations = this.annotationPersistence.getByClaimId(claimId);
    } else {
      annotations = this.annotationPersistence.getByFile(filePath!, originId);
    }

    this.sendJsonSuccess(res, { annotations });
  }

  /**
   * Create a new annotation
   * POST /annotations
   */
  private async handleCreateAnnotation(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (!this.annotationPersistence) {
      this.sendJsonError(res, 500, 'Annotation persistence not initialized');
      return;
    }

    const data = await this.parseJsonBody<Omit<Annotation, 'id' | 'createdAt'>>(req, res);
    if (!data) return;

    if (data.isClaimAnnotation) {
      if (!data.claimId || !data.comment) {
        this.sendJsonError(res, 400, 'Missing required fields for claims annotation (claimId, comment)');
        return;
      }
      if (data.artifact && (data.x === undefined || data.y === undefined)) {
        this.sendJsonError(res, 400, 'Image annotation requires x and y coordinates');
        return;
      }
    } else if (!data.filePath || !data.comment) {
      this.sendJsonError(res, 400, 'Missing required fields');
      return;
    }

    try {
      const annotation = this.annotationPersistence.add(data);

      res.writeHead(201, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(JSON.stringify({ annotation }));
    } catch (error: any) {
      this.sendJsonError(res, 500, error.message);
    }
  }

  /**
   * Update an annotation
   * PUT /annotations/:id
   */
  private async handleUpdateAnnotation(id: string, req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (!this.annotationPersistence) {
      this.sendJsonError(res, 500, 'Annotation persistence not initialized');
      return;
    }

    const updates = await this.parseJsonBody<Partial<Annotation>>(req, res);
    if (!updates) return;

    const annotation = this.annotationPersistence.update(id, updates);
    if (!annotation) {
      this.sendJsonError(res, 404, 'Annotation not found');
      return;
    }

    this.sendJsonSuccess(res, { annotation });
  }

  /**
   * Delete an annotation
   * DELETE /annotations/:id
   */
  private async handleDeleteAnnotation(id: string, res: ServerResponse): Promise<void> {
    if (!this.annotationPersistence) {
      this.sendJsonError(res, 500, 'Annotation persistence not initialized');
      return;
    }

    const annotation = this.annotationPersistence.delete(id);
    if (!annotation) {
      this.sendJsonError(res, 404, 'Annotation not found');
      return;
    }

    this.sendJsonSuccess(res, { success: true });
  }

  // Callback for creating new workers
  private onCreateNewWorker: ((cityPath: string, originId: string) => Promise<string>) | null = null;

  // Callback for focusing a session in Kitty
  private onFocusSession: ((sessionId: string) => void) | null = null;

  /**
   * Set callback for creating new workers
   * Returns the tmux session name of the created worker
   */
  setOnCreateNewWorker(fn: (cityPath: string, originId: string) => Promise<string>): void {
    this.onCreateNewWorker = fn;
  }

  /**
   * Set callback for focusing a session in Kitty
   */
  setOnFocusSession(fn: (sessionId: string) => void): void {
    this.onFocusSession = fn;
  }

  /**
   * Send annotations to a worker via tmux send-keys
   * POST /send-annotations
   * Supports either workerId (existing worker) or createNewWorker (new worker)
   */
  private async handleSendAnnotations(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (!this.sessionLookup) {
      this.sendJsonError(res, 500, 'Session lookup not initialized');
      return;
    }

    const data = await this.parseJsonBody<{
      workerId?: string;
      createNewWorker?: boolean;
      filePath: string;
      originId: string;
      annotations: Annotation[];
      globalComment?: string;
      cityName?: string;
      isClaimsSend?: boolean;
    }>(req, res);
    if (!data) return;

    const { workerId, createNewWorker, filePath, originId, annotations, globalComment, cityName, isClaimsSend } = data;

    const hasContent = (annotations && annotations.length > 0) || (globalComment && globalComment.trim().length > 0);
    if (!hasContent) {
      this.sendJsonError(res, 400, 'No content to send');
      return;
    }

    if (!workerId && !createNewWorker) {
      this.sendJsonError(res, 400, 'Must specify workerId or createNewWorker');
      return;
    }

    let tmuxSession: string;
    const isRemote = originId !== 'local' && !!originId;
    let sshHost: string | undefined;

    if (createNewWorker) {
      if (!this.onCreateNewWorker) {
        this.sendJsonError(res, 500, 'New worker creation not configured');
        return;
      }

      try {
        const cityPath = filePath.substring(0, filePath.lastIndexOf('/'));
        tmuxSession = await this.onCreateNewWorker(cityPath, originId);
        console.log(`[SendAnnotations] Created new worker: ${tmuxSession}`);

        // Wait for Claude to start up (4s for remote systems)
        await new Promise(resolve => setTimeout(resolve, 4000));
      } catch (error: any) {
        this.sendJsonError(res, 500, 'Failed to create worker: ' + error.message);
        return;
      }
    } else {
      const session = this.sessionLookup.findSession(workerId!);
      if (!session) {
        this.sendJsonError(res, 404, 'Worker not found');
        return;
      }
      tmuxSession = session.tmuxSession;
    }

    if (isRemote) {
      const origin = this.originLookup.getOrigin(originId);
      sshHost = origin?.sshHost;
    }

    const formattedMessage = isClaimsSend
      ? this.formatClaimsAnnotationsForClaude(cityName || 'unknown', annotations, globalComment)
      : this.formatAnnotationsForClaude(filePath, annotations, globalComment);

    try {
      const escaped = shellEscape(tmuxSession);

      if (!isRemote) {
        execSync(`tmux load-buffer -`, { input: formattedMessage, timeout: 5000 });
        execSync(`tmux paste-buffer -t ${escaped}`, { timeout: 5000 });
      } else {
        if (!sshHost) {
          this.sendJsonError(res, 404, 'Origin not found');
          return;
        }

        // execFileSync avoids local shell — command strings interpreted only by remote shell
        execFileSync('ssh', [sshHost, 'tmux load-buffer -'], { input: formattedMessage, timeout: 10000 });
        execFileSync('ssh', [sshHost, `tmux paste-buffer -t ${escaped}`], { timeout: 10000 });
      }

      if (workerId && this.onFocusSession) {
        this.onFocusSession(workerId);
      }

      this.sendJsonSuccess(res, { success: true });
    } catch (error: any) {
      console.error('Failed to send annotations:', error.message);
      this.sendJsonError(res, 500, 'Failed to send annotations: ' + error.message);
    }
  }

  /**
   * POST /send-message
   * Send a chat message to a worker's tmux session
   */
  private async handleSendMessage(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (!this.sessionLookup) {
      this.sendJsonError(res, 500, 'Session lookup not initialized');
      return;
    }

    const data = await this.parseJsonBody<{ sessionId: string; message: string }>(req, res);
    if (!data) return;

    const { sessionId, message } = data;

    if (!sessionId || !message?.trim()) {
      this.sendJsonError(res, 400, 'sessionId and message are required');
      return;
    }

    const session = this.sessionLookup.findSession(sessionId);
    if (!session) {
      this.sendJsonError(res, 404, 'Session not found');
      return;
    }

    const tmuxSession = session.tmuxSession;
    const isRemote = session.originId !== 'local';
    let sshHost: string | undefined;

    if (isRemote) {
      const origin = this.originLookup.getOrigin(session.originId);
      sshHost = origin?.sshHost;
      if (!sshHost) {
        this.sendJsonError(res, 404, 'Origin not found for remote session');
        return;
      }
    }

    try {
      const escaped = shellEscape(tmuxSession);

      if (!isRemote) {
        execSync(`tmux load-buffer -`, { input: message, timeout: 5000 });
        execSync(`tmux paste-buffer -t ${escaped}`, { timeout: 5000 });
        execSync(`tmux send-keys -t ${escaped} Enter`, { timeout: 5000 });
      } else {
        execFileSync('ssh', [sshHost!, 'tmux load-buffer -'], { input: message, timeout: 10000 });
        execFileSync('ssh', [sshHost!, `tmux paste-buffer -t ${escaped}`], { timeout: 10000 });
        execFileSync('ssh', [sshHost!, `tmux send-keys -t ${escaped} Enter`], { timeout: 10000 });
      }

      this.sendJsonSuccess(res, { success: true });
    } catch (error: any) {
      console.error('Failed to send message:', error.message);
      this.sendJsonError(res, 500, 'Failed to send message: ' + error.message);
    }
  }

  /**
   * Format annotations as markdown for Claude
   * Includes full file path and optional global comment
   */
  private formatAnnotationsForClaude(filePath: string, annotations: Annotation[], globalComment?: string): string {
    const lines = [
      '',  // Start with newline for clean separation
      `# Feedback on ${filePath}`,
      '',
    ];

    if (globalComment) {
      lines.push(globalComment);
      lines.push('');
    }

    if (annotations && annotations.length > 0) {
      lines.push(`I've reviewed this file and have ${annotations.length} piece${annotations.length === 1 ? '' : 's'} of feedback:`);
      lines.push('');

      annotations.forEach((ann, i) => {
        if (ann.isImageAnnotation) {
          // Image annotation - show position
          const posRef = ann.x !== undefined && ann.y !== undefined
            ? ` at position (${ann.x.toFixed(0)}%, ${ann.y.toFixed(0)}%)`
            : '';
          lines.push(`## ${i + 1}. Image annotation${posRef}`);
          lines.push(`> ${ann.comment}`);
          lines.push('');
        } else {
          // Text annotation - show selected text with start...end format for multiline
          let contextText: string;
          const text = ann.originalText || '';
          const isMultiline = text.includes('\n');

          if (isMultiline) {
            // For multiline: show "start text...end text"
            const lines_arr = text.split('\n');
            const startText = lines_arr[0].slice(0, 30).trim();
            const endText = lines_arr[lines_arr.length - 1].slice(-30).trim();
            contextText = `${startText}...${endText}`;
          } else if (text.length > 60) {
            // Single line but long: truncate
            contextText = text.slice(0, 57) + '...';
          } else {
            contextText = text;
          }

          // Format line reference: show range if multiline
          let lineRef = '';
          if (ann.line) {
            if (ann.endLine && ann.endLine !== ann.line) {
              lineRef = ` (L${ann.line}-${ann.endLine})`;
            } else {
              lineRef = ` (L${ann.line})`;
            }
          }

          lines.push(`## ${i + 1}.${lineRef} Feedback on: "${contextText}"`);
          lines.push(`> ${ann.comment}`);
          lines.push('');
        }
      });
    }

    lines.push('---');
    return lines.join('\n');
  }

  /**
   * Format claims annotations as markdown for Claude
   * Groups by claim title, shows text selections and image pins
   */
  private formatClaimsAnnotationsForClaude(cityName: string, annotations: Annotation[], globalComment?: string): string {
    const lines = [
      '',
      `# Claims review: ${cityName}`,
      '',
    ];

    if (globalComment) {
      lines.push(globalComment);
      lines.push('');
    }

    if (annotations && annotations.length > 0) {
      lines.push(`I've reviewed the claims dashboard and have ${annotations.length} piece${annotations.length === 1 ? '' : 's'} of feedback:`);
      lines.push('');

      // Group annotations by claim (preserves insertion order)
      const grouped = new Map<string, Annotation[]>();
      for (const ann of annotations) {
        const key = ann.claimId || 'unknown';
        const group = grouped.get(key);
        if (group) group.push(ann);
        else grouped.set(key, [ann]);
      }

      let claimNum = 0;
      for (const [, group] of grouped) {
        claimNum++;
        const title = group[0].claimTitle || group[0].claimId || 'Unknown claim';
        const filePath = group[0].filePath;
        const header = filePath ? filePath : `[${title}]`;
        lines.push(`## ${claimNum}. ${header}`);

        for (let i = 0; i < group.length; i++) {
          const ann = group[i];
          if (i > 0) lines.push('>');

          // Line reference
          let lineRef = '';
          if (ann.line) {
            if (ann.endLine && ann.endLine !== ann.line) {
              lineRef = ` (L${ann.line}-${ann.endLine})`;
            } else {
              lineRef = ` (L${ann.line})`;
            }
          }

          if (ann.artifact) {
            const posRef = ann.x !== undefined && ann.y !== undefined
              ? ` (at ${ann.x.toFixed(0)}%, ${ann.y.toFixed(0)}%)`
              : '';
            lines.push(`> On plot: ${ann.artifact}${posRef}`);
          } else if (ann.selectedText) {
            const truncated = ann.selectedText.length > 60
              ? ann.selectedText.slice(0, 60) + '…'
              : ann.selectedText;
            lines.push(`>${lineRef} On text: "${truncated}"`);
          }
          lines.push(`> ${ann.comment}`);
        }
        lines.push('');
      }
    }

    lines.push('---');
    return lines.join('\n');
  }

  // ============================================================================
  // File as Fiber Endpoint
  // ============================================================================

  /**
   * File annotations as a felt fiber
   * POST /file-as-fiber
   * Body: { filePath, originId, title, body, kind }
   */
  private async handleFileAsFiber(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const data = await this.parseJsonBody<{
      filePath: string;
      originId: string;
      cityPath?: string;
      title: string;
      body: string;
      kind?: string;
    }>(req, res);
    if (!data) return;

    const { filePath, originId, title, body, kind = 'task' } = data;

    if (!filePath || !title || !body) {
      this.sendJsonError(res, 400, 'Missing required fields');
      return;
    }

    const cityPath = data.cityPath || filePath.substring(0, filePath.lastIndexOf('/'));
    const isRemote = originId !== 'local' && !!originId;

    try {
      let fiberId: string;

      const feltCmd = `cd ${shellEscape(cityPath)} && felt add ${shellEscape(title)} -t ${shellEscape(kind)} -b ${shellEscape(body)}`;

      if (!isRemote) {
        const { stdout } = await execAsync(feltCmd, { timeout: 10000, maxBuffer: 1024 * 1024 });
        fiberId = stdout.trim();
      } else {
        const origin = this.originLookup.getOrigin(originId);
        if (!origin?.sshHost) {
          this.sendJsonError(res, 404, 'Origin not found or not connected');
          return;
        }

        // execFileAsync avoids local shell — feltCmd is interpreted only by the remote shell
        const { stdout } = await execFileAsync(
          'ssh', [origin.sshHost, feltCmd],
          { timeout: 30000, maxBuffer: 1024 * 1024 }
        );
        fiberId = stdout.trim();
      }

      this.sendJsonSuccess(res, { success: true, fiberId });
    } catch (error: any) {
      console.error('Failed to file as fiber:', error.message);
      this.sendJsonError(res, 500, 'Failed to file as fiber: ' + error.message);
    }
  }

  // ============================================================================
  // Promote to Felt
  // ============================================================================

  /**
   * Promote a claims annotation to a felt comment on the claim's fiber.
   * POST /promote-to-felt
   * Body: { claimId, comment, cityId }
   */
  private async handlePromoteToFelt(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const data = await this.parseJsonBody<{ claimId: string; comment: string; cityId: string }>(req, res);
    if (!data) return;

    const { claimId, comment, cityId } = data;

    if (!claimId || !comment || !cityId) {
      this.sendJsonError(res, 400, 'Missing required fields (claimId, comment, cityId)');
      return;
    }

    const city = this.cityLookup.getCityById(cityId);
    if (!city) {
      this.sendJsonError(res, 404, 'City not found');
      return;
    }

    const isRemote = city.originId !== 'local' && !!city.originId;
    const cityPath = city.path;

    try {
      const feltCmd = `cd ${shellEscape(cityPath)} && felt comment ${shellEscape(claimId)} ${shellEscape(comment)}`;
      if (!isRemote) {
        await execAsync(feltCmd, { timeout: 10000 });
      } else {
        const sshHost = this.getSshHost(city);
        // execFileAsync avoids local shell — feltCmd is interpreted only by the remote shell
        await execFileAsync('ssh', [sshHost, feltCmd], { timeout: 30000 });
      }

      this.sendJsonSuccess(res, { success: true });
    } catch (error: any) {
      console.error('Failed to promote to felt:', error.message);
      this.sendJsonError(res, 500, 'Failed to promote to felt: ' + error.message);
    }
  }

  // ============================================================================
  // Playground Endpoints
  // ============================================================================

  /**
   * List available playgrounds for a city
   * GET /playground-list?cityId=xxx
   */
  private async handlePlaygroundList(url: URL, res: ServerResponse): Promise<void> {
    const cityId = url.searchParams.get('cityId');
    if (!cityId) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing cityId parameter' }));
      return;
    }

    const city = this.cityLookup.getCityById(cityId);
    if (!city) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'City not found' }));
      return;
    }

    const playgroundsDir = `${city.path}/.portolan/playgrounds`;

    try {
      let files: string[];
      if (city.originId === 'local') {
        const { readdirSync } = await import('fs');
        files = readdirSync(playgroundsDir).filter(f => f.endsWith('.html'));
      } else {
        const sshHost = this.getSshHost(city);
        const { stdout } = await execFileAsync(
          'ssh', [sshHost, `ls ${shellEscape(playgroundsDir)}/*.html 2>/dev/null || true`],
          { timeout: 10000 }
        );
        files = stdout.trim().split('\n')
          .filter(Boolean)
          .map(f => f.split('/').pop()!)
          .filter(f => f.endsWith('.html'));
      }

      // Sort by name, most recently modified first would be nice but simpler to just sort alphabetically
      files.sort();

      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(JSON.stringify({ playgrounds: files }));
    } catch (error: any) {
      console.error('Failed to list playgrounds:', error.message);
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(JSON.stringify({ playgrounds: [] }));
    }
  }

  /**
   * Serve a playground HTML file
   * GET /playground?cityId=xxx&name=playground.html
   */
  private async handlePlayground(url: URL, res: ServerResponse): Promise<void> {
    const cityId = url.searchParams.get('cityId');
    const name = url.searchParams.get('name');

    if (!cityId || !name) {
      res.writeHead(400, { 'Content-Type': 'text/plain' });
      res.end('Missing cityId or name parameter');
      return;
    }

    // Security: prevent directory traversal
    if (name.includes('/') || name.includes('..')) {
      res.writeHead(400, { 'Content-Type': 'text/plain' });
      res.end('Invalid playground name');
      return;
    }

    const city = this.cityLookup.getCityById(cityId);
    if (!city) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('City not found');
      return;
    }

    const playgroundPath = `${city.path}/.portolan/playgrounds/${name}`;

    try {
      let html: string;
      if (city.originId === 'local') {
        html = await readFile(playgroundPath, 'utf-8');
      } else {
        const sshHost = this.getSshHost(city);
        const { stdout } = await execFileAsync(
          'ssh', [sshHost, `cat ${shellEscape(playgroundPath)}`],
          { maxBuffer: 10 * 1024 * 1024, timeout: 30000 }
        );
        html = stdout;
      }

      res.writeHead(200, {
        'Content-Type': 'text/html',
        'Access-Control-Allow-Origin': '*',
      });
      res.end(html);
    } catch (error: any) {
      console.error('Failed to fetch playground:', error.message);
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Playground not found');
    }
  }

  /**
   * Get conversation history for a session
   * Query params: sessionId, tmuxSession, limit
   */
  private async handleConversation(url: URL, res: ServerResponse): Promise<void> {
    const sessionId = url.searchParams.get('sessionId') ?? undefined;
    const tmuxSession = url.searchParams.get('tmuxSession') ?? undefined;
    const limit = parseInt(url.searchParams.get('limit') || '50', 10);

    if (!sessionId && !tmuxSession) {
      this.sendJsonError(res, 400, 'Missing sessionId or tmuxSession parameter');
      return;
    }

    try {
      const messages = await this.resolveConversationMessages(sessionId, limit, tmuxSession);
      this.sendJsonSuccess(res, { messages });
    } catch (error: any) {
      console.error('Failed to fetch conversation:', error.message);
      this.sendJsonError(res, 500, 'Failed to fetch conversation');
    }
  }

  /**
   * Resolve conversation messages using multiple lookup strategies
   *
   * Priority: sessionId > remote lookup > TranscriptReader
   * Session-specific lookup is strongly preferred. Tmux aggregation removed —
   * it caused cross-contamination between sessions sharing a tmux name after
   * disconnects/reconnects.
   */
  private async resolveConversationMessages(sessionId: string | undefined, limit: number, tmuxSessionParam?: string): Promise<any[]> {
    const session = sessionId ? this.sessionLookup?.findSession(sessionId) : undefined;
    const isRemote = session?.originId && session.originId !== 'local';

    // 1. ConversationCache by sessionId (primary path)
    if (this.conversationCache && sessionId) {
      const messages = this.conversationCache.getMessages(sessionId, limit);
      if (messages.length > 0) return messages;
    }

    // 2. ConversationCache by prefixed tmux session (remote hooks path)
    // Remote hooks store messages under Claude's own sessionId (a UUID),
    // but portolan session IDs are constructed differently (e.g. "remote-host-tmuxName").
    // Fall back to tmux-based lookup using the originId/tmuxSession key.
    if (this.conversationCache && isRemote && session) {
      const prefixedTmux = `${session.originId}/${session.tmuxSession}`;
      const messages = this.conversationCache.getMessagesByTmux(prefixedTmux, limit);
      if (messages.length > 0) return messages;
    }

    // Remaining lookups require sessionId
    if (!sessionId) return [];

    // 3. Remote conversation lookup (legacy agent-proxied path)
    if (isRemote) {
      const cached = this.remoteConversationLookup?.(sessionId);
      if (cached && cached.length > 0) return cached.slice(-limit);
    }

    // 3. TranscriptReader (legacy fallback for local sessions)
    if (this.transcriptReader && session && !isRemote) {
      await this.updateTranscriptMapping(sessionId, session);
      return this.transcriptReader.getRecentMessages(session.cwd, limit, sessionId);
    }

    return [];
  }

  /**
   * Update transcript mapping if a new transcript is detected
   */
  private async updateTranscriptMapping(sessionId: string, session: Session): Promise<void> {
    if (!this.transcriptReader || !session.tmuxSession) return;

    const currentMapping = this.transcriptReader.getSessionTranscript(sessionId);
    const detected = await this.transcriptReader.detectTranscriptFromTmux(session.tmuxSession, session.cwd);

    if (detected && detected !== currentMapping) {
      this.transcriptReader.setSessionTranscript(sessionId, detected);
    }
  }

  private async handleDebugTranscripts(res: ServerResponse): Promise<void> {
    const mappings = this.transcriptReader?.getAllSessionMappings() || new Map();
    const sessions = this.sessionLookup?.getAllSessions() || [];

    const debug = {
      mappings: Object.fromEntries(
        [...mappings.entries()].map(([id, path]) => [id, path.split('/').pop()])
      ),
      sessions: sessions.map(s => ({
        id: s.id,
        name: s.name,
        tmuxSession: s.tmuxSession,
        cwd: s.cwd,
        status: s.status,
        mappedTranscript: mappings.get(s.id)?.split('/').pop() || null
      }))
    };

    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(JSON.stringify(debug, null, 2));
  }

  // ============================================================================
  // Hook Endpoints for Conversation Capture
  // ============================================================================

  /**
   * POST /hook/message
   * Receive conversation messages from Claude Code hooks
   * Body: { sessionId, tmuxSession, cwd, messages: [{ type, content, timestamp }] }
   */
  private async handleHookMessage(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (!this.conversationCache) {
      this.sendJsonError(res, 500, 'Conversation cache not configured');
      return;
    }

    const parseResult = await this.parseJsonBody<{
      sessionId: string;
      tmuxSession: string;
      cwd: string;
      messages: CachedMessage[];
    }>(req, res);

    if (!parseResult) return;

    const { sessionId, tmuxSession, cwd, messages } = parseResult;

    if (!sessionId || !tmuxSession || !messages || !Array.isArray(messages)) {
      this.sendJsonError(res, 400, 'Missing required fields: sessionId, tmuxSession, messages');
      return;
    }

    try {
      // Auto-detect remote origin for hooks arriving via SSH tunnel.
      // Remote hooks POST directly to :4004 with unprefixed tmuxSession,
      // but ConversationCache keys need the originId/ prefix for lookup.
      let effectiveTmux = tmuxSession;
      if (this.sessionLookup) {
        const remoteSession = this.sessionLookup.getAllSessions().find(
          s => s.tmuxSession === tmuxSession && s.originId !== 'local'
        );
        if (remoteSession) {
          effectiveTmux = `${remoteSession.originId}/${tmuxSession}`;
        }
      }

      this.conversationCache.addMessages(sessionId, effectiveTmux, cwd || '', messages);
      this.sendJsonSuccess(res, { success: true, count: messages.length });
    } catch (error: any) {
      console.error('[Hook] Failed to add messages:', error.message);
      this.sendJsonError(res, 500, 'Failed to add messages');
    }
  }

  /**
   * Parse JSON body from request, sending error response if invalid
   */
  private async parseJsonBody<T>(req: IncomingMessage, res: ServerResponse): Promise<T | null> {
    let body = '';
    for await (const chunk of req) {
      body += chunk;
    }

    try {
      return JSON.parse(body) as T;
    } catch {
      this.sendJsonError(res, 400, 'Invalid JSON body');
      return null;
    }
  }

  /**
   * Send JSON error response
   */
  private sendJsonError(res: ServerResponse, status: number, error: string): void {
    res.writeHead(status, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(JSON.stringify({ error }));
  }

  /**
   * Send JSON success response
   */
  private sendJsonSuccess(res: ServerResponse, data: Record<string, unknown>): void {
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(JSON.stringify(data));
  }

  /**
   * GET /hook/health
   * Debug endpoint showing last event time per session
   */
  private async handleHookHealth(res: ServerResponse): Promise<void> {
    if (!this.conversationCache) {
      this.sendJsonError(res, 500, 'Conversation cache not configured');
      return;
    }

    const health = this.conversationCache.getHealthInfo();
    this.sendJsonSuccess(res, health as Record<string, unknown>);
  }

  // ============================================================================
  // Card State Persistence
  // ============================================================================

  /**
   * GET /card-states - Get all saved card states
   */
  private async handleGetCardStates(res: ServerResponse): Promise<void> {
    if (!this.cardStatePersistence) {
      this.sendJsonError(res, 500, 'Card state persistence not configured');
      return;
    }

    const states = this.cardStatePersistence.getAll();
    this.sendJsonSuccess(res, { states });
  }

  /**
   * GET /card-state/:workerId - Get card state for a worker
   */
  private async handleGetCardState(workerId: string, res: ServerResponse): Promise<void> {
    if (!this.cardStatePersistence) {
      this.sendJsonError(res, 500, 'Card state persistence not configured');
      return;
    }

    const state = this.cardStatePersistence.get(workerId);
    if (!state) {
      this.sendJsonError(res, 404, 'Card state not found');
      return;
    }

    this.sendJsonSuccess(res, { state });
  }

  /**
   * PUT /card-state/:workerId - Save card state for a worker
   */
  private async handleSaveCardState(
    workerId: string,
    req: IncomingMessage,
    res: ServerResponse
  ): Promise<void> {
    if (!this.cardStatePersistence) {
      this.sendJsonError(res, 500, 'Card state persistence not configured');
      return;
    }

    const body = await this.parseJsonBody<{
      size?: { width: number; height: number };
      swarmOffset?: { x: number; z: number };
    }>(req, res);

    if (!body) return;

    const state = this.cardStatePersistence.set(workerId, {
      size: body.size,
      swarmOffset: body.swarmOffset,
    });
    this.sendJsonSuccess(res, { state });
  }

  /**
   * DELETE /card-state/:workerId - Delete card state for a worker
   */
  private async handleDeleteCardState(workerId: string, res: ServerResponse): Promise<void> {
    if (!this.cardStatePersistence) {
      this.sendJsonError(res, 500, 'Card state persistence not configured');
      return;
    }

    const deleted = this.cardStatePersistence.delete(workerId);
    this.sendJsonSuccess(res, { deleted });
  }
}
