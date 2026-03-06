/**
 * AnnotationPersistence - Persist file annotations across sessions
 *
 * Annotations are stored in ~/.portolan/annotations.json
 * Each annotation captures a text selection with context for re-anchoring.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync } from 'fs';
import { homedir } from 'os';
import { join, resolve } from 'path';
import { randomUUID } from 'crypto';

// ============================================================================
// Types
// ============================================================================

export interface Annotation {
  id: string;
  originId: string;        // 'local' or 'remote-{hostname}'
  comment: string;
  createdAt: number;

  // File anchoring (required for file annotations, absent for claims)
  filePath?: string;       // Full file path
  from?: number;           // char offset at creation
  to?: number;
  line?: number;           // line number at 'from' (1-indexed)
  endLine?: number;        // line number at 'to' (1-indexed)
  originalText?: string;   // the selected text
  contextBefore?: string;  // ~20 chars for re-anchoring
  contextAfter?: string;   // ~20 chars for re-anchoring

  // Image annotation fields
  x?: number;              // percentage 0-100
  y?: number;              // percentage 0-100
  isImageAnnotation?: boolean;

  // Claims annotation fields (mutually exclusive with file anchoring)
  claimId?: string;        // claim identifier from dashboard
  claimTitle?: string;     // human-readable claim title
  selectedText?: string;   // text user highlighted in rendered claim
  artifact?: string;       // plot filename if annotating an image
  isClaimAnnotation?: boolean;
}

interface PersistenceFile {
  version: 1;
  annotations: Annotation[];
  // History: track files that were annotated (even if annotations later deleted)
  annotationHistory?: Array<{
    filePath: string;
    originId: string;
    lastAnnotatedAt: number;
  }>;
}

// ============================================================================
// AnnotationPersistence
// ============================================================================

export class AnnotationPersistence {
  private readonly dataDir: string;
  private readonly filePath: string;
  private annotations: Map<string, Annotation> = new Map(); // key = annotation.id
  // Track files that were annotated (persists even if annotations deleted)
  private annotationHistory: Map<string, { filePath: string; originId: string; lastAnnotatedAt: number }> = new Map();

  constructor() {
    this.dataDir = join(homedir(), '.portolan');
    this.filePath = join(this.dataDir, 'annotations.json');
  }

  /**
   * Make file key from originId and path
   */
  private makeFileKey(originId: string, filePath: string): string {
    return `${originId}:${resolve(filePath)}`;
  }

  /**
   * Load annotations from disk
   */
  load(): Annotation[] {
    this.annotations.clear();
    this.annotationHistory.clear();

    if (!existsSync(this.filePath)) {
      return [];
    }

    try {
      const content = readFileSync(this.filePath, 'utf-8');
      const data: PersistenceFile = JSON.parse(content);

      if (data.version !== 1) {
        console.warn(`Unknown annotations.json version: ${data.version}`);
        return [];
      }

      for (const annotation of data.annotations) {
        this.annotations.set(annotation.id, annotation);
      }

      // Load annotation history
      if (data.annotationHistory) {
        for (const entry of data.annotationHistory) {
          const key = this.makeFileKey(entry.originId, entry.filePath);
          this.annotationHistory.set(key, entry);
        }
      }

      console.log(`Loaded ${this.annotations.size} annotations, ${this.annotationHistory.size} history entries`);
      return this.getAll();
    } catch (error) {
      console.error('Failed to load annotations:', error);
      return [];
    }
  }

  /**
   * Save annotations to disk (atomic write)
   */
  private save(): void {
    // Ensure directory exists
    if (!existsSync(this.dataDir)) {
      mkdirSync(this.dataDir, { recursive: true });
    }

    const data: PersistenceFile = {
      version: 1,
      annotations: this.getAll(),
      annotationHistory: [...this.annotationHistory.values()],
    };

    const tmpPath = this.filePath + '.tmp';

    try {
      writeFileSync(tmpPath, JSON.stringify(data, null, 2), 'utf-8');
      renameSync(tmpPath, this.filePath);
    } catch (error) {
      console.error('Failed to save annotations:', error);
      throw error;
    }
  }

  /**
   * Get all annotations
   */
  getAll(): Annotation[] {
    return [...this.annotations.values()];
  }

  /**
   * Get annotations for a specific file
   */
  getByFile(filePath: string, originId: string = 'local'): Annotation[] {
    const fileKey = this.makeFileKey(originId, filePath);
    return this.getAll().filter(
      (a) => a.filePath && this.makeFileKey(a.originId, a.filePath) === fileKey
    );
  }

  /**
   * Get annotations for a specific claim
   */
  getByClaimId(claimId: string): Annotation[] {
    return this.getAll().filter(a => a.isClaimAnnotation && a.claimId === claimId);
  }

  /**
   * Get all claims annotations
   */
  getAllClaims(): Annotation[] {
    return this.getAll().filter(a => a.isClaimAnnotation);
  }

  /**
   * Add a new annotation
   */
  add(
    annotation: Omit<Annotation, 'id' | 'createdAt'>
  ): Annotation {
    const newAnnotation: Annotation = {
      ...annotation,
      id: randomUUID(),
      createdAt: Date.now(),
    };

    // File annotations: resolve path and update history
    const isFileAnnotation = !newAnnotation.isClaimAnnotation && !!newAnnotation.filePath;
    if (isFileAnnotation) {
      newAnnotation.filePath = resolve(newAnnotation.filePath!);

      const historyKey = this.makeFileKey(newAnnotation.originId, newAnnotation.filePath!);
      this.annotationHistory.set(historyKey, {
        filePath: newAnnotation.filePath!,
        originId: newAnnotation.originId,
        lastAnnotatedAt: newAnnotation.createdAt,
      });
    }

    this.annotations.set(newAnnotation.id, newAnnotation);

    this.save();

    const target = newAnnotation.isClaimAnnotation
      ? `claim ${newAnnotation.claimId}`
      : `${newAnnotation.filePath} at ${newAnnotation.from}-${newAnnotation.to}`;
    console.log(`Added annotation to ${target}`);
    return newAnnotation;
  }

  /**
   * Update an existing annotation
   */
  update(id: string, updates: Partial<Annotation>): Annotation | null {
    const annotation = this.annotations.get(id);
    if (!annotation) {
      return null;
    }

    // Apply updates
    Object.assign(annotation, updates);
    this.save();
    console.log(`Updated annotation ${id}`);
    return annotation;
  }

  /**
   * Delete an annotation
   */
  delete(id: string): Annotation | null {
    const annotation = this.annotations.get(id);
    if (!annotation) {
      return null;
    }

    this.annotations.delete(id);
    this.save();
    console.log(`Deleted annotation ${id}`);
    return annotation;
  }

  /**
   * Get recently annotated files, grouped by file with most recent annotation time
   * Optionally filter by originId to show only files for a specific city
   *
   * Shows files with current annotations first, then fills remaining slots
   * with historically annotated files (those whose annotations were deleted)
   */
  getRecentFiles(originId?: string, limit: number = 10): Array<{
    filePath: string;
    originId: string;
    annotationCount: number;
    mostRecentAt: number;
  }> {
    // Group current annotations by file
    const fileMap = new Map<string, {
      filePath: string;
      originId: string;
      count: number;
      mostRecentAt: number;
    }>();

    for (const annotation of this.annotations.values()) {
      // Skip claims annotations (no file path)
      if (annotation.isClaimAnnotation || !annotation.filePath) continue;

      // Filter by originId if provided
      if (originId && annotation.originId !== originId) {
        continue;
      }

      const key = this.makeFileKey(annotation.originId, annotation.filePath);
      const existing = fileMap.get(key);

      if (existing) {
        existing.count++;
        existing.mostRecentAt = Math.max(existing.mostRecentAt, annotation.createdAt);
      } else {
        fileMap.set(key, {
          filePath: annotation.filePath,
          originId: annotation.originId,
          count: 1,
          mostRecentAt: annotation.createdAt,
        });
      }
    }

    // Get files with current annotations
    const filesWithAnnotations = [...fileMap.values()]
      .sort((a, b) => b.mostRecentAt - a.mostRecentAt)
      .slice(0, limit)
      .map(f => ({
        filePath: f.filePath,
        originId: f.originId,
        annotationCount: f.count,
        mostRecentAt: f.mostRecentAt,
      }));

    // If we have fewer than limit, fill with history (files whose annotations were deleted)
    if (filesWithAnnotations.length < limit) {
      const seenKeys = new Set(filesWithAnnotations.map(f => this.makeFileKey(f.originId, f.filePath)));

      // Get historical entries not in current annotations
      const historyEntries = [...this.annotationHistory.values()]
        .filter(h => {
          if (originId && h.originId !== originId) return false;
          const key = this.makeFileKey(h.originId, h.filePath);
          return !seenKeys.has(key);
        })
        .sort((a, b) => b.lastAnnotatedAt - a.lastAnnotatedAt)
        .slice(0, limit - filesWithAnnotations.length)
        .map(h => ({
          filePath: h.filePath,
          originId: h.originId,
          annotationCount: 0, // annotations were deleted
          mostRecentAt: h.lastAnnotatedAt,
        }));

      return [...filesWithAnnotations, ...historyEntries];
    }

    return filesWithAnnotations;
  }
}
