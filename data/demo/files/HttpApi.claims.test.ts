/**
 * HttpApi claims annotation tests
 *
 * Tests the claims-specific behavior in HttpApi:
 * - GET /annotations?claimId= returns claim annotations
 * - POST /annotations creates claims annotations with validation
 * - formatClaimsAnnotationsForClaude output format
 * - /tapestry endpoint (DAG with fibers, evidence, staleness)
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AnnotationPersistence, Annotation } from '../AnnotationPersistence.js';
import { HttpApi } from '../HttpApi.js';
import { shellEscape } from '../KittyIntegration.js';
import { existsSync, mkdirSync, rmSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import {
  makePersistence as _makePersistence,
  httpRequest,
  makeCityLookup,
  writeFiber,
  stubOriginLookup,
  stubPersistenceLookup,
} from './test-utils.js';

const TEST_DIR = join(homedir(), '.portolan-test-httpapi-claims');
const TEST_FILE = join(TEST_DIR, 'annotations.json');

// Minimal stub for HttpApi constructor — city lookup with no results
const stubCityLookup = {
  getCityById: () => null,
};

function makePersistence(): AnnotationPersistence {
  return _makePersistence(TEST_DIR, TEST_FILE);
}

/** Factory: create an Annotation with claim defaults, overriding only what matters per test */
function makeClaimAnnotation(overrides: Partial<Annotation> = {}): Annotation {
  return {
    id: '1',
    originId: 'local',
    comment: 'test comment',
    createdAt: Date.now(),
    isClaimAnnotation: true,
    claimId: 'c1',
    ...overrides,
  };
}

describe('HttpApi — claims annotations', () => {
  let persistence: AnnotationPersistence;
  let api: HttpApi;

  beforeEach(() => {
    if (!existsSync(TEST_DIR)) {
      mkdirSync(TEST_DIR, { recursive: true });
    }
    persistence = makePersistence();
    persistence.load();

    api = new HttpApi(stubCityLookup as any, stubOriginLookup as any, stubPersistenceLookup as any);
    api.setAnnotationPersistence(persistence);
  });

  afterEach(() => {
    if (existsSync(TEST_DIR)) {
      rmSync(TEST_DIR, { recursive: true, force: true });
    }
  });

  /** Access private formatClaimsAnnotationsForClaude via api instance */
  function formatClaims(cityName: string, annotations: Annotation[], globalComment?: string): string {
    return (api as any).formatClaimsAnnotationsForClaude(cityName, annotations, globalComment);
  }


  // ────────────────────────────────────────────────────────────
  // POST /annotations — create claims annotation
  // ────────────────────────────────────────────────────────────

  describe('POST /annotations (claims)', () => {
    it('creates a claim annotation with required fields', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Seems low',
        isClaimAnnotation: true,
        claimId: 'claim-1',
        claimTitle: 'B-modes consistent with zero',
        selectedText: 'PTE 0.29',
      });

      expect(res.status).toBe(201);
      expect(res.data.annotation).toBeDefined();
      expect(res.data.annotation.isClaimAnnotation).toBe(true);
      expect(res.data.annotation.claimId).toBe('claim-1');
      expect(res.data.annotation.claimTitle).toBe('B-modes consistent with zero');
      expect(res.data.annotation.selectedText).toBe('PTE 0.29');
    });

    it('creates a claim image annotation', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Edge effects visible',
        isClaimAnnotation: true,
        claimId: 'claim-2',
        claimTitle: 'Galaxy generation pipeline',
        artifact: 'galaxy_fields.png',
        x: 45.2,
        y: 31.8,
        isImageAnnotation: true,
      });

      expect(res.status).toBe(201);
      expect(res.data.annotation.artifact).toBe('galaxy_fields.png');
      expect(res.data.annotation.x).toBe(45.2);
      expect(res.data.annotation.y).toBe(31.8);
    });

    it('rejects claim annotation without claimId', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Missing claimId',
        isClaimAnnotation: true,
      });

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/claimId/i);
    });

    it('rejects claim annotation without comment', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        isClaimAnnotation: true,
        claimId: 'claim-1',
      });

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/comment/i);
    });
  });

  // ────────────────────────────────────────────────────────────
  // GET /annotations?claimId=
  // ────────────────────────────────────────────────────────────

  describe('GET /annotations?claimId=', () => {
    it('returns annotations for a specific claim', async () => {
      persistence.add(makeClaimAnnotation({
        comment: 'First',
        claimId: 'claim-1',
        claimTitle: 'Test claim',
      }));
      persistence.add(makeClaimAnnotation({
        id: '2',
        comment: 'Second',
        claimId: 'claim-1',
        claimTitle: 'Test claim',
      }));
      persistence.add(makeClaimAnnotation({
        id: '3',
        comment: 'Other',
        claimId: 'claim-2',
        claimTitle: 'Other claim',
      }));

      const res = await httpRequest(api, 'GET', '/annotations?claimId=claim-1');

      expect(res.status).toBe(200);
      expect(res.data.annotations).toHaveLength(2);
      expect(res.data.annotations.every((a: any) => a.claimId === 'claim-1')).toBe(true);
    });

    it('returns empty array for unknown claim', async () => {
      const res = await httpRequest(api, 'GET', '/annotations?claimId=nonexistent');

      expect(res.status).toBe(200);
      expect(res.data.annotations).toHaveLength(0);
    });

    it('does not return file annotations when querying by claimId', async () => {
      persistence.add({
        originId: 'local',
        comment: 'File annotation',
        filePath: '/test/file.ts',
        from: 0,
        to: 10,
      } as any);
      persistence.add(makeClaimAnnotation({
        comment: 'Claim annotation',
        claimId: 'claim-1',
        claimTitle: 'Test',
      }));

      const res = await httpRequest(api, 'GET', '/annotations?claimId=claim-1');

      expect(res.status).toBe(200);
      expect(res.data.annotations).toHaveLength(1);
      expect(res.data.annotations[0].isClaimAnnotation).toBe(true);
    });

    it('returns 400 when neither path nor claimId provided', async () => {
      const res = await httpRequest(api, 'GET', '/annotations');

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/path.*claimId|claimId.*path/i);
    });
  });

  // ────────────────────────────────────────────────────────────
  // formatClaimsAnnotationsForClaude
  // ────────────────────────────────────────────────────────────

  describe('formatClaimsAnnotationsForClaude', () => {
    it('formats text annotations', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'Seems low — recheck with different bin edges',
          claimId: 'claim-1',
          claimTitle: 'B-modes consistent with zero',
          selectedText: 'PTE 0.29',
        }),
      ];

      const output = formatClaims('pure-eb', annotations);

      expect(output).toContain('# Claims review: pure-eb');
      expect(output).toContain('## 1. [B-modes consistent with zero]');
      expect(output).toContain('> On text: "PTE 0.29"');
      expect(output).toContain('> Seems low — recheck with different bin edges');
      expect(output).toContain('---');
    });

    it('formats image/artifact annotations with position', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'Check edge effects on velocity',
          claimId: 'claim-2',
          claimTitle: 'Galaxy generation pipeline',
          artifact: 'galaxy_fields.png',
          x: 45,
          y: 32,
        }),
      ];

      const output = formatClaims('pure-eb', annotations);

      expect(output).toContain('## 1. [Galaxy generation pipeline]');
      expect(output).toContain('> On plot: galaxy_fields.png (at 45%, 32%)');
      expect(output).toContain('> Check edge effects on velocity');
    });

    it('formats multiple annotations with numbering', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'First', claimTitle: 'Claim A', selectedText: 'text A',
        }),
        makeClaimAnnotation({
          id: '2', comment: 'Second',
          claimId: 'c2', claimTitle: 'Claim B', artifact: 'plot.png', x: 10, y: 20,
        }),
      ];

      const output = formatClaims('test', annotations);

      expect(output).toContain('## 1. [Claim A]');
      expect(output).toContain('## 2. [Claim B]');
      expect(output).toContain('2 pieces of feedback');
    });

    it('includes global comment when provided', () => {
      const output = formatClaims('test', [], 'Overall the analysis looks solid.');

      expect(output).toContain('# Claims review: test');
      expect(output).toContain('Overall the analysis looks solid.');
    });

    it('handles single annotation pluralization', () => {
      const annotations = [
        makeClaimAnnotation({ comment: 'Just one', claimTitle: 'Single' }),
      ];

      const output = formatClaims('test', annotations);
      expect(output).toContain('1 piece of feedback');
    });

    it('truncates long selected text', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'Too long', claimTitle: 'Long Text',
          selectedText: 'A'.repeat(100),
        }),
      ];

      const output = formatClaims('test', annotations);
      // selectedText is sliced to 60 chars with ellipsis
      expect(output).toContain('"' + 'A'.repeat(60) + '…"');
      expect(output).not.toContain('A'.repeat(100));
    });

    it('falls back to claimId when claimTitle missing', () => {
      const annotations = [
        makeClaimAnnotation({ comment: 'No title', claimId: 'claim-xyz' }),
      ];

      const output = formatClaims('test', annotations);
      expect(output).toContain('[claim-xyz]');
    });

    it('groups multiple annotations under the same claim heading', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'First note', claimId: 'c1', claimTitle: 'B-modes',
          selectedText: 'PTE 0.29',
        }),
        makeClaimAnnotation({
          id: '2', comment: 'Second note', claimId: 'c1', claimTitle: 'B-modes',
          artifact: 'b_modes.png', x: 50, y: 25,
        }),
        makeClaimAnnotation({
          id: '3', comment: 'Other claim', claimId: 'c2', claimTitle: 'Galaxy pipeline',
        }),
      ];

      const output = formatClaims('test', annotations);

      // Single heading for B-modes (not repeated)
      expect(output).toContain('## 1. [B-modes]');
      expect(output).toContain('## 2. [Galaxy pipeline]');
      // Both annotations under claim 1
      expect(output).toContain('> On text: "PTE 0.29"');
      expect(output).toContain('> On plot: b_modes.png (at 50%, 25%)');
      // B-modes heading appears only once
      const headingMatches = output.match(/\[B-modes\]/g);
      expect(headingMatches).toHaveLength(1);
      // 3 total annotations
      expect(output).toContain('3 pieces of feedback');
    });
  });

  // ────────────────────────────────────────────────────────────
  // Claims annotation CRUD round-trip
  // ────────────────────────────────────────────────────────────

  describe('CRUD round-trip', () => {
    it('create → get → update → delete', async () => {
      // Create
      const createRes = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Initial',
        isClaimAnnotation: true,
        claimId: 'claim-crud',
        claimTitle: 'CRUD Test',
      });
      expect(createRes.status).toBe(201);
      const id = createRes.data.annotation.id;

      // Get
      const getRes = await httpRequest(api, 'GET', '/annotations?claimId=claim-crud');
      expect(getRes.status).toBe(200);
      expect(getRes.data.annotations).toHaveLength(1);
      expect(getRes.data.annotations[0].comment).toBe('Initial');

      // Update
      const updateRes = await httpRequest(api, 'PUT', `/annotations/${id}`, {
        comment: 'Revised',
      });
      expect(updateRes.status).toBe(200);
      expect(updateRes.data.annotation.comment).toBe('Revised');

      // Delete
      const deleteRes = await httpRequest(api, 'DELETE', `/annotations/${id}`);
      expect(deleteRes.status).toBe(200);

      // Verify gone
      const finalRes = await httpRequest(api, 'GET', '/annotations?claimId=claim-crud');
      expect(finalRes.data.annotations).toHaveLength(0);
    });
  });

  // ────────────────────────────────────────────────────────────
  // GET /annotations?claims=true — all claims annotations
  // ────────────────────────────────────────────────────────────

  describe('GET /annotations?claims=true', () => {
    it('returns all claims annotations across claims', async () => {
      persistence.add(makeClaimAnnotation({
        comment: 'A',
        claimId: 'c1',
        claimTitle: 'First',
      }));
      persistence.add(makeClaimAnnotation({
        id: '2',
        comment: 'B',
        claimId: 'c2',
        claimTitle: 'Second',
      }));
      persistence.add({
        originId: 'local', comment: 'File only',
        filePath: '/test/file.ts', from: 0, to: 10,
      } as any);

      const res = await httpRequest(api, 'GET', '/annotations?claims=true');

      expect(res.status).toBe(200);
      expect(res.data.annotations).toHaveLength(2);
      expect(res.data.annotations.every((a: any) => a.isClaimAnnotation)).toBe(true);
    });

    it('returns empty array when no claims exist', async () => {
      persistence.add({
        originId: 'local', comment: 'File only',
        filePath: '/test/file.ts', from: 0, to: 10,
      } as any);

      const res = await httpRequest(api, 'GET', '/annotations?claims=true');

      expect(res.status).toBe(200);
      expect(res.data.annotations).toHaveLength(0);
    });
  });

  // ────────────────────────────────────────────────────────────
  // Image annotation validation
  // ────────────────────────────────────────────────────────────

  describe('POST /annotations — image validation', () => {
    it('rejects claim image annotation without x coordinate', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Missing position',
        isClaimAnnotation: true,
        claimId: 'claim-img',
        artifact: 'plot.png',
        y: 50,
      });

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/x and y/i);
    });

    it('rejects claim image annotation without y coordinate', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Missing position',
        isClaimAnnotation: true,
        claimId: 'claim-img',
        artifact: 'plot.png',
        x: 50,
      });

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/x and y/i);
    });

    it('accepts claim image annotation with both coordinates', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Valid image',
        isClaimAnnotation: true,
        claimId: 'claim-img',
        artifact: 'plot.png',
        x: 45, y: 32,
      });

      expect(res.status).toBe(201);
    });

    it('accepts claim text annotation without artifact', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'Just text',
        isClaimAnnotation: true,
        claimId: 'claim-txt',
        selectedText: 'some text',
      });

      expect(res.status).toBe(201);
    });
  });

  // ────────────────────────────────────────────────────────────
  // Delete annotation lifecycle
  // ────────────────────────────────────────────────────────────

  describe('DELETE /annotations/:id (claims)', () => {
    it('deletes a claims annotation and removes it from claimId query', async () => {
      const createRes = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'To delete',
        isClaimAnnotation: true,
        claimId: 'claim-del',
      });
      expect(createRes.status).toBe(201);
      const id = createRes.data.annotation.id;

      const deleteRes = await httpRequest(api, 'DELETE', `/annotations/${id}`);
      expect(deleteRes.status).toBe(200);

      const getRes = await httpRequest(api, 'GET', '/annotations?claimId=claim-del');
      expect(getRes.data.annotations).toHaveLength(0);

      // Also gone from all-claims query
      const allRes = await httpRequest(api, 'GET', '/annotations?claims=true');
      expect(allRes.data.annotations).toHaveLength(0);
    });
  });

  // ────────────────────────────────────────────────────────────
  // POST /promote-to-felt
  // ────────────────────────────────────────────────────────────

  describe('POST /promote-to-felt', () => {
    it('rejects request without required fields', async () => {
      const res = await httpRequest(api, 'POST', '/promote-to-felt', {
        claimId: 'claim-1',
      });

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/claimId.*comment.*cityId|Missing required/i);
    });

    it('rejects request with empty body', async () => {
      const res = await httpRequest(api, 'POST', '/promote-to-felt', {});

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/Missing required/i);
    });

    it('returns 404 for unknown city', async () => {
      const res = await httpRequest(api, 'POST', '/promote-to-felt', {
        claimId: 'claim-1',
        comment: 'Test comment',
        cityId: 'nonexistent-city',
      });

      expect(res.status).toBe(404);
      expect(res.data.error).toMatch(/City not found/i);
    });
  });

  // ────────────────────────────────────────────────────────────
  // formatClaimsAnnotationsForClaude — additional edge cases
  // ────────────────────────────────────────────────────────────

  describe('formatClaimsAnnotationsForClaude — edge cases', () => {
    it('formats comment-only annotations (no selectedText or artifact)', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'This claim needs more evidence',
          claimTitle: 'Dark energy equation of state',
        }),
      ];

      const output = formatClaims('test', annotations);

      expect(output).toContain('## 1. [Dark energy equation of state]');
      expect(output).toContain('> This claim needs more evidence');
      // Should not contain "On text:" or "On plot:" lines
      expect(output).not.toContain('On text:');
      expect(output).not.toContain('On plot:');
    });

    it('handles empty annotations array', () => {
      const output = formatClaims('test', []);

      expect(output).toContain('# Claims review: test');
      expect(output).toContain('---');
      expect(output).not.toContain('pieces of feedback');
    });

    it('handles mixed annotation types under the same claim', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'General note',
          claimId: 'c1',
          claimTitle: 'Multi-type claim',
        }),
        makeClaimAnnotation({
          id: '2',
          comment: 'Text note',
          claimId: 'c1',
          claimTitle: 'Multi-type claim',
          selectedText: 'PTE = 0.3',
        }),
        makeClaimAnnotation({
          id: '3',
          comment: 'Image note',
          claimId: 'c1',
          claimTitle: 'Multi-type claim',
          artifact: 'spectrum.png',
          x: 50,
          y: 25,
        }),
      ];

      const output = formatClaims('test', annotations);

      // Single heading for all three
      const headingMatches = output.match(/\[Multi-type claim\]/g);
      expect(headingMatches).toHaveLength(1);
      // All three annotation types present
      expect(output).toContain('> General note');
      expect(output).toContain('> On text: "PTE = 0.3"');
      expect(output).toContain('> On plot: spectrum.png (at 50%, 25%)');
      expect(output).toContain('3 pieces of feedback');
    });

    it('handles image annotation without coordinates', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'General plot note',
          artifact: 'overview.png',
        }),
      ];

      const output = formatClaims('test', annotations);

      expect(output).toContain('> On plot: overview.png');
      // No position reference when x/y are undefined
      expect(output).not.toContain('at');
    });
  });

  // ────────────────────────────────────────────────────────────
  // Tapestry asset serving — security
  // ────────────────────────────────────────────────────────────

  describe('GET /tapestry-asset (security)', () => {
    it('rejects path traversal in nested segments', async () => {
      const res = await httpRequest(api, 'GET', '/tapestry-asset/sub/..%2F..%2Fetc/passwd?cityId=test');

      // Either 400 (caught by traversal check) or 404 (city not found) — never serves the file
      expect([400, 404]).toContain(res.status);
    });

    it('rejects shell injection via dollar substitution', async () => {
      // /tapestry-asset requires {specName}/{filename} — single segment is rejected early
      const res = await httpRequest(api, 'GET', '/tapestry-asset/spec/$(id).png?cityId=test');

      expect(res.status).toBe(400);
      expect(res.data).toContain('Invalid asset path');
    });

    it('rejects shell injection via backticks', async () => {
      const res = await httpRequest(api, 'GET', '/tapestry-asset/spec/`whoami`.png?cityId=test');

      expect(res.status).toBe(400);
      expect(res.data).toContain('Invalid asset path');
    });

    it('accepts clean asset paths', async () => {
      // Will 404 (city not found in stub) but should pass the security check
      const res = await httpRequest(api, 'GET', '/tapestry-asset/spec-name/plot.png?cityId=test');

      // 404 because stub city lookup returns null — but NOT 400
      expect(res.status).toBe(404);
    });
  });

  // ────────────────────────────────────────────────────────────
  // Special characters in annotation comments
  // ────────────────────────────────────────────────────────────

  describe('annotations with special characters', () => {
    it('handles single quotes in comment', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: "it's got single quotes",
        isClaimAnnotation: true,
        claimId: 'claim-sq',
        claimTitle: 'Single quotes test',
      });

      expect(res.status).toBe(201);
      expect(res.data.annotation.comment).toBe("it's got single quotes");
    });

    it('handles double quotes in comment', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'the "scare quotes" issue',
        isClaimAnnotation: true,
        claimId: 'claim-dq',
        claimTitle: 'Double quotes test',
      });

      expect(res.status).toBe(201);
      expect(res.data.annotation.comment).toBe('the "scare quotes" issue');
    });

    it('handles unicode in selectedText', async () => {
      const res = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment: 'check this',
        isClaimAnnotation: true,
        claimId: 'claim-uni',
        selectedText: 'σ = 2.3 ± 0.1',
      });

      expect(res.status).toBe(201);
      expect(res.data.annotation.selectedText).toBe('σ = 2.3 ± 0.1');
    });

    it('round-trips special chars through CRUD', async () => {
      const comment = "it's a \"complex\" note — with em-dash & symbols <>";

      const createRes = await httpRequest(api, 'POST', '/annotations', {
        originId: 'local',
        comment,
        isClaimAnnotation: true,
        claimId: 'claim-special',
        claimTitle: 'Special chars',
        selectedText: 'PTE < 0.05 & σ > 3',
      });
      expect(createRes.status).toBe(201);
      const id = createRes.data.annotation.id;

      const getRes = await httpRequest(api, 'GET', '/annotations?claimId=claim-special');
      expect(getRes.status).toBe(200);
      expect(getRes.data.annotations[0].comment).toBe(comment);
      expect(getRes.data.annotations[0].selectedText).toBe('PTE < 0.05 & σ > 3');

      // Verify format output handles special chars
      const formatClaims = (api as any).formatClaimsAnnotationsForClaude.bind(api);
      const formatted = formatClaims('test', getRes.data.annotations);
      expect(formatted).toContain('PTE < 0.05 & σ > 3');
      expect(formatted).toContain(comment);

      // Clean up
      await httpRequest(api, 'DELETE', `/annotations/${id}`);
    });
  });

  // ────────────────────────────────────────────────────────────
  // shellEscape — imported from KittyIntegration
  // ────────────────────────────────────────────────────────────

  describe('shellEscape', () => {
    it('wraps in single quotes and escapes embedded single quotes', () => {
      expect(shellEscape("it's")).toBe("'it'\\''s'");
    });

    it('wraps clean strings in single quotes', () => {
      expect(shellEscape('/home/user/results/claims/index.html')).toBe("'/home/user/results/claims/index.html'");
    });

    it('escapes multiple single quotes', () => {
      expect(shellEscape("a'b'c")).toBe("'a'\\''b'\\''c'");
    });

    it('handles strings with double quotes (safe inside single quotes)', () => {
      expect(shellEscape('say "hello"')).toBe("'say \"hello\"'");
    });

    it('handles backticks (safe inside single quotes)', () => {
      expect(shellEscape('`whoami`')).toBe("'`whoami`'");
    });

    it('handles dollar substitution (safe inside single quotes)', () => {
      expect(shellEscape('$(id)')).toBe("'$(id)'");
    });

    it('handles path with spaces', () => {
      expect(shellEscape('/home/user/my project/file.txt')).toBe("'/home/user/my project/file.txt'");
    });

    it('escapes single quotes in paths with dangerous chars', () => {
      // Inside single quotes, only ' needs escaping.
      // Double quotes, backticks, $ are all literal inside single quotes.
      expect(shellEscape("path'with\"dangerous`chars$(id)")).toBe("'path'\\''with\"dangerous`chars$(id)'");
    });
  });

  // ────────────────────────────────────────────────────────────
  // GET /tapestry endpoint
  // ────────────────────────────────────────────────────────────

  describe('GET /tapestry', () => {
    const TAPESTRY_CITY_DIR = join(TEST_DIR, 'tapestry-city');
    const TAPESTRY_FELT_DIR = join(TAPESTRY_CITY_DIR, '.felt');

    function makeTapestryApi(cityId: string, cityDir: string) {
      return new HttpApi(
        makeCityLookup(cityId, cityDir, 'TapestryCity') as any,
        stubOriginLookup as any, stubPersistenceLookup as any,
      );
    }

    it('returns 400 without cityId', async () => {
      const tapestryApi = makeTapestryApi('test', TAPESTRY_CITY_DIR);
      const res = await httpRequest(tapestryApi, 'GET', '/tapestry');

      expect(res.status).toBe(400);
      expect(res.data.error).toMatch(/cityId/i);
    });

    it('returns 404 for unknown city', async () => {
      const tapestryApi = makeTapestryApi('test', TAPESTRY_CITY_DIR);
      const res = await httpRequest(tapestryApi, 'GET', '/tapestry?cityId=nonexistent');

      expect(res.status).toBe(404);
    });

    it('returns DAG with nodes, links, and downstream for tapestry: fibers', async () => {
      // Set up two tapestry: fibers with a dependency
      writeFiber(TAPESTRY_FELT_DIR, 'fiber-a', `---
title: Fiber A
status: active
kind: spec
tags:
  - tapestry:fiber_a
---
Body of fiber A.
`);
      writeFiber(TAPESTRY_FELT_DIR, 'fiber-b', `---
title: Fiber B
status: open
kind: spec
tags:
  - tapestry:fiber_b
depends-on:
  - fiber-a
---
Body of fiber B depends on A.
`);
      // A non-rule fiber that depends on a rule fiber (downstream concern)
      writeFiber(TAPESTRY_FELT_DIR, 'task-c', `---
title: Task C
status: open
kind: task
depends-on:
  - fiber-a
---
Downstream task.
`);

      const tapestryApi = makeTapestryApi('tapestry-test', TAPESTRY_CITY_DIR);
      const res = await httpRequest(tapestryApi, 'GET', '/tapestry?cityId=tapestry-test');

      expect(res.status).toBe(200);

      // Nodes: only tapestry: fibers (fiber-a, fiber-b), not task-c
      expect(res.data.nodes).toHaveLength(2);
      const nodeIds = res.data.nodes.map((n: any) => n.id);
      expect(nodeIds).toContain('fiber-a');
      expect(nodeIds).toContain('fiber-b');

      // Each node has expected fields
      const nodeA = res.data.nodes.find((n: any) => n.id === 'fiber-a');
      expect(nodeA.title).toBe('Fiber A');
      expect(nodeA.body).toContain('Body of fiber A');
      expect(nodeA.specName).toBe('fiber_a');
      expect(nodeA.staleness).toBeDefined();

      // Links: fiber-a → fiber-b (source: fiber-a, target: fiber-b)
      expect(res.data.links).toHaveLength(1);
      expect(res.data.links[0]).toEqual({ source: 'fiber-a', target: 'fiber-b' });

      // Downstream: fiber-b and task-c both depend on fiber-a
      expect(res.data.downstream['fiber-a']).toBeDefined();
      expect(res.data.downstream['fiber-a']).toHaveLength(2);
      const downstreamIds = res.data.downstream['fiber-a'].map((d: any) => d.id);
      expect(downstreamIds).toContain('fiber-b');
      expect(downstreamIds).toContain('task-c');
    });

    it('returns empty DAG when no tapestry: fibers exist', async () => {
      const emptyDir = join(TEST_DIR, 'empty-city');
      const emptyFeltDir = join(emptyDir, '.felt');
      mkdirSync(emptyFeltDir, { recursive: true });
      writeFiber(emptyFeltDir, 'plain-task', `---
title: Just a task
status: open
kind: task
---
No rule tag.
`);

      const emptyApi = makeTapestryApi('empty-test', emptyDir);
      const res = await httpRequest(emptyApi, 'GET', '/tapestry?cityId=empty-test');

      expect(res.status).toBe(200);
      expect(res.data.nodes).toHaveLength(0);
      expect(res.data.links).toHaveLength(0);
      expect(res.data.downstream).toEqual({});
    });
  });

  // ────────────────────────────────────────────────────────────
  // formatClaimsAnnotationsForClaude — send-to-worker format
  // ────────────────────────────────────────────────────────────

  describe('formatClaimsAnnotationsForClaude for send-to-worker', () => {
    it('produces the expected worker paste format', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'Seems low — recheck with different bin edges',
          claimId: 'c1',
          claimTitle: 'B-modes consistent with zero',
          selectedText: 'PTE 0.29',
        }),
        makeClaimAnnotation({
          id: '2',
          comment: 'Check edge effects on velocity profile',
          claimId: 'c2',
          claimTitle: 'Galaxy generation pipeline',
          artifact: 'galaxy_fields.png',
          x: 45,
          y: 32,
        }),
      ];

      const output = formatClaims('pure-eb', annotations);

      // Verify structure matches spec: "# Claims review: {cityName}"
      expect(output).toMatch(/^[\n]*# Claims review: pure-eb/);
      // "I've reviewed..." introduction
      expect(output).toContain("I've reviewed the claims dashboard and have 2 pieces of feedback:");
      // Claim headings with numbering
      expect(output).toContain('## 1. [B-modes consistent with zero]');
      expect(output).toContain('## 2. [Galaxy generation pipeline]');
      // Text annotation format
      expect(output).toContain('> On text: "PTE 0.29"');
      expect(output).toContain('> Seems low — recheck with different bin edges');
      // Image annotation format with position
      expect(output).toContain('> On plot: galaxy_fields.png (at 45%, 32%)');
      expect(output).toContain('> Check edge effects on velocity profile');
      // Ends with separator
      expect(output).toMatch(/---\s*$/);
    });

    it('produces readable output with global comment', () => {
      const annotations = [
        makeClaimAnnotation({
          comment: 'Minor issue',
          claimTitle: 'Test claim',
        }),
      ];

      const output = formatClaims('KineLens', annotations, 'Overall the analysis is solid, just a few notes.');

      expect(output).toContain('# Claims review: KineLens');
      expect(output).toContain('Overall the analysis is solid, just a few notes.');
      expect(output).toContain('## 1. [Test claim]');
      expect(output).toContain('> Minor issue');
    });
  });
});
