/**
 * EvidenceReader — reads evidence.json and artifact metadata for claim fibers.
 *
 * Works both locally and via SSH for remote cities.
 * Evidence lives at: {cityPath}/results/claims/{specName}/evidence.json
 * Artifacts are images and PDFs in the same directory.
 */

import { readFile, stat } from 'fs/promises';
import { join } from 'path';
import { promisify } from 'util';
import { execFile } from 'child_process';
import { shellEscape } from './KittyIntegration.js';

const execFileAsync = promisify(execFile);

export interface Evidence {
  specName: string;
  metrics: Record<string, unknown>;
  artifacts: Record<string, string>; // output name → filename (images and PDFs)
  mtime: number;                     // ms since epoch — evidence.json mtime
  generated: string | null;           // ISO timestamp from evidence.json
}

/**
 * Read evidence for a single spec directory.
 */
export async function readEvidence(
  cityPath: string,
  specName: string,
  sshHost?: string,
): Promise<Evidence | null> {
  const evidenceDir = `${cityPath}/results/claims/${specName}`;

  try {
    if (sshHost) {
      return await readRemoteEvidence(sshHost, evidenceDir, specName);
    }
    return await readLocalEvidence(evidenceDir, specName);
  } catch {
    return null;
  }
}

const ARTIFACT_RE = /\.(png|jpe?g|pdf)$/i;

function buildEvidence(
  specName: string,
  data: Record<string, unknown>,
  mtime: number,
): Evidence {
  // Artifacts come from the `output` field only — these are the rule's
  // declared outputs. Directory scans are no longer used.
  const output = (data.output || {}) as Record<string, string | string[]>;
  const artifacts: Record<string, string> = {};
  for (const [key, val] of Object.entries(output)) {
    if (typeof val === 'string' && ARTIFACT_RE.test(val)) {
      artifacts[key] = val;
    }
  }

  return {
    specName,
    metrics: (data.evidence as Record<string, unknown>) || {},
    artifacts,
    mtime,
    generated: (data.generated as string) ?? null,
  };
}

async function readLocalEvidence(evidenceDir: string, specName: string): Promise<Evidence | null> {
  const evidencePath = join(evidenceDir, 'evidence.json');

  let mtime: number;
  try {
    const s = await stat(evidencePath);
    mtime = s.mtimeMs;
  } catch {
    return null;
  }

  const content = await readFile(evidencePath, 'utf-8');
  const data = JSON.parse(content) as Record<string, unknown>;

  return buildEvidence(specName, data, mtime);
}

async function readRemoteEvidence(
  sshHost: string,
  evidenceDir: string,
  specName: string,
): Promise<Evidence | null> {
  const escapedPath = shellEscape(evidenceDir + '/evidence.json');
  const cmd = [
    `stat -c '%Y' ${escapedPath} 2>/dev/null || stat -f '%m' ${escapedPath} 2>/dev/null`,
    `cat ${escapedPath}`,
  ].join(' && echo "---SEPARATOR---" && ');

  const { stdout } = await execFileAsync(
    'ssh', [sshHost, cmd],
    { maxBuffer: 5 * 1024 * 1024, timeout: 15000 },
  );

  const parts = stdout.split('---SEPARATOR---');
  if (parts.length < 2) return null;

  const mtime = parseInt(parts[0].trim(), 10) * 1000; // seconds to ms
  if (isNaN(mtime)) return null;

  const data = JSON.parse(parts[1].trim()) as Record<string, unknown>;

  return buildEvidence(specName, data, mtime);
}

/**
 * Batch-read evidence for multiple specNames in a single SSH call.
 * Avoids SSH connection exhaustion from parallel per-spec calls.
 */
export async function readEvidenceBatch(
  cityPath: string,
  specNames: string[],
  sshHost: string,
): Promise<Map<string, Evidence | null>> {
  const results = new Map<string, Evidence | null>();
  if (specNames.length === 0) return results;

  // Build a shell loop that emits delimited blocks per spec
  const claimsDir = shellEscape(`${cityPath}/results/claims`);
  const perSpec = specNames.map(spec => {
    const ej = `${claimsDir}/${shellEscape(spec)}/evidence.json`;
    return [
      `echo "===SPEC:${spec}==="`,
      `stat -c '%Y' ${ej} 2>/dev/null || stat -f '%m' ${ej} 2>/dev/null || echo 'NO_STAT'`,
      `echo "---SEP---"`,
      `cat ${ej} 2>/dev/null || echo 'NO_FILE'`,
    ].join(' && ');
  }).join(' && ');

  try {
    const { stdout } = await execFileAsync(
      'ssh', [sshHost, perSpec],
      { maxBuffer: 10 * 1024 * 1024, timeout: 30000 },
    );

    // Split into per-spec blocks
    const blocks = stdout.split(/===SPEC:([^=]+)===/);
    // blocks: ['', specName1, content1, specName2, content2, ...]
    for (let i = 1; i < blocks.length; i += 2) {
      const specName = blocks[i];
      const content = blocks[i + 1] || '';
      const parts = content.split('---SEP---');

      const mtimeStr = (parts[0] || '').trim();
      const jsonStr = (parts[1] || '').trim();

      if (mtimeStr === 'NO_STAT' || jsonStr === 'NO_FILE') {
        results.set(specName, null);
        continue;
      }

      const mtime = parseInt(mtimeStr, 10) * 1000;
      if (isNaN(mtime)) {
        results.set(specName, null);
        continue;
      }

      try {
        const data = JSON.parse(jsonStr) as Record<string, unknown>;
        results.set(specName, buildEvidence(specName, data, mtime));
      } catch {
        results.set(specName, null);
      }
    }
  } catch (err) {
    console.error('Evidence batch read failed:', err);
    for (const spec of specNames) {
      results.set(spec, null);
    }
  }

  return results;
}

/**
 * Extract the spec name from a fiber's tapestry: tag.
 * e.g., "tapestry:cosebis_data_vector" → "cosebis_data_vector"
 * Also accepts legacy "rule:" prefix for backwards compatibility.
 */
export function getSpecName(tags: string[]): string | undefined {
  for (const tag of tags) {
    if (tag.startsWith('tapestry:')) return tag.slice(9);
    if (tag.startsWith('rule:')) return tag.slice(5);
  }
  return undefined;
}

/**
 * Compute staleness: a fiber is stale if any upstream dependency has
 * evidence with a newer mtime than this fiber's evidence.
 */
export function computeStaleness(
  fiberId: string,
  dependsOn: string[],
  evidenceMap: Map<string, Evidence | null>,
  fiberSpecMap: Map<string, string>, // fiberId → specName
): 'fresh' | 'stale' | 'no-evidence' {
  const specName = fiberSpecMap.get(fiberId);
  if (!specName) return 'no-evidence';

  const myEvidence = evidenceMap.get(specName);
  if (!myEvidence) return 'no-evidence';

  for (const depId of dependsOn) {
    const depSpec = fiberSpecMap.get(depId);
    if (!depSpec) continue;

    const depEvidence = evidenceMap.get(depSpec);
    if (!depEvidence) continue;

    if (depEvidence.mtime > myEvidence.mtime) {
      return 'stale';
    }
  }

  return 'fresh';
}
