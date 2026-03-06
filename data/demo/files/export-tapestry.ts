#!/usr/bin/env tsx
// Export tapestry data, artifact images, and linked files from a running portolan server.
// Usage: npx tsx scripts/export-tapestry.ts <cityName|cityId>

import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { execFileSync } from 'node:child_process'

const force = process.argv.includes('--force')
const asIdx = process.argv.indexOf('--as')
const exportName = asIdx !== -1 ? process.argv[asIdx + 1] : null
const cityArg = process.argv.filter(a => !a.startsWith('--') && a !== exportName)[2]
if (!cityArg) {
  console.error('Usage: npx tsx scripts/export-tapestry.ts <cityName|cityId> [--force] [--as <name>]')
  process.exit(1)
}
const outputName = exportName || cityArg

const API_BASE = process.env.PORTOLAN_URL || 'http://localhost:4004'
const BASE_OUT = path.resolve(import.meta.dirname, '..', 'docs', 'data')
const OUT_DIR = path.join(BASE_OUT, outputName)

interface CityInfo {
  id: string
  path: string
  sshHost?: string
}

function resolveCity(nameOrId: string): CityInfo {
  const citiesPath = path.join(os.homedir(), '.portolan', 'cities.json')
  if (!fs.existsSync(citiesPath)) {
    console.error(`Cannot resolve city: ${citiesPath} not found`)
    process.exit(1)
  }
  const data = JSON.parse(fs.readFileSync(citiesPath, 'utf-8'))
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-/.test(nameOrId)
  const match = data.cities.find((c: any) =>
    isUuid ? c.id === nameOrId : c.name === nameOrId
  )
  if (!match) {
    const names = data.cities.map((c: any) => c.name).join(', ')
    console.error(`City "${nameOrId}" not found. Available: ${names}`)
    process.exit(1)
  }
  if (!isUuid) console.log(`Resolved "${nameOrId}" → ${match.id}`)
  return { id: match.id, path: match.path, sshHost: match.sshHost }
}

/** Find non-URL, non-.md links in markdown body text. */
function findLinkedFiles(body: string): string[] {
  const files: string[] = []
  const add = (href: string) => {
    if (/^https?:\/\//.test(href)) return
    if (/\.md$/.test(href)) return
    if (!files.includes(href)) files.push(href)
  }

  // Markdown links: [text](path)
  const linkRe = /\[[^\]]+\]\(([^)]+)\)/g
  let m
  while ((m = linkRe.exec(body)) !== null) add(m[1])

  // Inline code file references: `path/to/file.ext` or `path/to/file.ext:L42`
  const codeRe = /`((?:\.{0,2}\/)?[\w.\-/]+\/[\w.\-]+\.[a-zA-Z]{1,10}(?::L?\d+(?:-\d+)?)?)`/g
  while ((m = codeRe.exec(body)) !== null) {
    // Strip line reference for the file path
    const path = m[1].replace(/:L?\d+(?:-\d+)?$/, '')
    add(path)
  }

  return files
}

/** Download a file from a city (local or remote via scp). */
function downloadFile(city: CityInfo, relativePath: string, outPath: string): boolean {
  const remotePath = `${city.path}/${relativePath}`
  try {
    fs.mkdirSync(path.dirname(outPath), { recursive: true })
    if (city.sshHost) {
      execFileSync('scp', [`${city.sshHost}:${remotePath}`, outPath], { timeout: 15000 })
    } else {
      fs.copyFileSync(remotePath, outPath)
    }
    return true
  } catch {
    return false
  }
}

/** Rewrite local file links in body to point to exported paths. */
function rewriteLinks(body: string, rewriteMap: Map<string, string>): string {
  // Rewrite markdown links
  let result = body.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, text, href) => {
    const rewritten = rewriteMap.get(href)
    return rewritten ? `[${text}](${rewritten})` : _match
  })
  // Rewrite inline code file references: `path/to/file.ext:L42` → `rewritten/file.ext:L42`
  result = result.replace(/`((?:\.{0,2}\/)?[\w.\-/]+\/[\w.\-]+\.[a-zA-Z]{1,10})((?::L?\d+(?:-\d+)?)?)`/g,
    (_match, filePath, lineSuffix) => {
      const rewritten = rewriteMap.get(filePath)
      return rewritten ? `\`${rewritten}${lineSuffix}\`` : _match
    })
  return result
}

const city = resolveCity(cityArg)

async function main() {
  console.log(`Fetching tapestry for city "${cityArg}"...`)
  const res = await fetch(`${API_BASE}/tapestry?cityId=${encodeURIComponent(city.id)}`)
  if (!res.ok) throw new Error(`Tapestry fetch failed: ${res.status} ${await res.text()}`)
  const data = await res.json()

  console.log(`  ${data.nodes.length} nodes, ${data.links.length} links`)

  // Download artifact images
  let artifactCount = 0
  for (const node of data.nodes) {
    if (!node.evidence?.artifacts || !node.specName) continue
    for (const [_name, filePath] of Object.entries(node.evidence.artifacts)) {
      if (typeof filePath !== 'string') continue
      const filename = filePath.split('/').pop() || ''
      const outDir = path.join(OUT_DIR, 'claims', node.specName)
      const outPath = path.join(outDir, filename)

      if (!force && fs.existsSync(outPath)) continue

      const url = `${API_BASE}/tapestry-asset/${encodeURIComponent(node.specName)}/${encodeURIComponent(filename)}?cityId=${encodeURIComponent(city.id)}`
      try {
        const imgRes = await fetch(url)
        if (!imgRes.ok) {
          console.warn(`  ⚠ ${node.specName}/${filename}: ${imgRes.status}`)
          continue
        }
        fs.mkdirSync(outDir, { recursive: true })
        const buffer = Buffer.from(await imgRes.arrayBuffer())
        fs.writeFileSync(outPath, buffer)
        artifactCount++
      } catch (err) {
        console.warn(`  ⚠ ${node.specName}/${filename}: ${(err as Error).message}`)
      }
    }
  }

  // Download linked files and rewrite body + outcome links (nodes + sidebar fibers)
  const allItems = [...data.nodes, ...(data.fibers || [])]
  let fileCount = 0
  for (const node of allItems) {
    const textFields = [node.body, node.outcome].filter(Boolean) as string[]
    if (textFields.length === 0) continue

    const allLinks = new Set<string>()
    for (const text of textFields) {
      for (const href of findLinkedFiles(text)) allLinks.add(href)
    }
    if (allLinks.size === 0) continue

    const rewriteMap = new Map<string, string>()
    for (const href of allLinks) {
      const filename = href.split('/').pop() || ''
      const outDir = path.join(OUT_DIR, 'files')
      const outPath = path.join(outDir, filename)

      if (!force && fs.existsSync(outPath)) {
        rewriteMap.set(href, `${outputName}/files/${filename}`)
        continue
      }

      if (downloadFile(city, href, outPath)) {
        rewriteMap.set(href, `${outputName}/files/${filename}`)
        fileCount++
        console.log(`  ↓ ${href}`)
      } else {
        console.warn(`  ⚠ ${href}: download failed`)
      }
    }

    if (rewriteMap.size > 0) {
      if (node.body) node.body = rewriteLinks(node.body, rewriteMap)
      if (node.outcome) node.outcome = rewriteLinks(node.outcome, rewriteMap)
    }
  }

  // Write JSON
  fs.mkdirSync(OUT_DIR, { recursive: true })
  fs.writeFileSync(path.join(OUT_DIR, 'tapestry.json'), JSON.stringify(data, null, 2))

  // Update manifest (list of all exported tapestries)
  const manifestPath = path.join(BASE_OUT, 'manifest.json')
  let manifest: { name: string; nodeCount: number; updated: string }[] = []
  if (fs.existsSync(manifestPath)) {
    try { manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) } catch {}
  }
  const entry = { name: outputName, nodeCount: data.nodes.length, updated: new Date().toISOString() }
  const idx = manifest.findIndex(m => m.name === outputName)
  if (idx >= 0) manifest[idx] = entry; else manifest.push(entry)
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2))

  console.log(`Exported: ${data.nodes.length} nodes, ${artifactCount} artifacts, ${fileCount} files → docs/data/${outputName}/`)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
