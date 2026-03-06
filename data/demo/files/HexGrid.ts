// HexGrid.ts - Hexagonal grid coordinate system
// Ported from v1, originally from hexarchy project

import type { HexCoord, CartesianCoord } from '../state/types'

export class HexGrid {
  size: number
  hexRadius: number

  // Precomputed hex geometry constants
  private readonly sqrt3 = Math.sqrt(3)

  constructor(size: number, hexRadius: number) {
    this.size = size
    this.hexRadius = hexRadius
  }

  /**
   * Convert axial hex coordinates to cartesian world coordinates
   */
  axialToCartesian(hex: HexCoord): CartesianCoord {
    const x = this.hexRadius * this.sqrt3 * (hex.q + hex.r / 2)
    const z = this.hexRadius * (3 / 2) * hex.r
    return { x, z }
  }

  /**
   * Convert cartesian world coordinates to axial hex coordinates
   */
  cartesianToHex(x: number, z: number): HexCoord {
    const q = (this.sqrt3 / 3 * x - z / 3) / this.hexRadius
    const r = (2 / 3 * z) / this.hexRadius
    return this.roundHex({ q, r })
  }

  /**
   * Round fractional hex coordinates to nearest hex
   */
  roundHex(hex: HexCoord): HexCoord {
    let q = Math.round(hex.q)
    let r = Math.round(hex.r)
    const s = Math.round(-hex.q - hex.r)

    const qDiff = Math.abs(q - hex.q)
    const rDiff = Math.abs(r - hex.r)
    const sDiff = Math.abs(s - (-hex.q - hex.r))

    if (qDiff > rDiff && qDiff > sDiff) {
      q = -r - s
    } else if (rDiff > sDiff) {
      r = -q - s
    }

    return { q, r }
  }

  /**
   * Get hex distance between two hexes
   */
  distance(a: HexCoord, b: HexCoord): number {
    return (
      Math.abs(a.q - b.q) +
      Math.abs(a.q + a.r - b.q - b.r) +
      Math.abs(a.r - b.r)
    ) / 2
  }

  /**
   * Get all hexes within radius of center
   */
  getHexesInRadius(center: HexCoord, radius: number): HexCoord[] {
    const hexes: HexCoord[] = []

    for (let q = -radius; q <= radius; q++) {
      for (let r = Math.max(-radius, -q - radius); r <= Math.min(radius, -q + radius); r++) {
        hexes.push({
          q: center.q + q,
          r: center.r + r,
        })
      }
    }

    return hexes
  }

  /**
   * Generate hex key for use in maps/sets
   */
  hexKey(hex: HexCoord): string {
    return `${hex.q},${hex.r}`
  }
}
