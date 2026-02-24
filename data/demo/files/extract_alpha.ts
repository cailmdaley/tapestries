#!/usr/bin/env npx tsx
/**
 * extract_alpha.ts - Difference matting for transparency extraction
 *
 * Given the same image on white and black backgrounds, mathematically
 * extracts the true alpha channel.
 *
 * Usage: npx tsx scripts/extract_alpha.ts <white.jpg> <black.jpg> <output.png>
 *
 * Based on: https://medium.com/@julien.deluca/transparent-assets-with-nano-banana
 */

import sharp from 'sharp'

export interface RGB { r: number; g: number; b: number }

export async function extractAlphaTwoPass(
  imgOnWhitePath: string,
  imgOnBlackPath: string,
  outputPath: string
): Promise<void> {
  const img1 = sharp(imgOnWhitePath)
  const img2 = sharp(imgOnBlackPath)

  // Ensure we are working with raw pixel data
  const { data: dataWhite, info: meta } = await img1
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  const { data: dataBlack } = await img2
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  if (dataWhite.length !== dataBlack.length) {
    throw new Error("Dimension mismatch: Images must be identical size")
  }

  const outputBuffer = Buffer.alloc(dataWhite.length)

  // Distance between White (255,255,255) and Black (0,0,0)
  // sqrt(255^2 + 255^2 + 255^2) ≈ 441.67
  const bgDist = Math.sqrt(3 * 255 * 255)

  for (let i = 0; i < meta.width * meta.height; i++) {
    const offset = i * 4

    // Get RGB values for the same pixel in both images
    const rW = dataWhite[offset]
    const gW = dataWhite[offset + 1]
    const bW = dataWhite[offset + 2]

    const rB = dataBlack[offset]
    const gB = dataBlack[offset + 1]
    const bB = dataBlack[offset + 2]

    // Calculate the distance between the two observed pixels
    const pixelDist = Math.sqrt(
      Math.pow(rW - rB, 2) +
      Math.pow(gW - gB, 2) +
      Math.pow(bW - bB, 2)
    )

    // THE FORMULA:
    // If the pixel is 100% opaque, it looks the same on Black and White (pixelDist = 0).
    // If the pixel is 100% transparent, it looks exactly like the backgrounds (pixelDist = bgDist).
    // Therefore:
    let alpha = 1 - (pixelDist / bgDist)

    // Clamp results to 0-1 range
    alpha = Math.max(0, Math.min(1, alpha))

    // Clean up noise: threshold very low alpha to zero
    // This handles slight variations between AI-generated images
    if (alpha < 0.15) {
      alpha = 0
    } else if (alpha < 0.3) {
      // Feather the threshold slightly
      alpha = (alpha - 0.15) / 0.15 * 0.3
    }

    // Color Recovery:
    // We use the image on black to recover the color, dividing by alpha
    // to un-premultiply it (brighten the semi-transparent pixels)
    let rOut = 0, gOut = 0, bOut = 0

    if (alpha > 0.01) {
       // Recover foreground color from the version on black
       // (C - (1-alpha) * BG) / alpha
       // Since BG is black (0,0,0), this simplifies to C / alpha
       rOut = rB / alpha
       gOut = gB / alpha
       bOut = bB / alpha
    }

    outputBuffer[offset] = Math.round(Math.min(255, rOut))
    outputBuffer[offset + 1] = Math.round(Math.min(255, gOut))
    outputBuffer[offset + 2] = Math.round(Math.min(255, bOut))
    outputBuffer[offset + 3] = Math.round(alpha * 255)
  }

  await sharp(outputBuffer, {
    raw: { width: meta.width, height: meta.height, channels: 4 }
  })
    .png()
    .toFile(outputPath)

  console.log(`✓ Extracted alpha: ${outputPath}`)
  console.log(`  Dimensions: ${meta.width}×${meta.height}`)
}

// CLI
const [,, whitePath, blackPath, outputPath] = process.argv

if (!whitePath || !blackPath || !outputPath) {
  console.log('Usage: npx tsx scripts/extract_alpha.ts <white.jpg> <black.jpg> <output.png>')
  console.log('')
  console.log('Difference matting: extracts true alpha from same image on white vs black backgrounds.')
  process.exit(1)
}

extractAlphaTwoPass(whitePath, blackPath, outputPath)
  .catch(err => {
    console.error('Error:', err.message)
    process.exit(1)
  })
