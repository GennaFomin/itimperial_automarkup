import assert from 'node:assert/strict'
import { createHash, randomBytes } from 'node:crypto'
import { sha256Hex } from '../src/lib/sha256'

assert.equal(sha256Hex(new Uint8Array()), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
assert.equal(sha256Hex(new TextEncoder().encode('abc')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
for (const size of [55, 56, 64, 1000, 4 * 1024 * 1024 + 3]) {
  const data = randomBytes(size)
  assert.equal(sha256Hex(new Uint8Array(data)), createHash('sha256').update(data).digest('hex'), `size ${size}`)
}
console.log('sha256: ok')
