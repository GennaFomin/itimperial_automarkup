import assert from 'node:assert/strict'
import {
  carveOut, splitSegment, moveBoundary, moveSegment, createSegment,
  mergeWithNext, validate, diffAgainstPrediction, type EditableSegment,
} from '../src/lib/segments'

const DUR = 60_000
const seg = (id: string, a: number, b: number, action = 'move'): EditableSegment => ({
  id, origin: 'model', start_ms: a, end_ms: b, action, object: 'part',
  keyframe_ms: Math.round((a + b) / 2),
  boundary_confidence: 0.9, action_confidence: 0.8, object_confidence: 0.7, edited: false,
})

const ok = (s: EditableSegment[], msg: string) => {
  const issues = validate(s, DUR)
  assert.equal(issues.length, 0, `${msg}: ${issues.map(i => i.message).join(', ')}`)
}

let passed = 0
const test = (name: string, fn: () => void) => {
  fn(); passed++; console.log('  ✓', name)
}

console.log('splitSegment')
test('режет на две части, сумма длин сохраняется', () => {
  const src = [seg('a', 1000, 5000)]
  const r = splitSegment(src, 'a', 3000)
  assert.equal(r.segments.length, 2)
  assert.deepEqual(r.segments.map(s => [s.start_ms, s.end_ms]), [[1000, 3000], [3000, 5000]])
  ok(r.segments, 'после split')
})
test('не режет вплотную к границе', () => {
  const src = [seg('a', 1000, 5000)]
  assert.equal(splitSegment(src, 'a', 1050).segments, src)
  assert.equal(splitSegment(src, 'a', 4990).segments, src)
})
test('keyframe остаётся внутри своей половины', () => {
  const src = [seg('a', 1000, 5000)] // keyframe 3000
  const r = splitSegment(src, 'a', 2000)
  assert.ok(r.segments.every(s => s.keyframe_ms! >= s.start_ms && s.keyframe_ms! <= s.end_ms))
})

console.log('carveOut — вырезание середины')
test('вырез внутри одного сегмента даёт три', () => {
  const r = carveOut([seg('a', 0, 10_000)], 4000, 6000, 'pick', 'tool')
  assert.equal(r.segments.length, 3)
  assert.deepEqual(r.segments.map(s => [s.start_ms, s.end_ms]), [[0, 4000], [4000, 6000], [6000, 10_000]])
  assert.equal(r.segments[1].action, 'pick')
  assert.equal(r.segments[1].origin, 'human')
  ok(r.segments, 'после carve внутри')
})
test('вырез через несколько сегментов подрезает края и удаляет накрытые', () => {
  const src = [seg('a', 0, 3000), seg('b', 3000, 6000), seg('c', 6000, 9000)]
  const r = carveOut(src, 2000, 7000, 'place', 'tray')
  assert.deepEqual(r.segments.map(s => [s.start_ms, s.end_ms]), [[0, 2000], [2000, 7000], [7000, 9000]])
  assert.equal(r.segments.find(s => s.id === 'b'), undefined, 'полностью накрытый удалён')
  ok(r.segments, 'после carve через несколько')
})
test('вырез по пустому месту просто создаёт сегмент', () => {
  const src = [seg('a', 0, 2000), seg('b', 8000, 9000)]
  const r = carveOut(src, 4000, 6000, 'pick', 'part')
  assert.equal(r.segments.length, 3)
  ok(r.segments, 'после carve в дыре')
})
test('слишком короткий вырез игнорируется', () => {
  const src = [seg('a', 0, 10_000)]
  assert.equal(carveOut(src, 4000, 4050, 'pick', 'part').newId, null)
})
test('огрызок короче минимума не остаётся', () => {
  const r = carveOut([seg('a', 0, 5000)], 50, 5000, 'pick', 'part')
  assert.ok(r.segments.every(s => s.end_ms - s.start_ms >= 120))
  ok(r.segments, 'огрызки')
})

console.log('moveBoundary')
test('упирается в соседа, не толкая его', () => {
  const src = [seg('a', 0, 3000), seg('b', 4000, 6000)]
  const r = moveBoundary(src, 'b', 'start', 1000, DUR)
  assert.equal(r.find(s => s.id === 'b')!.start_ms, 3000)
  assert.equal(r.find(s => s.id === 'a')!.end_ms, 3000, 'сосед не сдвинулся')
  ok(r, 'после moveBoundary')
})
test('не схлопывает сегмент короче минимума', () => {
  const r = moveBoundary([seg('a', 0, 3000)], 'a', 'end', 10, DUR)
  assert.equal(r[0].end_ms, 120)
})
test('не вылезает за длительность', () => {
  const r = moveBoundary([seg('a', 0, 3000)], 'a', 'end', 999_999, DUR)
  assert.equal(r[0].end_ms, DUR)
  ok(r, 'граница у конца')
})

console.log('moveSegment')
test('сохраняет длину и двигает keyframe', () => {
  const src = [seg('a', 1000, 3000)]
  const r = moveSegment(src, 'a', 5000, DUR)
  assert.deepEqual([r[0].start_ms, r[0].end_ms], [5000, 7000])
  assert.equal(r[0].keyframe_ms, 6000)
})
test('не наезжает на соседа', () => {
  const src = [seg('a', 0, 2000), seg('b', 3000, 4000)]
  const r = moveSegment(src, 'a', 2500, DUR)
  assert.ok(r.find(s => s.id === 'a')!.end_ms <= 3000)
  ok(r, 'после moveSegment')
})

console.log('createSegment')
test('в дыре создаёт, поверх сегмента — нет', () => {
  const src = [seg('a', 0, 2000), seg('b', 5000, 7000)]
  assert.ok(createSegment(src, 3000, 4000, 'pick', 'part', DUR).newId)
  assert.equal(createSegment(src, 1000, 4000, 'pick', 'part', DUR).newId, null)
})

console.log('mergeWithNext')
test('поглощает дыру между сегментами', () => {
  const src = [seg('a', 0, 2000), seg('b', 5000, 7000)]
  const r = mergeWithNext(src, 'a')
  assert.equal(r.length, 1)
  assert.deepEqual([r[0].start_ms, r[0].end_ms], [0, 7000])
  ok(r, 'после merge')
})

console.log('validate')
test('ловит пересечение', () => {
  const bad = [seg('a', 0, 5000), seg('b', 3000, 7000)]
  assert.ok(validate(bad, DUR).some(i => i.message.includes('Пересекается')))
})
test('ловит keyframe вне сегмента', () => {
  const s = seg('a', 0, 1000); s.keyframe_ms = 5000
  assert.ok(validate([s], DUR).some(i => i.message.includes('Ключевой кадр')))
})

console.log('diffAgainstPrediction')
test('считает правки по полям', () => {
  const pred = [{
    id: 'a', start_ms: 0, end_ms: 3000, boundary_confidence: 0.9,
    action: { value: 'move', confidence: 0.8 }, object: { value: 'part', confidence: 0.7 },
    keyframe_ms: 1500, keyframe_confidence: 0.6,
  }, {
    id: 'b', start_ms: 4000, end_ms: 6000, boundary_confidence: 0.9,
    action: { value: 'pick', confidence: 0.8 }, object: { value: 'tray', confidence: 0.7 },
    keyframe_ms: 5000, keyframe_confidence: 0.6,
  }]
  const cur = [
    { ...seg('a', 0, 3500), action: 'place' },  // граница + действие
    seg('c', 8000, 9000),                        // добавлен, 'b' удалён
  ]
  const d = diffAgainstPrediction(pred as never, cur)
  assert.deepEqual(d, {
    boundaries_edited: 1, actions_changed: 1, objects_changed: 0,
    keyframes_moved: 1, segments_added: 1, segments_deleted: 1, segments_untouched: 0,
  })
})

console.log(`\n${passed} тестов пройдено`)
