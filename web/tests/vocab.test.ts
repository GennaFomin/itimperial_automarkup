import assert from 'node:assert/strict'
import { frequentValues, mergeVocab, objectsForAction, outOfVocab } from '../src/lib/vocab'

const actions = [
  { id: 'take', label_ru: 'взял', color: '#111' },
  { id: 'put', label_ru: 'положил', color: '#222' },
]
const objects = [
  { id: 'cup', label_ru: 'чашка' },
  { id: 'bottle', label_ru: 'бутылка' },
  { id: 'unknown', label_ru: 'Неизвестно' },
]
const segs = [
  { id: 's1', action: 'take', object: 'cup' },
  { id: 's2', action: 'повернул', object: 'cup' },
  { id: 's3', action: 'take', object: 'губка' },
  { id: 's4', action: 'put', object: 'unknown' },
]

// mergeVocab: чужие значения добавляются с пометкой, порядок словаря не меняется
{
  const merged = mergeVocab(actions, objects, segs)
  assert.deepEqual(merged.actions.map((a) => a.id), ['take', 'put', 'повернул'])
  assert.equal(merged.actions[2].unknown, true)
  assert.equal(merged.actions[0].unknown, undefined)
  assert.deepEqual(merged.objects.map((o) => o.id), ['cup', 'bottle', 'unknown', 'губка'])
}

// outOfVocab: при открытом словаре пусто, при закрытом — по полям
{
  const closed = { version: '1', actions, objects, open: false }
  assert.deepEqual(outOfVocab(segs, { ...closed, open: true }), [])
  assert.deepEqual(outOfVocab(segs, null), [])
  assert.deepEqual(outOfVocab(segs, closed), [
    { segmentId: 's2', field: 'action', value: 'повернул' },
    { segmentId: 's3', field: 'object', value: 'губка' },
  ])
}

// frequentValues: по убыванию частоты, unknown не считается, лимит соблюдается
{
  assert.deepEqual(frequentValues(segs, 'action'), ['take', 'повернул', 'put'])
  assert.deepEqual(frequentValues(segs, 'object'), ['cup', 'губка'])
  assert.deepEqual(frequentValues(segs, 'action', 1), ['take'])
}

// objectsForAction: без карты — всё; с картой — разрешённые плюс текущее и unknown
{
  assert.equal(objectsForAction(objects, undefined, 'take', ['cup']), objects)
  assert.equal(objectsForAction(objects, null, 'take', ['cup']), objects)
  const pairs = { take: ['bottle'] }
  assert.deepEqual(
    objectsForAction(objects, pairs, 'take', ['cup', 'unknown']).map((o) => o.id),
    ['cup', 'bottle', 'unknown'],
  )
  // Действие вне карты — весь список.
  assert.equal(objectsForAction(objects, pairs, 'put', []), objects)
}

console.log('vocab: ok')
