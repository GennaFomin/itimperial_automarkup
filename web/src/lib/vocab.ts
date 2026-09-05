/**
 * Словарь и разметка: чистые функции без React, чтобы их можно было проверить
 * тестом и переиспользовать в инспекторе, в окне «Вырезать» и в шапке.
 */
import type { VocabAction, VocabObject, Vocabulary } from '../api/types'

/** Цвет для `unknown` и всего, у чего цвета нет: нейтральный, чтобы не спорить с палитрой. */
export const UNKNOWN_COLOR = '#9AA3AD'

/**
 * Устойчивый цвет для метки вне словаря — та же идея, что `color_for` на бэкенде:
 * при открытой лексике у ролика десяток своих глаголов, и серые одинаковые полосы
 * на таймлайне не давали отличить «взял» от «положил». Хэш строки задаёт тон,
 * яркость и насыщенность фиксированы, чтобы цвета сидели в одной палитре.
 */
export function colorFor(label: string): string {
  if (!label || label === 'unknown') return UNKNOWN_COLOR
  let hash = 2166136261
  for (let i = 0; i < label.length; i++) {
    hash ^= label.charCodeAt(i)
    hash = Math.imul(hash, 16777619) >>> 0
  }
  return `hsl(${hash % 360} 62% 58%)`
}

interface Labelled {
  action: string
  object: string
}

/**
 * Значения, которых нет в словаре задачи, не подменяем на unknown и не роняем
 * интерфейс (контракт §1): добавляем в списки как есть, с пометкой `unknown`.
 * Берём их из текущей разметки, а не только из прогноза: свободный ввод человека
 * тоже должен получить плашку и цвет.
 */
export function mergeVocab(
  actions: VocabAction[],
  objects: VocabObject[],
  segments: Labelled[],
): { actions: VocabAction[]; objects: VocabObject[] } {
  const actionIds = new Set(actions.map((a) => a.id))
  const objectIds = new Set(objects.map((o) => o.id))
  const extraActions: VocabAction[] = []
  const extraObjects: VocabObject[] = []
  for (const seg of segments) {
    if (seg.action && !actionIds.has(seg.action)) {
      actionIds.add(seg.action)
      extraActions.push({ id: seg.action, label_ru: seg.action, color: colorFor(seg.action), unknown: true })
    }
    if (seg.object && !objectIds.has(seg.object)) {
      objectIds.add(seg.object)
      extraObjects.push({ id: seg.object, label_ru: seg.object, unknown: true })
    }
  }
  return { actions: [...actions, ...extraActions], objects: [...objects, ...extraObjects] }
}

export interface OutOfVocab {
  segmentId: string
  field: 'action' | 'object'
  value: string
}

/**
 * Что в разметке не сходится со словарём. При открытом словаре — ничего: модель
 * и человек отвечают своими словами, и это норма, а не предупреждение.
 */
export function outOfVocab(
  segments: Array<Labelled & { id: string }>,
  vocab: Vocabulary | null,
): OutOfVocab[] {
  if (!vocab || vocab.open) return []
  const actionIds = new Set(vocab.actions.map((a) => a.id))
  const objectIds = new Set(vocab.objects.map((o) => o.id))
  const result: OutOfVocab[] = []
  for (const seg of segments) {
    if (seg.action && !actionIds.has(seg.action)) {
      result.push({ segmentId: seg.id, field: 'action', value: seg.action })
    }
    if (seg.object && !objectIds.has(seg.object)) {
      result.push({ segmentId: seg.id, field: 'object', value: seg.object })
    }
  }
  return result
}

/**
 * Самые частые значения в этом ролике: у настольной съёмки один и тот же десяток
 * слов повторяется, и именно их надо держать под рукой, а не весь список.
 * При равной частоте раньше идёт то, что встретилось первым.
 */
export function frequentValues(
  segments: Labelled[],
  field: 'action' | 'object',
  limit = 8,
): string[] {
  const counts = new Map<string, number>()
  for (const seg of segments) {
    const value = seg[field]
    if (!value || value === 'unknown') continue
    counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([value]) => value)
}

/**
 * Объекты, допустимые для действия по карте `pairs` словаря. Текущее значение и
 * `unknown` остаются всегда: иначе выбранный объект пропал бы из списка.
 * Без `pairs` — весь список.
 */
export function objectsForAction(
  objects: VocabObject[],
  pairs: Vocabulary['pairs'] | undefined,
  actionId: string,
  keep: string[],
): VocabObject[] {
  const allowed = pairs?.[actionId]
  if (!allowed) return objects
  const keepSet = new Set([...allowed, ...keep.filter(Boolean)])
  return objects.filter((o) => keepSet.has(o.id))
}
