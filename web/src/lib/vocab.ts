/**
 * Словарь и разметка: чистые функции без React, чтобы их можно было проверить
 * тестом и переиспользовать в инспекторе, в окне «Вырезать» и в шапке.
 */
import type { VocabAction, VocabObject, Vocabulary } from '../api/types'

/** Цвет для значения, которого нет в словаре: нейтральный, чтобы не спорить с палитрой. */
export const UNKNOWN_COLOR = '#9AA3AD'

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
      extraActions.push({ id: seg.action, label_ru: seg.action, color: UNKNOWN_COLOR, unknown: true })
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
