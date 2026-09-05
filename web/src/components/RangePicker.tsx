import { useEffect, useMemo, useState } from 'react'
import type { VocabAction, VocabObject, Vocabulary } from '../api/types'
import { formatShort } from '../lib/time'
import { objectsForAction } from '../lib/vocab'
import { LabelPicker } from './LabelPicker'
import './RangePicker.css'

interface Props {
  actions: VocabAction[]
  objects: VocabObject[]
  /** Какие объекты допустимы для действия; без карты — все. */
  pairs?: Vocabulary['pairs']
  /** Словарь открытый: в поле можно вписать своё значение. */
  allowFree: boolean
  /** Частые значения ролика — плашками под полями. */
  frequentActions: string[]
  frequentObjects: string[]
  x: number
  range: [number, number]
  /** carve — диапазон накрывает существующие сегменты, create — пустое место. */
  mode: 'carve' | 'create'
  onApply: (actionId: string, objectId: string, mode: 'carve' | 'create') => void
  onCancel: () => void
}

/**
 * Всплывающий выбор класса для только что выделенного диапазона.
 * Появляется сразу после отпускания мыши: вырезать середину и назвать её —
 * это один жест, а не поход в боковую панель.
 */
export function RangePicker({
  actions,
  objects,
  pairs,
  allowFree,
  frequentActions,
  frequentObjects,
  x,
  range,
  mode,
  onApply,
  onCancel,
}: Props) {
  const [actionId, setActionId] = useState(frequentActions[0] ?? actions[0]?.id ?? 'unknown')
  const [objectId, setObjectId] = useState(frequentObjects[0] ?? objects[0]?.id ?? 'unknown')
  const allowedObjects = useMemo(
    () => objectsForAction(objects, pairs, actionId, [objectId, 'unknown']),
    [objects, pairs, actionId, objectId],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      // Внутри поля ввода клавиши принадлежат полю: цифры и Enter там — текст и выбор.
      if (target && target.tagName === 'INPUT') return
      if (e.key === 'Escape') {
        e.stopPropagation()
        onCancel()
      }
      if (e.key === 'Enter') {
        e.stopPropagation()
        onApply(actionId, objectId, mode)
      }
      // Цифра выбирает действие по порядку — то же соответствие, что в инспекторе.
      const digit = Number(e.key)
      if (Number.isInteger(digit) && digit >= 1 && digit <= 9 && actions[digit - 1]) {
        setActionId(actions[digit - 1].id)
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [actionId, objectId, mode, actions, onApply, onCancel])

  const pickAction = (id: string) => {
    setActionId(id)
    // Пары словаря: если прежний объект для нового действия недопустим, берём первый допустимый.
    const allowed = pairs?.[id]
    if (allowed && !allowed.includes(objectId)) setObjectId(allowed[0] ?? 'unknown')
  }

  return (
    <div className="rpick" style={{ left: x }} onPointerDown={(e) => e.stopPropagation()}>
      <div>
        <div className="rpick__title">
          {mode === 'carve' ? 'Вырезать и переклассифицировать' : 'Новый сегмент'}
        </div>
        <div className="rpick__sub mono">
          {formatShort(range[0])} → {formatShort(range[1])} · {formatShort(range[1] - range[0])}
        </div>
      </div>

      <div>
        <div className="rpick__sub" style={{ marginBottom: 5 }}>Действие</div>
        <LabelPicker
          options={actions}
          value={actionId}
          onChange={pickAction}
          allowFree={allowFree}
          frequent={frequentActions}
          hotkeys
          placeholder="найти действие"
        />
      </div>

      <div>
        <div className="rpick__sub" style={{ marginBottom: 5 }}>Объект</div>
        <LabelPicker
          options={allowedObjects}
          value={objectId}
          onChange={setObjectId}
          allowFree={allowFree}
          frequent={frequentObjects}
          placeholder="найти объект"
        />
      </div>

      <div className="rpick__row">
        <button className="btn btn--sm" onClick={onCancel}>
          Отмена
        </button>
        <button className="btn btn--sm btn--primary" onClick={() => onApply(actionId, objectId, mode)}>
          {mode === 'carve' ? 'Вырезать' : 'Создать'} ⏎
        </button>
      </div>
    </div>
  )
}
