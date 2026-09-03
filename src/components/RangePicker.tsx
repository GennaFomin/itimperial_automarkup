import { useEffect, useState } from 'react'
import type { VocabAction, VocabObject } from '../api/types'
import { formatShort } from '../lib/time'
import './RangePicker.css'

interface Props {
  actions: VocabAction[]
  objects: VocabObject[]
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
export function RangePicker({ actions, objects, x, range, mode, onApply, onCancel }: Props) {
  const [actionId, setActionId] = useState(actions[0]?.id ?? 'unknown')
  const [objectId, setObjectId] = useState(objects[0]?.id ?? 'unknown')

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
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
        <div className="rpick__grid">
          {actions.map((a, i) => (
            <button
              key={a.id}
              className={`rpick__opt${a.id === actionId ? ' rpick__opt--on' : ''}`}
              onClick={() => setActionId(a.id)}
              title={i < 9 ? `Клавиша ${i + 1}` : undefined}
            >
              <span className="chip__dot" style={{ background: a.color }} />
              {a.label_ru}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="rpick__sub" style={{ marginBottom: 5 }}>Объект</div>
        <div className="rpick__grid">
          {objects.map((o) => (
            <button
              key={o.id}
              className={`rpick__opt${o.id === objectId ? ' rpick__opt--on' : ''}`}
              onClick={() => setObjectId(o.id)}
            >
              {o.label_ru}
            </button>
          ))}
        </div>
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
