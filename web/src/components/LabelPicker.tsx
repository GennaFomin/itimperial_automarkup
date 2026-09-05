import { useEffect, useMemo, useRef, useState } from 'react'
import type { VocabAction, VocabObject } from '../api/types'
import './LabelPicker.css'

type Option = VocabAction | VocabObject

interface Props {
  /** Полный список словаря (плюс значения из разметки). Порядок задаёт цифры 1–9. */
  options: Option[]
  value: string
  onChange: (id: string) => void
  /** Словарь открытый: Enter с текстом, которого нет в списке, всё равно принимается. */
  allowFree: boolean
  /** Что показать плашками под полем: самые частые значения в этом ролике. */
  frequent: string[]
  /** Подсказки цифр у первых девяти вариантов — только для действий. */
  hotkeys?: boolean
  placeholder?: string
  autoFocus?: boolean
}

const CHIPS = 8

function labelOf(option: Option | undefined, id: string): string {
  return option?.label_ru ?? id
}

function colorOf(option: Option | undefined): string | null {
  return option && 'color' in option ? option.color : null
}

/**
 * Выбор метки при большом словаре: одно поле с автодополнением и несколько плашек
 * самых частых значений. Шестьдесят плашек в колонке были стеной, за которой
 * терялись границы и ключевой кадр; теперь весь список открывается по вводу.
 *
 * Клавиатура внутри поля: ↑/↓ по списку, Enter — выбрать (или принять свой текст,
 * если словарь открытый), Esc — закрыть и вернуть прежнее значение. Пока фокус в
 * поле, глобальные горячие клавиши редактора не срабатывают — это их правило.
 */
export function LabelPicker({
  options,
  value,
  onChange,
  allowFree,
  frequent,
  hotkeys = false,
  placeholder,
  autoFocus = false,
}: Props) {
  const byId = useMemo(() => new Map(options.map((o) => [o.id, o])), [options])
  const current = byId.get(value)
  const [text, setText] = useState(() => labelOf(current, value))
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // Значение поменяли снаружи (цифрой, плашкой, undo) — поле показывает его, а не старый ввод.
  useEffect(() => {
    if (!open) setText(labelOf(byId.get(value), value))
  }, [value, byId, open])

  const query = text.trim().toLowerCase()
  const matches = useMemo(() => {
    const currentLabel = labelOf(current, value).toLowerCase()
    // Пока текст равен подписи выбранного значения, фильтровать нечего — показываем всё.
    if (!query || query === currentLabel) return options.map((o, index) => ({ o, index }))
    return options
      .map((o, index) => ({ o, index }))
      .filter(({ o }) => o.label_ru.toLowerCase().includes(query) || o.id.toLowerCase().includes(query))
  }, [options, query, current, value])

  useEffect(() => setActive(0), [query])
  useEffect(() => {
    const row = listRef.current?.children[active] as HTMLElement | undefined
    row?.scrollIntoView({ block: 'nearest' })
  }, [active, open])

  const commit = (id: string) => {
    if (id && id !== value) onChange(id)
    setOpen(false)
    setText(labelOf(byId.get(id), id))
    inputRef.current?.blur()
  }

  const cancel = () => {
    setOpen(false)
    setText(labelOf(current, value))
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      e.stopPropagation()
      if (!open) setOpen(true)
      if (matches.length === 0) return
      const step = e.key === 'ArrowDown' ? 1 : -1
      setActive((i) => (i + step + matches.length) % matches.length)
      return
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      e.stopPropagation()
      const exact = options.find(
        (o) => o.label_ru.toLowerCase() === query || o.id.toLowerCase() === query,
      )
      const chosen = open && matches[active] ? matches[active].o : exact
      if (chosen) return commit(chosen.id)
      if (exact) return commit(exact.id)
      const free = text.trim()
      if (allowFree && free) return commit(free)
      return
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      e.stopPropagation()
      cancel()
      inputRef.current?.blur()
    }
  }

  const chips = useMemo(() => {
    const ids: string[] = []
    for (const id of [value, ...frequent]) {
      if (id && id !== 'unknown' && !ids.includes(id)) ids.push(id)
      if (ids.length >= CHIPS) break
    }
    return ids
  }, [value, frequent])

  const indexOf = (id: string) => options.findIndex((o) => o.id === id)

  return (
    <div className="lp">
      <div className="lp__field">
        <input
          ref={inputRef}
          className="input lp__input"
          value={text}
          placeholder={placeholder}
          autoFocus={autoFocus}
          role="combobox"
          aria-expanded={open}
          aria-autocomplete="list"
          onFocus={(e) => {
            setOpen(true)
            e.target.select()
          }}
          onBlur={() => window.setTimeout(cancel, 0)}
          onChange={(e) => {
            setText(e.target.value)
            setOpen(true)
          }}
          onKeyDown={onKeyDown}
        />
        {open && (
          <div className="lp__list" role="listbox" ref={listRef}>
            {matches.length === 0 && (
              <div className="lp__hint">
                {allowFree ? 'Нет в словаре — Enter примет ваш текст' : 'Ничего не найдено'}
              </div>
            )}
            {matches.map(({ o, index }, i) => {
              const color = colorOf(o)
              return (
                <div
                  key={o.id}
                  role="option"
                  aria-selected={i === active}
                  className={`lp__row${i === active ? ' lp__row--active' : ''}${o.id === value ? ' lp__row--on' : ''}`}
                  onMouseDown={(e) => {
                    e.preventDefault() // не отдавать фокус до commit
                    commit(o.id)
                  }}
                  onMouseEnter={() => setActive(i)}
                >
                  {color && <span className="chip__dot" style={{ background: color }} />}
                  <span className="lp__label">{o.label_ru}</span>
                  {o.unknown && !allowFree && <span className="lp__oov">вне словаря</span>}
                  {hotkeys && index < 9 && <span className="opt__key">{index + 1}</span>}
                </div>
              )
            })}
          </div>
        )}
      </div>
      {chips.length > 0 && (
        <div className="opts">
          {chips.map((id) => {
            const o = byId.get(id)
            const color = colorOf(o)
            const index = indexOf(id)
            return (
              <button
                key={id}
                type="button"
                className={`opt${id === value ? ' opt--on' : ''}`}
                onClick={() => commit(id)}
                title={o?.unknown && !allowFree ? 'Значения нет в словаре' : undefined}
              >
                {color && <span className="chip__dot" style={{ background: color }} />}
                {labelOf(o, id)}
                {hotkeys && index >= 0 && index < 9 && <span className="opt__key">{index + 1}</span>}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
