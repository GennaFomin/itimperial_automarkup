import { useEffect, useRef, useState } from 'react'
import './Menu.css'

export interface MenuItem {
  label: string
  onClick: () => void
  disabled?: boolean
  /** Черта над пунктом: отделяет группы. */
  divider?: boolean
}

interface Props {
  label: React.ReactNode
  items: MenuItem[]
  className?: string
  title?: string
}

/** Кнопка с выпадающим списком. Закрывается по Esc, клику вне и после выбора. */
export function Menu({ label, items, className = 'btn btn--sm', title }: Props) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="menu" ref={rootRef}>
      <button className={className} onClick={() => setOpen((v) => !v)} title={title} aria-haspopup="menu" aria-expanded={open}>
        {label}
      </button>
      {open && (
        <div className="menu__list" role="menu">
          {items.map((item, i) => (
            <div key={i}>
              {item.divider && <hr className="divider menu__divider" />}
              <button
                role="menuitem"
                className="btn btn--ghost btn--sm menu__item"
                disabled={item.disabled}
                onClick={() => {
                  setOpen(false)
                  item.onClick()
                }}
              >
                {item.label}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
