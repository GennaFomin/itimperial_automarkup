import { useToastStore } from '../store/toastStore'
import './Toaster.css'

export function Toaster() {
  const toasts = useToastStore((s) => s.toasts)
  const dismiss = useToastStore((s) => s.dismiss)
  if (toasts.length === 0) return null
  return (
    <div className="toaster" role="status" aria-live="polite">
      {toasts.map((t) => (
        <button key={t.id} className={`toast toast--${t.kind}`} onClick={() => dismiss(t.id)}>
          <div className="toast__msg">{t.message}</div>
          {t.lines && t.lines.length > 0 && (
            <ul className="toast__lines">
              {t.lines.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          )}
        </button>
      ))}
    </div>
  )
}
