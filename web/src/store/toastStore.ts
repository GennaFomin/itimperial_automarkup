/**
 * Уведомления вместо alert(): человек видит, что именно не так, и продолжает
 * работать. Список живёт в сторе, чтобы сообщать можно было откуда угодно —
 * из обработчика сохранения, из карточки задачи, из горячей клавиши.
 */
import { create } from 'zustand'

export type ToastKind = 'info' | 'warn' | 'error'

export interface Toast {
  id: number
  kind: ToastKind
  message: string
  /** Подробности второй строкой: список замечаний, текст ошибки. */
  lines?: string[]
  ttlMs: number
}

interface ToastState {
  toasts: Toast[]
  push: (toast: Omit<Toast, 'id' | 'ttlMs'> & { ttlMs?: number }) => number
  dismiss: (id: number) => void
}

const TTL: Record<ToastKind, number> = { info: 4000, warn: 8000, error: 10000 }
let nextId = 1

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (toast) => {
    const id = nextId++
    const ttlMs = toast.ttlMs ?? TTL[toast.kind]
    set((state) => ({ toasts: [...state.toasts, { ...toast, id, ttlMs }] }))
    window.setTimeout(() => {
      set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }))
    }, ttlMs)
    return id
  },
  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}))

const push = (kind: ToastKind) => (message: string, lines?: string[]) =>
  useToastStore.getState().push({ kind, message, lines })

/** Короткая форма для вызова вне React: `toast.error('Не сохранилось', [причина])`. */
export const toast = { info: push('info'), warn: push('warn'), error: push('error') }
