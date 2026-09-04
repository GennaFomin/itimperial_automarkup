/**
 * Сколько человек реально работал над роликом.
 *
 * Перенесено из редактора пайплайна (`web/src/useReviewTimer.ts`, коммит f599cf1)
 * с двумя добавлениями: периодический сброс на сервер и отправка при уходе
 * вкладки в фон. Оригинал отправлял замер только при размонтировании, и закрытая
 * вкладка теряла его целиком — а это единственная доказательная база для KPI
 * «в три раза быстрее».
 *
 * Считаются только активные секунды: вкладка на экране и была активность за
 * последние тридцать секунд. Иначе счётчик рос бы, пока человек пьёт кофе, и
 * метрика показывала бы обратное тому, что мы хотим доказать.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { reportActivity } from '../api/client'

const IDLE_MS = 30_000
const TICK_MS = 1000
/** Как часто сбрасывать накопленное на сервер, не дожидаясь ухода со страницы. */
const FLUSH_MS = 30_000

const ACTIVITY_EVENTS = ['mousemove', 'mousedown', 'keydown', 'wheel', 'touchstart'] as const

export function useActivityTimer(jobId: string | null, mode: 'review' | 'scratch') {
  const [seconds, setSeconds] = useState(0)
  const lastActivity = useRef(Date.now())
  const reported = useRef(0)

  useEffect(() => {
    const touch = () => {
      lastActivity.current = Date.now()
    }
    ACTIVITY_EVENTS.forEach((event) =>
      window.addEventListener(event, touch, { passive: true }),
    )

    const interval = window.setInterval(() => {
      // Пока задача не готова, работать не над чем: без этой проверки ожидание
      // авторазметки копилось бы в счётчике и целиком уходило на сервер первым
      // же отчётом, завышая ровно ту метрику, ради которой замер и делается.
      if (!jobId) return
      const active = !document.hidden && Date.now() - lastActivity.current < IDLE_MS
      if (active) setSeconds((value) => value + TICK_MS / 1000)
    }, TICK_MS)

    return () => {
      ACTIVITY_EVENTS.forEach((event) => window.removeEventListener(event, touch))
      window.clearInterval(interval)
    }
  }, [jobId])

  /** Отправляем дельту, а не сумму: сервер складывает события сам. */
  const report = useCallback(() => {
    if (!jobId) return
    const delta = seconds - reported.current
    if (delta < 1) return
    reported.current = seconds
    void reportActivity(jobId, mode, Math.round(delta))
  }, [jobId, mode, seconds])

  // Замыкание в ref: обработчики ниже вешаются один раз, а report меняется
  // каждую секунду вместе с seconds.
  const latest = useRef(report)
  latest.current = report

  useEffect(() => {
    const flush = () => latest.current()
    const onHidden = () => {
      if (document.hidden) flush()
    }
    const timer = window.setInterval(flush, FLUSH_MS)
    document.addEventListener('visibilitychange', onHidden)
    window.addEventListener('pagehide', flush)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onHidden)
      window.removeEventListener('pagehide', flush)
      flush()
    }
  }, [])

  return { seconds, report }
}
