import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from './api'

const IDLE_MS = 30_000
const TICK_MS = 1000

/** Сколько человек реально работал над роликом.
 *
 * Считаются только активные секунды: вкладка на экране и была активность за последние
 * тридцать секунд. Это число — вся доказательная база для KPI «в три раза быстрее»,
 * поэтому оно не должно расти, пока пользователь пьёт кофе.
 */
export function useReviewTimer(videoId: string) {
  const [seconds, setSeconds] = useState(0)
  const lastActivity = useRef(Date.now())
  const reported = useRef(0)

  useEffect(() => {
    const touch = () => {
      lastActivity.current = Date.now()
    }
    const events = ['mousemove', 'mousedown', 'keydown', 'wheel', 'touchstart']
    events.forEach((event) => window.addEventListener(event, touch, { passive: true }))

    const interval = window.setInterval(() => {
      const active = !document.hidden && Date.now() - lastActivity.current < IDLE_MS
      if (active) setSeconds((value) => value + TICK_MS / 1000)
    }, TICK_MS)

    return () => {
      events.forEach((event) => window.removeEventListener(event, touch))
      window.clearInterval(interval)
    }
  }, [])

  const report = useCallback(
    (extra: Record<string, unknown> = {}) => {
      const delta = seconds - reported.current
      if (delta < 1) return
      reported.current = seconds
      void api.logEvent(videoId, 'review_seconds', { seconds: Math.round(delta), ...extra })
    },
    [seconds, videoId],
  )

  const latest = useRef(report)
  latest.current = report
  useEffect(() => () => latest.current({ reason: 'leave' }), [])

  return { seconds, report }
}
