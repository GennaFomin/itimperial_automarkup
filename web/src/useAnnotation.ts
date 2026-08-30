import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from './api'
import type { Annotation } from './types'

export type SaveState = 'loading' | 'clean' | 'dirty' | 'saving' | 'error'

const AUTOSAVE_MS = 800
const HISTORY_LIMIT = 100

/** Состояние разметки: правки, история, автосохранение и счётчик изменений. */
export function useAnnotation(videoId: string) {
  const [annotation, setAnnotation] = useState<Annotation | null>(null)
  const [problems, setProblems] = useState<string[]>([])
  const [saveState, setSaveState] = useState<SaveState>('loading')
  const [editCount, setEditCount] = useState(0)
  const [history, setHistory] = useState({ undo: 0, redo: 0 })

  const current = useRef<Annotation | null>(null)
  const undoStack = useRef<Annotation[]>([])
  const redoStack = useRef<Annotation[]>([])
  const timer = useRef<number | undefined>(undefined)

  const put = useCallback((next: Annotation) => {
    current.current = next
    setAnnotation(next)
    setSaveState('dirty')
    setHistory({ undo: undoStack.current.length, redo: redoStack.current.length })
  }, [])

  useEffect(() => {
    let cancelled = false
    setSaveState('loading')
    api
      .getAnnotation(videoId)
      .then(({ annotation: loaded, problems: found }) => {
        if (cancelled) return
        current.current = loaded
        undoStack.current = []
        redoStack.current = []
        setAnnotation(loaded)
        setProblems(found)
        setSaveState('clean')
      })
      .catch(() => !cancelled && setSaveState('error'))
    return () => {
      cancelled = true
    }
  }, [videoId])

  /** Единственная точка изменения разметки: пишет историю и считает правки. */
  const apply = useCallback(
    (mutate: (annotation: Annotation) => Annotation | null) => {
      const previous = current.current
      if (!previous) return
      const next = mutate(previous)
      if (!next || next === previous) return

      undoStack.current.push(previous)
      if (undoStack.current.length > HISTORY_LIMIT) undoStack.current.shift()
      redoStack.current = []
      setEditCount((count) => count + 1)
      put(next)
    },
    [put],
  )

  const undo = useCallback(() => {
    const previous = undoStack.current.pop()
    if (!previous || !current.current) return
    redoStack.current.push(current.current)
    put(previous)
  }, [put])

  const redo = useCallback(() => {
    const next = redoStack.current.pop()
    if (!next || !current.current) return
    undoStack.current.push(current.current)
    put(next)
  }, [put])

  const save = useCallback(async () => {
    const value = current.current
    if (!value) return
    setSaveState('saving')
    try {
      const result = await api.saveAnnotation(videoId, value)
      setProblems(result.problems)
      setSaveState(current.current === value ? 'clean' : 'dirty')
    } catch {
      setSaveState('error')
    }
  }, [videoId])

  useEffect(() => {
    if (saveState !== 'dirty') return
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(save, AUTOSAVE_MS)
    return () => window.clearTimeout(timer.current)
  }, [saveState, annotation, save])

  return { annotation, problems, saveState, editCount, history, apply, undo, redo, save }
}
