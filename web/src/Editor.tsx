import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import * as api from './api'
import { StepList } from './StepList'
import { Timeline } from './Timeline'
import { motionMinima } from './motion'
import {
  deleteStep,
  mergeWithNext,
  moveBoundary,
  nextUnverified,
  setVerified,
  sortSteps,
  splitStep,
  updateStep,
  verifiedCount,
  verifyAll,
  withSteps,
} from './steps'
import type { Step, VideoRecord, Vocabulary } from './types'
import { useAnnotation } from './useAnnotation'
import { useReviewTimer } from './useReviewTimer'

const SAVE_LABEL: Record<string, string> = {
  loading: 'загрузка…',
  clean: 'сохранено',
  dirty: 'есть правки',
  saving: 'сохраняю…',
  error: 'ошибка сохранения',
}

export function Editor({ videoId, onBack }: { videoId: string; onBack: () => void }) {
  const [record, setRecord] = useState<VideoRecord | null>(null)

  useEffect(() => {
    let stop = false
    const poll = async () => {
      const next = await api.getVideo(videoId)
      if (stop) return
      setRecord(next)
      if (next.status === 'queued' || next.status === 'processing') setTimeout(poll, 1000)
    }
    void poll()
    return () => {
      stop = true
    }
  }, [videoId])

  if (!record) return <div className="pending">загрузка ролика…</div>
  if (record.status === 'failed')
    return (
      <div className="pending error">
        обработка не удалась: {record.error}
        <button onClick={onBack}>назад</button>
      </div>
    )
  if (record.status !== 'done') return <div className="pending">размечаю ролик…</div>
  return <EditorBody record={record} onBack={onBack} />
}

function EditorBody({ record, onBack }: { record: VideoRecord; onBack: () => void }) {
  const videoId = record.id
  const video = useRef<HTMLVideoElement>(null)
  const [vocabulary, setVocabulary] = useState<Vocabulary | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const { annotation, problems, saveState, editCount, history, apply, undo, redo, save } =
    useAnnotation(videoId)
  const { seconds, report } = useReviewTimer(videoId)

  useEffect(() => {
    void api.getVocabulary().then(setVocabulary)
  }, [])

  const candidates = useMemo(
    () => motionMinima(record.motion, record.duration_sec),
    [record.motion, record.duration_sec],
  )

  const steps = annotation ? sortSteps(annotation.steps) : []
  const checked = verifiedCount(steps)

  useEffect(() => {
    if (selectedId === null && steps.length) setSelectedId(steps[0].id)
  }, [selectedId, steps])

  const seek = useCallback((time: number) => {
    const element = video.current
    if (!element) return
    element.currentTime = Math.max(0, time)
    setCurrentTime(Math.max(0, time))
  }, [])

  useEffect(() => {
    let frame = 0
    const tick = () => {
      const element = video.current
      if (element && !element.paused) setCurrentTime(element.currentTime)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [])

  const mutateSteps = useCallback(
    (change: (steps: Step[]) => Step[] | null) =>
      apply((current) => {
        const next = change(current.steps)
        return next ? withSteps(current, next) : null
      }),
    [apply],
  )

  const onMoveBoundary = useCallback(
    (leftId: number, time: number) => mutateSteps((all) => moveBoundary(all, leftId, time)),
    [mutateSteps],
  )

  const onSplit = useCallback(
    (id: number) => mutateSteps((all) => splitStep(all, id, currentTime)),
    [mutateSteps, currentTime],
  )

  const onMerge = useCallback((id: number) => mutateSteps((all) => mergeWithNext(all, id)), [mutateSteps])

  const onDelete = useCallback(
    (id: number) => {
      mutateSteps((all) => deleteStep(all, id))
      setSelectedId(null)
    },
    [mutateSteps],
  )

  const onUpdate = useCallback(
    (id: number, patch: Partial<Step>) => mutateSteps((all) => updateStep(all, id, patch)),
    [mutateSteps],
  )

  const onKeyframeHere = useCallback(
    (id: number) => mutateSteps((all) => updateStep(all, id, { keyframe_sec: round3(currentTime) })),
    [mutateSteps, currentTime],
  )

  const onVerify = useCallback(
    (id: number, verified: boolean) => mutateSteps((all) => setVerified(all, id, verified)),
    [mutateSteps],
  )

  const onVerifyAll = useCallback(() => mutateSteps((all) => verifyAll(all)), [mutateSteps])

  /** Перейти к следующему непроверенному шагу и встать плеером на его начало. */
  const goToNextUnverified = useCallback(() => {
    const target = nextUnverified(steps, selectedId)
    if (!target) return
    setSelectedId(target.id)
    seek(target.start_sec)
  }, [steps, selectedId, seek])

  /** Разбор ролика: подтвердить текущий шаг и сразу прыгнуть на следующий непроверенный. */
  const verifyAndAdvance = useCallback(() => {
    if (selectedId === null) return
    const current = steps.find((step) => step.id === selectedId)
    if (current?.verified) {
      onVerify(selectedId, false)
      return
    }
    onVerify(selectedId, true)
    const target = nextUnverified(steps, selectedId)
    if (target && target.id !== selectedId) {
      setSelectedId(target.id)
      seek(target.start_sec)
    }
  }, [selectedId, steps, onVerify, seek])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return

      const element = video.current
      const index = steps.findIndex((step) => step.id === selectedId)
      const frame = 1 / (record.fps || 30)

      if (event.key === ' ') {
        event.preventDefault()
        if (element?.paused) void element.play()
        else element?.pause()
        return
      }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault()
        const delta = (event.shiftKey ? 1 : frame) * (event.key === 'ArrowLeft' ? -1 : 1)
        seek(Math.min(Math.max(currentTime + delta, 0), record.duration_sec))
        return
      }
      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        event.preventDefault()
        if (!steps.length) return
        const next = Math.min(
          Math.max(index + (event.key === 'ArrowDown' ? 1 : -1), 0),
          steps.length - 1,
        )
        setSelectedId(steps[next].id)
        seek(steps[next].start_sec)
        return
      }
      if (event.key === 'Tab') {
        event.preventDefault()
        goToNextUnverified()
        return
      }
      if (event.ctrlKey && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        event.shiftKey ? redo() : undo()
        return
      }
      if (selectedId === null) return

      switch (event.key.toLowerCase()) {
        case '[':
          if (index > 0) onMoveBoundary(steps[index - 1].id, currentTime)
          break
        case ']':
          if (index < steps.length - 1) onMoveBoundary(steps[index].id, currentTime)
          break
        case 's':
          onSplit(selectedId)
          break
        case 'm':
          onMerge(selectedId)
          break
        case 'k':
          onKeyframeHere(selectedId)
          break
        case 'v':
          verifyAndAdvance()
          break
        case 'delete':
        case 'backspace':
          event.preventDefault()
          onDelete(selectedId)
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [
    steps,
    selectedId,
    currentTime,
    record.fps,
    record.duration_sec,
    seek,
    undo,
    redo,
    onMoveBoundary,
    onSplit,
    onMerge,
    onDelete,
    onKeyframeHere,
    goToNextUnverified,
    verifyAndAdvance,
  ])

  const onExport = (format: 'json' | 'csv') => {
    void save()
    report({ reason: `export-${format}` })
  }

  return (
    <div className="editor">
      <header className="toolbar">
        <button onClick={onBack}>← ролики</button>
        <strong>{record.filename}</strong>
        <span className="muted">
          {record.width}×{record.height} · {record.duration_sec.toFixed(1)} с ·{' '}
          {record.fps.toFixed(0)} fps
        </span>
        <span className="grow" />
        <span className={`progress ${checked === steps.length && steps.length ? 'done' : ''}`}>
          проверено {checked} из {steps.length}
        </span>
        <button onClick={onVerifyAll} disabled={checked === steps.length} title="Подтвердить все оставшиеся шаги">
          подтвердить остальные
        </button>
        <span className="muted">проверка: {formatTime(seconds)}</span>
        <span className="muted">правок: {editCount}</span>
        <button onClick={undo} disabled={!history.undo}>
          ↶
        </button>
        <button onClick={redo} disabled={!history.redo}>
          ↷
        </button>
        <span className={`save ${saveState}`}>{SAVE_LABEL[saveState]}</span>
        <a className="button" href={api.exportUrl(videoId, 'json')} onClick={() => onExport('json')}>
          JSON
        </a>
        <a className="button" href={api.exportUrl(videoId, 'csv')} onClick={() => onExport('csv')}>
          CSV
        </a>
      </header>

      {problems.length > 0 && (
        <div className="banner">
          {problems.length} значений вне словаря: {problems.slice(0, 3).join('; ')}
        </div>
      )}

      <div className="stage">
        <div className="player">
          <video
            ref={video}
            src={api.mediaUrl(videoId)}
            controls
            onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
          />
          {annotation && (
            <Timeline
              videoId={videoId}
              duration={record.duration_sec}
              steps={annotation.steps}
              motion={record.motion}
              filmstrip={record.filmstrip}
              candidates={candidates}
              currentTime={currentTime}
              selectedId={selectedId}
              onSeek={seek}
              onSelect={setSelectedId}
              onMoveBoundary={onMoveBoundary}
            />
          )}
          <div className="hints">
            space — играть · ←/→ — кадр (shift — секунда) · ↑/↓ — шаг · [ и ] — границы шага ·
            V — проверено и дальше · Tab — следующий непроверенный · S — разделить · M — слить ·
            K — ключевой кадр · Del — удалить · Alt при перетаскивании отключает магнит
          </div>
          {annotation && (
            <div className="provenance">
              пайплайн {annotation.provenance.pipeline} · словарь{' '}
              {annotation.provenance.vocabulary} · бэкенд {annotation.provenance.backend} ·
              обработка {annotation.provenance.processing_sec?.toFixed(1)} с
            </div>
          )}
        </div>

        <aside className="sidebar">
          {annotation && (
            <StepList
              videoId={videoId}
              steps={annotation.steps}
              vocabulary={vocabulary}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onSeek={seek}
              onUpdate={onUpdate}
              onSplit={onSplit}
              onMerge={onMerge}
              onDelete={onDelete}
              onKeyframeHere={onKeyframeHere}
              onVerify={onVerify}
            />
          )}
        </aside>
      </div>
    </div>
  )
}

const round3 = (value: number) => Math.round(value * 1000) / 1000

function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}
