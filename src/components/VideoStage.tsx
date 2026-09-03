import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { segmentAt } from '../lib/segments'
import { formatPrecise } from '../lib/time'
import type { VocabAction, VocabObject } from '../api/types'
import './VideoStage.css'

const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4]

export interface VideoHandle {
  seek: (ms: number) => void
  togglePlay: () => void
  play: () => void
  pause: () => void
  stepFrames: (frames: number) => void
}

interface Props {
  src: string | null
  fileName: string | null
  fps: number
  actions: VocabAction[]
  objects: VocabObject[]
  /** Пользователь выбрал файл заново — после перезагрузки blob-URL теряется. */
  onPickFile: (file: File) => void
}

/**
 * Плеер и транспорт. Единственный владелец времени воспроизведения: playhead в
 * сторе — зеркало текущей позиции, а не независимое состояние.
 *
 * Когда файла нет (вкладку перезагрузили — blob-URL умер), плеер честно
 * сообщает об этом и продолжает работать как «виртуальные часы»: таймлайн,
 * правки и экспорт остаются полностью рабочими.
 */
export const VideoStage = forwardRef<VideoHandle, Props>(function VideoStage(
  { src, fileName, fps, actions, objects, onPickFile },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const durationMs = useEditorStore((s) => s.durationMs)
  const playheadMs = useEditorStore((s) => s.playheadMs)
  const setPlayhead = useEditorStore((s) => s.setPlayhead)
  const segments = useEditorStore((s) => s.segments)

  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  /** Виртуальные часы, когда видео нет: id кадра анимации. */
  const clockRef = useRef<number | null>(null)
  const clockStateRef = useRef({ lastTs: 0 })

  useImperativeHandle(ref, () => ({
    seek: (ms) => {
      setPlayhead(ms)
      const v = videoRef.current
      if (v) v.currentTime = Math.max(0, ms / 1000)
    },
    togglePlay: () => (playing ? pause() : play()),
    play,
    pause,
    stepFrames: (frames) => {
      pause()
      const step = 1000 / (fps || 30)
      setPlayhead(playheadMs + frames * step)
      const v = videoRef.current
      if (v) v.currentTime = Math.max(0, (playheadMs + frames * step) / 1000)
    },
  }))

  function play() {
    const v = videoRef.current
    if (v) {
      v.playbackRate = speed
      void v.play().catch(() => setPlaying(false))
    }
    setPlaying(true)
  }

  function pause() {
    videoRef.current?.pause()
    setPlaying(false)
  }

  // Позиция реального видео -> стор.
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    let raf = 0
    const tick = () => {
      setPlayhead(v.currentTime * 1000)
      raf = requestAnimationFrame(tick)
    }
    const onPlay = () => {
      setPlaying(true)
      raf = requestAnimationFrame(tick)
    }
    const onPause = () => {
      setPlaying(false)
      cancelAnimationFrame(raf)
      setPlayhead(v.currentTime * 1000)
    }
    v.addEventListener('play', onPlay)
    v.addEventListener('pause', onPause)
    v.addEventListener('seeked', onPause)
    return () => {
      cancelAnimationFrame(raf)
      v.removeEventListener('play', onPlay)
      v.removeEventListener('pause', onPause)
      v.removeEventListener('seeked', onPause)
    }
  }, [src, setPlayhead])

  // Виртуальные часы, если файла нет.
  useEffect(() => {
    if (src || !playing) {
      if (clockRef.current) cancelAnimationFrame(clockRef.current)
      clockRef.current = null
      return
    }
    clockStateRef.current.lastTs = performance.now()
    const tick = (ts: number) => {
      const dt = ts - clockStateRef.current.lastTs
      clockStateRef.current.lastTs = ts
      const state = useEditorStore.getState()
      const next = state.playheadMs + dt * speed
      if (next >= durationMs) {
        state.setPlayhead(durationMs)
        setPlaying(false)
        return
      }
      state.setPlayhead(next)
      clockRef.current = requestAnimationFrame(tick)
    }
    clockRef.current = requestAnimationFrame(tick)
    return () => {
      if (clockRef.current) cancelAnimationFrame(clockRef.current)
      clockRef.current = null
    }
  }, [src, playing, speed, durationMs])

  useEffect(() => {
    const v = videoRef.current
    if (v) v.playbackRate = speed
  }, [speed])

  const current = segmentAt(segments, playheadMs)
  const action = current ? actions.find((a) => a.id === current.action) : null
  const object = current ? objects.find((o) => o.id === current.object) : null

  return (
    <div className="stage">
      <div className="stage__frame">
        {src ? (
          <video
            ref={videoRef}
            className="stage__video"
            src={src}
            preload="auto"
            playsInline
            onClick={() => (playing ? pause() : play())}
          />
        ) : (
          <div className="stage__missing">
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-muted)' }}>
              Видео недоступно в этой вкладке
            </div>
            <p>
              Файл {fileName ? <b>{fileName}</b> : 'ролика'} остался на вашем компьютере: браузер не
              хранит его между перезагрузками. Разметка, таймлайн и экспорт работают и без него —
              выберите файл заново, чтобы видеть кадры.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="visually-hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) onPickFile(f)
              }}
            />
            <button className="btn" onClick={() => fileInputRef.current?.click()}>
              Выбрать файл
            </button>
          </div>
        )}

        <div className={`stage__overlay${current ? '' : ' stage__overlay--idle'}`}>
          {current ? (
            <>
              <span className="chip__dot" style={{ background: action?.color ?? '#9AA3AD' }} />
              <span className="stage__overlay-action">{action?.label_ru ?? current.action}</span>
              <span className="stage__overlay-object">{object?.label_ru ?? current.object}</span>
            </>
          ) : (
            <span className="stage__overlay-object">Вне сегментов — idle</span>
          )}
        </div>

        <div className="stage__tc">{formatPrecise(playheadMs)}</div>
      </div>

      <div className="transport">
        <button className="transport__play" onClick={() => (playing ? pause() : play())} title="Пробел">
          {playing ? '❚❚' : '▶'}
        </button>

        <div className="transport__group">
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => {
              pause()
              const step = 1000 / (fps || 30)
              const next = playheadMs - step
              setPlayhead(next)
              if (videoRef.current) videoRef.current.currentTime = Math.max(0, next / 1000)
            }}
            title="Кадр назад (,)"
          >
            ◀|
          </button>
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => {
              pause()
              const step = 1000 / (fps || 30)
              const next = playheadMs + step
              setPlayhead(next)
              if (videoRef.current) videoRef.current.currentTime = next / 1000
            }}
            title="Кадр вперёд (.)"
          >
            |▶
          </button>
        </div>

        <div className="transport__time">
          {formatPrecise(playheadMs)} <span>/ {formatPrecise(durationMs)}</span>
        </div>

        <div className="transport__speeds" style={{ marginLeft: 'auto' }}>
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={`transport__speed${s === speed ? ' transport__speed--on' : ''}`}
              onClick={() => setSpeed(s)}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  )
})
