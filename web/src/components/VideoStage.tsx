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
  /**
   * Видео отдаёт бэкенд, поэтому источник переживает перезагрузку вкладки.
   * null — работа против фикстур, где ролика нет вовсе: таймлайн, правки и
   * экспорт при этом полностью рабочие, а плеер честно говорит, что показывать
   * нечего, вместо битой картинки.
   */
  src: string | null
  fps: number
  actions: VocabAction[]
  objects: VocabObject[]
}

/**
 * Плеер и транспорт. Единственный владелец времени воспроизведения: playhead в
 * сторе — зеркало текущей позиции, а не независимое состояние.
 */
export const VideoStage = forwardRef<VideoHandle, Props>(function VideoStage(
  { src, fps, actions, objects },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const durationMs = useEditorStore((s) => s.durationMs)
  const playheadMs = useEditorStore((s) => s.playheadMs)
  const setPlayhead = useEditorStore((s) => s.setPlayhead)
  const segments = useEditorStore((s) => s.segments)

  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

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
    // Без ролика двигать нечего: раньше кнопка переключалась в «пауза», а
    // playhead стоял на месте — интерфейс сообщал о том, чего не происходит.
    if (!v) return
    v.playbackRate = speed
    void v.play().catch(() => setPlaying(false))
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
              Работа на фикстурах
            </div>
            <p>
              Видео отдаёт бэкенд, а сейчас его нет. Разметка, таймлайн и экспорт работают
              как обычно — не хватает только картинки.
            </p>
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
        <button
          className="transport__play"
          onClick={() => (playing ? pause() : play())}
          disabled={!src}
          title={src ? 'Пробел' : 'Видео недоступно'}
        >
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
