import { useCallback, useEffect, useRef } from 'react'

import { stripUrl } from './api'
import { snap } from './motion'
import { isUncertain, sortSteps } from './steps'
import type { Step } from './types'

const SNAP_PIXELS = 9

interface Props {
  videoId: string
  duration: number
  steps: Step[]
  motion: number[]
  filmstrip: string[]
  candidates: number[]
  currentTime: number
  selectedId: number | null
  onSeek: (time: number) => void
  onSelect: (id: number) => void
  onMoveBoundary: (leftId: number, time: number) => void
}

export function Timeline(props: Props) {
  const { duration, steps, motion, filmstrip, candidates, currentTime, selectedId } = props
  const track = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const dragging = useRef<{ leftId: number } | null>(null)

  const ordered = sortSteps(steps)
  const percent = (time: number) => `${(time / duration) * 100}%`

  const timeAt = useCallback(
    (clientX: number, withSnap: boolean) => {
      const element = track.current
      if (!element) return 0
      const rect = element.getBoundingClientRect()
      const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1)
      const time = ratio * duration
      if (!withSnap) return time
      return snap(time, candidates, (SNAP_PIXELS / rect.width) * duration)
    },
    [candidates, duration],
  )

  useEffect(() => {
    const element = canvas.current
    if (!element) return
    const width = element.clientWidth
    const height = element.clientHeight
    const ratio = window.devicePixelRatio || 1
    element.width = width * ratio
    element.height = height * ratio

    const context = element.getContext('2d')
    if (!context) return
    context.scale(ratio, ratio)
    context.clearRect(0, 0, width, height)

    if (motion.length > 1) {
      context.beginPath()
      context.moveTo(0, height)
      motion.forEach((value, index) => {
        const x = (index / (motion.length - 1)) * width
        context.lineTo(x, height - value * (height - 2) - 1)
      })
      context.lineTo(width, height)
      context.closePath()
      context.fillStyle = 'rgba(96, 165, 250, 0.25)'
      context.fill()
      context.strokeStyle = 'rgba(96, 165, 250, 0.8)'
      context.lineWidth = 1
      context.stroke()
    }

    context.strokeStyle = 'rgba(250, 204, 21, 0.55)'
    candidates.forEach((time) => {
      const x = (time / duration) * width
      context.beginPath()
      context.moveTo(x, height - 6)
      context.lineTo(x, height)
      context.stroke()
    })
  }, [motion, candidates, duration])

  const onTrackPointerDown = (event: React.PointerEvent) => {
    if (dragging.current) return
    ;(event.target as HTMLElement).setPointerCapture?.(event.pointerId)
    props.onSeek(timeAt(event.clientX, false))
  }

  const onTrackPointerMove = (event: React.PointerEvent) => {
    if (dragging.current || event.buttons !== 1) return
    props.onSeek(timeAt(event.clientX, false))
  }

  const onHandleDown = (event: React.PointerEvent, leftId: number) => {
    event.stopPropagation()
    ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
    dragging.current = { leftId }
  }

  const onHandleMove = (event: React.PointerEvent) => {
    const state = dragging.current
    if (!state) return
    event.stopPropagation()
    props.onMoveBoundary(state.leftId, timeAt(event.clientX, !event.altKey))
  }

  const onHandleUp = () => {
    dragging.current = null
  }

  return (
    <div className="timeline">
      <div className="filmstrip">
        {filmstrip.map((name) => (
          <img key={name} src={stripUrl(props.videoId, name)} alt="" draggable={false} />
        ))}
      </div>

      <div
        className="track"
        ref={track}
        onPointerDown={onTrackPointerDown}
        onPointerMove={onTrackPointerMove}
      >
        <canvas className="motion" ref={canvas} />

        <div className="segments">
          {ordered.map((step, index) => (
            <div
              key={step.id}
              className={[
                'segment',
                `hue-${step.id % 6}`,
                step.id === selectedId ? 'selected' : '',
                isUncertain(step) ? 'uncertain' : '',
              ].join(' ')}
              style={{ left: percent(step.start_sec), width: percent(step.end_sec - step.start_sec) }}
              onPointerDown={(event) => {
                event.stopPropagation()
                props.onSelect(step.id)
                props.onSeek(timeAt(event.clientX, false))
              }}
              title={`${step.action}${step.object ? ` · ${step.object}` : ''}`}
            >
              <span className="segment-label">
                {index + 1}. {step.action}
                {step.object ? ` · ${step.object}` : ''}
              </span>
              {isUncertain(step) && <span className="segment-flag">проверить</span>}
            </div>
          ))}

          {ordered.slice(0, -1).map((step) => (
            <div
              key={`handle-${step.id}`}
              className="handle"
              style={{ left: percent(step.end_sec) }}
              onPointerDown={(event) => onHandleDown(event, step.id)}
              onPointerMove={onHandleMove}
              onPointerUp={onHandleUp}
              onPointerCancel={onHandleUp}
              title="Перетащить границу (Alt — без магнита)"
            />
          ))}
        </div>

        <div className="playhead" style={{ left: percent(currentTime) }} />
      </div>

      <div className="ruler">
        {Array.from({ length: Math.min(11, Math.max(2, Math.round(duration) + 1)) }).map(
          (_, index, all) => {
            const time = (duration * index) / (all.length - 1)
            return (
              <span key={index} style={{ left: percent(time) }}>
                {time.toFixed(1)}s
              </span>
            )
          },
        )}
      </div>
    </div>
  )
}
