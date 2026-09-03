import { useEffect, useRef, useState } from 'react'
import { USING_MOCK, createJob, getLimits, getVocab } from '../api/client'
import type { Limits, Vocabulary } from '../api/types'
import { formatDuration } from '../lib/time'
import { useTasksStore } from '../store/tasksStore'
import './CreateTaskDialog.css'

interface Picked {
  file: File
  url: string
  durationMs: number | null
  height: number | null
  posterUrl: string | null
}

interface Props {
  onClose: () => void
  onCreated: () => void
}

export function CreateTaskDialog({ onClose, onCreated }: Props) {
  const setTitle = useTasksStore((s) => s.setTitle)

  const [limits, setLimits] = useState<Limits | null>(null)
  const [vocab, setVocab] = useState<Vocabulary | null>(null)
  const [picked, setPicked] = useState<Picked | null>(null)
  const [title, setTitleValue] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [scenario, setScenario] = useState('ok')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const inputRef = useRef<HTMLInputElement>(null)
  const handedOff = useRef(false)

  useEffect(() => {
    void getLimits().then(setLimits).catch(() => setLimits(null))
    void getVocab().then(setVocab).catch(() => setVocab(null))
  }, [])

  useEffect(() => {
    return () => {
      if (handedOff.current || !picked) return
      URL.revokeObjectURL(picked.url)
      if (picked.posterUrl) URL.revokeObjectURL(picked.posterUrl)
    }
  }, [picked])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const extensions = limits?.allowed_extensions ?? ['.mp4', '.mov']

  /**
   * Проверка до отправки — это удобство, а не контроль: решает всё равно сервер.
   * Поэтому непрочитанные браузером метаданные не блокируют загрузку, а вот
   * заведомо неподходящий ролик лучше отклонить сразу, чем ждать ответа.
   */
  function reject(file: File, probe: { durationMs: number | null; height: number | null }): string | null {
    const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    if (!extensions.includes(suffix)) {
      return `Поддерживаются только ${extensions.join(' и ')} — этот файл ${suffix || 'без расширения'}.`
    }
    if (limits && probe.durationMs !== null && probe.durationMs > limits.max_duration_ms) {
      return `Ролик длиннее ${Math.round(limits.max_duration_ms / 1000)} с (в файле ${formatDuration(probe.durationMs)}).`
    }
    if (limits && probe.height !== null && probe.height < limits.min_height) {
      return `Нужно не меньше ${limits.min_height}p, а здесь ${probe.height}p.`
    }
    return null
  }

  async function handleFile(file: File) {
    setError(null)
    const probe = await probeVideo(URL.createObjectURL(file)).catch(() => null)
    const url = URL.createObjectURL(file)
    const info = {
      durationMs: probe?.durationMs ?? null,
      height: probe?.height ?? null,
    }
    const problem = reject(file, info)
    if (problem) {
      URL.revokeObjectURL(url)
      setError(problem)
      setPicked(null)
      return
    }
    setPicked({ file, url, ...info, posterUrl: probe?.posterUrl ?? null })
    if (!title.trim()) setTitleValue(file.name.replace(/\.[^.]+$/, ''))
  }

  async function submit() {
    if (!picked) {
      setError('Загрузите видео.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const { job_id } = await createJob({
        file: picked.file,
        scenario,
        durationMs: picked.durationMs,
      })
      handedOff.current = true
      if (title.trim()) setTitle(job_id, title.trim())
      onCreated()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Не удалось создать задачу')
      setBusy(false)
    }
  }

  return (
    <div className="modal__backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Новая задача">
        <div className="modal__head">
          <div>
            <h2 className="modal__title">Новая задача</h2>
            <p className="modal__desc">
              Видео уйдёт в очередь на авторазметку. Задача откроется, когда прогноз будет готов.
            </p>
          </div>
          <button className="btn btn--ghost btn--icon" onClick={onClose} aria-label="Закрыть">
            ✕
          </button>
        </div>

        <div className="modal__body">
          <div className="field">
            <span className="field__label">Видео</span>
            <input
              ref={inputRef}
              type="file"
              accept={`${extensions.join(',')},video/mp4,video/quicktime`}
              className="visually-hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void handleFile(f)
              }}
            />
            <div
              className={`drop${dragOver ? ' drop--over' : ''}${picked ? ' drop--filled' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                const f = e.dataTransfer.files?.[0]
                if (f) void handleFile(f)
              }}
            >
              {picked ? (
                <>
                  {picked.posterUrl ? (
                    <img className="drop__thumb" src={picked.posterUrl} alt="" />
                  ) : (
                    <div className="drop__thumb" />
                  )}
                  <div style={{ minWidth: 0 }}>
                    <div className="drop__title" style={{ wordBreak: 'break-all' }}>
                      {picked.file.name}
                    </div>
                    <div className="drop__hint mono">
                      {picked.durationMs ? formatDuration(picked.durationMs) : 'длина неизвестна'}
                      {picked.height ? ` · ${picked.height}p` : ''} ·{' '}
                      {(picked.file.size / 1024 / 1024).toFixed(1)} МБ
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="drop__title">Перетащите видео или нажмите, чтобы выбрать</div>
                  <div className="drop__hint">{describeLimits(limits)}</div>
                </>
              )}
            </div>
          </div>

          <label className="field">
            <span className="field__label">Название задачи</span>
            <input
              className="input"
              value={title}
              placeholder="Например: Сборка узла, камера 2"
              onChange={(e) => setTitleValue(e.target.value)}
            />
            <span className="field__hint">
              Только для вашего удобства: сервер знает ролик по имени файла.
            </span>
          </label>

          <div className="field">
            <span className="field__label">Словарь действий</span>
            {vocab ? (
              <>
                <div className="vocab-preview">
                  {vocab.actions.slice(0, 12).map((a) => (
                    <span className="chip" key={a.id}>
                      <span className="chip__dot" style={{ background: a.color }} />
                      {a.label_ru}
                    </span>
                  ))}
                  {vocab.actions.length > 12 && (
                    <span className="chip" style={{ color: 'var(--text-dim)' }}>
                      +{vocab.actions.length - 12}
                    </span>
                  )}
                </div>
                <span className="field__hint">
                  Таксономию задаёт сервер (PRAXIS_VOCAB).{' '}
                  {vocab.open === true
                    ? 'Словарь открытый: модель отвечает своими словами, список — подсказка, а не ограничение.'
                    : 'Список закрытый: значения вне него будут помечены.'}
                </span>
              </>
            ) : (
              <span className="field__hint">Словарь не загрузился.</span>
            )}
          </div>

          {USING_MOCK && (
            <label className="field">
              <span className="field__label">Сценарий мока</span>
              <select
                className="select"
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
              >
                <option value="ok">Обычный прогноз</option>
                <option value="with_errors">done_with_errors — часть keyframe не посчитана</option>
                <option value="empty">Пустой результат — ни одного сегмента</option>
                <option value="failed">Джоба падает</option>
                <option value="slow">Долгая обработка — дольше 10 минут</option>
              </select>
              <span className="field__hint">
                Обязательные сценарии контракта §8: живой бэкенд их по требованию не воспроизводит.
              </span>
            </label>
          )}

          {error && <div className="alert">{error}</div>}
        </div>

        <div className="modal__foot">
          <span className="field__hint">{describeLimits(limits)}</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={onClose} disabled={busy}>
              Отмена
            </button>
            <button className="btn btn--primary" onClick={submit} disabled={busy || !picked}>
              {busy ? 'Загружаем…' : 'В очередь на разметку'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Требования печатаются из ответа сервера, чтобы не разойтись с ним числами. */
function describeLimits(limits: Limits | null): string {
  if (!limits) return 'Требования к ролику загружаются…'
  return [
    limits.allowed_extensions.join(' или '),
    `не длиннее ${Math.round(limits.max_duration_ms / 1000)} с`,
    `не ниже ${limits.min_height}p`,
  ].join(' · ')
}

/** Длительность, высота и кадр-обложка — без показа служебного плеера. */
function probeVideo(
  url: string,
): Promise<{ durationMs: number | null; height: number | null; posterUrl: string | null }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.src = url

    video.onerror = () => reject(new Error('Не удалось прочитать метаданные видео'))

    video.onloadedmetadata = () => {
      const durationMs = Number.isFinite(video.duration) ? Math.round(video.duration * 1000) : null
      const height = video.videoHeight || null
      // Кадр на 10% длины: первый кадр часто чёрный.
      video.currentTime = Math.min(video.duration * 0.1, 3)
      video.onseeked = () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = 320
          canvas.height = Math.round((320 * video.videoHeight) / (video.videoWidth || 1)) || 180
          const ctx = canvas.getContext('2d')
          if (!ctx) return resolve({ durationMs, height, posterUrl: null })
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          canvas.toBlob(
            (blob) =>
              resolve({ durationMs, height, posterUrl: blob ? URL.createObjectURL(blob) : null }),
            'image/jpeg',
            0.7,
          )
        } catch {
          resolve({ durationMs, height, posterUrl: null })
        }
      }
      // Если seek не случится (кодек без произвольного доступа) — отдаём хотя бы метаданные.
      setTimeout(() => resolve({ durationMs, height, posterUrl: null }), 2500)
    }
  })
}
