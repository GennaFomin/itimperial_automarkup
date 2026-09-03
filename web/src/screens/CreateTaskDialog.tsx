import { useEffect, useMemo, useRef, useState } from 'react'
import { createJob, getVocab } from '../api/client'
import type { MockScenario } from '../api/mock'
import type { VocabAction, VocabObject, Vocabulary } from '../api/types'
import {
  loadSavedVocabs,
  saveVocab,
  setTaskVideoUrl,
  useTasksStore,
  type SavedVocab,
  type TaskVocabulary,
} from '../store/tasksStore'
import { formatDuration } from '../lib/time'
import './CreateTaskDialog.css'

/** Палитра для классов, которых нет в словаре бэкенда. */
const PALETTE = ['#FF5A1F', '#3DDCC8', '#7C6BFF', '#F2C14E', '#4A9DFF', '#E85D9B', '#35C07D', '#FF8A47']

type VocabMode = 'default' | 'saved' | 'new'

interface Picked {
  file: File
  url: string
  durationMs: number | null
  posterUrl: string | null
}

export function CreateTaskDialog({ onClose }: { onClose: () => void }) {
  const addTask = useTasksStore((s) => s.addTask)

  const [baseVocab, setBaseVocab] = useState<Vocabulary | null>(null)
  const [picked, setPicked] = useState<Picked | null>(null)
  const [title, setTitle] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [mode, setMode] = useState<VocabMode>('default')
  const [savedVocabs] = useState<SavedVocab[]>(() => loadSavedVocabs())
  const [savedIndex, setSavedIndex] = useState(0)
  const [newName, setNewName] = useState('')
  const [newActions, setNewActions] = useState('Взять\nПереместить\nПоложить')
  const [newObjects, setNewObjects] = useState('Деталь\nЛоток\nИнструмент')
  const [scenario, setScenario] = useState<MockScenario>('ok')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const inputRef = useRef<HTMLInputElement>(null)
  // Ревокаем blob-URL-ы только при отмене: у созданной задачи URL забирает стор.
  const handedOff = useRef(false)

  useEffect(() => {
    getVocab().then(setBaseVocab).catch(() => setBaseVocab(null))
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

  const vocab: TaskVocabulary = useMemo(() => {
    if (mode === 'saved' && savedVocabs[savedIndex]) return savedVocabs[savedIndex]
    if (mode === 'new') {
      return {
        name: newName.trim() || 'Без названия',
        actions: parseActions(newActions),
        objects: parseObjects(newObjects),
      }
    }
    return {
      name: baseVocab ? `Общий v${baseVocab.version}` : 'Общий',
      actions: baseVocab?.actions ?? [],
      objects: baseVocab?.objects ?? [],
    }
  }, [mode, savedVocabs, savedIndex, newName, newActions, newObjects, baseVocab])

  async function handleFile(file: File) {
    setError(null)
    if (!file.type.startsWith('video/')) {
      setError('Нужен видеофайл. Поддерживаются форматы, которые умеет браузер: mp4, webm, mov.')
      return
    }
    const url = URL.createObjectURL(file)
    const probe = await probeVideo(url).catch(() => null)
    setPicked({
      file,
      url,
      durationMs: probe?.durationMs ?? null,
      posterUrl: probe?.posterUrl ?? null,
    })
    if (!title.trim()) setTitle(file.name.replace(/\.[^.]+$/, ''))
  }

  async function submit() {
    if (!picked) {
      setError('Загрузите видео.')
      return
    }
    if (vocab.actions.length === 0) {
      setError('В словаре нет ни одного действия.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const { job_id } = await createJob({
        file: picked.file,
        options: { vocab_version: baseVocab?.version ?? '1.0' },
        scenario,
        durationMs: picked.durationMs,
      })
      const taskId = `t_${job_id}`
      setTaskVideoUrl(taskId, picked.url)
      handedOff.current = true
      if (mode === 'new' && newName.trim()) {
        saveVocab({ ...vocab, name: newName.trim() })
      }
      addTask({
        id: taskId,
        job_id,
        title: title.trim() || picked.file.name,
        duration_ms: picked.durationMs,
        created_at: new Date().toISOString(),
        status: 'queued',
        progress: 0,
        vocab,
        file_name: picked.file.name,
        scenario,
        reviewed_at: null,
      })
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
              accept="video/*"
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
                      {picked.durationMs ? formatDuration(picked.durationMs) : 'длина неизвестна'} ·{' '}
                      {(picked.file.size / 1024 / 1024).toFixed(1)} МБ
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="drop__title">Перетащите видео или нажмите, чтобы выбрать</div>
                  <div className="drop__hint">mp4, webm, mov — то, что играет браузер</div>
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
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>

          <div className="field">
            <span className="field__label">Словарь действий</span>
            <div className="seg-toggle">
              <button
                className={`seg-toggle__btn${mode === 'default' ? ' seg-toggle__btn--active' : ''}`}
                onClick={() => setMode('default')}
              >
                Общий
              </button>
              <button
                className={`seg-toggle__btn${mode === 'saved' ? ' seg-toggle__btn--active' : ''}`}
                onClick={() => setMode('saved')}
                disabled={savedVocabs.length === 0}
              >
                Сохранённый {savedVocabs.length > 0 && `(${savedVocabs.length})`}
              </button>
              <button
                className={`seg-toggle__btn${mode === 'new' ? ' seg-toggle__btn--active' : ''}`}
                onClick={() => setMode('new')}
              >
                Новый
              </button>
            </div>

            {mode === 'saved' && savedVocabs.length > 0 && (
              <select
                className="select"
                value={savedIndex}
                onChange={(e) => setSavedIndex(Number(e.target.value))}
              >
                {savedVocabs.map((v, i) => (
                  <option key={v.name} value={i}>
                    {v.name} — {v.actions.length} действий
                  </option>
                ))}
              </select>
            )}

            {mode === 'new' && (
              <>
                <input
                  className="input"
                  value={newName}
                  placeholder="Название словаря"
                  onChange={(e) => setNewName(e.target.value)}
                />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <label className="field">
                    <span className="field__label">Действия — по одному в строке</span>
                    <textarea
                      className="textarea"
                      value={newActions}
                      onChange={(e) => setNewActions(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    <span className="field__label">Объекты — по одному в строке</span>
                    <textarea
                      className="textarea"
                      value={newObjects}
                      onChange={(e) => setNewObjects(e.target.value)}
                    />
                  </label>
                </div>
                <span className="field__hint">
                  Можно задать id явно: <span className="mono">pick — Взять</span>. Иначе id
                  сгенерируется из названия. «Неизвестно» добавляется автоматически.
                </span>
              </>
            )}

            {mode !== 'new' && (
              <div className="vocab-preview">
                {vocab.actions.map((a) => (
                  <span className="chip" key={a.id}>
                    <span className="chip__dot" style={{ background: a.color }} />
                    {a.label_ru}
                  </span>
                ))}
                {vocab.actions.length === 0 && (
                  <span className="field__hint">Словарь не загрузился.</span>
                )}
              </div>
            )}
          </div>

          <label className="field">
            <span className="field__label">Сценарий мока</span>
            <select
              className="select"
              value={scenario}
              onChange={(e) => setScenario(e.target.value as MockScenario)}
            >
              <option value="ok">Обычный прогноз</option>
              <option value="with_errors">done_with_errors — часть keyframe не посчитана</option>
              <option value="empty">Пустой результат — ни одного сегмента</option>
              <option value="failed">Джоба падает</option>
              <option value="slow">Долгая обработка — дольше 10 минут</option>
            </select>
            <span className="field__hint">
              Пока бэкенд не подключён — так проверяются обязательные сценарии из контракта.
            </span>
          </label>

          {error && <div className="alert">{error}</div>}
        </div>

        <div className="modal__foot">
          <span className="field__hint">
            {vocab.actions.length} действий · {vocab.objects.length} объектов
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={onClose} disabled={busy}>
              Отмена
            </button>
            <button className="btn btn--primary" onClick={submit} disabled={busy || !picked}>
              {busy ? 'Создаём…' : 'В очередь на разметку'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Достаём длительность и кадр-обложку, не показывая пользователю служебный плеер. */
function probeVideo(url: string): Promise<{ durationMs: number | null; posterUrl: string | null }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.src = url

    const fail = () => reject(new Error('Не удалось прочитать метаданные видео'))
    video.onerror = fail

    video.onloadedmetadata = () => {
      const durationMs = Number.isFinite(video.duration) ? Math.round(video.duration * 1000) : null
      // Кадр на 10% длины: первый кадр часто чёрный.
      video.currentTime = Math.min(video.duration * 0.1, 3)
      video.onseeked = () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = 320
          canvas.height = Math.round((320 * video.videoHeight) / (video.videoWidth || 1)) || 180
          const ctx = canvas.getContext('2d')
          if (!ctx) return resolve({ durationMs, posterUrl: null })
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          canvas.toBlob(
            (blob) =>
              resolve({ durationMs, posterUrl: blob ? URL.createObjectURL(blob) : null }),
            'image/jpeg',
            0.7,
          )
        } catch {
          resolve({ durationMs, posterUrl: null })
        }
      }
      // Если seek не случится (кодек без произвольного доступа) — отдаём хотя бы длину.
      setTimeout(() => resolve({ durationMs, posterUrl: null }), 2500)
    }
  })
}

const slug = (s: string, i: number) => {
  const ascii = s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  return ascii || `class_${i + 1}`
}

/** Строка вида `pick — Взять` или просто `Взять`. */
function parseLine(line: string): { id: string | null; label: string } | null {
  const trimmed = line.trim()
  if (!trimmed) return null
  const m = trimmed.match(/^([a-zA-Z0-9_-]+)\s*[—–-]\s*(.+)$/)
  if (m) return { id: m[1], label: m[2].trim() }
  return { id: null, label: trimmed }
}

function parseActions(text: string): VocabAction[] {
  const out: VocabAction[] = []
  const seen = new Set<string>()
  text.split('\n').forEach((line, i) => {
    const parsed = parseLine(line)
    if (!parsed) return
    const id = parsed.id ?? slug(parsed.label, i)
    if (seen.has(id)) return
    seen.add(id)
    out.push({ id, label_ru: parsed.label, color: PALETTE[out.length % PALETTE.length] })
  })
  // unknown обязателен: модель возвращает его вместо уверенной выдумки.
  if (!seen.has('unknown')) {
    out.push({ id: 'unknown', label_ru: 'Неизвестно', color: '#9AA3AD' })
  }
  return out
}

function parseObjects(text: string): VocabObject[] {
  const out: VocabObject[] = []
  const seen = new Set<string>()
  text.split('\n').forEach((line, i) => {
    const parsed = parseLine(line)
    if (!parsed) return
    const id = parsed.id ?? slug(parsed.label, i)
    if (seen.has(id)) return
    seen.add(id)
    out.push({ id, label_ru: parsed.label })
  })
  if (!seen.has('unknown')) out.push({ id: 'unknown', label_ru: 'Неизвестно' })
  return out
}
