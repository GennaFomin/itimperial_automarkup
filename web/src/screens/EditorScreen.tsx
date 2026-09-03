import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  ApiError,
  USING_MOCK,
  cancelJob,
  frameUrl,
  getPrediction,
  getVocab,
  mediaUrl,
  pollJob,
  saveReview,
} from '../api/client'
import { JOB_STAGES, STAGE_LABEL, type Job, type Prediction, type VocabAction, type VocabObject } from '../api/types'
import { Timeline } from '../components/Timeline'
import { VideoStage, type VideoHandle } from '../components/VideoStage'
import { KeyframePanel } from '../components/KeyframePanel'
import { SegmentInspector } from '../components/SegmentInspector'
import { buildExportCsv, buildExportJson, buildReview, download } from '../lib/export'
import { coverage, diffAgainstPrediction, validate, verifiedCount } from '../lib/segments'
import { formatDuration } from '../lib/time'
import { useActivityTimer } from '../lib/useActivityTimer'
import { useEditorStore } from '../store/editorStore'
import { useTasksStore } from '../store/tasksStore'
import '../components/SidePanel.css'
import './EditorScreen.css'

/** После этого времени в очереди показываем «долго», но джобу не убиваем (§2). */
const SLOW_AFTER_MS = 10 * 60_000

type Phase =
  | { kind: 'waiting'; job: Job | null; slow: boolean }
  | { kind: 'ready'; prediction: Prediction }
  | { kind: 'error'; message: string; code: string }

export function EditorScreen() {
  const { taskId: jobId = '' } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  // Экран не зависит от того, успел ли загрузиться список: название — лишь
  // украшение, а всё остальное берётся у сервера по идентификатору задания.
  const task = useTasksStore((s) => s.tasks.find((t) => t.job_id === jobId))
  const refreshTasks = useTasksStore((s) => s.refresh)

  const mode: 'review' | 'scratch' = params.get('mode') === 'scratch' ? 'scratch' : 'review'
  const [phase, setPhase] = useState<Phase>({ kind: 'waiting', job: null, slow: false })
  const [tab, setTab] = useState<'keyframes' | 'inspector'>('keyframes')
  const [saving, setSaving] = useState(false)
  const [savedAt, setSavedAt] = useState<string | null>(null)

  const videoRef = useRef<VideoHandle>(null)
  const load = useEditorStore((s) => s.load)
  const reset = useEditorStore((s) => s.reset)

  useEffect(() => reset, [reset, jobId])

  /* ---------- Ожидание авторазметки и загрузка прогноза ---------- */
  // Зависим только от идентификаторов: объект задачи пересоздаётся на каждом
  // обновлении прогресса, и зависимость от него перезапускала бы поллинг
  // бесконечно, так и не дойдя до готового прогноза.
  useEffect(() => {
    if (!jobId) return
    const startedAt = Date.now()
    let cancelled = false

    const stop = pollJob(
      jobId,
      (job) => {
        if (cancelled) return

        if (job.status === 'failed') {
          setPhase({
            kind: 'error',
            code: job.error?.code ?? 'INTERNAL',
            message: job.error?.message ?? 'Обработка завершилась ошибкой',
          })
          return
        }
        if (job.status === 'cancelled') {
          setPhase({ kind: 'error', code: 'CANCELLED', message: 'Обработка отменена' })
          return
        }
        if (job.status === 'done' || job.status === 'done_with_errors') {
          void Promise.all([getPrediction(jobId), getVocab()])
            .then(([prediction, vocabDoc]) => {
              if (cancelled) return
              load(prediction, vocabDoc, mode)
              setPhase({ kind: 'ready', prediction })
            })
            .catch((e: unknown) => {
              if (cancelled) return
              const err = e instanceof ApiError ? e : null
              setPhase({
                kind: 'error',
                code: err?.code ?? 'INTERNAL',
                message: err?.message ?? 'Не удалось получить прогноз',
              })
            })
          return
        }
        setPhase({ kind: 'waiting', job, slow: Date.now() - startedAt > SLOW_AFTER_MS })
      },
      (err) => {
        if (!cancelled) setPhase({ kind: 'error', code: err.code, message: err.message })
      },
    )

    return () => {
      cancelled = true
      stop()
    }
  }, [jobId, load, mode])

  /* ---------- Словарь, дополненный значениями вне его ---------- */
  // При открытой лексике модель отвечает своими словами, поэтому значения вне
  // словаря — норма, а не сбой: контракт (§1) требует показать их как есть.
  const vocab = useEditorStore((s) => s.vocab)
  const { actions, objects, unknownValues } = useMemo(
    () =>
      mergeVocab(
        vocab?.actions ?? [],
        vocab?.objects ?? [],
        phase.kind === 'ready' ? phase.prediction : null,
      ),
    [vocab, phase],
  )

  const timer = useActivityTimer(phase.kind === 'ready' ? jobId : null, mode)

  const seek = useCallback((ms: number) => videoRef.current?.seek(ms), [])

  const selectAndSeek = useCallback(
    (segmentId: string, ms: number) => {
      useEditorStore.getState().select(segmentId)
      useEditorStore.getState().zoomToSegment(segmentId)
      seek(ms)
      setTab('inspector')
    },
    [seek],
  )

  useKeyboardShortcuts(phase.kind === 'ready', videoRef, actions, seek)

  if (!task) {
    return (
      <div className="ed__loading">
        <div className="empty__title">Задача не найдена</div>
        <Link className="btn" to="/">
          К списку задач
        </Link>
      </div>
    )
  }

  if (phase.kind === 'error') {
    return (
      <div className="ed__loading">
        <div className="empty__title">{phase.message}</div>
        <div className="mono" style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          {phase.code}
        </div>
        <Link className="btn" to="/">
          К списку задач
        </Link>
      </div>
    )
  }

  if (phase.kind === 'waiting') {
    const progress = phase.job?.progress ?? 0
    const currentStage = phase.job?.stage
    return (
      <div className="ed__loading">
        <div>
          <div className="empty__title" style={{ textAlign: 'center' }}>
            {phase.slow ? 'Обработка идёт дольше обычного' : 'Идёт авторазметка'}
          </div>
          <p style={{ color: 'var(--text-dim)', textAlign: 'center', marginTop: 6 }}>
            {phase.slow
              ? 'Задача не потеряна и продолжает считаться. Можно подождать или отменить.'
              : task.title}
          </p>
        </div>
        <div className="ed__loading-bar">
          <div className="ed__loading-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
        <div className="ed__stages">
          {JOB_STAGES.map((s) => {
            const i = JOB_STAGES.indexOf(s)
            const cur = currentStage ? JOB_STAGES.indexOf(currentStage) : -1
            const cls = i === cur ? ' ed__stage--on' : i < cur ? ' ed__stage--done' : ''
            return (
              <span key={s} className={`ed__stage${cls}`}>
                {STAGE_LABEL[s]}
              </span>
            )
          })}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Link className="btn" to="/">
            К списку
          </Link>
          <button
            className="btn btn--danger"
            onClick={() => {
              void cancelJob(task.job_id).then(() => navigate('/'))
            }}
          >
            Отменить обработку
          </button>
        </div>
      </div>
    )
  }

  const prediction = phase.prediction
  return (
    <EditorBody
      prediction={prediction}
      actions={actions}
      objects={objects}
      unknownValues={unknownValues}
      // Против фикстур ролика нет: запрашивать медиа и кадры не у кого, и
      // молчаливые 404 в консоли только маскировали бы настоящие ошибки.
      videoSrc={USING_MOCK ? null : mediaUrl(jobId)}
      frameUrl={USING_MOCK ? null : (ms) => frameUrl(jobId, ms)}
      videoRef={videoRef}
      title={task?.title ?? jobId}
      mode={mode}
      seconds={timer.seconds}
      tab={tab}
      setTab={setTab}
      saving={saving}
      savedAt={savedAt}
      onSeek={seek}
      onPickKeyframe={selectAndSeek}
      onExitScratch={() => navigate(`/task/${jobId}`)}
      onSave={async () => {
        setSaving(true)
        try {
          const state = useEditorStore.getState()
          const review = buildReview(prediction, state.segments, 'user_12', timer.seconds * 1000)
          review.mode = mode
          const res = await saveReview(jobId, review)
          setSavedAt(res.saved_at)
          // Замер времени уходит отдельным событием: сложить его с отправкой
          // значило бы посчитать один интервал дважды.
          timer.report()
          void refreshTasks()
          if (res.problems.length) {
            console.warn('Замечания по словарю:', res.problems)
          }
        } catch (e) {
          alert(e instanceof Error ? e.message : 'Не удалось сохранить правку')
        } finally {
          setSaving(false)
        }
      }}
    />
  )
}

interface BodyProps {
  prediction: Prediction
  actions: VocabAction[]
  objects: VocabObject[]
  unknownValues: string[]
  videoSrc: string | null
  frameUrl: ((ms: number) => string) | null
  videoRef: React.RefObject<VideoHandle | null>
  title: string
  mode: 'review' | 'scratch'
  /** Активные секунды работы — то, из чего считается ускорение. */
  seconds: number
  tab: 'keyframes' | 'inspector'
  setTab: (t: 'keyframes' | 'inspector') => void
  saving: boolean
  savedAt: string | null
  onSeek: (ms: number) => void
  onPickKeyframe: (segmentId: string, ms: number) => void
  onExitScratch: () => void
  onSave: () => void
}

function EditorBody({
  prediction,
  actions,
  objects,
  unknownValues,
  videoSrc,
  frameUrl,
  videoRef,
  title,
  mode,
  seconds,
  tab,
  setTab,
  saving,
  savedAt,
  onSeek,
  onPickKeyframe,
  onExitScratch,
  onSave,
}: BodyProps) {
  const segments = useEditorStore((s) => s.segments)
  const durationMs = useEditorStore((s) => s.durationMs)
  const undo = useEditorStore((s) => s.undo)
  const redo = useEditorStore((s) => s.redo)
  const canUndo = useEditorStore((s) => s.past.length > 0)
  const canRedo = useEditorStore((s) => s.future.length > 0)
  // Считаем производные здесь, а не селектором стора: обе функции возвращают
  // новый объект на каждый вызов, и как селекторы они зациклили бы рендер.
  const issues = useMemo(() => validate(segments, durationMs), [segments, durationMs])
  const diff = useMemo(
    () => diffAgainstPrediction(prediction.segments, segments),
    [prediction, segments],
  )
  const [exportOpen, setExportOpen] = useState(false)

  const applyVerifyAll = useEditorStore((s) => s.applyVerifyAll)
  const gotoNextUnverified = useEditorStore((s) => s.gotoNextUnverified)

  const edited = segments.filter((s) => s.edited).length
  const checked = verifiedCount(segments)
  const missingKeyframes = segments.filter((s) => s.keyframe_ms === null).length
  const cov = Math.round(coverage(segments, durationMs) * 100)

  const doExport = (format: 'json' | 'csv', source: 'review' | 'prediction') => {
    const base = title.replace(/[^\wа-яА-ЯёЁ-]+/g, '_').slice(0, 60) || 'export'
    if (format === 'json') {
      download(
        `${base}_${source}.json`,
        buildExportJson(prediction, segments, source),
        'application/json',
      )
    } else {
      download(
        `${base}_${source}.csv`,
        buildExportCsv(prediction, segments, source, prediction.job_id),
        'text/csv',
      )
    }
    setExportOpen(false)
  }

  return (
    <div className="ed">
      <header className="ed__top">
        <Link className="btn btn--ghost btn--sm ed__back" to="/">
          ← Задачи
        </Link>
        <div className="ed__titles">
          <div className="ed__title">{title}</div>
          <div className="ed__sub">
            {prediction.model_version} · vocab {prediction.vocab_version} ·{' '}
            {formatDuration(prediction.video.duration_ms)}
          </div>
        </div>

        <div className="ed__stats">
          <div className="ed__stat">
            <span className="ed__stat-val">{segments.length}</span>
            <span className="ed__stat-key">сегментов</span>
          </div>
          <div className="ed__stat">
            <span className="ed__stat-val">{edited}</span>
            <span className="ed__stat-key">правок</span>
          </div>
          <div className="ed__stat">
            <span className="ed__stat-val">{cov}%</span>
            <span className="ed__stat-key">покрытие</span>
          </div>
          <div className="ed__stat">
            <span
              className="ed__stat-val"
              style={{ color: checked === segments.length && segments.length ? 'var(--ok)' : undefined }}
            >
              {checked}/{segments.length}
            </span>
            <span className="ed__stat-key">проверено</span>
          </div>
          <div className="ed__stat" title="Только активное время: паузы не считаются">
            <span className="ed__stat-val">{formatClock(seconds)}</span>
            <span className="ed__stat-key">{mode === 'scratch' ? 'с нуля' : 'проверка'}</span>
          </div>
          {issues.length > 0 && (
            <div className="ed__stat">
              <span className="ed__stat-val" style={{ color: 'var(--danger)' }}>
                {issues.length}
              </span>
              <span className="ed__stat-key">проблем</span>
            </div>
          )}
        </div>

        <div className="ed__right">
          <button className="btn btn--sm" onClick={undo} disabled={!canUndo} title="Ctrl+Z">
            ↶
          </button>
          <button className="btn btn--sm" onClick={redo} disabled={!canRedo} title="Ctrl+Shift+Z">
            ↷
          </button>

          <button
            className="btn btn--sm"
            onClick={() => gotoNextUnverified()}
            disabled={checked === segments.length}
            title="Следующий непроверенный (Shift+Tab)"
          >
            ⇥ Непроверенный
          </button>
          <button
            className="btn btn--sm"
            onClick={applyVerifyAll}
            disabled={checked === segments.length}
            title="Подтвердить все оставшиеся"
          >
            ✓ Все
          </button>

          <div style={{ position: 'relative' }}>
            <button className="btn btn--sm" onClick={() => setExportOpen((v) => !v)}>
              Экспорт ▾
            </button>
            {exportOpen && (
              <div
                style={{
                  position: 'absolute',
                  right: 0,
                  top: 'calc(100% + 6px)',
                  zIndex: 30,
                  background: 'var(--surface-2)',
                  border: '1px solid var(--line-strong)',
                  borderRadius: 'var(--r-md)',
                  boxShadow: 'var(--shadow)',
                  padding: 6,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 2,
                  minWidth: 210,
                }}
              >
                <button className="btn btn--ghost btn--sm" onClick={() => doExport('json', 'review')}>
                  JSON — правка
                </button>
                <button className="btn btn--ghost btn--sm" onClick={() => doExport('csv', 'review')}>
                  CSV — правка
                </button>
                <hr className="divider" style={{ margin: '4px 0' }} />
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => doExport('json', 'prediction')}
                >
                  JSON — исходный прогноз
                </button>
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => doExport('csv', 'prediction')}
                >
                  CSV — исходный прогноз
                </button>
              </div>
            )}
          </div>

          <button
            className="btn btn--primary btn--sm"
            onClick={onSave}
            disabled={saving || issues.length > 0}
            title={issues.length > 0 ? 'Сначала исправьте проблемы разметки' : 'Ctrl+S'}
          >
            {saving
              ? 'Сохраняем…'
              : savedAt
                ? 'Сохранено ✓'
                : mode === 'scratch'
                  ? 'Завершить замер'
                  : 'Отправить правку'}
          </button>
        </div>
      </header>

      {mode === 'scratch' && (
        <div className="banner banner--scratch">
          ⏱ Замер ручной разметки: прогноз скрыт, идёт отсчёт времени. Результат не
          заменит настоящую правку — он нужен только для сравнения скорости.
          <button className="btn btn--sm" style={{ marginLeft: 'auto' }} onClick={onExitScratch}>
            Выйти к разметке
          </button>
        </div>
      )}
      {mode !== 'scratch' && prediction.errors.length > 0 && (
        <div className="banner banner--warn">
          ⚠ {prediction.errors.map((e) => e.message).join('; ')}
          {missingKeyframes > 0 && ` — ${missingKeyframes} сегментов без ключевого кадра`}
        </div>
      )}
      {unknownValues.length > 0 && (
        <div className="banner banner--warn">
          ⚠ Значения вне словаря: {unknownValues.join(', ')}. Показаны как есть — при
          открытой лексике модель отвечает своими словами.
        </div>
      )}
      {issues.length > 0 && (
        <div className="banner banner--err">
          ⚠ {issues[0].message}
          {issues.length > 1 && ` и ещё ${issues.length - 1}`} — отправка review заблокирована
        </div>
      )}

      <div className="ed__main">
        <div className="ed__left">
          <VideoStage
            ref={videoRef}
            src={videoSrc}
            fps={prediction.video.fps}
            actions={actions}
            objects={objects}
          />
          <Timeline actions={actions} objects={objects} onSeek={onSeek} />
        </div>

        <aside className="panel">
          <div className="panel__tabs">
            <button
              className={`panel__tab${tab === 'keyframes' ? ' panel__tab--on' : ''}`}
              onClick={() => setTab('keyframes')}
            >
              Ключевые кадры
              <span className="panel__badge">{segments.length}</span>
            </button>
            <button
              className={`panel__tab${tab === 'inspector' ? ' panel__tab--on' : ''}`}
              onClick={() => setTab('inspector')}
            >
              Сегмент
            </button>
          </div>

          <div className="panel__body">
            {tab === 'keyframes' ? (
              <KeyframePanel frameUrl={frameUrl} actions={actions} onPick={onPickKeyframe} />
            ) : (
              <SegmentInspector actions={actions} objects={objects} onSeek={onSeek} />
            )}
          </div>

          {/* Дифф с прогнозом — та самая телеметрия правок из контракта §4. */}
          <div className="ed__diff">
            <span>границ: {diff.boundaries_edited}</span>
            <span>действий: {diff.actions_changed}</span>
            <span>объектов: {diff.objects_changed}</span>
            <span>кадров: {diff.keyframes_moved}</span>
            <span>+{diff.segments_added}</span>
            <span>−{diff.segments_deleted}</span>
          </div>
        </aside>
      </div>
    </div>
  )
}

/** Активное время в виде м:сс — оно же уходит в метрику ускорения. */
function formatClock(seconds: number): string {
  const total = Math.round(seconds)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

/**
 * Значения, которых нет в словаре задачи, не подменяем на unknown и не роняем
 * интерфейс (контракт §1) — добавляем в список с нейтральным цветом и
 * предупреждаем баннером.
 */
function mergeVocab(
  actions: VocabAction[],
  objects: VocabObject[],
  prediction: Prediction | null,
): { actions: VocabAction[]; objects: VocabObject[]; unknownValues: string[] } {
  if (!prediction) return { actions, objects, unknownValues: [] }
  const actionIds = new Set(actions.map((a) => a.id))
  const objectIds = new Set(objects.map((o) => o.id))
  const extraActions: VocabAction[] = []
  const extraObjects: VocabObject[] = []
  const unknown: string[] = []

  for (const seg of prediction.segments) {
    if (!actionIds.has(seg.action.value)) {
      actionIds.add(seg.action.value)
      extraActions.push({ id: seg.action.value, label_ru: seg.action.value, color: '#9AA3AD' })
      unknown.push(seg.action.value)
    }
    if (!objectIds.has(seg.object.value)) {
      objectIds.add(seg.object.value)
      extraObjects.push({ id: seg.object.value, label_ru: seg.object.value })
      unknown.push(seg.object.value)
    }
  }
  return {
    actions: [...actions, ...extraActions],
    objects: [...objects, ...extraObjects],
    unknownValues: unknown,
  }
}

/** Горячие клавиши: ручная проверка должна идти с клавиатуры, а не мышью. */
function useKeyboardShortcuts(
  active: boolean,
  videoRef: React.RefObject<VideoHandle | null>,
  actions: VocabAction[],
  seek: (ms: number) => void,
) {
  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return
      }
      const store = useEditorStore.getState()
      const mod = e.metaKey || e.ctrlKey

      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.shiftKey ? store.redo() : store.undo()
        return
      }
      if (mod) return

      switch (e.key) {
        case ' ':
          e.preventDefault()
          videoRef.current?.togglePlay()
          return
        case ',':
          e.preventDefault()
          videoRef.current?.stepFrames(-1)
          return
        case '.':
          e.preventDefault()
          videoRef.current?.stepFrames(1)
          return
        case 'ArrowLeft':
          e.preventDefault()
          seek(store.playheadMs - (e.shiftKey ? 5000 : 1000))
          return
        case 'ArrowRight':
          e.preventDefault()
          seek(store.playheadMs + (e.shiftKey ? 5000 : 1000))
          return
        case 'Tab': {
          e.preventDefault()
          const sorted = store.segments
          if (!sorted.length) return
          // Shift+Tab ведёт по непроверенным: это основной маршрут просмотра,
          // и он важнее обхода назад, который делает то же, что стрелки.
          if (e.shiftKey) {
            const next = store.gotoNextUnverified()
            if (next) seek(next.keyframe_ms ?? next.start_ms)
            return
          }
          const i = sorted.findIndex((s) => s.id === store.selectedId)
          const next = sorted[(i + 1) % sorted.length]
          store.select(next.id)
          store.zoomToSegment(next.id)
          seek(next.keyframe_ms ?? next.start_ms)
          return
        }
        case 'Delete':
        case 'Backspace':
          if (store.selectedId) {
            e.preventDefault()
            store.applyDelete(store.selectedId)
          }
          return
        case '0':
          e.preventDefault()
          store.zoomToFit()
          return
        case '+':
        case '=':
          e.preventDefault()
          store.zoomAt(store.playheadMs, 1 / 1.4)
          return
        case '-':
          e.preventDefault()
          store.zoomAt(store.playheadMs, 1.4)
          return
      }

      const key = e.key.toLowerCase()
      if (key === 's') {
        e.preventDefault()
        // Режем сегмент под курсором, даже если он не выделен: так быстрее.
        const target = store.segments.find(
          (x) => store.playheadMs > x.start_ms && store.playheadMs < x.end_ms,
        )
        if (target) store.applySplit(target.id, store.playheadMs)
        return
      }
      if (key === 'c') {
        e.preventDefault()
        store.setTool('carve')
        return
      }
      if (key === 'v') {
        e.preventDefault()
        store.setTool('select')
        return
      }
      // «v» уже занята инструментом выбора, поэтому отметка проверки — на «y».
      if (key === 'y' && store.selectedId) {
        e.preventDefault()
        const current = store.segments.find((x) => x.id === store.selectedId)
        store.applyVerify(store.selectedId, !current?.verified)
        const next = store.gotoNextUnverified()
        if (next) seek(next.keyframe_ms ?? next.start_ms)
        return
      }
      if (key === 'f' && store.selectedId) {
        e.preventDefault()
        store.zoomToSegment(store.selectedId)
        return
      }

      const digit = Number(e.key)
      if (Number.isInteger(digit) && digit >= 1 && digit <= 9 && store.selectedId) {
        const action = actions[digit - 1]
        if (action) {
          e.preventDefault()
          store.applyUpdate(store.selectedId, { action: action.id })
        }
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [active, videoRef, actions, seek])
}
