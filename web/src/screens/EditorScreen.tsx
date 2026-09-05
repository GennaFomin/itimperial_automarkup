import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  ApiError,
  USING_MOCK,
  cancelJob,
  frameUrl,
  getAnnotation,
  getPrediction,
  getVocab,
  mediaUrl,
  pollJob,
  saveReview,
} from '../api/client'
import { JOB_STAGES, STAGE_LABEL, type Job, type Prediction, type VocabAction, type VocabObject } from '../api/types'
import { Menu } from '../components/Menu'
import { Timeline } from '../components/Timeline'
import { VideoStage, type VideoHandle } from '../components/VideoStage'
import { KeyframePanel } from '../components/KeyframePanel'
import { SegmentInspector } from '../components/SegmentInspector'
import { buildExportCsv, buildExportJson, buildReview, download } from '../lib/export'
import { coverage, diffAgainstPrediction, validate, verifiedCount } from '../lib/segments'
import { formatDuration } from '../lib/time'
import { useActivityTimer } from '../lib/useActivityTimer'
import { mergeVocab, outOfVocab } from '../lib/vocab'
import { useEditorStore } from '../store/editorStore'
import { useTasksStore } from '../store/tasksStore'
import { toast } from '../store/toastStore'
import '../components/SidePanel.css'
import './EditorScreen.css'

/** После этого времени в очереди показываем «долго», но джобу не убиваем (§2). */
const SLOW_AFTER_MS = 10 * 60_000

type Phase =
  /** Статус ещё не пришёл или разметка готова и догружается: считать заново ничего не нужно. */
  | { kind: 'loading' }
  | { kind: 'waiting'; job: Job; slow: boolean }
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
  const [phase, setPhase] = useState<Phase>({ kind: 'loading' })
  const [tab, setTab] = useState<'keyframes' | 'inspector'>('keyframes')
  const [saving, setSaving] = useState(false)
  const [savedAt, setSavedAt] = useState<string | null>(null)
  // Имя файла с сервера — запасной заголовок: при прямом открытии ссылки или
  // перезагрузке список задач ещё не загружен, и без него в шапке светился бы
  // голый идентификатор.
  const [filename, setFilename] = useState<string | null>(null)
  const title = task?.title ?? filename?.replace(/\.[^.]+$/, '') ?? jobId

  // Список задач нужен ради названия, которое живёт только на клиенте.
  useEffect(() => {
    if (!task) void refreshTasks()
    // Только при первом открытии: дальше список обновляет сам экран задач.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

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
        if (job.filename) setFilename(job.filename)

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
          // Дорожку заполняет актуальная разметка, а прогноз нужен отдельно:
          // по нему считается дифф правок и он остаётся неизменяемым.
          // В режиме замера прогноз не показывается, но база сравнения та же.
          setPhase({ kind: 'loading' })
          void Promise.all([getAnnotation(jobId), getPrediction(jobId), getVocab()])
            .then(([annotation, prediction, vocabDoc]) => {
              if (cancelled) return
              load(annotation, vocabDoc, mode)
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
  const segments = useEditorStore((s) => s.segments)
  const { actions, objects } = useMemo(
    () => mergeVocab(vocab?.actions ?? [], vocab?.objects ?? [], segments),
    [vocab, segments],
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

  const prediction = phase.kind === 'ready' ? phase.prediction : null
  const save = useCallback(async () => {
    if (!prediction || saving) return
    const state = useEditorStore.getState()
    if (validate(state.segments, state.durationMs).length > 0) {
      toast.warn('Сначала исправьте проблемы разметки')
      return
    }
    setSaving(true)
    try {
      const review = buildReview(prediction, state.segments, 'user_12', timer.seconds * 1000)
      review.mode = mode
      const res = await saveReview(jobId, review)
      setSavedAt(res.saved_at)
      // Замер времени уходит отдельным событием: сложить его с отправкой
      // значило бы посчитать один интервал дважды.
      timer.report()
      void refreshTasks()
      if (res.problems.length) {
        toast.warn(`Замечания по словарю: ${res.problems.length}`, res.problems.slice(0, 5))
      }
    } catch (e) {
      toast.error('Не удалось сохранить правку', [e instanceof Error ? e.message : String(e)])
    } finally {
      setSaving(false)
    }
  }, [prediction, saving, timer, mode, jobId, refreshTasks])
  // Горячая клавиша зовёт актуальную версию, а не ту, что была при подписке.
  const saveRef = useRef(save)
  saveRef.current = save

  useKeyboardShortcuts(phase.kind === 'ready', videoRef, actions, seek, saveRef)

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

  if (phase.kind === 'loading') {
    return (
      <div className="ed__loading">
        <div className="empty__title">Загружаю разметку</div>
        <p style={{ color: 'var(--text-dim)' }}>{title}</p>
        <Link className="btn" to="/">
          К списку
        </Link>
      </div>
    )
  }

  if (phase.kind === 'waiting') {
    const progress = phase.job.progress
    const currentStage = phase.job.stage
    return (
      <div className="ed__loading">
        <div>
          <div className="empty__title" style={{ textAlign: 'center' }}>
            {phase.slow ? 'Обработка идёт дольше обычного' : 'Идёт авторазметка'}
          </div>
          <p style={{ color: 'var(--text-dim)', textAlign: 'center', marginTop: 6 }}>
            {phase.slow
              ? 'Задача не потеряна и продолжает считаться. Можно подождать или отменить.'
              : (title)}
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
              void cancelJob(jobId).then(() => navigate('/'))
            }}
          >
            Отменить обработку
          </button>
        </div>
      </div>
    )
  }

  return (
    <EditorBody
      prediction={phase.prediction}
      actions={actions}
      objects={objects}
      // Против фикстур ролика нет: запрашивать медиа и кадры не у кого, и
      // молчаливые 404 в консоли только маскировали бы настоящие ошибки.
      videoSrc={USING_MOCK ? null : mediaUrl(jobId)}
      frameUrl={USING_MOCK ? null : (ms) => frameUrl(jobId, ms)}
      videoRef={videoRef}
      title={title}
      mode={mode}
      seconds={timer.seconds}
      tab={tab}
      setTab={setTab}
      saving={saving}
      savedAt={savedAt}
      onSeek={seek}
      onPickKeyframe={selectAndSeek}
      onExitScratch={() => navigate(`/task/${jobId}`)}
      onSave={() => void save()}
    />
  )
}

interface BodyProps {
  prediction: Prediction
  actions: VocabAction[]
  objects: VocabObject[]
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
  const vocab = useEditorStore((s) => s.vocab)
  const oov = useMemo(() => outOfVocab(segments, vocab), [segments, vocab])
  const [oovOpen, setOovOpen] = useState(false)

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
  }

  const jumpTo = (segmentId: string) => {
    const seg = segments.find((x) => x.id === segmentId)
    if (seg) onPickKeyframe(seg.id, seg.keyframe_ms ?? seg.start_ms)
  }

  const { width, height } = prediction.video
  const videoAspect = width > 0 && height > 0 ? `${width} / ${height}` : undefined

  return (
    <div className="ed" style={{ '--video-ar': videoAspect } as React.CSSProperties}>
      <header className="ed__top">
        <Link className="btn btn--ghost btn--sm ed__back" to="/">
          ← Задачи
        </Link>
        <div className="ed__titles">
          <div className="ed__title" title={title}>{title}</div>
          <div className="ed__sub">
            {prediction.model_version} · vocab {prediction.vocab_version} ·{' '}
            {formatDuration(prediction.video.duration_ms)}
          </div>
        </div>

        <div className="ed__stats">
          <div className="ed__stat" title="Сегментов на дорожке">
            <span className="ed__stat-val">{segments.length}</span>
            <span className="ed__stat-key">сегментов</span>
          </div>
          <div className="ed__stat" title="Сегментов, которые человек менял относительно прогноза">
            <span className="ed__stat-val">{edited}</span>
            <span className="ed__stat-key">правок</span>
          </div>
          <div className="ed__stat" title="Какая доля ролика накрыта сегментами">
            <span className="ed__stat-val">{cov}%</span>
            <span className="ed__stat-key">покрытие</span>
          </div>
          <div className="ed__stat" title="Сегментов, которые человек просмотрел и подтвердил">
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
            <div className="ed__stat" title="Проблемы разметки: список под шапкой">
              <span className="ed__stat-val" style={{ color: 'var(--danger)' }}>
                {issues.length}
              </span>
              <span className="ed__stat-key">проблем</span>
            </div>
          )}
          {oov.length > 0 && (
            <div className="ed__stat ed__stat--pop">
              <button
                className="ed__stat-btn"
                onClick={() => setOovOpen((v) => !v)}
                title="Значения, которых нет в словаре задачи. Показаны как есть, при экспорте не заменяются."
              >
                <span className="ed__stat-val" style={{ color: 'var(--warn)' }}>
                  {oov.length}
                </span>
                <span className="ed__stat-key">вне словаря</span>
              </button>
              {oovOpen && (
                <div className="ed__pop menu__list" role="menu">
                  {oov.slice(0, 20).map((item) => (
                    <button
                      key={`${item.segmentId}-${item.field}`}
                      className="btn btn--ghost btn--sm menu__item"
                      onClick={() => {
                        setOovOpen(false)
                        jumpTo(item.segmentId)
                      }}
                    >
                      <span className="mono" style={{ color: 'var(--text-dim)' }}>
                        {item.segmentId}
                      </span>
                      <span>{item.field === 'action' ? 'действие' : 'объект'}: {item.value}</span>
                    </button>
                  ))}
                  {oov.length > 20 && <div className="field__hint">и ещё {oov.length - 20}</div>}
                </div>
              )}
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

          <div className="ed__secondary">
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
          </div>
          <div className="ed__more">
            <Menu
              label="⋯"
              title="Ещё"
              items={[
                {
                  label: '⇥ Следующий непроверенный',
                  onClick: () => void gotoNextUnverified(),
                  disabled: checked === segments.length,
                },
                {
                  label: '✓ Подтвердить все',
                  onClick: applyVerifyAll,
                  disabled: checked === segments.length,
                },
              ]}
            />
          </div>

          <Menu
            label="Экспорт ▾"
            items={[
              { label: 'JSON — правка', onClick: () => doExport('json', 'review') },
              { label: 'CSV — правка', onClick: () => doExport('csv', 'review') },
              { label: 'JSON — исходный прогноз', onClick: () => doExport('json', 'prediction'), divider: true },
              { label: 'CSV — исходный прогноз', onClick: () => doExport('csv', 'prediction') },
            ]}
          />

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

      {/* Баннеры в одном блоке: у сетки ровно три строки, и что бы ни появилось
          сверху, видео с таймлайном получают остаток высоты, а не обрезаются. */}
      <div className="ed__banners">
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
        {issues.length > 0 && (
          <div className="banner banner--err banner--list">
            <span>⚠ Отправка правки заблокирована:</span>
            {issues.slice(0, 5).map((issue, i) => (
              <button
                key={i}
                className="banner__link"
                disabled={!issue.segmentId}
                onClick={() => issue.segmentId && jumpTo(issue.segmentId)}
              >
                {issue.message}
              </button>
            ))}
            {issues.length > 5 && <span>и ещё {issues.length - 5}</span>}
          </div>
        )}
      </div>

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

/** Горячие клавиши: ручная проверка должна идти с клавиатуры, а не мышью. */
function useKeyboardShortcuts(
  active: boolean,
  videoRef: React.RefObject<VideoHandle | null>,
  actions: VocabAction[],
  seek: (ms: number) => void,
  saveRef: React.RefObject<() => Promise<void>>,
) {
  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      // Сохранение работает и из поля ввода: иначе Ctrl+S открывал бы «сохранить страницу».
      if (mod && e.key.toLowerCase() === 's') {
        e.preventDefault()
        void saveRef.current()
        return
      }
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return
      }
      const store = useEditorStore.getState()

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
  }, [active, videoRef, actions, seek, saveRef])
}
