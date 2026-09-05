import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { USING_MOCK, getStats } from '../api/client'
import { STATUS_LABEL, type JobStatus, type Stats } from '../api/types'
import { formatDuration } from '../lib/time'
import { useTasksStore, type Task } from '../store/tasksStore'
import { toast } from '../store/toastStore'
import { CreateTaskDialog } from './CreateTaskDialog'
import './TasksScreen.css'

/** Пока в очереди есть задачи, список опрашивается — как требует контракт §2. */
const POLL_MS = 2000

/** Короткие подписи для карточки; полное название приходит с сервера в /limits. */
const PIPELINE_SHORT: Record<string, string> = {
  'learned-boundaries': 'детектор',
  'tsm-kernel': 'change-point',
}

type Filter = 'all' | 'running' | 'ready' | 'failed'
const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: 'all', label: 'Все' },
  { id: 'running', label: 'В работе' },
  { id: 'ready', label: 'Готовые' },
  { id: 'failed', label: 'Ошибки' },
]

function matches(task: Task, filter: Filter): boolean {
  switch (filter) {
    case 'running':
      return task.status === 'queued' || task.status === 'running'
    case 'ready':
      return task.status === 'done' || task.status === 'done_with_errors'
    case 'failed':
      return task.status === 'failed' || task.status === 'cancelled'
    default:
      return true
  }
}

export function TasksScreen() {
  const tasks = useTasksStore((s) => s.tasks)
  const error = useTasksStore((s) => s.error)
  const refresh = useTasksStore((s) => s.refresh)
  const [creating, setCreating] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)
  const [filter, setFilter] = useState<Filter>('all')

  const pending = tasks.some((task) => task.status === 'queued' || task.status === 'running')
  // Число проверенных задач — сигнал, что статистика ускорения могла измениться.
  const reviewedCount = tasks.filter((task) => task.reviewed).length

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    const load = () => void getStats().then(setStats).catch(() => setStats(null))
    load()
    window.addEventListener('focus', load)
    return () => window.removeEventListener('focus', load)
  }, [reviewedCount])

  const shown = tasks.filter((task) => matches(task, filter))

  // Один запрос на весь список вместо отдельного поллинга на каждую карточку.
  useEffect(() => {
    if (!pending) return
    const timer = setInterval(() => void refresh(), POLL_MS)
    return () => clearInterval(timer)
  }, [pending, refresh])

  return (
    <div className="tasks">
      <header className="tasks__top">
        <div className="tasks__brand">
          <div className="tasks__mark">A</div>
          <div>
            <div className="tasks__name">Automarkup</div>
            <div className="tasks__sub">Предразметка действий по видео</div>
          </div>
        </div>
        {stats?.speedup !== undefined && <Speedup stats={stats} />}
        <button className="btn btn--primary" onClick={() => setCreating(true)}>
          + Новая задача
        </button>
      </header>

      <div className="tasks__body">
        <div className="tasks__head">
          <div>
            <h1 className="tasks__title">Задачи</h1>
            <div className="tasks__count">
              {tasks.length === 0
                ? 'Пока пусто'
                : `${shown.length} ${plural(shown.length, 'задача', 'задачи', 'задач')}`}
            </div>
          </div>
          {tasks.length > 0 && (
            <div className="tasks__filters">
              {FILTERS.map((f) => {
                const count = tasks.filter((task) => matches(task, f.id)).length
                return (
                  <button
                    key={f.id}
                    className={`chip tasks__filter${filter === f.id ? ' tasks__filter--on' : ''}`}
                    onClick={() => setFilter(f.id)}
                  >
                    {f.label}
                    <span className="mono" style={{ color: 'var(--text-dim)' }}>{count}</span>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {error && <div className="banner banner--err">⚠ {error}</div>}

        {tasks.length === 0 && !error ? (
          <div className="empty">
            <div className="empty__title">Ещё ни одной задачи</div>
            <p>Загрузите видео — оно попадёт в очередь на авторазметку.</p>
            <button className="btn btn--primary" onClick={() => setCreating(true)}>
              Создать задачу
            </button>
          </div>
        ) : shown.length === 0 ? (
          <div className="empty">
            <div className="empty__title">Нет задач с таким статусом</div>
          </div>
        ) : (
          <div className="tasks__grid">
            {shown.map((task) => (
              <TaskCard key={task.job_id} task={task} />
            ))}
          </div>
        )}

        {USING_MOCK && (
          <div className="tasks__foot-note">
            <span className="chip">
              <span className="chip__dot" style={{ background: 'var(--warn)' }} />
              мок-бэкенд
            </span>
            Данные из фикстур. Уберите VITE_API_MOCK, чтобы работать с пайплайном.
          </div>
        )}
      </div>

      {creating && (
        <CreateTaskDialog
          onClose={() => setCreating(false)}
          onCreated={() => void refresh()}
        />
      )}
    </div>
  )
}

/**
 * Целевая метрика кейса на виду. Без неё весь механизм замера времени невидим,
 * а значит и не используется.
 */
function Speedup({ stats }: { stats: Stats }) {
  const value = stats.speedup ?? 0
  const reached = value >= 3
  return (
    <div className="speedup" title="Медиана правки против медианы разметки с нуля">
      <span className="speedup__value" style={{ color: reached ? 'var(--ok)' : 'var(--warn)' }}>
        {value.toFixed(1)}×
      </span>
      <span className="speedup__label">
        быстрее ручной
        <br />
        разметки · цель 3×
      </span>
    </div>
  )
}

function TaskCard({ task }: { task: Task }) {
  const navigate = useNavigate()
  const remove = useTasksStore((s) => s.remove)
  const cancel = useTasksStore((s) => s.cancel)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const ready = task.status === 'done' || task.status === 'done_with_errors'
  const running = task.status === 'queued' || task.status === 'running'

  const onDelete = async () => {
    setBusy(true)
    try {
      await remove(task.job_id)
    } catch (e) {
      toast.error('Не удалось удалить задачу', [e instanceof Error ? e.message : String(e)])
      setBusy(false)
      setConfirming(false)
    }
  }

  const onCancel = async () => {
    setBusy(true)
    try {
      await cancel(task.job_id)
    } catch (e) {
      toast.error('Не удалось отменить обработку', [e instanceof Error ? e.message : String(e)])
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className={`card${ready ? '' : ' card--disabled'}`}>
      <div className="card__main">
        <div className="card__row">
          <div style={{ minWidth: 0 }}>
            <div className="card__title">{task.title}</div>
            <div className="card__meta">
              <span className="mono">
                {task.duration_ms ? formatDuration(task.duration_ms) : 'длина неизвестна'}
              </span>
              <span className="card__dot" />
              <span>{new Date(task.created_at).toLocaleDateString('ru-RU')}</span>
              {task.reviewed && (
                <>
                  <span className="card__dot" />
                  <span style={{ color: 'var(--ok)' }}>проверено</span>
                </>
              )}
              {task.pipeline && (
                <>
                  <span className="card__dot" />
                  <span
                    className="mono"
                    title={
                      task.tasThreshold != null
                        ? `Нарезка ${task.pipeline}, порог ${task.tasThreshold}`
                        : `Нарезка ${task.pipeline}`
                    }
                  >
                    {PIPELINE_SHORT[task.pipeline] ?? task.pipeline}
                    {task.tasThreshold != null ? ` · ${task.tasThreshold}` : ''}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {task.warnings.length > 0 && (
          <div className="card__warn" title={task.warnings.join('; ')}>
            ⚠ {task.warnings[0]}
          </div>
        )}
      </div>

      {running && (
        <div className="progress">
          <div className="progress__bar" style={{ width: `${Math.round(task.progress * 100)}%` }} />
        </div>
      )}

      <div className="card__foot">
        {confirming ? (
          // Подтверждение прямо на карточке: удаление стирает прогон на минуты GPU,
          // одно случайное нажатие не должно его стоить.
          <div className="card__confirm">
            <span>Удалить задачу и её файлы?</span>
            <button className="btn btn--sm btn--danger" disabled={busy} onClick={() => void onDelete()}>
              {busy ? 'Удаляем…' : 'Да, удалить'}
            </button>
            <button className="btn btn--sm btn--ghost" disabled={busy} onClick={() => setConfirming(false)}>
              Отмена
            </button>
          </div>
        ) : (
          <>
            <StatusBadge status={task.status} progress={task.progress} />
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className="btn btn--sm btn--ghost card__delete"
                title="Удалить задачу"
                aria-label="Удалить задачу"
                onClick={() => setConfirming(true)}
              >
                🗑
              </button>
              {running && (
                <button className="btn btn--sm btn--danger" disabled={busy} onClick={() => void onCancel()}>
                  Отменить
                </button>
              )}
              {ready && (
                <button
                  className="btn btn--sm"
                  title="Замер: разметить этот ролик с нуля, без подсказки модели"
                  onClick={() => navigate(`/task/${task.job_id}?mode=scratch`)}
                >
                  С нуля
                </button>
              )}
              <button
                className="btn btn--sm"
                disabled={!ready}
                onClick={() => navigate(`/task/${task.job_id}`)}
              >
                {ready ? 'Открыть разметку' : 'Ждём авторазметку'}
              </button>
            </div>
          </>
        )}
      </div>
    </article>
  )
}

function StatusBadge({ status, progress }: { status: JobStatus; progress: number }) {
  return (
    <span className={`status status--${status}`}>
      <span className="status__dot" />
      {STATUS_LABEL[status]}
      {status === 'running' && (
        <span className="mono" style={{ color: 'var(--text-dim)' }}>
          {Math.round(progress * 100)}%
        </span>
      )}
    </span>
  )
}

function plural(n: number, one: string, few: string, many: string) {
  const mod10 = n % 10
  const mod100 = n % 100
  if (mod10 === 1 && mod100 !== 11) return one
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few
  return many
}
