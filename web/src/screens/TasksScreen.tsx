import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { USING_MOCK, getStats } from '../api/client'
import { STATUS_LABEL, type JobStatus, type Stats } from '../api/types'
import { formatDuration } from '../lib/time'
import { useTasksStore, type Task } from '../store/tasksStore'
import { CreateTaskDialog } from './CreateTaskDialog'
import './TasksScreen.css'

/** Пока в очереди есть задачи, список опрашивается — как требует контракт §2. */
const POLL_MS = 2000

export function TasksScreen() {
  const tasks = useTasksStore((s) => s.tasks)
  const error = useTasksStore((s) => s.error)
  const refresh = useTasksStore((s) => s.refresh)
  const [creating, setCreating] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)

  const pending = tasks.some((task) => task.status === 'queued' || task.status === 'running')

  useEffect(() => {
    void refresh()
    void getStats().then(setStats).catch(() => setStats(null))
  }, [refresh])

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
                : `${tasks.length} ${plural(tasks.length, 'задача', 'задачи', 'задач')}`}
            </div>
          </div>
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
        ) : (
          <div className="tasks__grid">
            {tasks.map((task) => (
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
  const ready = task.status === 'done' || task.status === 'done_with_errors'
  const running = task.status === 'queued' || task.status === 'running'

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
        <StatusBadge status={task.status} progress={task.progress} />
        <div style={{ display: 'flex', gap: 6 }}>
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
