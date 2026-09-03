import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { USING_MOCK, pollJob } from '../api/client'
import { STATUS_LABEL, type JobStatus } from '../api/types'
import { formatDuration } from '../lib/time'
import { useTasksStore, type Task } from '../store/tasksStore'
import { CreateTaskDialog } from './CreateTaskDialog'
import './TasksScreen.css'

export function TasksScreen() {
  const tasks = useTasksStore((s) => s.tasks)
  const updateTask = useTasksStore((s) => s.updateTask)
  const [creating, setCreating] = useState(false)

  // Задачи в очереди опрашиваем прямо со списка: карточка показывает прогресс
  // авторазметки, и открыть её можно ровно тогда, когда прогноз готов.
  //
  // Ключ — строка, а не массив: обновление прогресса создаёт новый массив задач,
  // и зависимость от массива перезапускала бы поллинг на каждом тике.
  const pendingKey = tasks
    .filter((t) => t.status === 'queued' || t.status === 'running')
    .map((t) => t.job_id)
    .join(',')

  useEffect(() => {
    if (!pendingKey) return
    const byJob = new Map(useTasksStore.getState().tasks.map((t) => [t.job_id, t.id]))
    const stops = pendingKey.split(',').map((jobId) =>
      pollJob(
        jobId,
        (job) => {
          const taskId = byJob.get(jobId)
          if (taskId) updateTask(taskId, { status: job.status, progress: job.progress })
        },
        () => {
          const taskId = byJob.get(jobId)
          if (taskId) updateTask(taskId, { status: 'failed' })
        },
      ),
    )
    return () => stops.forEach((stop) => stop())
  }, [pendingKey, updateTask])

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

        {tasks.length === 0 ? (
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
              <TaskCard key={task.id} task={task} />
            ))}
          </div>
        )}

        {USING_MOCK && (
          <div className="tasks__foot-note">
            <span className="chip">
              <span className="chip__dot" style={{ background: 'var(--warn)' }} />
              мок-бэкенд
            </span>
            Данные из фикстур. Реальный бэкенд подключается переменной VITE_API_BASE.
          </div>
        )}
      </div>

      {creating && <CreateTaskDialog onClose={() => setCreating(false)} />}
    </div>
  )
}

function TaskCard({ task }: { task: Task }) {
  const navigate = useNavigate()
  const removeTask = useTasksStore((s) => s.removeTask)
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
              {task.reviewed_at && (
                <>
                  <span className="card__dot" />
                  <span style={{ color: 'var(--ok)' }}>проверено</span>
                </>
              )}
            </div>
          </div>
          <button
            className="btn btn--ghost btn--sm btn--danger"
            title="Удалить задачу"
            onClick={() => removeTask(task.id)}
          >
            ✕
          </button>
        </div>

        <div>
          <div className="card__vocab-name" style={{ marginBottom: 7 }}>
            Словарь{task.vocab.name ? ` · ${task.vocab.name}` : ''}
          </div>
          <div className="card__vocab">
            {task.vocab.actions.slice(0, 7).map((a) => (
              <span className="chip" key={a.id}>
                <span className="chip__dot" style={{ background: a.color }} />
                {a.label_ru}
              </span>
            ))}
            {task.vocab.actions.length > 7 && (
              <span className="chip" style={{ color: 'var(--text-dim)' }}>
                +{task.vocab.actions.length - 7}
              </span>
            )}
          </div>
        </div>
      </div>

      {running && (
        <div className="progress">
          <div className="progress__bar" style={{ width: `${Math.round(task.progress * 100)}%` }} />
        </div>
      )}

      <div className="card__foot">
        <StatusBadge status={task.status} progress={task.progress} />
        <button
          className="btn btn--sm"
          disabled={!ready}
          onClick={() => navigate(`/task/${task.id}`)}
        >
          {ready ? 'Открыть разметку' : 'Ждём авторазметку'}
        </button>
      </div>
    </article>
  )
}

function StatusBadge({ status, progress }: { status: JobStatus; progress: number }) {
  const showPercent = status === 'running'
  return (
    <span className={`status status--${status}`}>
      <span className="status__dot" />
      {STATUS_LABEL[status]}
      {showPercent && <span className="mono" style={{ color: 'var(--text-dim)' }}>{Math.round(progress * 100)}%</span>}
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
