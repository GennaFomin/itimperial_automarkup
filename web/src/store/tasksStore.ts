/**
 * Список задач — проекция сервера, а не отдельная база на клиенте.
 *
 * Раньше список жил в localStorage вместе с blob-URL загруженного файла, и это
 * ломалось предсказуемо: после перезагрузки вкладки карточки оставались, а видео
 * к ним — нет. Теперь единственный источник истины — `GET /api/v1/jobs`, а видео
 * отдаёт сам бэкенд по `/media`.
 *
 * Локально хранится ровно одно: понятное человеку название ролика. Бэкенд знает
 * только имя файла, а называть задачу своими словами удобно — и терять это
 * название при перезагрузке не стоит.
 */
import { create } from 'zustand'
import { listJobs } from '../api/client'
import type { JobStatus, JobSummary } from '../api/types'

export interface Task {
  job_id: string
  title: string
  filename: string
  duration_ms: number
  created_at: string
  status: JobStatus
  progress: number
  /** Прогон прошёл не в полную силу: причина — здесь, а не в статусе. */
  warnings: string[]
  reviewed: boolean
}

const TITLES_KEY = 'automarkup.titles.v2'

function loadTitles(): Record<string, string> {
  try {
    const raw = localStorage.getItem(TITLES_KEY)
    return raw ? (JSON.parse(raw) as Record<string, string>) : {}
  } catch {
    return {}
  }
}

function persistTitles(titles: Record<string, string>) {
  try {
    localStorage.setItem(TITLES_KEY, JSON.stringify(titles))
  } catch {
    // Приватный режим: название просто не переживёт перезагрузку.
  }
}

const toTask = (job: JobSummary, titles: Record<string, string>): Task => ({
  job_id: job.job_id,
  title: titles[job.job_id] || job.filename.replace(/\.[^.]+$/, ''),
  filename: job.filename,
  duration_ms: job.duration_ms,
  created_at: job.created_at,
  status: job.status,
  progress: job.progress,
  warnings: job.warnings ?? [],
  reviewed: job.reviewed,
})

interface TasksState {
  tasks: Task[]
  loading: boolean
  error: string | null
  titles: Record<string, string>
  refresh: () => Promise<void>
  setTitle: (jobId: string, title: string) => void
  getTask: (jobId: string) => Task | undefined
}

export const useTasksStore = create<TasksState>((set, get) => ({
  tasks: [],
  loading: false,
  error: null,
  titles: loadTitles(),

  refresh: async () => {
    set({ loading: true })
    try {
      const jobs = await listJobs()
      const titles = get().titles
      set({ tasks: jobs.map((job) => toTask(job, titles)), loading: false, error: null })
    } catch (e) {
      set({ loading: false, error: e instanceof Error ? e.message : 'Не удалось получить список' })
    }
  },

  setTitle: (jobId, title) =>
    set((state) => {
      const titles = { ...state.titles, [jobId]: title }
      persistTitles(titles)
      return {
        titles,
        tasks: state.tasks.map((task) =>
          task.job_id === jobId ? { ...task, title } : task,
        ),
      }
    }),

  getTask: (jobId) => get().tasks.find((task) => task.job_id === jobId),
}))
