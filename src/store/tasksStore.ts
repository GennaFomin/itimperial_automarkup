/**
 * Список задач. Метаданные переживают перезагрузку через localStorage,
 * а вот сам файл видео — нет: blob-URL живёт только в текущей вкладке.
 * Поэтому после перезагрузки задача остаётся, а видео просит перевыбрать файл.
 */
import { create } from 'zustand'
import type { MockScenario } from '../api/mock'
import type { JobStatus, VocabAction, VocabObject } from '../api/types'

export interface TaskVocabulary {
  /** null — задача использует общий словарь с бэкенда. */
  name: string | null
  actions: VocabAction[]
  objects: VocabObject[]
}

export interface Task {
  id: string
  job_id: string
  title: string
  /** Длительность из метаданных файла; null — пока не известна. */
  duration_ms: number | null
  created_at: string
  status: JobStatus
  progress: number
  vocab: TaskVocabulary
  /** Имя исходного файла — показываем, когда blob потерян после перезагрузки. */
  file_name: string | null
  scenario: MockScenario
  /** Отправляли ли review — на карточке видно, что задача закрыта. */
  reviewed_at: string | null
}

const STORAGE_KEY = 'automarkup.tasks.v1'

/** blob-URL-ы держим вне стора: они не сериализуются и живут одну вкладку. */
const videoUrls = new Map<string, string>()

export const setTaskVideoUrl = (taskId: string, url: string) => {
  const prev = videoUrls.get(taskId)
  if (prev && prev !== url) URL.revokeObjectURL(prev)
  videoUrls.set(taskId, url)
}
export const getTaskVideoUrl = (taskId: string) => videoUrls.get(taskId) ?? null

function load(): Task[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as Task[]) : []
  } catch {
    return []
  }
}

function persist(tasks: Task[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks))
  } catch {
    // Приватный режим или переполнение — список просто не переживёт перезагрузку.
  }
}

interface TasksState {
  tasks: Task[]
  addTask: (task: Task) => void
  updateTask: (id: string, patch: Partial<Task>) => void
  removeTask: (id: string) => void
  getTask: (id: string) => Task | undefined
}

export const useTasksStore = create<TasksState>((set, get) => ({
  tasks: load(),

  addTask: (task) =>
    set((s) => {
      const tasks = [task, ...s.tasks]
      persist(tasks)
      return { tasks }
    }),

  updateTask: (id, patch) =>
    set((s) => {
      const tasks = s.tasks.map((t) => (t.id === id ? { ...t, ...patch } : t))
      persist(tasks)
      return { tasks }
    }),

  removeTask: (id) =>
    set((s) => {
      const url = videoUrls.get(id)
      if (url) {
        URL.revokeObjectURL(url)
        videoUrls.delete(id)
      }
      const tasks = s.tasks.filter((t) => t.id !== id)
      persist(tasks)
      return { tasks }
    }),

  getTask: (id) => get().tasks.find((t) => t.id === id),
}))

/** Сохранённые словари для повторного использования при создании задачи. */
const VOCAB_KEY = 'automarkup.vocabs.v1'

export interface SavedVocab extends TaskVocabulary {
  name: string
}

export function loadSavedVocabs(): SavedVocab[] {
  try {
    const raw = localStorage.getItem(VOCAB_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as SavedVocab[]) : []
  } catch {
    return []
  }
}

export function saveVocab(vocab: SavedVocab) {
  const existing = loadSavedVocabs().filter((v) => v.name !== vocab.name)
  try {
    localStorage.setItem(VOCAB_KEY, JSON.stringify([vocab, ...existing]))
  } catch {
    // Не критично: словарь просто не попадёт в список сохранённых.
  }
}
