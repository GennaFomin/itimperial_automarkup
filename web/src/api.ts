import type { Annotation, VideoRecord, Vocabulary } from './types'

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(detail.detail ?? `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const listVideos = () => fetch('/api/videos').then(json<VideoRecord[]>)

export const getVideo = (id: string) => fetch(`/api/videos/${id}`).then(json<VideoRecord>)

export const getVocabulary = () => fetch('/api/vocabulary').then(json<Vocabulary>)

export const getStats = () =>
  fetch('/api/stats').then(json<{ videos: number; total_sec: number; median_sec: number }>)

export function uploadVideo(file: File): Promise<{ id: string }> {
  const body = new FormData()
  body.append('file', file)
  return fetch('/api/videos', { method: 'POST', body }).then(json<{ id: string }>)
}

export interface Alternative {
  action: string
  object: string | null
}

export const getAnnotation = (id: string) =>
  fetch(`/api/videos/${id}/annotation`).then(
    json<{
      annotation: Annotation
      problems: string[]
      alternatives: Record<string, Alternative[]>
    }>,
  )

export const saveAnnotation = (id: string, annotation: Annotation) =>
  fetch(`/api/videos/${id}/annotation`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotation),
  }).then(json<{ saved: boolean; problems: string[] }>)

export function logEvent(id: string, kind: string, payload: Record<string, unknown> = {}) {
  return fetch(`/api/videos/${id}/events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, payload }),
    keepalive: true,
  })
}

export const mediaUrl = (id: string) => `/api/videos/${id}/media`
export const frameUrl = (id: string, t: number) => `/api/videos/${id}/frame?t=${t.toFixed(3)}`
export const stripUrl = (id: string, name: string) => `/api/videos/${id}/strip/${name}`
export const exportUrl = (id: string, format: 'json' | 'csv') =>
  `/api/videos/${id}/export.${format}`
