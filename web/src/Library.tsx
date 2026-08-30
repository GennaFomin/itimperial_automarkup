import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from './api'
import type { VideoRecord } from './types'

const STATUS: Record<string, string> = {
  queued: 'в очереди',
  processing: 'обрабатывается',
  done: 'готово',
  failed: 'ошибка',
}

export function Library({ onOpen }: { onOpen: (id: string) => void }) {
  const [videos, setVideos] = useState<VideoRecord[]>([])
  const [stats, setStats] = useState<{ videos: number; median_sec: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    const [list, aggregated] = await Promise.all([api.listVideos(), api.getStats()])
    setVideos(list)
    setStats(aggregated)
  }, [])

  useEffect(() => {
    void refresh()
    const timer = setInterval(refresh, 2000)
    return () => clearInterval(timer)
  }, [refresh])

  const upload = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return
      setBusy(true)
      setError(null)
      try {
        for (const file of Array.from(files)) await api.uploadVideo(file)
        await refresh()
      } catch (failure) {
        setError((failure as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [refresh],
  )

  return (
    <div className="library">
      <h1>Praxis</h1>
      <p className="muted">
        Автоматическая разметка действий: ролик делится на шаги, каждому определяются
        действие, объект и ключевой кадр. Дальше вы правите, а не размечаете с нуля.
      </p>

      <div
        className={`dropzone ${busy ? 'busy' : ''}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          void upload(event.dataTransfer.files)
        }}
        onClick={() => input.current?.click()}
      >
        {busy ? 'загружаю…' : 'перетащите MP4 или MOV сюда — до 30 секунд, от 720p'}
        <input
          ref={input}
          type="file"
          accept="video/mp4,video/quicktime"
          multiple
          hidden
          onChange={(event) => void upload(event.target.files)}
        />
      </div>

      {error && <div className="banner">{error}</div>}

      {stats && stats.videos > 0 && (
        <div className="stats">
          проверено роликов: {stats.videos} · медианное время проверки:{' '}
          {stats.median_sec.toFixed(0)} с
        </div>
      )}

      <div className="grid">
        {videos.map((video) => (
          <button
            key={video.id}
            className={`card ${video.status}`}
            onClick={() => onOpen(video.id)}
            disabled={video.status === 'failed'}
          >
            {video.status === 'done' ? (
              <img src={api.frameUrl(video.id, video.duration_sec / 2)} alt="" />
            ) : (
              <div className="thumb-placeholder">{STATUS[video.status]}</div>
            )}
            <span className="name">{video.filename}</span>
            <span className="muted">
              {video.duration_sec.toFixed(1)} с · {STATUS[video.status]}
              {video.processing_sec ? ` · ${video.processing_sec.toFixed(1)} с обработки` : ''}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
