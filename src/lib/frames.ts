/**
 * Извлечение кадров из локального видео для панели ключевых кадров.
 *
 * Один скрытый <video> на весь экран и последовательная очередь: параллельные
 * seek по одному элементу гонятся друг с другом, а несколько декодеров сразу
 * кладут вкладку на длинном ролике.
 */

export interface FrameRequest {
  key: string
  timeMs: number
}

const THUMB_WIDTH = 168

export class FrameExtractor {
  private video: HTMLVideoElement | null = null
  private queue: FrameRequest[] = []
  private running = false
  private cache = new Map<string, string>()
  private pending = new Set<string>()
  private disposed = false

  constructor(
    private src: string,
    private onFrame: (key: string, url: string) => void,
  ) {}

  get(key: string): string | undefined {
    return this.cache.get(key)
  }

  /** Поставить кадр в очередь. Уже готовые и уже запрошенные пропускаем. */
  request(req: FrameRequest) {
    if (this.disposed || this.cache.has(req.key) || this.pending.has(req.key)) return
    this.pending.add(req.key)
    this.queue.push(req)
    void this.drain()
  }

  /** Видимые кадры важнее: двигаем их в голову очереди. */
  prioritize(keys: string[]) {
    if (this.queue.length < 2) return
    const set = new Set(keys)
    this.queue.sort((a, b) => Number(set.has(b.key)) - Number(set.has(a.key)))
  }

  private async ensureVideo(): Promise<HTMLVideoElement> {
    if (this.video) return this.video
    const v = document.createElement('video')
    v.preload = 'auto'
    v.muted = true
    v.playsInline = true
    v.crossOrigin = 'anonymous'
    v.src = this.src
    this.video = v
    await new Promise<void>((resolve, reject) => {
      v.onloadeddata = () => resolve()
      v.onerror = () => reject(new Error('Видео не открылось'))
    })
    return v
  }

  private async drain() {
    if (this.running || this.disposed) return
    this.running = true
    try {
      const video = await this.ensureVideo()
      const canvas = document.createElement('canvas')
      while (this.queue.length && !this.disposed) {
        const req = this.queue.shift()!
        try {
          const url = await this.grab(video, canvas, req.timeMs)
          if (this.disposed) {
            URL.revokeObjectURL(url)
            break
          }
          this.cache.set(req.key, url)
          this.onFrame(req.key, url)
        } catch {
          // Кадр не сняли — панель покажет заглушку, это не повод падать.
        } finally {
          this.pending.delete(req.key)
        }
      }
    } catch {
      // Видео недоступно целиком: чистим очередь, чтобы не крутиться впустую.
      this.pending.clear()
      this.queue = []
    } finally {
      this.running = false
    }
  }

  private grab(video: HTMLVideoElement, canvas: HTMLCanvasElement, timeMs: number): Promise<string> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('seek timeout')), 4000)
      const onSeeked = () => {
        clearTimeout(timer)
        video.removeEventListener('seeked', onSeeked)
        try {
          const ratio = video.videoHeight / (video.videoWidth || 1)
          canvas.width = THUMB_WIDTH
          canvas.height = Math.max(1, Math.round(THUMB_WIDTH * (ratio || 0.5625)))
          const ctx = canvas.getContext('2d')
          if (!ctx) return reject(new Error('no 2d context'))
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          canvas.toBlob(
            (blob) => (blob ? resolve(URL.createObjectURL(blob)) : reject(new Error('no blob'))),
            'image/jpeg',
            0.72,
          )
        } catch (e) {
          reject(e instanceof Error ? e : new Error('draw failed'))
        }
      }
      video.addEventListener('seeked', onSeeked)
      video.currentTime = Math.max(0, timeMs / 1000)
    })
  }

  dispose() {
    this.disposed = true
    this.queue = []
    this.pending.clear()
    for (const url of this.cache.values()) URL.revokeObjectURL(url)
    this.cache.clear()
    if (this.video) {
      this.video.removeAttribute('src')
      this.video.load()
      this.video = null
    }
  }
}
