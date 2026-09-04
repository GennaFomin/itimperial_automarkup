/** Всё время — целые миллисекунды от начала видео (контракт, §1). */

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/** `1:23.4` — компактный таймкод для плотных мест вроде подписей на таймлайне. */
export function formatShort(ms: number): string {
  const total = Math.max(0, Math.round(ms))
  const m = Math.floor(total / 60_000)
  const s = Math.floor((total % 60_000) / 1000)
  const d = Math.floor((total % 1000) / 100)
  return `${m}:${String(s).padStart(2, '0')}.${d}`
}

/** `01:23.456` — точный таймкод для инспектора и полей ввода. */
export function formatPrecise(ms: number): string {
  const total = Math.max(0, Math.round(ms))
  const m = Math.floor(total / 60_000)
  const s = Math.floor((total % 60_000) / 1000)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(total % 1000).padStart(3, '0')}`
}

/** `2 мин 04 с` — для карточек списка задач. */
export function formatDuration(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000))
  const m = Math.floor(total / 60)
  const s = total % 60
  if (m === 0) return `${s} с`
  return `${m} мин ${String(s).padStart(2, '0')} с`
}

/** Разбор `mm:ss.mmm`, `ss.mmm` или голых миллисекунд. Null — ввод не понят. */
export function parseTimecode(input: string): number | null {
  const raw = input.trim().replace(',', '.')
  if (!raw) return null
  const m = raw.match(/^(?:(\d+):)?(\d{1,2})(?:\.(\d{1,3}))?$/)
  if (m) {
    const min = m[1] ? Number(m[1]) : 0
    const sec = Number(m[2])
    const frac = m[3] ? Number(m[3].padEnd(3, '0')) : 0
    return min * 60_000 + sec * 1000 + frac
  }
  if (/^\d+$/.test(raw)) return Number(raw)
  return null
}

/** Шаг сетки таймлайна: «круглый» интервал, дающий метки примерно через 90 px. */
export function tickStepMs(msPerPx: number): number {
  const target = msPerPx * 90
  const steps = [
    100, 200, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000,
    300_000, 600_000, 900_000, 1_800_000,
  ]
  return steps.find((s) => s >= target) ?? steps[steps.length - 1]
}
