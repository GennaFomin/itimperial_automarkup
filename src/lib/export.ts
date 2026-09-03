/**
 * Экспорт review в JSON и CSV — контракт §5.
 * Порядок колонок CSV фиксирован и не меняется без бампа версии схемы.
 */
import type { Prediction, Review } from '../api/types'
import type { EditableSegment } from './segments'
import { toReviewSegments } from './segments'

export function buildReview(
  prediction: Prediction,
  segments: EditableSegment[],
  reviewer: string,
  timeSpentMs: number,
): Review {
  return {
    schema_version: '1.0',
    prediction_id: prediction.prediction_id,
    reviewer,
    submitted_at: new Date().toISOString(),
    segments: toReviewSegments(segments),
    time_spent_ms: Math.max(0, Math.round(timeSpentMs)),
  }
}

export function buildExportJson(
  prediction: Prediction,
  segments: EditableSegment[],
  source: 'review' | 'prediction',
): string {
  const doc =
    source === 'review'
      ? {
          schema_version: '1.0',
          source,
          video: prediction.video,
          segments: toReviewSegments(segments),
          model_version: prediction.model_version,
          vocab_version: prediction.vocab_version,
          exported_at: new Date().toISOString(),
        }
      : {
          schema_version: '1.0',
          source,
          video: prediction.video,
          segments: prediction.segments,
          model_version: prediction.model_version,
          vocab_version: prediction.vocab_version,
          exported_at: new Date().toISOString(),
        }
  return JSON.stringify(doc, null, 2)
}

const CSV_HEADER =
  'video_id,segment_id,start_ms,end_ms,action,object,keyframe_ms,confidence_action,confidence_object,origin'

/** Экранируем по RFC 4180: кавычки удваиваем, поле берём в кавычки при need. */
function csvCell(value: string | number | null): string {
  if (value === null) return ''
  const s = String(value)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function buildExportCsv(
  prediction: Prediction,
  segments: EditableSegment[],
  source: 'review' | 'prediction',
  videoId: string,
): string {
  const rows =
    source === 'review'
      ? toReviewSegments(segments).map((s) => [
          videoId,
          s.id,
          s.start_ms,
          s.end_ms,
          s.action,
          s.object,
          s.keyframe_ms,
          // Для source=review колонки confidence пустые: человек не выдаёт вероятностей.
          null,
          null,
          s.origin,
        ])
      : prediction.segments.map((s) => [
          videoId,
          s.id,
          s.start_ms,
          s.end_ms,
          s.action.value,
          s.object.value,
          s.keyframe_ms,
          s.action.confidence,
          s.object.confidence,
          'model',
        ])

  const body = rows.map((r) => r.map(csvCell).join(',')).join('\n')
  // UTF-8 с BOM — иначе Excel ломает кириллицу.
  return `﻿${CSV_HEADER}\n${body}${rows.length ? '\n' : ''}`
}

export function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Отзываем на следующем тике: Safari не успевает начать скачивание синхронно.
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
