export type Level = 'coarse' | 'fine'
export type Source = 'auto' | 'edited' | 'manual'

export interface Step {
  id: number
  level: Level
  parent_id: number | null
  start_sec: number
  end_sec: number
  action: string
  object: string | null
  keyframe_sec: number | null
  confidence: number | null
  source: Source
  verified: boolean
}

export interface VideoMeta {
  id: string
  filename: string
  duration_sec: number
  fps: number
  width: number
  height: number
}

export interface Provenance {
  app_version: string
  pipeline: string
  vocabulary: string
  models: Record<string, string>
  backend: string | null
  processing_sec: number | null
  created_at: string
}

export interface Annotation {
  video: VideoMeta
  steps: Step[]
  provenance: Provenance
}

export interface VideoRecord extends VideoMeta {
  status: 'queued' | 'processing' | 'done' | 'failed'
  error: string | null
  processing_sec: number | null
  motion: number[]
  filmstrip: string[]
}

export interface Vocabulary {
  name: string
  version: number
  description: string
  actions: string[]
  objects: string[]
  pairs: Record<string, string[]> | null
}
