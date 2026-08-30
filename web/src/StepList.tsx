import { frameUrl } from './api'
import { isUncertain, sortSteps } from './steps'
import type { Step, Vocabulary } from './types'

interface Props {
  videoId: string
  steps: Step[]
  vocabulary: Vocabulary | null
  selectedId: number | null
  onSelect: (id: number) => void
  onSeek: (time: number) => void
  onUpdate: (id: number, patch: Partial<Step>) => void
  onSplit: (id: number) => void
  onMerge: (id: number) => void
  onDelete: (id: number) => void
  onKeyframeHere: (id: number) => void
}

export function StepList(props: Props) {
  const { vocabulary } = props
  const ordered = sortSteps(props.steps)

  const objectsFor = (action: string, current: string | null): string[] => {
    const allowed = vocabulary?.pairs?.[action] ?? vocabulary?.objects ?? []
    return current && !allowed.includes(current) ? [current, ...allowed] : allowed
  }

  return (
    <div className="steps">
      {ordered.map((step, index) => {
        const selected = step.id === props.selectedId
        return (
          <div
            key={step.id}
            className={`step ${selected ? 'selected' : ''} ${isUncertain(step) ? 'uncertain' : ''}`}
            onClick={() => props.onSelect(step.id)}
          >
            <div className="step-head">
              <span className={`badge hue-${step.id % 6}`}>{index + 1}</span>
              <button className="time" onClick={() => props.onSeek(step.start_sec)}>
                {step.start_sec.toFixed(2)} — {step.end_sec.toFixed(2)} с
              </button>
              <span className="grow" />
              {step.source !== 'auto' && <span className="tag">правка</span>}
              {step.confidence !== null && (
                <span className="conf" title="Уверенность модели">
                  {Math.round(step.confidence * 100)}%
                </span>
              )}
            </div>

            <div className="step-body">
              {step.keyframe_sec !== null && (
                <img
                  className="keyframe"
                  src={frameUrl(props.videoId, step.keyframe_sec)}
                  alt="ключевой кадр"
                  onClick={() => props.onSeek(step.keyframe_sec as number)}
                />
              )}

              <div className="fields">
                <label>
                  действие
                  <select
                    value={step.action}
                    onChange={(event) => {
                      const action = event.target.value
                      const allowed = vocabulary?.pairs?.[action] ?? vocabulary?.objects ?? []
                      const keepObject = step.object && allowed.includes(step.object)
                      props.onUpdate(step.id, {
                        action,
                        object: keepObject ? step.object : (allowed[0] ?? null),
                      })
                    }}
                  >
                    {(vocabulary?.actions ?? [step.action]).map((action) => (
                      <option key={action} value={action}>
                        {action}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  объект
                  <select
                    value={step.object ?? ''}
                    onChange={(event) =>
                      props.onUpdate(step.id, { object: event.target.value || null })
                    }
                  >
                    <option value="">—</option>
                    {objectsFor(step.action, step.object).map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>

            {selected && (
              <div className="step-actions">
                <button onClick={() => props.onKeyframeHere(step.id)} title="Клавиша K">
                  кадр сюда
                </button>
                <button onClick={() => props.onSplit(step.id)} title="Клавиша S">
                  разделить
                </button>
                <button onClick={() => props.onMerge(step.id)} title="Клавиша M">
                  слить со следующим
                </button>
                <button className="danger" onClick={() => props.onDelete(step.id)} title="Delete">
                  удалить
                </button>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
