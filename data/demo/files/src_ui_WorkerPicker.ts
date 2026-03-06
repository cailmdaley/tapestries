import { escapeHtml } from './utils'

export interface WorkerInfo {
  id: string
  name: string
  tmuxSession: string
}

export interface WorkerPickerCallbacks {
  onSelectWorker: (workerId: string) => void
  onNewWorker: () => void
}

/**
 * Show a modal overlay listing available workers.
 * Returns immediately; callbacks fire on selection.
 */
export function showWorkerPicker(
  workers: WorkerInfo[],
  annotationCount: number,
  callbacks: WorkerPickerCallbacks
): HTMLElement {
  const picker = document.createElement('div')
  picker.className = 'worker-picker-overlay'
  picker.innerHTML = `
    <div class="worker-picker">
      <div class="worker-picker-header">
        <span>Send ${annotationCount} annotation${annotationCount === 1 ? '' : 's'} to worker</span>
        <button class="worker-picker-close">&times;</button>
      </div>
      <div class="worker-picker-list">
        <button class="worker-picker-item worker-picker-new" data-action="new">
          <span class="worker-name">+ New Worker</span>
          <span class="worker-session">Create new worker and send</span>
        </button>
        ${workers.map(w => `
          <button class="worker-picker-item" data-worker-id="${escapeHtml(w.id)}">
            <span class="worker-name">${escapeHtml(w.name)}</span>
            <span class="worker-session">${escapeHtml(w.tmuxSession)}</span>
          </button>
        `).join('')}
      </div>
    </div>
  `

  document.body.appendChild(picker)

  picker.querySelector('.worker-picker-close')?.addEventListener('click', () => {
    picker.remove()
  })

  picker.addEventListener('click', (e) => {
    if (e.target === picker) picker.remove()
  })

  picker.querySelector('.worker-picker-new')?.addEventListener('click', () => {
    picker.remove()
    callbacks.onNewWorker()
  })

  picker.querySelectorAll('.worker-picker-item:not(.worker-picker-new)').forEach(item => {
    item.addEventListener('click', () => {
      const workerId = item.getAttribute('data-worker-id')!
      picker.remove()
      callbacks.onSelectWorker(workerId)
    })
  })

  return picker
}
