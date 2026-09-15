const state = { sessionId: null, busy: false };

const landing = document.querySelector('#landing');
const chatView = document.querySelector('#chat-view');
const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const fileName = document.querySelector('#file-name');
const uploadStatus = document.querySelector('#upload-status');
const messages = document.querySelector('#messages');
const questionForm = document.querySelector('#question-form');
const questionInput = document.querySelector('#question-input');
const replaceButton = document.querySelector('#replace-button');
const sendButton = document.querySelector('.send-button');

function setStatus(message, kind = '') {
  uploadStatus.textContent = message;
  uploadStatus.className = `upload-status ${kind}`;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' })[character]);
}

function formatAnswer(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(\[S\d+\])/g, '<span class="citation">$1</span>')
    .replace(/\n/g, '<br>');
}

function scrollMessages() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function addMessage(role, content, sources = []) {
  const message = document.createElement('article');
  message.className = `message ${role}`;
  const marker = role === 'assistant' ? 'PAPERLENS' : 'YOU';
  const label = role === 'assistant' ? 'GROUNDED RESPONSE' : 'YOUR QUESTION';
  message.innerHTML = `<div class="message-marker">${marker}</div><div class="message-content"><div class="message-label">${label}</div><div class="message-text">${formatAnswer(content)}</div></div>`;
  messages.append(message);
  if (sources.length) addSources(message.querySelector('.message-content'), sources);
  scrollMessages();
  return message;
}

function addSources(parent, sources) {
  const wrapper = document.createElement('section');
  wrapper.className = 'sources';
  const toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.type = 'button';
  toggle.setAttribute('aria-expanded', 'false');
  toggle.innerHTML = `<span>SOURCE EVIDENCE / ${sources.length} PASSAGE${sources.length === 1 ? '' : 'S'}</span><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m6 9 6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="square"/></svg>`;
  const list = document.createElement('div');
  list.className = 'sources-list';
  list.setAttribute('aria-hidden', 'true');
  sources.forEach((source) => {
    const item = document.createElement('article');
    item.className = 'source';
    item.innerHTML = `<div class="source-header"><span class="source-id">[${escapeHtml(source.label)}]</span><span class="source-location"></span><span class="source-score">MATCH ${Number(source.score).toFixed(2)}</span></div><div class="source-text"></div>`;
    item.querySelector('.source-location').textContent = source.location;
    item.querySelector('.source-text').textContent = source.text;
    list.append(item);
  });
  toggle.addEventListener('click', () => {
    const isOpen = list.classList.toggle('is-open');
    list.setAttribute('aria-hidden', String(!isOpen));
    toggle.setAttribute('aria-expanded', String(isOpen));
    toggle.querySelector('span').textContent = `${isOpen ? 'HIDE EVIDENCE' : 'SOURCE EVIDENCE'} / ${sources.length} PASSAGE${sources.length === 1 ? '' : 'S'}`;
  });
  wrapper.append(toggle, list);
  parent.append(wrapper);
}

function showEmptyChat() {
  messages.replaceChildren();
  const empty = document.createElement('p');
  empty.className = 'empty-chat';
  empty.textContent = 'Document indexed. Ask about methods, findings, limitations, or any claim in the paper.';
  messages.append(empty);
}

function showChat(filename, chunkCount, retrievalMode) {
  document.querySelector('#active-filename').textContent = filename;
  const retrievalLabel = retrievalMode === 'local_fallback' ? 'LOCAL RETRIEVAL FALLBACK ACTIVE' : 'SEMANTIC RETRIEVAL INDEXED';
  document.querySelector('#chunk-count').textContent = `${chunkCount} PASSAGES · ${retrievalLabel}`;
  landing.hidden = true;
  chatView.hidden = false;
  showEmptyChat();
  questionInput.focus();
}

function apiError(payload) {
  return payload?.detail || 'The request could not be completed. Please try again.';
}

async function uploadFile(file) {
  if (!file || state.busy) return;
  state.busy = true;
  fileName.textContent = file.name;
  setStatus('CREATING SEMANTIC DOCUMENT INDEX...', 'is-working');
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await fetch('/api/upload', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(apiError(payload));
    state.sessionId = payload.session_id;
    setStatus(payload.retrieval_mode === 'local_fallback' ? 'DOCUMENT READY — LOCAL RETRIEVAL FALLBACK ACTIVE' : 'DOCUMENT READY', '');
    showChat(payload.filename, payload.chunk_count, payload.retrieval_mode);
  } catch (error) {
    setStatus(error.message.toUpperCase(), 'is-error');
    fileInput.value = '';
    fileName.textContent = 'NO FILE SELECTED';
  } finally {
    state.busy = false;
  }
}

fileInput.addEventListener('change', () => uploadFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('drag-over'); }));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('drag-over'); }));
dropZone.addEventListener('drop', (event) => uploadFile(event.dataTransfer.files[0]));

questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
});

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    questionForm.requestSubmit();
  }
});

questionForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !state.sessionId || state.busy) return;
  state.busy = true;
  sendButton.disabled = true;
  const empty = messages.querySelector('.empty-chat');
  if (empty) empty.remove();
  addMessage('user', question);
  questionInput.value = '';
  questionInput.style.height = 'auto';
  const pending = addMessage('assistant', '<span class="typing"><span></span><span></span><span></span></span>');
  pending.querySelector('.message-text').innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, question, result_count: 5 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(apiError(payload));
    pending.remove();
    addMessage('assistant', payload.answer, payload.sources);
  } catch (error) {
    pending.remove();
    addMessage('assistant', `Request unavailable: ${error.message}`);
  } finally {
    state.busy = false;
    sendButton.disabled = false;
    questionInput.focus();
  }
});

replaceButton.addEventListener('click', async () => {
  if (state.sessionId) await fetch(`/api/session/${state.sessionId}`, { method: 'DELETE' });
  state.sessionId = null;
  chatView.hidden = true;
  landing.hidden = false;
  fileInput.value = '';
  fileName.textContent = 'NO FILE SELECTED';
  setStatus('TEXT-BASED PDFS PRESERVE PAGE REFERENCES');
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
