const nextBtn = document.getElementById('next-btn');
const searchBtn = document.getElementById('search-btn');
const networkSection = document.getElementById('network-section');
const querySection = document.getElementById('query-section');
const resultsDiv = document.getElementById('results');
const netInfo = document.getElementById('netinfo');
const resultsList = document.getElementById('results-list');
const API_BASE = "http://aps105.ece.utoronto.ca:61427/api";

// header buttons
const backBtn = document.getElementById('back-btn');
const closeBtn = document.getElementById('close-popup');

// init header button states
if (backBtn) backBtn.style.display = 'none';

// close
closeBtn?.addEventListener('click', () => window.close());

// back
backBtn?.addEventListener('click', () => {
  querySection.classList.add('hidden');
  networkSection.classList.remove('hidden');
  resultsDiv.classList.add('hidden');
  resultsList.innerHTML = '';
  backBtn.style.display = 'none';
});

// tooltip toggle on click
netInfo?.addEventListener('click', () => {
  netInfo.classList.toggle('show');
});

// registration check
async function isRegisteredNetwork(id) {
  const r = await fetch(`${API_BASE}/is-registered?network_id=${encodeURIComponent(id)}`);
  const j = await r.json();
  return !!j.registered;
}

// after entering network id, go to query page (only if registered)
nextBtn.addEventListener('click', async () => {
  const netId = document.getElementById('network-id').value.trim();
  if (!netId) { alert('Please enter your Course Network ID.'); return; }

  const ok = await isRegisteredNetwork(netId);
  if (!ok) { alert('This course is not registered yet.'); return; }

  chrome?.storage?.local.set({ networkId: netId });
  networkSection.classList.add('hidden');
  querySection.classList.remove('hidden');
  if (backBtn) backBtn.style.display = 'inline-block';
});

// call backend and render results
searchBtn.addEventListener('click', async () => {
  const q = document.getElementById('query').value.trim();
  if (!q) { alert('Please type a question.'); return; }

  // get the network id
  let netId = document.getElementById('network-id').value.trim();
  if (!netId) {
    try {
      const obj = await new Promise(resolve => chrome.storage.local.get(['networkId'], resolve));
      netId = (obj.networkId || '').trim();
    } catch { }
  }
  if (!netId) { alert('No network ID found.'); return; }

  // searching
  searchBtn.disabled = true;
  const oldText = searchBtn.textContent;
  searchBtn.textContent = 'Searching…';

  try {
    // call backend
    const res = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ network_id: netId, query: q, k: 10 })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Search failed');

    // render results as links

    const urlBase = `https://piazza.com/class/${encodeURIComponent(netId)}/post/`;

    for (const item of data.results) {
      const li = document.createElement('li');

      const a = document.createElement('a');
      a.href = urlBase + encodeURIComponent(item.post_id);
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = `#${item.post_id}: ${item.subject}`;

      const score = document.createElement('span');
      score.style.opacity = '0.6';
      score.textContent = ` (score: ${item.score.toFixed(4)})`;

      li.appendChild(a);
      li.appendChild(score);
      resultsList.appendChild(li);
    }
    resultsDiv.classList.remove('hidden');
  } catch (e) {
    alert(e.message);
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = oldText;
  }
});

// open result links in a background tab
resultsList.addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  e.preventDefault();

  const url = a.href;
  if (chrome?.tabs?.create) {
    chrome.tabs.create({ url, active: false }); // open in background
  } else {
    window.open(url, '_blank');
  }
});
