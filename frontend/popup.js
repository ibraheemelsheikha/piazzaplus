const nextBtn = document.getElementById('next-btn');
const searchBtn = document.getElementById('search-btn');
const networkSection = document.getElementById('network-section');
const querySection = document.getElementById('query-section');
const resultsDiv = document.getElementById('results');
const netInfo = document.getElementById('netinfo');
const resultsList = document.getElementById('results-list');
const API_BASE = "http://aps105v2.ece.utoronto.ca:61427/api";

// header buttons
const backBtn = document.getElementById('back-btn');
const closeBtn = document.getElementById('close-popup');

const networkInput = document.getElementById('network-id');
const courseLabelInput = document.getElementById('course-label');
const savedCoursesSection = document.getElementById('saved-courses');
const savedCoursesList = document.getElementById('saved-courses-list');

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

function loadSavedCourses() {
  if (!chrome?.storage?.local) return;

  chrome.storage.local.get(['savedCourses', 'networkId'], (data) => {
    const saved = data.savedCourses || {};
    const entries = Object.values(saved);

    if (!entries.length) {
      savedCoursesSection.classList.add('hidden');
      savedCoursesList.innerHTML = '';
      return;
    }

    entries.sort((a, b) => (b.lastUsed || 0) - (a.lastUsed || 0));

    savedCoursesSection.classList.remove('hidden');
    savedCoursesList.innerHTML = '';

    for (const course of entries) {
      const li = document.createElement('li');
      li.className = 'saved-course';
      li.dataset.nid = course.networkId;

      // Create a span for the text content
      const textSpan = document.createElement('span');
      textSpan.textContent = course.displayName || course.networkId;
      li.appendChild(textSpan);

      // Create the Delete (X) Button
      const delBtn = document.createElement('button');
      delBtn.className = 'delete-course-btn';
      delBtn.innerHTML = '&times;';
      delBtn.title = 'Remove course';

      delBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering the course selection
        deleteCourse(course.networkId);
      });

      li.tabIndex = 0;
      li.setAttribute('role', 'button');
      li.title = 'Click to select this course';

      li.appendChild(delBtn);
      savedCoursesList.appendChild(li);
    }
  });
}

function deleteCourse(networkId) {
  if (!chrome?.storage?.local) return;
  chrome.storage.local.get(['savedCourses'], (data) => {
    const saved = data.savedCourses || {};
    delete saved[networkId];
    chrome.storage.local.set({ savedCourses: saved }, () => {
      loadSavedCourses();
    });
  });
}

function saveCourse(networkId, displayName) {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) return resolve();

    chrome.storage.local.get(['savedCourses'], (data) => {
      const saved = data.savedCourses || {};

      saved[networkId] = {
        networkId,
        displayName: displayName || networkId,
        lastUsed: Date.now(),
      };

      chrome.storage.local.set(
        { savedCourses: saved, networkId },
        () => resolve()
      );
    });
  });
}

// registration check
async function isRegisteredNetwork(id) {
  const r = await fetch(`${API_BASE}/is-registered?network_id=${encodeURIComponent(id)}`);
  const j = await r.json();
  return !!j.registered;
}

// after entering network id, go to query page (only if registered)
nextBtn.addEventListener('click', async () => {
  const netId = networkInput.value.trim();
  if (!netId) { alert('Please enter your Course Network ID.'); return; }

  const ok = await isRegisteredNetwork(netId);
  if (!ok) { alert('This course is not registered yet.'); return; }

  const displayName =
    (courseLabelInput && courseLabelInput.value.trim()) || netId;

  await saveCourse(netId, displayName);
  loadSavedCourses(); // refresh the list in the UI

  chrome?.storage?.local.set({ networkId: netId });
  networkSection.classList.add('hidden');
  querySection.classList.remove('hidden');
  if (backBtn) backBtn.style.display = 'inline-block';
});

// call backend and render results
searchBtn.addEventListener('click', async () => {
  const q = document.getElementById('query').value.trim();
  if (!q) { alert('Please type a question.'); return; }

  resultsList.innerHTML = '';
  resultsDiv.classList.add('hidden');
  resultsList.scrollTop = 0;
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
      body: JSON.stringify({ network_id: netId, query: q, k: 20 })
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

      li.appendChild(a);
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

document.addEventListener('DOMContentLoaded', () => {
  loadSavedCourses();
});

// clicking a saved course fills fields
if (savedCoursesList) {
  savedCoursesList.addEventListener('click', (e) => {
    const li = e.target.closest('.saved-course');
    const delBtn = e.target.closest('.delete-course-btn');
    if (!li || delBtn) return; // ignore if clicking delete button

    const nid = li.dataset.nid;
    if (!nid) return;

    // fill network ID and label fields
    if (networkInput) {
      networkInput.value = nid;
    }

    // extract text from the span
    const span = li.querySelector('span');
    if (courseLabelInput && span) {
      courseLabelInput.value = span.textContent || '';
    }

    nextBtn?.click();
  });
}

resultsList.addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  e.preventDefault();

  const url = a.href;

  // open in background
  if (chrome?.tabs?.create) {
    chrome.tabs.create({ url, active: false });
  } else {
    window.open(url, '_blank');
  }
});