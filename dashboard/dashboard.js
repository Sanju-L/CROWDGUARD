/* ═══════════════════════════════════════════════
   CROWDGUARD — AI Crowd Management Dashboard
   Connected to YOLO + ByteTrack Python Backend
   ═══════════════════════════════════════════════ */

(function () {
    'use strict';

    // ══════════════════════════════════════════
    //  CONFIGURATION
    // ══════════════════════════════════════════
    const API_BASE = '';
    const POLL_INTERVAL = 500; // ms between API polls

    async function readApiJson(response, operation) {
        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
            if (response.status === 404) {
                throw new Error(
                    `${operation}: camera API not found. Restart Flask from the project folder to load the multi-camera backend.`,
                );
            }
            throw new Error(`${operation}: server returned HTTP ${response.status} instead of JSON.`);
        }

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`${operation}: server returned invalid JSON (HTTP ${response.status}).`);
        }
        if (!response.ok) {
            throw new Error(data.msg || `${operation} failed (HTTP ${response.status}).`);
        }
        return data;
    }

    const CONFIG = {
        zones: [
            { id: 'A', name: 'ZONE A', color: '#00e676', capacity: 200, x: 5, y: 180, w: 565, h: 100 },
            { id: 'B', name: 'ZONE B', color: '#00d4ff', capacity: 200, x: 630, y: 180, w: 565, h: 100 },
            { id: 'C', name: 'ZONE C', color: '#ffab00', capacity: 350, x: 5, y: 330, w: 565, h: 200 },
            { id: 'D', name: 'ZONE D', color: '#7c4dff', capacity: 350, x: 630, y: 330, w: 565, h: 200 },
            { id: 'E', name: 'ZONE E', color: '#ff6b35', capacity: 250, x: 5, y: 580, w: 1190, h: 120 },
        ],
        exits: [],
        svgViewBox: { w: 1200, h: 800 },
        maxHistoryPoints: 60,
    };

    // ══════════════════════════════════════════
    //  GLOBAL STATE — Updated from backend API
    // ══════════════════════════════════════════
    const state = {
        totalPeople: 0,
        density: 0,
        alerts: 0,
        criticalZones: 0,
        flowRate: 0,
        riskScore: 0,
        riskLevel: 'WAITING',
        zones: {},
        showHeatmap: true,
        showFlow: true,
        showPressure: false,
        mapZoom: 1,
        mapPanX: 0,
        mapPanY: 0,
        isPanning: false,
        panStart: { x: 0, y: 0 },
        densityHistory: [],
        alertHistory: [],
        videoLoaded: false,
        backendConnected: false,
        aiRunning: false,
        healthEndpointAvailable: null,
        tracks: [],         // YOLO tracks from backend
        personFlows: [],    // Flow data from backend
        frameInfo: {},
    };

    // ══════════════════════════════════════════
    //  VIDEO STATE
    // ══════════════════════════════════════════
    const video = {
        showTracking: true,
        showHeatmap: true,
        uploadedFilename: null,
    };

    // ══════════════════════════════════════════
    //  LOADING SCREEN
    // ══════════════════════════════════════════
    function runLoadingSequence() {
        const fill = document.getElementById('loader-fill');
        const status = document.getElementById('loader-status');
        const steps = [
            { pct: 20, msg: 'Loading dashboard interface...' },
            { pct: 45, msg: 'Preparing crowd monitoring panels...' },
            { pct: 70, msg: 'Preparing video analysis controls...' },
            { pct: 90, msg: 'Checking backend connection...' },
            { pct: 100, msg: 'Ready — Upload a video to begin.' },
        ];
        let i = 0;
        const interval = setInterval(() => {
            if (i >= steps.length) {
                clearInterval(interval);
                setTimeout(() => {
                    document.getElementById('loading-screen').classList.add('hidden');
                    document.getElementById('dashboard').classList.add('visible');
                    startDashboard();
                }, 400);
                return;
            }
            fill.style.width = steps[i].pct + '%';
            status.textContent = steps[i].msg;
            i++;
        }, 350);
    }

    // ══════════════════════════════════════════
    //  CLOCK
    // ══════════════════════════════════════════
    function updateClock() {
        const now = new Date();
        const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
        document.getElementById('clock-date').textContent =
            `${String(now.getDate()).padStart(2,'0')} ${months[now.getMonth()]} ${now.getFullYear()}`;
        document.getElementById('clock-time').textContent =
            `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    }

    // ══════════════════════════════════════════
    //  BG PARTICLES
    // ══════════════════════════════════════════
    function initBgParticles() {
        const c = document.getElementById('bg-particles-canvas');
        const ctx = c.getContext('2d');
        let pts = [];
        function resize() { c.width = window.innerWidth; c.height = window.innerHeight; }
        resize(); window.addEventListener('resize', resize);
        for (let i = 0; i < 40; i++) pts.push({
            x: Math.random()*c.width, y: Math.random()*c.height,
            vx: (Math.random()-0.5)*0.2, vy: (Math.random()-0.5)*0.2,
            r: Math.random()*1.2+0.5, a: Math.random()*0.2+0.05
        });
        (function loop() {
            ctx.clearRect(0,0,c.width,c.height);
            pts.forEach(p => {
                p.x+=p.vx; p.y+=p.vy;
                if(p.x<0)p.x=c.width; if(p.x>c.width)p.x=0;
                if(p.y<0)p.y=c.height; if(p.y>c.height)p.y=0;
                ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
                ctx.fillStyle=`rgba(0,212,255,${p.a})`; ctx.fill();
            });
            requestAnimationFrame(loop);
        })();
    }

    // ══════════════════════════════════════════
    //  VIDEO MODULE — Upload to Backend
    // ══════════════════════════════════════════
    function initVideoModule() {
        const fileInput = document.getElementById('video-file-input');
        const btnPlay = document.getElementById('btn-video-play');
        const btnPause = document.getElementById('btn-video-pause');
        const progress = document.getElementById('video-progress');
        const videoEl = document.getElementById('video-element');

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // Show uploading status
            document.getElementById('video-status-badge').className = 'video-status-badge';
            document.getElementById('video-status-text').textContent = 'UPLOADING...';
            addLogEntry(`Uploading: ${file.name}...`);

            try {
                // 1. Upload file to backend
                const formData = new FormData();
                formData.append('file', file);
                const uploadRes = await fetch(`${API_BASE}/api/upload`, {
                    method: 'POST',
                    body: formData
                });
                const uploadData = await uploadRes.json();

                if (uploadData.status !== 'ok') {
                    addLogEntry(`Upload failed: ${uploadData.msg}`);
                    document.getElementById('video-status-text').textContent = 'UPLOAD FAILED';
                    return;
                }

                video.uploadedFilename = uploadData.filename;
                addLogEntry(`Uploaded: ${uploadData.filename}`);

                // 2. Start the AI pipeline on the backend
                const startRes = await fetch(`${API_BASE}/api/command`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'start', source: uploadData.filename })
                });
                const startData = await startRes.json();

                if (startData.status !== 'ok') {
                    addLogEntry(`Start failed: ${startData.msg}`);
                    document.getElementById('video-status-text').textContent = 'START FAILED';
                    return;
                }

                // 3. Connect video feed from backend (MJPEG stream with YOLO boxes drawn)
                videoEl.style.display = 'none'; // Hide HTML5 video element
                const feedImg = document.getElementById('video-feed-img') || createFeedImage();
                feedImg.src = `${API_BASE}/video_feed`;
                feedImg.style.display = 'block';

                // 4. Update UI
                state.videoLoaded = true;
                state.backendConnected = true;
                state.aiRunning = true;
                document.getElementById('video-placeholder').classList.add('hidden');
                document.getElementById('video-info-bar').style.display = 'flex';
                document.getElementById('video-progress').style.display = 'block';

                const mapPlaceholder = document.getElementById('map-placeholder');
                if (mapPlaceholder) mapPlaceholder.classList.add('hidden');
                const mapScene = document.getElementById('map-scene');
                if (mapScene) mapScene.style.display = 'block';

                btnPlay.disabled = false;
                btnPause.disabled = false;

                document.getElementById('video-status-badge').className = 'video-status-badge analyzing';
                document.getElementById('video-status-text').textContent = 'AI ANALYZING';
                addLogEntry('YOLO + ByteTrack AI pipeline started');
                addAlertEntry('info', `AI analysis running on "${uploadData.filename}"`);

                // 5. Start polling backend for state data
                startPolling();

            } catch (err) {
                console.error('Backend connection error:', err);
                document.getElementById('video-status-text').textContent = 'BACKEND ERROR';
                addLogEntry(`Error: ${err.message}. Is the Flask backend running on port 5000?`);
                addAlertEntry('critical', 'Cannot connect to backend. Run: python app.py');
            }
        });

        btnPlay.addEventListener('click', async () => {
            if (!video.uploadedFilename) return;
            try {
                const response = await fetch(`${API_BASE}/api/command`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'start', source: video.uploadedFilename })
                });
                const result = await response.json();
                if (!response.ok || result.status !== 'ok') throw new Error(result.msg || 'Could not start AI analysis');
                state.aiRunning = true;
                document.getElementById('video-status-badge').className = 'video-status-badge analyzing';
                document.getElementById('video-status-text').textContent = 'AI ANALYZING';
                refreshBackendStatus();
            } catch(e) { console.error(e); }
        });

        btnPause.addEventListener('click', async () => {
            try {
                const response = await fetch(`${API_BASE}/api/command`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'stop' })
                });
                const result = await response.json();
                if (!response.ok || result.status !== 'ok') throw new Error(result.msg || 'Could not stop AI analysis');
                state.aiRunning = false;
                document.getElementById('video-status-badge').className = 'video-status-badge active';
                document.getElementById('video-status-text').textContent = 'PAUSED';
                refreshBackendStatus();
            } catch(e) { console.error(e); }
        });

        document.getElementById('btn-toggle-tracking').addEventListener('click', function() {
            video.showTracking = !video.showTracking;
            this.classList.toggle('active', video.showTracking);
            document.getElementById('video-tracking-canvas').classList.toggle('hidden', !video.showTracking);
        });

        document.getElementById('btn-toggle-video-heatmap').addEventListener('click', function() {
            video.showHeatmap = !video.showHeatmap;
            this.classList.toggle('active', video.showHeatmap);
            document.getElementById('video-heatmap-canvas').classList.toggle('hidden', !video.showHeatmap);
        });
    }

    let cameraPollTimer = null;
    const cameraTiles = new Map();

    function initMultiCameraDashboard() {
        const form = document.getElementById('camera-add-form');
        const sourceType = document.getElementById('camera-source-type');
        const sourceLabel = document.getElementById('camera-source-label');
        const sourceInput = document.getElementById('camera-source');

        sourceType.addEventListener('change', () => {
            const isStream = sourceType.value === 'stream';
            sourceLabel.textContent = isStream ? 'Stream URL' : 'Device index';
            sourceInput.type = isStream ? 'url' : 'text';
            sourceInput.inputMode = isStream ? 'url' : 'numeric';
            sourceInput.placeholder = isStream ? 'rtsp://host:554/stream' : '0';
            sourceInput.value = isStream ? '' : '0';
        });

        form.addEventListener('submit', async event => {
            event.preventDefault();
            const button = document.getElementById('camera-add-button');
            const error = document.getElementById('camera-error');
            const source = sourceInput.value.trim();
            const nameInput = document.getElementById('camera-name');
            if (!source) {
                error.textContent = 'Enter a camera device index or stream URL.';
                return;
            }

            button.disabled = true;
            button.textContent = 'Connecting...';
            error.textContent = '';
            try {
                const response = await fetch(`${API_BASE}/api/cameras`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: nameInput.value.trim(),
                        source_type: sourceType.value,
                        source,
                        area: document.getElementById('camera-area').value,
                    }),
                });
                const result = await readApiJson(response, 'Add camera');
                if (result.status !== 'ok') throw new Error(result.msg || 'Camera could not be added.');
                nameInput.value = '';
                await refreshCameraDashboard();
            } catch (cameraError) {
                error.textContent = cameraError.message;
            } finally {
                button.disabled = false;
                button.textContent = 'Add camera';
            }
        });

        cameraPollTimer = setInterval(refreshCameraDashboard, 1200);
        refreshCameraDashboard();
    }

    function createCameraTile(camera) {
        const tile = document.createElement('article');
        tile.className = 'camera-tile';
        tile.dataset.cameraId = camera.id;

        const header = document.createElement('div');
        header.className = 'camera-tile-header';
        const name = document.createElement('span');
        name.className = 'camera-tile-name';
        const area = document.createElement('span');
        area.className = 'camera-tile-area';
        const status = document.createElement('span');
        status.className = 'camera-tile-status';
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'camera-remove-button';
        remove.textContent = 'Remove';
        remove.setAttribute('aria-label', `Remove ${camera.name}`);
        remove.addEventListener('click', async () => {
            remove.disabled = true;
            try {
                const response = await fetch(`${API_BASE}/api/cameras/${encodeURIComponent(camera.id)}`, {
                    method: 'DELETE',
                });
                const result = await readApiJson(response, `Remove ${camera.name}`);
                if (result.status !== 'ok') throw new Error(result.msg || `Could not remove ${camera.name}.`);
                await refreshCameraDashboard();
            } catch (error) {
                document.getElementById('camera-error').textContent = error.message;
            } finally {
                remove.disabled = false;
            }
        });
        header.append(name, area, status, remove);

        const feed = document.createElement('div');
        feed.className = 'camera-tile-feed';
        const image = document.createElement('img');
        image.src = `${API_BASE}/api/cameras/${encodeURIComponent(camera.id)}/feed`;
        image.alt = `Live video feed: ${camera.name}`;
        image.loading = 'lazy';
        const offline = document.createElement('div');
        offline.className = 'camera-tile-offline';
        offline.hidden = true;
        feed.append(image, offline);

        const stats = document.createElement('div');
        stats.className = 'camera-tile-stats';
        const people = document.createElement('span');
        const density = document.createElement('span');
        const risk = document.createElement('span');
        const flow = document.createElement('span');
        stats.append(people, density, risk, flow);

        const route = document.createElement('div');
        route.className = 'camera-tile-route';
        tile.append(header, feed, stats, route);

        const elements = { tile, name, area, status, image, offline, people, density, risk, flow, route };
        cameraTiles.set(camera.id, elements);
        return elements;
    }

    async function refreshCameraDashboard() {
        try {
            const response = await fetch(`${API_BASE}/api/cameras`, { cache: 'no-store' });
            const data = await readApiJson(response, 'Load cameras');
            const grid = document.getElementById('camera-grid');
            const currentIds = new Set(data.cameras.map(camera => camera.id));

            for (const [id, elements] of cameraTiles) {
                if (!currentIds.has(id)) {
                    elements.tile.remove();
                    cameraTiles.delete(id);
                }
            }

            if (data.cameras.length === 0) {
                cameraTiles.clear();
                grid.innerHTML = '';
                const empty = document.createElement('div');
                empty.className = 'camera-empty-state';
                empty.textContent = 'Connect one or more webcams or network camera streams to begin multi-area analysis.';
                grid.appendChild(empty);
            } else {
                grid.querySelector('.camera-empty-state')?.remove();
                data.cameras.forEach(camera => {
                    const elements = cameraTiles.get(camera.id) || createCameraTile(camera);
                    if (!elements.tile.isConnected) grid.appendChild(elements.tile);
                    elements.name.textContent = camera.name;
                    elements.name.title = camera.name;
                    elements.area.textContent = camera.area;
                    elements.status.textContent = camera.status;
                    elements.status.classList.toggle(
                        'error',
                        camera.status === 'ERROR' || camera.status === 'DISCONNECTED',
                    );
                    elements.offline.hidden = camera.frame_ready && camera.status === 'ANALYZING';
                    elements.offline.textContent = camera.error || camera.status;
                    elements.people.textContent = `${camera.people} people`;
                    elements.density.textContent = `${camera.density}%`;
                    elements.risk.textContent = `Risk ${camera.risk_level}`;
                    elements.flow.textContent = camera.dominant_flow;
                    const path = camera.route?.path || [];
                    elements.route.textContent = camera.route?.direction
                        ? `${camera.route.direction}${path.length ? ` · ${path.join(' → ')}` : ''}`
                        : 'Calculating safe direction…';
                });
            }

            document.getElementById('camera-count').textContent =
                `${data.camera_count} ${data.camera_count === 1 ? 'CAMERA' : 'CAMERAS'} · ${data.analyzing_count} ANALYZING`;
            const onlineDot = document.querySelector('#camera-online-badge .status-dot');
            onlineDot.className = `status-dot ${data.analyzing_count ? 'cyan-dot' : 'idle-dot'}`;
            document.getElementById('camera-online-status').textContent =
                `CAMERAS ONLINE: ${data.analyzing_count}`;
            document.getElementById('camera-summary').textContent =
                `${data.total_people} people observed across connected camera areas`;

            const directions = document.getElementById('camera-direction-list');
            directions.replaceChildren();
            if (data.directions.length) {
                data.directions.forEach(recommendation => {
                    const item = document.createElement('div');
                    item.className = 'camera-direction-item';
                    const heading = document.createElement('strong');
                    heading.textContent = `${recommendation.camera} (${recommendation.area}): `;
                    const message = document.createElement('span');
                    message.textContent = recommendation.direction;
                    const context = document.createElement('span');
                    context.textContent =
                        ` — ${recommendation.people} people, ${recommendation.density}% estimated area capacity, ${recommendation.risk_level} risk`;
                    item.append(heading, message, context);
                    directions.appendChild(item);
                });
            } else {
                directions.textContent =
                    'Recommendations appear after a camera is analyzing its assigned area.';
            }
            document.getElementById('camera-recommendation-note').textContent =
                data.recommendation_note;
        } catch (error) {
            document.getElementById('camera-error').textContent = error.message;
        }
    }

    function createFeedImage() {
        const container = document.getElementById('video-player-container');
        const img = document.createElement('img');
        img.id = 'video-feed-img';
        img.className = 'video-feed-img';
        img.style.cssText = 'width:100%;height:100%;object-fit:contain;position:absolute;top:0;left:0;z-index:1;';
        container.insertBefore(img, container.firstChild);
        return img;
    }

    // ══════════════════════════════════════════
    //  BACKEND POLLING — Real YOLO Data
    // ══════════════════════════════════════════
    let pollTimer = null;

    function startPolling() {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollBackend, POLL_INTERVAL);
    }

    async function pollBackend() {
        if (!state.backendConnected) return;

        try {
            const res = await fetch(`${API_BASE}/api/state`);
            const data = await res.json();

            // ── Update state from real AI data ──
            const m = data.metrics || {};
            state.totalPeople = m.people || 0;
            state.density = m.density || 0;
            state.riskScore = m.risk_score || 0;
            state.riskLevel = m.risk_level || 'SAFE';
            state.flowRate = m.speed || 0;

            // Tracks (real YOLO bounding boxes)
            state.tracks = data.tracks || [];
            state.personFlows = data.person_flows || [];

            // Zones from backend
            const zones = data.zones || [];
            state.criticalZones = 0;
            state.alerts = 0;
            zones.forEach(z => {
                const id = z.name.replace('Zone ', '');
                state.zones[id] = {
                    density: z.density || 0,
                    people: Math.round((z.density || 0) / 20),
                    status: z.status || 'SAFE',
                    color: CONFIG.zones.find(cz => cz.id === id)?.color || '#00d4ff',
                };
                if (z.status === 'CRITICAL') state.criticalZones++;
                if (z.status === 'WARNING' || z.status === 'CRITICAL') state.alerts++;
            });

            CONFIG.exits = (data.gates || []).map(gate => ({
                ...gate,
                name: gate.id,
            }));
            renderDetectedGates();

            // Frame info
            state.frameInfo = data.frame_info || {};

            // Density history
            state.densityHistory.push({
                time: state.frameInfo.frame_idx || 0,
                density: state.density,
                people: state.totalPeople,
            });
            if (state.densityHistory.length > CONFIG.maxHistoryPoints) state.densityHistory.shift();

            // Timeline / alerts from backend
            if (data.timeline) {
                data.timeline.slice(0, 5).forEach(ev => {
                    const type = ev.event_type.includes('CRITICAL') ? 'critical' :
                                 ev.event_type.includes('COUNTER') || ev.event_type.includes('BOTTLENECK') ? 'warning' : 'info';
                    // Only add if not already shown
                    if (!state.alertHistory.includes(ev.timestamp + ev.detail)) {
                        state.alertHistory.push(ev.timestamp + ev.detail);
                    }
                });
            }

            // Update video HUD
            const fi = state.frameInfo;
            document.getElementById('vid-time').textContent = `${fi.time || '00:00'} / ${fi.total_time || '00:00'}`;
            document.getElementById('vid-frame').textContent = fi.frame_idx || 0;
            document.getElementById('vid-fps').textContent = fi.fps || 0;
            document.getElementById('vid-people').textContent = state.totalPeople;
            document.getElementById('vid-density').textContent = Math.round(state.density) + '%';

            if (fi.progress) {
                document.getElementById('video-progress-fill').style.width = fi.progress + '%';
            }

            // Update all dashboard panels
            updateUI();
            renderTrackingOverlay();
            updateZoneOverlays();
            renderMapHeatmap();
            updateMapParticles();
            renderDensityChart();

        } catch (err) {
            // Backend disconnected - don't spam console
        }
    }

    // ══════════════════════════════════════════
    //  TRACKING OVERLAY — Real YOLO Boxes
    // ══════════════════════════════════════════
    function renderTrackingOverlay() {
        if (!video.showTracking || !state.tracks.length) return;
        const c = document.getElementById('video-tracking-canvas');
        if (!c) return;

        // Resize canvas to match container
        const container = document.getElementById('video-player-container');
        if (container) {
            const r = container.getBoundingClientRect();
            if (c.width !== r.width || c.height !== r.height) {
                c.width = r.width;
                c.height = r.height;
            }
        }

        const ctx = c.getContext('2d');
        const cw = c.width, ch = c.height;
        ctx.clearRect(0, 0, cw, ch);

        // Get actual video dimensions from backend
        const vidW = state.frameInfo.width || 1280;
        const vidH = state.frameInfo.height || 720;

        // Calculate scale to fit canvas (maintaining aspect ratio)
        const scale = Math.min(cw / vidW, ch / vidH);
        const offsetX = (cw - vidW * scale) / 2;
        const offsetY = (ch - vidH * scale) / 2;

        state.tracks.forEach(track => {
            const [x1, y1, x2, y2] = track.bbox;

            // Scale bbox to canvas coordinates
            const sx1 = x1 * scale + offsetX;
            const sy1 = y1 * scale + offsetY;
            const sx2 = x2 * scale + offsetX;
            const sy2 = y2 * scale + offsetY;
            const bw = sx2 - sx1;
            const bh = sy2 - sy1;

            const color = '#00d4ff';
            const cl = Math.min(12, bw * 0.25);

            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;

            // Corner brackets
            ctx.beginPath(); ctx.moveTo(sx1,sy1+cl); ctx.lineTo(sx1,sy1); ctx.lineTo(sx1+cl,sy1); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(sx2-cl,sy1); ctx.lineTo(sx2,sy1); ctx.lineTo(sx2,sy1+cl); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(sx1,sy2-cl); ctx.lineTo(sx1,sy2); ctx.lineTo(sx1+cl,sy2); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(sx2-cl,sy2); ctx.lineTo(sx2,sy2); ctx.lineTo(sx2,sy2-cl); ctx.stroke();

            // Subtle fill
            ctx.fillStyle = 'rgba(0,212,255,0.06)';
            ctx.fillRect(sx1, sy1, bw, bh);

            // Label
            const label = `ID-${track.id}`;
            ctx.font = '9px JetBrains Mono, monospace';
            const tw = ctx.measureText(label).width + 8;
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(sx1, sy1 - 14, tw, 14);
            ctx.fillStyle = color;
            ctx.fillText(label, sx1 + 4, sy1 - 3);

            // Center dot
            const cx = (sx1 + sx2) / 2, cy = (sy1 + sy2) / 2;
            ctx.beginPath(); ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            ctx.fillStyle = color; ctx.fill();
        });

        // Status bar
        if (state.tracks.length > 0) {
            ctx.fillStyle = 'rgba(0,0,0,0.65)';
            ctx.fillRect(6, ch - 24, 200, 18);
            ctx.fillStyle = '#00d4ff';
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.fillText(`YOLO TRACKING: ${state.tracks.length} people`, 10, ch - 10);
        }

        // Warning overlay for high density
        if (state.totalPeople > 8) {
            const sev = state.riskLevel === 'CRITICAL' ? 'CRITICAL' : 'WARNING';
            const col = sev === 'CRITICAL' ? '#ff1744' : '#ffab00';
            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(6, 6, 240, 34);
            ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.strokeRect(6, 6, 240, 34);
            ctx.fillStyle = col; ctx.font = 'bold 10px Inter, sans-serif';
            ctx.fillText(`⚠ ${sev} DENSITY`, 14, 21);
            ctx.fillStyle = '#aaa'; ctx.font = '9px Inter, sans-serif';
            ctx.fillText(`${state.totalPeople} people • Risk: ${state.riskScore}`, 14, 34);
        }
    }

    // ══════════════════════════════════════════
    //  MAP INTERACTIONS
    // ══════════════════════════════════════════
    function initMapInteractions() {
        const viewport = document.getElementById('map-viewport');
        viewport.addEventListener('mousedown', (e) => {
            if (e.target.closest('.ctrl-btn')) return;
            state.isPanning = true;
            state.panStart.x = e.clientX - state.mapPanX;
            state.panStart.y = e.clientY - state.mapPanY;
            viewport.style.cursor = 'grabbing';
        });
        window.addEventListener('mousemove', (e) => {
            if (!state.isPanning) return;
            state.mapPanX = e.clientX - state.panStart.x;
            state.mapPanY = e.clientY - state.panStart.y;
            updateMapTransform();
        });
        window.addEventListener('mouseup', () => { state.isPanning = false; });

        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            state.mapZoom = Math.max(0.5, Math.min(3, state.mapZoom + (e.deltaY > 0 ? -0.1 : 0.1)));
            updateMapTransform();
            document.getElementById('zoom-label').textContent = Math.round(state.mapZoom * 100) + '%';
        }, { passive: false });

        document.getElementById('btn-zoom-in').addEventListener('click', () => {
            state.mapZoom = Math.min(3, state.mapZoom + 0.2); updateMapTransform();
            document.getElementById('zoom-label').textContent = Math.round(state.mapZoom * 100) + '%';
        });
        document.getElementById('btn-zoom-out').addEventListener('click', () => {
            state.mapZoom = Math.max(0.5, state.mapZoom - 0.2); updateMapTransform();
            document.getElementById('zoom-label').textContent = Math.round(state.mapZoom * 100) + '%';
        });
        document.getElementById('btn-reset-view').addEventListener('click', () => {
            state.mapZoom = 1; state.mapPanX = 0; state.mapPanY = 0;
            updateMapTransform();
            document.getElementById('zoom-label').textContent = '100%';
        });

        const toggles = [
            ['btn-toggle-heatmap', 'showHeatmap', () => document.getElementById('heatmap-canvas').classList.toggle('hidden', !state.showHeatmap)],
            ['btn-toggle-flow', 'showFlow', () => document.getElementById('flow-arrows-layer').classList.toggle('hidden', !state.showFlow)],
            ['btn-toggle-pressure', 'showPressure', () => document.getElementById('pressure-wave-layer').style.display = state.showPressure ? '' : 'none'],
        ];
        toggles.forEach(([id, key, cb]) => {
            document.getElementById(id).addEventListener('click', () => {
                state[key] = !state[key];
                document.getElementById(id).classList.toggle('active', state[key]);
                if (cb) cb();
            });
        });
    }

    function updateMapTransform() {
        const scene = document.getElementById('map-scene');
        if (scene) scene.style.transform = `translate(${state.mapPanX}px, ${state.mapPanY}px) scale(${state.mapZoom})`;
    }

    async function refreshBackendStatus() {
        const backendBadge = document.getElementById('backend-status');
        const aiBadge = document.getElementById('ai-status');
        const backendDot = backendBadge.querySelector('.status-dot');
        const aiDot = aiBadge.querySelector('.status-dot');

        try {
            let aiRunning;
            if (state.healthEndpointAvailable !== false) {
                const response = await fetch(`${API_BASE}/api/health`, { cache: 'no-store' });
                if (response.status === 404) {
                    state.healthEndpointAvailable = false;
                } else {
                    if (!response.ok) throw new Error(`Backend health check failed (${response.status})`);
                    const health = await response.json();
                    state.healthEndpointAvailable = true;
                    state.backendConnected = health.status === 'ok';
                    aiRunning = health.ai_running;
                    state.aiRunning = aiRunning;
                }
            }

            if (state.healthEndpointAvailable === false) {
                const stateResponse = await fetch(`${API_BASE}/api/state`, { cache: 'no-store' });
                if (!stateResponse.ok) throw new Error(`Backend state check failed (${stateResponse.status})`);
                await stateResponse.json();
                state.backendConnected = true;
                aiRunning = state.aiRunning;
            }
            backendDot.className = `status-dot ${state.backendConnected ? 'green-dot' : 'red-dot'}`;
            backendBadge.lastChild.textContent = state.backendConnected ? ' BACKEND CONNECTED' : ' BACKEND OFFLINE';

            aiDot.className = `status-dot ${aiRunning ? 'blue-dot' : state.backendConnected ? 'idle-dot' : 'red-dot'}`;
            aiBadge.lastChild.textContent = aiRunning ? ' AI ANALYZING' : state.backendConnected ? ' AI IDLE' : ' AI OFFLINE';
        } catch (error) {
            state.backendConnected = false;
            state.aiRunning = false;
            backendDot.className = 'status-dot red-dot';
            backendBadge.lastChild.textContent = ' BACKEND OFFLINE';
            aiDot.className = 'status-dot red-dot';
            aiBadge.lastChild.textContent = ' AI OFFLINE';
        }
    }

    function renderDetectedGates() {
        const layer = document.getElementById('detected-gates-layer');
        if (!layer) return;
        layer.replaceChildren();

        const markerId = 'detected-gate-arrow';
        if (!document.getElementById(markerId)) {
            const defs = document.querySelector('#map-svg defs');
            if (defs) {
                const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                marker.setAttribute('id', markerId);
                marker.setAttribute('viewBox', '0 0 10 10');
                marker.setAttribute('refX', '8');
                marker.setAttribute('refY', '5');
                marker.setAttribute('markerWidth', '5');
                marker.setAttribute('markerHeight', '5');
                marker.setAttribute('orient', 'auto-start-reverse');
                const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
                arrow.setAttribute('fill', 'context-stroke');
                marker.appendChild(arrow);
                defs.appendChild(marker);
            }
        }

        CONFIG.exits.forEach(gate => {
            if (!Array.isArray(gate.corners) || gate.corners.length !== 4) return;
            const points = gate.corners.map(([x, y]) =>
                `${x * CONFIG.svgViewBox.w},${y * CONFIG.svgViewBox.h}`
            ).join(' ');
            const centerX = gate.cx * CONFIG.svgViewBox.w;
            const centerY = gate.cy * CONFIG.svgViewBox.h;
            const color = gate.occluded || gate.confidence < 0.65 ? '#ffab00' : '#00e676';
            const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            polygon.setAttribute('points', points);
            polygon.setAttribute('class', 'detected-gate-boundary');
            polygon.setAttribute('stroke', color);
            polygon.setAttribute('aria-label', `${gate.name}, ${Math.round(gate.confidence * 100)}% confidence`);

            const top = gate.corners[0];
            const bottom = gate.corners[2];
            const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            arrow.setAttribute('x1', String(((top[0] + gate.corners[1][0]) / 2) * CONFIG.svgViewBox.w));
            arrow.setAttribute('y1', String(((top[1] + gate.corners[1][1]) / 2) * CONFIG.svgViewBox.h));
            arrow.setAttribute('x2', String(((bottom[0] + gate.corners[3][0]) / 2) * CONFIG.svgViewBox.w));
            arrow.setAttribute('y2', String(((bottom[1] + gate.corners[3][1]) / 2) * CONFIG.svgViewBox.h));
            arrow.setAttribute('class', 'detected-gate-arrow');
            arrow.setAttribute('stroke', color);
            arrow.setAttribute('marker-end', `url(#${markerId})`);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', String(centerX));
            label.setAttribute('y', String(centerY));
            label.setAttribute('class', 'detected-gate-label');
            label.setAttribute('fill', color);
            label.setAttribute('text-anchor', 'middle');
            label.textContent = `${gate.name} · ${Math.round(gate.confidence * 100)}%${gate.occluded ? ' · LOW CONFIDENCE' : ''}`;
            layer.append(polygon, arrow, label);
        });
    }

    // ══════════════════════════════════════════
    //  MAP HEATMAP
    // ══════════════════════════════════════════
    let heatmapCanvas, heatmapCtx;
    function resizeHeatmapCanvas() {
        const vp = document.getElementById('map-viewport');
        heatmapCanvas = document.getElementById('heatmap-canvas');
        if (!heatmapCanvas || !vp) return;
        heatmapCanvas.width = vp.clientWidth;
        heatmapCanvas.height = vp.clientHeight;
        heatmapCtx = heatmapCanvas.getContext('2d');
    }

    function renderMapHeatmap() {
        if (!state.showHeatmap || !heatmapCtx || !state.videoLoaded) return;
        const w = heatmapCanvas.width, h = heatmapCanvas.height;
        heatmapCtx.clearRect(0, 0, w, h);

        const svg = document.getElementById('map-svg');
        const vp = document.getElementById('map-viewport');
        if (!svg || !vp) return;
        const vR = vp.getBoundingClientRect(), sR = svg.getBoundingClientRect();
        const sx = sR.width / CONFIG.svgViewBox.w, sy = sR.height / CONFIG.svgViewBox.h;
        const ox = sR.left - vR.left, oy = sR.top - vR.top;

        CONFIG.zones.forEach(zone => {
            const zs = state.zones[zone.id];
            if (!zs || zs.density < 5) return;
            const cx = (zone.x + zone.w/2) * sx + ox;
            const cy = (zone.y + zone.h/2) * sy + oy;
            const r = Math.max(zone.w/2 * sx, zone.h/2 * sy) * 0.8;

            let cr, cg, cb;
            if (zs.density < 40) { cr=0; cg=230; cb=118; }
            else if (zs.density < 70) { cr=255; cg=171; cb=0; }
            else { cr=255; cg=23; cb=68; }

            const alpha = Math.min(0.45, zs.density / 160);
            const grad = heatmapCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
            grad.addColorStop(0, `rgba(${cr},${cg},${cb},${alpha})`);
            grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
            heatmapCtx.beginPath(); heatmapCtx.arc(cx, cy, r, 0, Math.PI*2);
            heatmapCtx.fillStyle = grad; heatmapCtx.fill();
        });
    }

    // ══════════════════════════════════════════
    //  MAP PARTICLES — From backend tracks
    // ══════════════════════════════════════════
    function updateMapParticles() {
        if (!state.videoLoaded) return;
        const layer = document.getElementById('crowd-particles-layer');
        if (!layer) return;
        layer.innerHTML = '';

        const vidW = state.frameInfo.width || 1280;
        const vidH = state.frameInfo.height || 720;

        state.tracks.forEach((track, i) => {
            const normX = track.cx / vidW;
            const normY = track.cy / vidH;
            const mapX = normX * CONFIG.svgViewBox.w;
            const mapY = 150 + normY * 600;

            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', mapX);
            circle.setAttribute('cy', mapY);
            circle.setAttribute('r', '5');
            circle.setAttribute('fill', 'rgba(0,212,255,0.6)');
            circle.setAttribute('stroke', 'rgba(0,212,255,0.3)');
            circle.setAttribute('stroke-width', '2');
            layer.appendChild(circle);
        });
    }

    // ══════════════════════════════════════════
    //  ZONE OVERLAYS
    // ══════════════════════════════════════════
    function updateZoneOverlays() {
        if (!state.videoLoaded) return;
        CONFIG.zones.forEach(zone => {
            const overlay = document.getElementById(`zone-${zone.id}-overlay`);
            if (!overlay) return;
            const zs = state.zones[zone.id] || { density: 0, status: 'SAFE' };
            let fill, stroke;
            if (zs.status === 'CRITICAL') { fill = 'rgba(255,23,68,0.12)'; stroke = 'rgba(255,23,68,0.4)'; }
            else if (zs.status === 'WARNING') { fill = 'rgba(255,171,0,0.1)'; stroke = 'rgba(255,171,0,0.35)'; }
            else { fill = `${zone.color}10`; stroke = `${zone.color}33`; }
            overlay.setAttribute('fill', fill);
            overlay.setAttribute('stroke', stroke);
        });
    }

    // ══════════════════════════════════════════
    //  DENSITY CHART
    // ══════════════════════════════════════════
    let chartCanvas, chartCtx;
    function initDensityChart() {
        chartCanvas = document.getElementById('density-chart');
        const container = document.getElementById('timeline-chart-panel');
        if (!chartCanvas || !container) return;
        chartCanvas.width = container.clientWidth - 28;
        chartCanvas.height = 90;
        chartCtx = chartCanvas.getContext('2d');
    }

    function renderDensityChart() {
        if (!chartCtx || state.densityHistory.length < 2) return;
        const w = chartCanvas.width, h = chartCanvas.height;
        chartCtx.clearRect(0, 0, w, h);
        const data = state.densityHistory;
        const stepX = w / (CONFIG.maxHistoryPoints - 1);

        chartCtx.strokeStyle = 'rgba(255,255,255,0.04)'; chartCtx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            chartCtx.beginPath(); chartCtx.moveTo(0, (i/4)*h); chartCtx.lineTo(w, (i/4)*h); chartCtx.stroke();
        }

        chartCtx.beginPath();
        data.forEach((d, i) => {
            const x = i * stepX, y = h - (d.density / 100) * h;
            i === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
        });
        chartCtx.strokeStyle = 'rgba(0,212,255,0.9)'; chartCtx.lineWidth = 2; chartCtx.stroke();
        chartCtx.lineTo((data.length-1)*stepX, h); chartCtx.lineTo(0, h); chartCtx.closePath();
        chartCtx.fillStyle = 'rgba(0,212,255,0.06)'; chartCtx.fill();

        chartCtx.setLineDash([4, 4]);
        chartCtx.beginPath(); chartCtx.moveTo(0, h*0.55); chartCtx.lineTo(w, h*0.55);
        chartCtx.strokeStyle = 'rgba(255,171,0,0.3)'; chartCtx.lineWidth = 1; chartCtx.stroke();
        chartCtx.beginPath(); chartCtx.moveTo(0, h*0.25); chartCtx.lineTo(w, h*0.25);
        chartCtx.strokeStyle = 'rgba(255,23,68,0.3)'; chartCtx.stroke();
        chartCtx.setLineDash([]);

        const last = data[data.length-1];
        const lx = (data.length-1)*stepX, ly = h - (last.density/100)*h;
        chartCtx.beginPath(); chartCtx.arc(lx, ly, 4, 0, Math.PI*2);
        chartCtx.fillStyle = '#00d4ff'; chartCtx.fill();
    }

    // ══════════════════════════════════════════
    //  UI UPDATE — All from backend state
    // ══════════════════════════════════════════
    function updateUI() {
        if (!state.videoLoaded) return;

        document.getElementById('metric-people').textContent = state.totalPeople;
        document.getElementById('metric-density').innerHTML = Math.round(state.density) + '<span class="mc-unit">%</span>';
        document.getElementById('metric-alerts').textContent = state.alerts;
        document.getElementById('metric-critical').textContent = state.criticalZones;
        document.getElementById('metric-flowrate').innerHTML = state.flowRate.toFixed(1) + '<span class="mc-unit"> px/f</span>';

        document.getElementById('bar-people').style.width = Math.min(100, state.totalPeople * 4) + '%';
        document.getElementById('bar-density').style.width = Math.min(100, state.density) + '%';

        // Risk gauge
        const fillLen = (state.riskScore / 100) * 251;
        document.getElementById('gauge-fill').setAttribute('stroke-dasharray', `${fillLen} 251`);
        document.getElementById('gauge-value').textContent = Math.round(state.riskScore);

        const badge = document.getElementById('risk-level-badge');
        badge.className = `risk-level-badge ${state.riskLevel.toLowerCase()}`;
        document.getElementById('risk-level-text').textContent = state.riskLevel;

        document.getElementById('risk-density').textContent = Math.round(state.density) + '%';
        document.getElementById('risk-counterflow').textContent = state.riskScore > 40 ? 'Detected' : 'None';
        document.getElementById('risk-bottleneck').textContent = state.criticalZones > 0 ? 'High' : 'Low';
        document.getElementById('risk-instability').textContent = state.riskScore > 50 ? 'Unstable' : 'Stable';

        document.getElementById('info-counterflow').textContent = state.riskScore > 40 ? 'YES' : 'NO';
        document.getElementById('info-compression').textContent = state.density > 60 ? 'HIGH' : state.density > 30 ? 'MODERATE' : 'LOW';
        document.getElementById('info-zones').textContent = Object.keys(state.zones).length;

        document.getElementById('pressure-value').textContent = state.density > 60 ? 'HIGH' : state.density > 30 ? 'MODERATE' : 'LOW';
        document.getElementById('pressure-value').style.color = state.density > 60 ? '#ff1744' : state.density > 30 ? '#ffab00' : '#00e676';
        document.getElementById('pressure-risk').textContent = state.riskLevel;
        document.getElementById('pressure-risk').style.color = state.riskScore > 60 ? '#ff1744' : state.riskScore > 35 ? '#ffab00' : '#00e676';

        const worstZone = Object.entries(state.zones).sort((a,b) => b[1].density - a[1].density)[0];
        document.getElementById('pressure-zone').textContent = worstZone ? `ZONE ${worstZone[0]}` : '—';
        document.getElementById('pressure-warning').textContent = state.density > 50 ? `~${Math.max(1, Math.floor(8 - state.density/12))} min` : '—';

        updateZoneList();
        updateExitList();
        updateZoneStats();
        updateActions();
    }

    function updateZoneList() {
        const c = document.getElementById('zone-list'); if (!c) return; c.innerHTML = '';
        CONFIG.zones.forEach(z => {
            const zs = state.zones[z.id] || { density: 0, status: 'SAFE' };
            const d = document.createElement('div');
            d.className = `zone-item ${zs.status === 'CRITICAL' ? 'critical' : zs.status === 'WARNING' ? 'warning' : ''}`;
            d.style.borderLeftColor = z.color;
            d.innerHTML = `<span class="zone-name">${z.name}</span><span class="zone-density" style="color:${zs.status==='CRITICAL'?'#ff1744':zs.status==='WARNING'?'#ffab00':'#00e676'}">${zs.density}%</span>`;
            c.appendChild(d);
        });
    }

    function updateExitList() {
        const c = document.getElementById('exit-list'); if (!c) return; c.innerHTML = '';
        const displayGates = CONFIG.exits;
        if (!displayGates.length) {
            const empty = document.createElement('div');
            empty.className = 'exit-item';
            empty.textContent = state.videoLoaded && state.aiRunning
                ? 'Scanning for physical gate geometry...'
                : state.videoLoaded
                    ? 'No gates confirmed in this footage'
                    : 'Upload footage to detect gates';
            c.appendChild(empty);
            return;
        }
        displayGates.forEach(e => {
            const d = document.createElement('div'); d.className = 'exit-item';
            const statusColor = e.occluded ? '#ffab00' : '#00e676';
            const statusText = e.occluded ? 'LOW CONFIDENCE' : 'GEOMETRY MATCH';
            const confText = e.confidence ? ` ${Math.round(e.confidence * 100)}%` : '';
            const peopleText = e.people_nearby > 0 ? ` · ${e.people_nearby}p` : '';
            d.innerHTML = `<span class="exit-dot" style="background:${statusColor}"></span><span class="exit-name">${e.name}${confText}</span><span class="exit-status" style="color:${statusColor}">${statusText}${peopleText}</span>`;
            c.appendChild(d);
        });
    }

    function updateZoneStats() {
        const c = document.getElementById('zone-stats'); if (!c) return; c.innerHTML = '';
        CONFIG.zones.forEach(z => {
            const zs = state.zones[z.id] || { density: 0, people: 0, status: 'SAFE' };
            const d = document.createElement('div'); d.className = 'zone-stat-card';
            d.innerHTML = `<div class="zs-name" style="color:${z.color}">${z.name}</div><div class="zs-people">${zs.people}</div><div class="zs-status ${zs.status.toLowerCase()}">${zs.status}</div>`;
            c.appendChild(d);
        });
    }

    function updateActions() {
        const c = document.getElementById('action-list');
        const actions = [];
        if (state.riskLevel === 'CRITICAL') {
            actions.push({ type:'critical', icon:'⚠', text:`Critical density: ${Math.round(state.density)}%. Activate dispersal protocol.` });
        } else if (state.riskLevel === 'WARNING' || state.riskLevel === 'HIGH') {
            actions.push({ type:'warning', icon:'⚡', text:`Monitor closely. Risk score: ${Math.round(state.riskScore)}` });
        } else if (state.videoLoaded) {
            actions.push({ type:'safe', icon:'✓', text:'All zones within safe capacity' });
        } else {
            actions.push({ type:'safe', icon:'📹', text:'Upload a video to begin analysis' });
        }
        c.innerHTML = actions.map(a => `<div class="action-item ${a.type}"><div class="action-icon">${a.icon}</div><div class="action-text">${a.text}</div></div>`).join('');
    }

    // ══════════════════════════════════════════
    //  ALERTS & LOGS
    // ══════════════════════════════════════════
    function addAlertEntry(type, msg) {
        const now = new Date();
        const time = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
        const c = document.getElementById('alert-timeline');
        const e = document.createElement('div'); e.className = `alert-entry ${type}`;
        e.innerHTML = `<span class="alert-time">${time}</span><span class="alert-msg">${msg}</span>`;
        c.insertBefore(e, c.firstChild);
        while (c.children.length > 15) c.removeChild(c.lastChild);
    }

    function addLogEntry(msg) {
        const now = new Date();
        const time = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
        const c = document.getElementById('log-entries');
        const e = document.createElement('div'); e.className = 'log-entry';
        e.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${msg}</span>`;
        c.insertBefore(e, c.firstChild);
        while (c.children.length > 10) c.removeChild(c.lastChild);
    }

    // ══════════════════════════════════════════
    //  INITIALIZATION
    // ══════════════════════════════════════════
    function startDashboard() {
        CONFIG.zones.forEach(z => {
            state.zones[z.id] = { density: 0, people: 0, status: 'SAFE', color: z.color };
        });

        resizeHeatmapCanvas();
        initDensityChart();
        initMapInteractions();
        initVideoModule();
        initMultiCameraDashboard();
        updateExitList();

        addLogEntry('CrowdGuard AI System initialized');
        addLogEntry('Backend: Flask API on the dashboard server');
        addLogEntry('Waiting for video upload...');
        addAlertEntry('info', 'System ready — upload a video to begin AI analysis');
        refreshBackendStatus();
        setInterval(refreshBackendStatus, 2000);

        updateClock();
        setInterval(updateClock, 1000);
        window.addEventListener('resize', () => {
            resizeHeatmapCanvas();
            initDensityChart();
        });
    }

    // ══════════════════════════════════════════
    //  BOOT
    // ══════════════════════════════════════════
    window.addEventListener('DOMContentLoaded', () => {
        initBgParticles();
        runLoadingSequence();
    });

})();
