document.addEventListener("DOMContentLoaded", () => {
    // ── UI Elements ───────────────────────────────────────
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const sourceSelect = document.getElementById('video-source');
    const videoFeed = document.getElementById('video-feed');

    // Canvas layers
    const trackingCanvas = document.getElementById('tracking-canvas');
    const heatmapCanvas = document.getElementById('heatmap-canvas');
    const tCtx = trackingCanvas.getContext('2d');
    const hCtx = heatmapCanvas.getContext('2d', { willReadFrequently: true });

    // Status Dots
    const dotLive = document.querySelector('.live-dot');
    const dotCamera = document.querySelector('.camera-dot');
    const dotAi = document.querySelector('.ai-dot');

    // Metrics
    const mPeople = document.getElementById('m-people');
    const mDensity = document.getElementById('m-density');
    const mSpeed = document.getElementById('m-speed');
    const mFlow = document.getElementById('m-flow');
    const mRisk = document.getElementById('m-risk');
    const riskCard = document.getElementById('risk-card');

    // Timeline
    const timelineContainer = document.getElementById('timeline-container');

    // State variables
    let isRunning = false;
    let loopId = null;
    let trackHistory = {}; // { id: [{x,y}, ...] } for motion trails

    // ── Load available videos on page load ──────────────────
    fetch('/api/videos')
        .then(r => r.json())
        .then(data => {
            // Remove placeholder
            sourceSelect.innerHTML = '';
            if (data.videos && data.videos.length > 0) {
                data.videos.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v;
                    opt.textContent = v;
                    sourceSelect.appendChild(opt);
                });
            } else {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No videos found';
                sourceSelect.appendChild(opt);
            }
            // Always add webcam option
            const camOpt = document.createElement('option');
            camOpt.value = 'camera';
            camOpt.textContent = 'Webcam';
            sourceSelect.appendChild(camOpt);
        })
        .catch(() => {
            sourceSelect.innerHTML = '<option value="camera">Webcam</option>';
        });

    // ── Resizing Canvas to match video ──────────────────────
    function resizeCanvas() {
        const rect = videoFeed.parentElement.getBoundingClientRect();
        trackingCanvas.width = rect.width;
        trackingCanvas.height = rect.height;
        heatmapCanvas.width = rect.width;
        heatmapCanvas.height = rect.height;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // ── Live Clock ──────────────────────────────────────────
    setInterval(() => {
        const now = new Date();
        document.getElementById('clock-time').textContent = now.toTimeString().split(' ')[0];
        document.getElementById('clock-date').textContent = now.toLocaleDateString('en-GB', {
            day: '2-digit', month: 'short', year: 'numeric'
        }).toUpperCase();
    }, 1000);

    // ── Background Particles ────────────────────────────────
    function initParticles() {
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
        const bg = document.getElementById('particles-bg');
        const particles = [];

        for (let i = 0; i < 40; i++) {
            const p = document.createElement('div');
            const size = Math.random() * 2.5 + 0.5;
            p.style.cssText = `
                position: absolute;
                width: ${size}px; height: ${size}px;
                background: rgba(0,212,255,0.3);
                border-radius: 50%;
                box-shadow: 0 0 4px rgba(0,212,255,0.5);
                pointer-events: none;
            `;
            const x = Math.random() * 100;
            const y = Math.random() * 100;
            p.style.left = x + '%';
            p.style.top = y + '%';
            bg.appendChild(p);
            particles.push({
                el: p,
                x: x,
                y: y,
                dx: (Math.random() - 0.5) * 0.03,
                dy: (Math.random() - 0.5) * 0.03 - 0.01
            });
        }

        function animateParticles() {
            for (const p of particles) {
                p.x += p.dx;
                p.y += p.dy;
                if (p.y < -2) p.y = 102;
                if (p.x < -2) p.x = 102;
                if (p.x > 102) p.x = -2;
                if (p.y > 102) p.y = -2;
                p.el.style.left = p.x + '%';
                p.el.style.top = p.y + '%';
            }
            requestAnimationFrame(animateParticles);
        }
        animateParticles();
    }
    initParticles();

    // ── File Upload ─────────────────────────────────────────
    const videoUpload = document.getElementById('video-upload');
    if (videoUpload) {
        videoUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            const origText = btnStart.innerHTML;
            btnStart.innerHTML = '⏳ Uploading...';
            btnStart.disabled = true;

            fetch('/api/upload', {
                method: 'POST',
                body: formData
            }).then(r => r.json()).then(data => {
                if (data.status === 'ok') {
                    // Add to dropdown and select it
                    const opt = document.createElement('option');
                    opt.value = data.filename;
                    opt.textContent = `📁 ${data.filename}`;
                    // Insert before the webcam option
                    const camOption = Array.from(sourceSelect.options).find(o => o.value === 'camera');
                    if (camOption) {
                        sourceSelect.insertBefore(opt, camOption);
                    } else {
                        sourceSelect.appendChild(opt);
                    }
                    sourceSelect.value = data.filename;
                    console.log('Upload complete:', data.filename);
                } else {
                    alert('Upload failed: ' + (data.msg || 'Unknown error'));
                }
            }).catch(err => {
                console.error('Upload error:', err);
                alert('Upload error');
            }).finally(() => {
                btnStart.innerHTML = origText;
                btnStart.disabled = false;
                videoUpload.value = '';
            });
        });
    }

    // ── Start / Stop ────────────────────────────────────────
    btnStart.addEventListener('click', () => {
        const source = sourceSelect.value;
        if (!source) {
            alert('Please select a video source');
            return;
        }

        btnStart.innerHTML = '⏳ Starting...';
        btnStart.disabled = true;

        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'start', source: source })
        }).then(r => r.json()).then(data => {
            if (data.status === 'ok') {
                isRunning = true;
                videoFeed.src = '/video_feed?' + Date.now();
                btnStart.innerHTML = '▶ Running';
                btnStop.disabled = false;
                dotLive.classList.add('active');
                dotCamera.classList.add('active');
                dotAi.classList.add('active');
                // Wait a moment for video to start, then resize canvas
                setTimeout(resizeCanvas, 500);
                loopId = setInterval(fetchState, 16); // ~60fps
            } else {
                alert('Start failed: ' + (data.msg || 'Unknown error'));
                btnStart.innerHTML = '▶ Start';
                btnStart.disabled = false;
            }
        }).catch(err => {
            console.error('Start error:', err);
            alert('Connection error');
            btnStart.innerHTML = '▶ Start';
            btnStart.disabled = false;
        });
    });

    btnStop.addEventListener('click', () => {
        fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'stop' })
        }).then(() => {
            isRunning = false;
            if (loopId) clearInterval(loopId);
            loopId = null;
            videoFeed.src = '/static/placeholder.jpg';
            btnStart.innerHTML = '▶ Start';
            btnStart.disabled = false;
            btnStop.disabled = true;
            dotLive.classList.remove('active');
            dotCamera.classList.remove('active');
            dotAi.classList.remove('active');
            tCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
            hCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
            trackHistory = {};
        });
    });

    // ── State Polling ───────────────────────────────────────
    function fetchState() {
        fetch('/api/state')
            .then(r => r.json())
            .then(state => {
                updateMetrics(state.metrics);
                updateTimeline(state.timeline);
                updateSessionInfo(state.summary, state.frame_info);
                update3DMap(state.zones, state.route);

                // Scale factor: map from original video resolution to canvas size
                const nw = (state.frame_info && state.frame_info.width) || 1280;
                const nh = (state.frame_info && state.frame_info.height) || 720;
                const sx = trackingCanvas.width / nw;
                const sy = trackingCanvas.height / nh;

                drawOverlays(state.tracks, state.person_flows, state.zones, state.route, state.pressure_waves, state.exits, sx, sy);
            })
            .catch(err => console.error("API error:", err));
    }

    // ── Update DOM ──────────────────────────────────────────

    function updateMetrics(m) {
        if (!m) return;
        mPeople.textContent = m.people;
        mDensity.textContent = m.density + '%';
        mSpeed.textContent = m.speed + ' px/s';
        mFlow.textContent = m.dominant_flow;
        mRisk.textContent = m.risk_level;

        // Risk card styling
        riskCard.className = 'metric-card glass-panel risk-card';
        if (m.risk_level === 'WARNING') riskCard.classList.add('risk-WARNING');
        else if (m.risk_level === 'HIGH' || m.risk_level === 'CRITICAL') riskCard.classList.add('risk-CRITICAL');
        else riskCard.classList.add('risk-SAFE');

        // Counter-flow indicator
        const cfEl = document.getElementById('la-cf');
        if (cfEl) {
            cfEl.textContent = m.cf_flag ? 'YES' : 'NO';
            cfEl.style.color = m.cf_flag ? 'var(--critical)' : 'var(--safe)';
        }
    }

    let lastTimelineLen = 0;
    function updateTimeline(events) {
        if (!events || events.length === lastTimelineLen) return;
        lastTimelineLen = events.length;

        timelineContainer.innerHTML = '';
        events.slice(0, 20).forEach(ev => {
            const div = document.createElement('div');
            div.className = 'timeline-event';
            div.innerHTML = `
                <span class="ts">${ev.timestamp}</span>
                <span class="title">${ev.event_type}</span>
                <br><span style="color:var(--text-muted)">${ev.zone ? ev.zone + ' — ' : ''}${ev.detail}</span>
            `;
            timelineContainer.appendChild(div);
        });
    }

    function updateSessionInfo(s, f) {
        if (!s || !s.duration) return;
        document.getElementById('s-duration').textContent = s.duration;
        document.getElementById('s-peak-people').textContent = s.peak_people;
        document.getElementById('s-peak-density').textContent = s.peak_density + '%';
        document.getElementById('s-routes').textContent = s.route_changes;

        if (f) {
            document.getElementById('v-time').textContent = f.time + (f.total_time ? ' / ' + f.total_time : '');
            document.getElementById('v-frame').textContent = f.frame_idx;
            document.getElementById('v-fps').textContent = f.fps;
            if (f.progress !== undefined) {
                document.getElementById('v-progress').style.width = f.progress + '%';
            }
        }
    }

    function update3DMap(zones, route) {
        if (!zones) return;
        // Update zone visuals
        zones.forEach(z => {
            // Zone name is like "Zone A", we need the letter
            const letter = z.name.replace('Zone ', '');
            const el = document.getElementById('zone-' + letter);
            if (el) {
                const elevation = Math.min(z.density * 0.8, 80);
                const positions = { 'A': 0, 'B': 120, 'C': 240, 'D': 360 };
                const x = positions[letter] || 0;
                el.style.transform = `translate3d(${x}px, 0, ${elevation}px)`;

                let color = 'rgba(0, 230, 118, 0.2)';
                if (z.status === 'NORMAL') color = 'rgba(0, 230, 118, 0.3)';
                if (z.status === 'WARNING') color = 'rgba(255, 171, 0, 0.4)';
                if (z.status === 'HIGH') color = 'rgba(255, 109, 0, 0.5)';
                if (z.status === 'CRITICAL') color = 'rgba(255, 23, 68, 0.5)';
                el.querySelector('.zone-heatmap').style.background = color;
            }
        });

        // Update route details
        const rDet = document.getElementById('route-details');
        if (route && route.path && route.path.length > 0) {
            rDet.innerHTML = `
                <div><strong>Exit:</strong> ${route.exit}</div>
                <div><strong>Path:</strong> ${route.path.join(' → ')}</div>
                <div><strong>Cost:</strong> ${route.cost}</div>
            `;
        }
    }

    window.setMapView = function (mode) {
        const v = document.getElementById('venue-3d');
        if (mode === 'top') v.className = 'venue-3d top-view';
        else v.className = 'venue-3d angle-view';
    };

    // ── Drawing Engine (Canvas Overlays) ────────────────────

    function drawZoneHeatmap(zones, ctx, sx, sy) {
        if (!zones) return;
        zones.forEach(z => {
            if (z.bounds) {
                const x = z.bounds[0] * sx;
                const y = z.bounds[1] * sy;
                const w = (z.bounds[2] - z.bounds[0]) * sx;
                const h = (z.bounds[3] - z.bounds[1]) * sy;
                
                // Opacity scales with density (capped at 0.85)
                const opacity = Math.min((z.density / 100) * 0.85, 0.85);
                
                // Color gets darker (closer to dark red/black) as density increases
                // Low density = soft transparent blue/green
                let r = 0, g = 150, b = 255; 
                if (z.status === 'CRITICAL') { r = 139; g = 0; b = 0; } // Dark red
                else if (z.status === 'HIGH') { r = 255; g = 69; b = 0; } // Red-orange
                else if (z.status === 'WARNING') { r = 255; g = 165; b = 0; } // Orange
                else if (z.status === 'NORMAL') { r = 0; g = 200; b = 100; } // Green
                
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
                ctx.fillRect(x, y, w, h);
                
                // Subtle label
                ctx.fillStyle = `rgba(255, 255, 255, 0.7)`;
                ctx.font = '12px JetBrains Mono, monospace';
                ctx.fillText(`${z.name} (${z.density.toFixed(0)}%)`, x + 5, y + 16);
            }
        });
    }

    function drawOverlays(tracks, flows, zones, route, waves, exits, sx, sy) {
        if (!tracks) return;

        // 1. Heatmap
        hCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        
        // Draw the zone density heatmap
        drawZoneHeatmap(zones, hCtx, sx, sy);
        
        // Individual person hot-spots
        tracks.forEach(t => {
            const cx = t.cx * sx;
            const cy = t.cy * sy;
            const r = 40 * sx;
            const grad = hCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
            grad.addColorStop(0, 'rgba(255, 255, 255, 0.15)');
            grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
            hCtx.fillStyle = grad;
            hCtx.beginPath();
            hCtx.arc(cx, cy, r, 0, Math.PI * 2);
            hCtx.fill();
        });

    function drawZoneGrid(zones, ctx, sx, sy) {
        if (!zones) return;
        zones.forEach(z => {
            if (z.bounds) {
                const x = z.bounds[0] * sx;
                const y = z.bounds[1] * sy;
                const w = (z.bounds[2] - z.bounds[0]) * sx;
                const h = (z.bounds[3] - z.bounds[1]) * sy;

                // Draw dashed grid line for the zone boundary
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(x, y, w, h);
                ctx.setLineDash([]); // Reset
                
                // Draw a prominent label for the Zone name
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.fillRect(x, y, 120, 30);
                
                ctx.fillStyle = '#00d4ff'; // Bright cyan for visibility
                ctx.font = 'bold 16px sans-serif';
                ctx.fillText(z.name, x + 10, y + 22);
            }
        });
    }

    // 2. Tracking overlays
    tCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);

    // Draw the zone grid
    drawZoneGrid(zones, tCtx, sx, sy);

        // Clean old tracks
        const currentIds = new Set(tracks.map(t => t.id));
        Object.keys(trackHistory).forEach(k => {
            if (!currentIds.has(parseInt(k))) delete trackHistory[k];
        });

        tracks.forEach(t => {
            const id = t.id;
            const x1 = t.bbox[0] * sx;
            const y1 = t.bbox[1] * sy;
            const x2 = t.bbox[2] * sx;
            const y2 = t.bbox[3] * sy;
            const cx = t.cx * sx;
            const cy = t.cy * sy;

            // Track history for trail
            if (!trackHistory[id]) trackHistory[id] = [];
            trackHistory[id].push({ x: cx, y: cy });
            if (trackHistory[id].length > 15) trackHistory[id].shift();

            // Bounding box
            tCtx.strokeStyle = 'rgba(0, 212, 255, 0.7)';
            tCtx.lineWidth = 2;
            tCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // ID label
            tCtx.fillStyle = 'rgba(0, 212, 255, 0.9)';
            tCtx.font = '10px JetBrains Mono, monospace';
            tCtx.fillText(`ID:${id}`, x1, y1 - 4);

            // Motion trail
            const hist = trackHistory[id];
            if (hist.length > 1) {
                tCtx.beginPath();
                tCtx.moveTo(hist[0].x, hist[0].y);
                for (let i = 1; i < hist.length; i++) {
                    tCtx.lineTo(hist[i].x, hist[i].y);
                }
                tCtx.strokeStyle = 'rgba(124, 77, 255, 0.5)';
                tCtx.lineWidth = 2;
                tCtx.stroke();
            }
        });

        // 3. Flow arrows
        if (flows) {
            flows.forEach(f => {
                const tr = tracks.find(x => x.id === f.id);
                if (tr && f.speed > 5) {
                    const cx = tr.cx * sx;
                    const cy = tr.cy * sy;
                    const dx = f.dx * sx;
                    const dy = f.dy * sy;

                    tCtx.beginPath();
                    tCtx.moveTo(cx, cy);
                    tCtx.lineTo(cx + dx * 2, cy + dy * 2);
                    tCtx.strokeStyle = f.direction === 'LEFT' || f.direction === 'UP'
                        ? 'rgba(255, 171, 0, 0.8)' : 'rgba(0, 255, 100, 0.8)';
                    tCtx.lineWidth = 2;
                    tCtx.stroke();
                }
            });
        }

        // 4. AI Exit Suggestion Overlay
        if (route && route.path && route.path.length > 0) {
            tCtx.fillStyle = 'rgba(0, 0, 0, 0.75)';
            tCtx.fillRect(10, 10, 400, 65);
            
            // Draw a bright indicator border
            tCtx.strokeStyle = '#00e676';
            tCtx.lineWidth = 2;
            tCtx.strokeRect(10, 10, 400, 65);
            
            tCtx.fillStyle = '#00e676';
            tCtx.font = 'bold 16px sans-serif';
            tCtx.fillText(`⚡ AI SUGGESTION: Redirect to ${route.exit}`, 20, 35);
            
            tCtx.fillStyle = '#ffffff';
            tCtx.font = '14px sans-serif';
            tCtx.fillText(`Safest Route: ${route.path.join(' → ')}`, 20, 58);
        }

        // 5. Pressure Wave Predictor Overlay (CPWP)
        if (waves && waves.length > 0) {
            waves.forEach(w => {
                const sz = zones.find(z => z.name === w.source);
                const tz = zones.find(z => z.name === w.target);
                
                if (sz && tz) {
                    const scx = (sz.bounds[0] + sz.bounds[2]) / 2 * sx;
                    const scy = (sz.bounds[1] + sz.bounds[3]) / 2 * sy;
                    const tcx = (tz.bounds[0] + tz.bounds[2]) / 2 * sx;
                    const tcy = (tz.bounds[1] + tz.bounds[3]) / 2 * sy;
                    
                    // Draw animated-looking propagation line
                    tCtx.beginPath();
                    tCtx.moveTo(scx, scy);
                    tCtx.lineTo(tcx, tcy);
                    tCtx.strokeStyle = w.severity === 'CRITICAL' ? 'rgba(255, 35, 68, 0.9)' : 'rgba(255, 171, 0, 0.9)';
                    tCtx.lineWidth = 5;
                    tCtx.setLineDash([15, 10]);
                    tCtx.stroke();
                    tCtx.setLineDash([]);
                    
                    // Draw Warning Box
                    tCtx.fillStyle = 'rgba(15, 15, 20, 0.85)';
                    tCtx.fillRect(tcx - 140, tcy - 50, 280, 80);
                    
                    tCtx.strokeStyle = w.severity === 'CRITICAL' ? '#ff2344' : '#ffab00';
                    tCtx.lineWidth = 2;
                    tCtx.strokeRect(tcx - 140, tcy - 50, 280, 80);
                    
                    tCtx.fillStyle = w.severity === 'CRITICAL' ? '#ff2344' : '#ffab00';
                    tCtx.font = 'bold 13px sans-serif';
                    tCtx.fillText(`⚠️ ${w.msg}`, tcx - 130, tcy - 28);
                    
                    tCtx.fillStyle = '#00d4ff';
                    tCtx.font = 'bold 11px sans-serif';
                    tCtx.fillText(`ACTION: ${w.recs[0]}`, tcx - 130, tcy - 8);
                    tCtx.fillText(`ACTION: ${w.recs[1]}`, tcx - 130, tcy + 8);
                    tCtx.fillText(`ACTION: ${w.recs[2]}`, tcx - 130, tcy + 24);
                }
            });
        }
        
        // 6. Draw Exits
        if (exits && exits.length > 0) {
            exits.forEach(ex => {
                const x = ex.bounds[0] * sx;
                const y = ex.bounds[1] * sy;
                const w = (ex.bounds[2] - ex.bounds[0]) * sx;
                const h = (ex.bounds[3] - ex.bounds[1]) * sy;
                
                let color = 'rgba(255, 255, 255, 0.5)';
                let stroke = '#ffffff';
                let icon = '🚪';
                if (ex.status === 'RECOMMENDED') {
                    color = 'rgba(0, 230, 118, 0.4)';
                    stroke = '#00e676';
                    icon = '✅';
                } else if (ex.status === 'BLOCKED') {
                    color = 'rgba(255, 23, 68, 0.4)';
                    stroke = '#ff1744';
                    icon = '⛔';
                } else if (ex.status === 'OPEN') {
                    color = 'rgba(0, 212, 255, 0.4)';
                    stroke = '#00d4ff';
                }
                
                tCtx.fillStyle = color;
                tCtx.fillRect(x, y, w, h);
                
                tCtx.strokeStyle = stroke;
                tCtx.lineWidth = 3;
                tCtx.strokeRect(x, y, w, h);
                
                tCtx.fillStyle = '#ffffff';
                tCtx.font = 'bold 16px sans-serif';
                // Center text roughly
                tCtx.fillText(`${icon} ${ex.name}`, x + 10, y + h/2 + 5);
                
                // Status badge
                tCtx.fillStyle = stroke;
                tCtx.font = 'bold 12px sans-serif';
                tCtx.fillText(`[${ex.status}]`, x + 10, y + h/2 + 25);
            });
        }
    }
});
