/**
 * JWST NIRSpec MSA Operability & Geometry Visualizer
 * High performance Canvas engine supporting 250,000 microshutters with smooth 60fps pan/zoom.
 */

// MSA Physical Constants (Arcseconds)
// Aligned with STScI JDox NIRSpec MSA & Fixed Slits Geometry:
// Active Area: ~3.6' (216") cross-dispersion x 3.4' (204") dispersion
// Quadrant active area: 365 cols * 0.27" = 98.55" width; 171 rows * 0.53" = 90.63" height
// Central Gap Y = 20.0" (mounting bar); Gap X = 23.0" (cross-dispersion gap)
// Total Active Array Height = 2 * 90.63" + 20.0" = 201.26" (~3.35' / ~3.4')
// Total Active Array Width = 2 * 98.55" + 23.0" = 220.10" (~3.67' / ~3.6')
const MSA_CONSTANTS = {
  COLS: 365,
  ROWS: 171,
  TOTAL_SHUTTERS: 249660,
  SHUTTER_WIDTH: 0.20,     // Open aperture width (arcsec)
  SHUTTER_HEIGHT: 0.46,    // Open aperture height (arcsec)
  PITCH_X: 0.27,           // Grid spacing horizontal (arcsec)
  PITCH_Y: 0.53,           // Grid spacing vertical (arcsec)
  BAR_WALL: 0.07,          // Margin between open apertures (0.27 - 0.20 = 0.07)
  GAP_X: 23.0,             // Cross-dispersion gap between Q3/Q4 and Q1/Q2 (arcsec)
  GAP_Y: 20.0,             // Dispersion gap between Q3/Q1 and Q4/Q2 (central mounting bar = 20.0")
  DETECTOR_GAP: 18.0,      // Physical gap between NRS1 and NRS2 detectors (arcsec)
  DETECTOR_SIZE: 211.0,    // Square Teledyne Hawaii-2RG active dimension (2048 px * ~0.10303"/px = 211.0")
  DETECTOR_PIXELS: 2048,   // 2048 x 2048 pixels per detector
  DETECTOR_PIXEL_SCALE: 0.103027 // arcsec / pixel
};

// Shutter State Codes:
// 0: Operable / Normal
// 1: Stuck Closed
// 2: Vignetted Normal
// 3: Vignetted Stuck Closed
// 4: Stuck Open
// 5: Vignetted Stuck Open

const STATE_NAMES = {
  0: { label: 'Operable (Closed)', class: 'badge-operable', color: '#d4b886' },
  1: { label: 'Stuck Closed', class: 'badge-stuck-closed', color: '#555e69' },
  2: { label: 'Vignetted (Operable)', class: 'badge-vignetted', color: '#a855f7' },
  3: { label: 'Vignetted (Stuck Closed)', class: 'badge-vignetted-closed', color: '#7e22ce' },
  4: { label: 'Stuck Open', class: 'badge-stuck-open', color: '#f85149' },
  5: { label: 'Vignetted (Stuck Open)', class: 'badge-stuck-open', color: '#f85149' }
};

// Fixed slits & IFU coordinates (arcsec in MSA global coordinate frame)
// Sourced from mos_trace / Jakobsen et al. (2022) physical layout
const FIXED_SLITS = [
  { name: 'S200A1', x: -73.0, y: 7.3, w: 0.20, h: 3.20, desc: 'High-precision MOS slit A1 (primary MOS)' },
  { name: 'S200A2', x: -92.8, y: 3.5, w: 0.20, h: 3.20, desc: 'High-precision MOS slit A2 (primary MOS)' },
  { name: 'S400A1', x: -79.3, y: -0.5, w: 0.40, h: 3.65, desc: 'Exoplanet Transit Slit' },
  { name: 'S1600A1', x: -76.2, y: -3.8, w: 1.60, h: 1.60, desc: 'Square Exoplanet Aperture' },
  { name: 'S200B1', x: 85.0, y: -7.0, w: 0.20, h: 3.20, desc: 'Redundant MOS slit B1 (opposite side)' },
  { name: 'IFU', x: -98.5, y: -2.0, w: 3.00, h: 3.00, desc: 'Integral Field Unit 3.0"x3.0" Entrance Aperture' }
];

class MSAVisualizer {
  constructor() {
    this.canvas = document.getElementById('msa-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.container = document.getElementById('viewport-container');

    // Camera transform: (x, y) is origin center in arcsec, scale is px per arcsec
    this.camera = {
      x: 0,
      y: 0,
      scale: 3.2 // Initial scale (px per arcsec)
    };

    // User customized open shutters (Set of string 'q-x-y')
    this.userOpenShutters = new Set();
    this.selectedShutter = null; // No shutter selected by default (only selected on explicit click)

    // Multi-map operability data repository
    this.allMapsData = null;
    this.currentVersion = '0018';
    this.availableVersions = ['0016', '0017', '0018'];
    this.blinkTimer = null;
    this.isBlinking = false;
    this.blinkInterval = 800; // ms

    // Grid data: 4 Uint8Array buffers (Q1=0, Q2=1, Q3=2, Q4=3)
    this.quadGrids = [
      new Uint8Array(MSA_CONSTANTS.COLS * MSA_CONSTANTS.ROWS),
      new Uint8Array(MSA_CONSTANTS.COLS * MSA_CONSTANTS.ROWS),
      new Uint8Array(MSA_CONSTANTS.COLS * MSA_CONSTANTS.ROWS),
      new Uint8Array(MSA_CONSTANTS.COLS * MSA_CONSTANTS.ROWS)
    ];

    this.stuckOpenList = [];
    this.dataLoaded = false;

    // Layer toggles (Default matching user configuration)
    this.layers = {
      fillOpen: false, // Fill Open Shutters (off by default)
      detectors: false, // Detectors off by default
      detectorPixels: false,
      vignetting: true,
      stuckOpen: true,
      stuckClosed: true,
      fixedSlits: true,
      magnetArm: false,
      grid: true,
      coordinates: true,
      dispersion: true
    };


    this.shutterOpacity = 0.40; // Default 40% opacity for open shutters


    // Load persisted settings from localStorage
    this.loadSettings();

    // Mouse / interaction state
    this.isDragging = false;
    this.dragStart = { x: 0, y: 0 };
    this.mousePos = { x: 0, y: 0 };
    this.hoveredShutter = null;

    this.init();
  }

  loadSettings() {
    try {
      const saved = localStorage.getItem('msa_viewer_settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.layers && typeof parsed.layers === 'object') {
          this.layers = { ...this.layers, ...parsed.layers };
        }
        if (typeof parsed.shutterOpacity === 'number') {
          this.shutterOpacity = Math.max(0, Math.min(1, parsed.shutterOpacity));
        }
        if (typeof parsed.blinkInterval === 'number') {
          this.blinkInterval = parsed.blinkInterval;
        }
        if (parsed.currentVersion) {
          this.currentVersion = parsed.currentVersion;
        }
      }
    } catch (e) {
      console.warn('Unable to load localStorage settings:', e);
    }
  }

  saveSettings() {
    try {
      const settings = {
        layers: this.layers,
        shutterOpacity: this.shutterOpacity,
        blinkInterval: this.blinkInterval,
        currentVersion: this.currentVersion
      };
      localStorage.setItem('msa_viewer_settings', JSON.stringify(settings));
    } catch (e) {
      console.warn('Unable to save localStorage settings:', e);
    }
  }

  applySettingsToUI() {
    // Sync layer checkboxes
    const layerMap = {
      'layer-fill-open': 'fillOpen',
      'layer-detectors': 'detectors',
      'layer-detector-pixels': 'detectorPixels',
      'layer-vignetting': 'vignetting',
      'layer-stuck-open': 'stuckOpen',
      'layer-stuck-closed': 'stuckClosed',
      'layer-fixed-slits': 'fixedSlits',
      'layer-magnet-arm': 'magnetArm',
      'layer-grid': 'grid',
      'layer-coordinates': 'coordinates',
      'layer-dispersion': 'dispersion'
    };


    for (const [id, key] of Object.entries(layerMap)) {
      const el = document.getElementById(id);
      if (el) {
        el.checked = !!this.layers[key];
      }
    }

    this.updateDispersionIndicator();

    // Sync Opacity slider & label
    const opacitySlider = document.getElementById('shutter-opacity-slider');
    const opacityLabel = document.getElementById('shutter-opacity-val');
    const pct = Math.round(this.shutterOpacity * 100);
    if (opacitySlider) opacitySlider.value = pct;
    if (opacityLabel) opacityLabel.innerText = `${pct}%`;

    // Sync Blink speed slider & label
    const blinkSlider = document.getElementById('blink-speed-slider');
    const blinkLabel = document.getElementById('blink-rate-label');
    if (blinkSlider) blinkSlider.value = this.blinkInterval;
    if (blinkLabel) blinkLabel.innerText = `${(this.blinkInterval / 1000).toFixed(1)}s`;
  }

  updateDispersionIndicator() {
    const el = document.getElementById('dispersion-indicator');
    if (el) {
      el.style.display = this.layers.dispersion ? 'flex' : 'none';
    }
  }

  async init() {
    this.setupResizeHandler();
    this.setupEventListeners();
    this.applySettingsToUI();
    await this.loadAllMapsData();
    this.fitView('msa');
    this.render();
  }


  setupResizeHandler() {
    const resize = () => {
      const rect = this.container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.width = rect.width;
      this.height = rect.height;
      this.dpr = dpr;
      this.render();
    };

    window.addEventListener('resize', resize);
    resize();
  }

  async loadAllMapsData() {
    try {
      const res = await fetch('data/msa_all_maps.json');
      this.allMapsData = await res.json();
      this.availableVersions = this.allMapsData.versions || ['0016', '0017', '0018'];
      
      // Default to latest version (e.g. 0018)
      this.currentVersion = this.availableVersions[this.availableVersions.length - 1];
      
      this.populateMapSelector();
      await this.setActiveMap(this.currentVersion);
    } catch (err) {
      console.error('Failed to load multi-map dataset, loading fallback compact map:', err);
      await this.loadFallbackSingleMap();
    }
  }

  getMapColor(version) {
    // Total versions: 0018 (newest -> green), 0017 (yellow-orange), 0016 (oldest -> red)
    const idx = this.availableVersions.indexOf(version);
    if (idx === -1) return '#58a6ff';
    const total = this.availableVersions.length;
    if (total <= 1) return '#22c55e';

    // ratio: 0 (oldest) -> 1 (newest)
    const ratio = idx / (total - 1);
    if (ratio >= 0.9) return '#22c55e'; // Green (newest)
    if (ratio >= 0.5) return '#eab308'; // Yellow
    if (ratio >= 0.25) return '#f97316'; // Orange
    return '#ef4444'; // Red (oldest)
  }

  populateMapSelector() {
    const select = document.getElementById('map-select');
    if (!select) return;
    select.innerHTML = '';

    for (const v of this.availableVersions) {
      const opt = document.createElement('option');
      opt.value = v;
      const mapMeta = this.allMapsData?.datasets?.[v];
      const useAfterStr = mapMeta?.useafter ? mapMeta.useafter.split('T')[0] : '';
      opt.textContent = `CRDS v${v} (${useAfterStr || 'In-Flight'})`;
      opt.style.color = this.getMapColor(v);
      opt.style.backgroundColor = '#161b22';
      opt.style.fontWeight = '600';
      if (v === this.currentVersion) opt.selected = true;
      select.appendChild(opt);
    }
    this.updateSelectColor();
  }

  updateSelectColor() {
    const select = document.getElementById('map-select');
    if (select) {
      const color = this.getMapColor(this.currentVersion);
      select.style.color = color;
      select.style.borderColor = `${color}66`;
    }
  }

  async setActiveMap(version) {
    if (!this.allMapsData || !this.allMapsData.datasets[version]) return;
    this.currentVersion = version;

    const dataset = this.allMapsData.datasets[version];
    this.stuckOpenList = dataset.stuck_open || [];

    // Decode compressed grids
    for (let q = 0; q < 4; q++) {
      const b64 = dataset.grids_zlib_b64[q];
      const binaryStr = atob(b64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }

      if (window.DecompressionStream) {
        const ds = new DecompressionStream('deflate-raw');
        const writer = ds.writable.getWriter();
        writer.write(bytes.subarray(2, bytes.length - 4));
        writer.close();
        const response = new Response(ds.readable);
        const decompressed = await response.arrayBuffer();
        this.quadGrids[q] = new Uint8Array(decompressed);
      } else {
        this.quadGrids[q] = bytes;
      }
    }

    // Update UI elements
    const select = document.getElementById('map-select');
    if (select && select.value !== version) select.value = version;
    this.updateSelectColor();

    // Update Sidebar Operability Statistics
    const stats = dataset.stats;
    if (stats) {
      document.getElementById('stat-total').textContent = stats.total.toLocaleString();
      const pctOperable = ((stats.operable / stats.total) * 100).toFixed(1);
      document.getElementById('stat-operable').textContent = `${stats.operable.toLocaleString()} (${pctOperable}%)`;
      const pctClosed = ((stats.stuck_closed / stats.total) * 100).toFixed(1);
      document.getElementById('stat-stuck-closed').textContent = `${stats.stuck_closed.toLocaleString()} (${pctClosed}%)`;
      const pctOpen = ((stats.stuck_open / stats.total) * 100).toFixed(2);
      document.getElementById('stat-stuck-open').textContent = `${stats.stuck_open.toLocaleString()} (${pctOpen}%)`;
      const pctVig = ((stats.vignetted / stats.total) * 100).toFixed(1);
      document.getElementById('stat-vignetted').textContent = `${stats.vignetted.toLocaleString()} (${pctVig}%)`;
      
      const stuckOpenLayerLabel = document.querySelector('#layer-stuck-open ~ .toggle-label');
      if (stuckOpenLayerLabel) {
        stuckOpenLayerLabel.textContent = `Stuck Open Badges & Glow (${stats.stuck_open})`;
      }
    }

    this.populateStuckOpenList();
    this.populateVignettedList();
    if (this.selectedShutter) {
      this.updateInspector(this.selectedShutter.q, this.selectedShutter.x, this.selectedShutter.y);
    } else {
      this.updateInspector(null);
    }
    this.dataLoaded = true;

    this.saveSettings();
    this.render();
  }

  stepMap(direction) {
    const idx = this.availableVersions.indexOf(this.currentVersion);
    if (idx === -1) return;
    let newIdx = idx + direction;
    if (newIdx >= this.availableVersions.length) newIdx = 0;
    if (newIdx < 0) newIdx = this.availableVersions.length - 1;
    this.setActiveMap(this.availableVersions[newIdx]);
  }

  toggleBlink(forceState = null) {
    const shouldBlink = (forceState !== null) ? forceState : !this.isBlinking;
    this.isBlinking = shouldBlink;

    const blinkBtn = document.getElementById('btn-blink-toggle');
    const container = document.getElementById('blink-slider-container');
    if (blinkBtn) blinkBtn.classList.toggle('active', this.isBlinking);
    if (container) container.classList.toggle('active', this.isBlinking);

    if (this.blinkTimer) {
      clearInterval(this.blinkTimer);
      this.blinkTimer = null;
    }

    if (this.isBlinking) {
      this.blinkTimer = setInterval(() => {
        this.stepMap(1);
      }, this.blinkInterval);
    }
  }

  setBlinkInterval(ms) {
    this.blinkInterval = Math.max(150, ms);
    const label = document.getElementById('blink-rate-label');
    if (label) {
      label.textContent = `${(this.blinkInterval / 1000).toFixed(1)}s`;
    }
    this.saveSettings();
    if (this.isBlinking) {
      clearInterval(this.blinkTimer);
      this.blinkTimer = setInterval(() => {
        this.stepMap(1);
      }, this.blinkInterval);
    }
  }

  async loadFallbackSingleMap() {
    try {
      const res = await fetch('data/msa_compact.json');
      const data = await res.json();
      this.stuckOpenList = data.stuck_open || [];
      this.populateStuckOpenList();
      this.dataLoaded = true;
      this.render();
    } catch (e) {
      this.buildFallbackData();
    }
  }

  buildFallbackData() {
    this.stuckOpenList = [
      { q: 2, x: 88, y: 116, vig: false, label: 'q2d88s116' },
      { q: 1, x: 38, y: 25, vig: false, label: 'q1d38s25' }
    ];
    this.dataLoaded = true;
  }

  toggleSidebar(forceState = null) {
    const sidebar = document.getElementById('sidebar');
    const toggleLeftBtn = document.getElementById('toggle-left-sidebar');
    if (!sidebar) return;

    if (forceState !== null) {
      sidebar.classList.toggle('collapsed', !forceState);
    } else {
      sidebar.classList.toggle('collapsed');
    }

    const isActive = !sidebar.classList.contains('collapsed');
    if (toggleLeftBtn) toggleLeftBtn.classList.toggle('active', isActive);
    setTimeout(() => window.dispatchEvent(new Event('resize')), 260);
  }

  toggleDrawer(forceState = null) {
    const drawer = document.getElementById('inspector-drawer');
    const toggleRightBtn = document.getElementById('toggle-right-drawer');
    if (!drawer) return;

    if (forceState !== null) {
      drawer.classList.toggle('collapsed', !forceState);
    } else {
      drawer.classList.toggle('collapsed');
    }

    const isActive = !drawer.classList.contains('collapsed');
    if (toggleRightBtn) toggleRightBtn.classList.toggle('active', isActive);
    setTimeout(() => window.dispatchEvent(new Event('resize')), 260);
  }

  setupEventListeners() {
    // Mouse pan & drag tracking
    let mouseDownPos = { x: 0, y: 0 };
    let dragDistance = 0;

    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this.isDragging = true;
        this.dragStart = { x: e.clientX, y: e.clientY };
        mouseDownPos = { x: e.clientX, y: e.clientY };
        dragDistance = 0;
        this.canvas.classList.add('dragging');
      }
    });

    window.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.mousePos = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };

      if (this.isDragging) {
        const moveDist = Math.hypot(e.clientX - mouseDownPos.x, e.clientY - mouseDownPos.y);
        dragDistance = Math.max(dragDistance, moveDist);

        const dx = (e.clientX - this.dragStart.x) / this.camera.scale;
        const dy = (e.clientY - this.dragStart.y) / this.camera.scale;
        this.camera.x -= dx;
        // Invert dy to match screen drag direction with canvas Y-inverted world space
        this.camera.y += dy;
        this.dragStart = { x: e.clientX, y: e.clientY };
        this.render();
      }

      this.updateHover();
    });

    window.addEventListener('mouseup', () => {
      if (this.isDragging) {
        this.isDragging = false;
        this.canvas.classList.remove('dragging');
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredShutter = null;
      const hudId = document.getElementById('hud-shutter-id');
      const hudBadge = document.getElementById('hud-status-badge');
      const hudQuad = document.getElementById('hud-quad');
      const hudQuadCol = document.getElementById('hud-quad-col');
      const hudQuadRow = document.getElementById('hud-quad-row');
      const hudGlobalCol = document.getElementById('hud-global-col');
      const hudGlobalRow = document.getElementById('hud-global-row');
      if (hudBadge) hudBadge.style.display = 'none';
      if (hudQuad) hudQuad.innerText = '-';
      if (hudQuadCol) hudQuadCol.innerText = '-';
      if (hudQuadRow) hudQuadRow.innerText = '-';
      if (hudGlobalCol) hudGlobalCol.innerText = '-';
      if (hudGlobalRow) hudGlobalRow.innerText = '-';
      this.render();
    });


    // Slowed down, gentle Wheel Zoom centered at cursor (reduced from 1.18 to 1.06)
    // Listened on the viewport container so wheel zoom works across canvas and floating overlays like DISPERSION
    this.container.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.06 : (1 / 1.06);
      this.zoomAt(this.mousePos.x, this.mousePos.y, zoomFactor);
      this.updateHover();
    }, { passive: false });



    // Keyboard navigation: Left/Right arrows switch operability maps, Cmd-J / Cmd-K toggle panels
    window.addEventListener('keydown', (e) => {
      // Allow Cmd-J / Cmd-K even if focused inside input
      const isCmdOrCtrl = e.metaKey || e.ctrlKey;
      if (isCmdOrCtrl) {
        if (e.key === 'j' || e.key === 'J') {
          e.preventDefault();
          this.toggleSidebar();
          return;
        }
        if (e.key === 'k' || e.key === 'K') {
          e.preventDefault();
          this.toggleDrawer();
          return;
        }
      }

      // Escape closes open modals or deselects inputs
      if (e.key === 'Escape') {
        const modal = document.getElementById('guide-modal');
        if (modal && !modal.classList.contains('hidden')) {
          e.preventDefault();
          modal.classList.add('hidden');
          return;
        }
      }

      // Don't intercept single-key shortcuts if Cmd/Ctrl/Alt is held down (e.g. Cmd-R / Ctrl-R for browser refresh)
      if (isCmdOrCtrl || e.altKey) {
        return;
      }

      // Don't intercept regular keys if user is typing in search box
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        this.stepMap(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        this.stepMap(1);
      } else if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        this.toggleBlink();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        const targetScale = Math.min(this.width / 240, this.height / 240);
        this.animateCameraTo(0, 0, targetScale, 350);
      } else if (e.key === 'h' || e.key === 'H' || e.key === '/' || e.key === '?') {
        e.preventDefault();
        this.openGuideModal();
      } else if (e.key === '+' || e.key === '=') {
        this.zoomAt(this.width / 2, this.height / 2, 1.12);
      } else if (e.key === '-' || e.key === '_') {
        this.zoomAt(this.width / 2, this.height / 2, 1 / 1.12);
      }





    });

    // Map switcher UI buttons & select dropdown
    const select = document.getElementById('map-select');
    if (select) {
      select.addEventListener('change', (e) => {
        this.setActiveMap(e.target.value);
      });
    }

    const prevBtn = document.getElementById('btn-prev-map');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => this.stepMap(-1));
    }

    const nextBtn = document.getElementById('btn-next-map');
    if (nextBtn) {
      nextBtn.addEventListener('click', () => this.stepMap(1));
    }

    // Blink mode toggle button & speed slider
    const blinkBtn = document.getElementById('btn-blink-toggle');
    if (blinkBtn) {
      blinkBtn.addEventListener('click', () => this.toggleBlink());
    }

    const blinkSlider = document.getElementById('blink-speed-slider');
    if (blinkSlider) {
      blinkSlider.addEventListener('input', (e) => {
        this.setBlinkInterval(parseInt(e.target.value));
      });
    }

    // Click to select/deselect shutter or deselect when clicking outside MSA
    this.canvas.addEventListener('click', (e) => {
      // Ignore click if it was part of a drag/pan motion (> 5px)
      if (dragDistance > 5) return;

      if (this.hoveredShutter) {
        // If clicking already selected shutter, toggle/deselect it
        if (
          this.selectedShutter &&
          this.selectedShutter.q === this.hoveredShutter.q &&
          this.selectedShutter.x === this.hoveredShutter.x &&
          this.selectedShutter.y === this.hoveredShutter.y
        ) {
          this.selectedShutter = null;
          this.updateInspector(null);
        } else {
          this.selectedShutter = { ...this.hoveredShutter };
          this.updateInspector(this.selectedShutter.q, this.selectedShutter.x, this.selectedShutter.y);
        }
      } else {
        // Clicked outside any microshutter -> deselect
        this.selectedShutter = null;
        this.updateInspector(null);
      }
      this.render();
    });

    // Double click to center view smoothly at clicked coordinate
    this.canvas.addEventListener('dblclick', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      const targetWorld = this.screenToWorld(clickX, clickY);

      // Smoothly animate camera to center targetWorld
      this.animateCameraTo(targetWorld.x, targetWorld.y, this.camera.scale);
    });
    let lastTouchDist = 0;
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.dragStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        lastTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    });

    this.canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && this.isDragging) {
        const dx = (e.touches[0].clientX - this.dragStart.x) / this.camera.scale;
        const dy = (e.touches[0].clientY - this.dragStart.y) / this.camera.scale;
        this.camera.x -= dx;
        this.camera.y += dy;
        this.dragStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.render();
      } else if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        if (lastTouchDist > 0) {
          const factor = dist / lastTouchDist;
          this.zoomAt(midX, midY, factor);
        }
        lastTouchDist = dist;
      }
    });

    this.canvas.addEventListener('touchend', () => {
      this.isDragging = false;
      lastTouchDist = 0;
    });

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.getAttribute('data-view');
        this.fitView(view);
      });
    });

    // Layer checkboxes
    const layerMap = {
      'layer-fill-open': 'fillOpen',
      'layer-detectors': 'detectors',
      'layer-detector-pixels': 'detectorPixels',
      'layer-vignetting': 'vignetting',
      'layer-stuck-open': 'stuckOpen',
      'layer-stuck-closed': 'stuckClosed',
      'layer-fixed-slits': 'fixedSlits',
      'layer-magnet-arm': 'magnetArm',
      'layer-grid': 'grid',
      'layer-coordinates': 'coordinates',
      'layer-dispersion': 'dispersion'
    };

    for (const [id, key] of Object.entries(layerMap)) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', (e) => {
          this.layers[key] = e.target.checked;
          this.updateDispersionIndicator();
          this.saveSettings();
          this.render();
        });
      }
    }

    // Open Shutter Opacity Slider
    const opacitySlider = document.getElementById('shutter-opacity-slider');
    const opacityLabel = document.getElementById('shutter-opacity-val');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        this.shutterOpacity = val / 100;
        if (opacityLabel) opacityLabel.innerText = `${val}%`;
        // If user adjusts opacity > 0, auto-enable Fill Open Shutters checkbox if not checked
        if (val > 0 && !this.layers.fillOpen) {
          this.layers.fillOpen = true;
          const fillEl = document.getElementById('layer-fill-open');
          if (fillEl) fillEl.checked = true;
        }
        this.saveSettings();
        this.render();
      });
    }


    // Dispersion Indicator Banner (click to hide)
    const dispersionEl = document.getElementById('dispersion-indicator');
    if (dispersionEl) {
      dispersionEl.addEventListener('click', () => {
        this.layers.dispersion = false;
        const layerDispEl = document.getElementById('layer-dispersion');
        if (layerDispEl) layerDispEl.checked = false;
        this.updateDispersionIndicator();
        this.saveSettings();
      });
    }




    // Canvas UI Buttons (Zoom In/Out, Reset View)
    document.getElementById('zoom-in-btn').addEventListener('click', () => {
      this.zoomAt(this.width / 2, this.height / 2, 1.3);
    });

    document.getElementById('zoom-out-btn').addEventListener('click', () => {
      this.zoomAt(this.width / 2, this.height / 2, 1 / 1.3);
    });

    const canvasResetBtn = document.getElementById('canvas-reset-btn');
    if (canvasResetBtn) {
      canvasResetBtn.addEventListener('click', () => {
        const targetScale = Math.min(this.width / 240, this.height / 240);
        this.animateCameraTo(0, 0, targetScale, 350);
      });
    }


    // Panel Layout Toggles (Left Sidebar & Right Inspector Drawer)
    const toggleLeftBtn = document.getElementById('toggle-left-sidebar');
    if (toggleLeftBtn) {
      toggleLeftBtn.addEventListener('click', () => this.toggleSidebar());
    }

    const toggleRightBtn = document.getElementById('toggle-right-drawer');
    if (toggleRightBtn) {
      toggleRightBtn.addEventListener('click', () => this.toggleDrawer());
    }

    const closeDrawerBtn = document.getElementById('close-drawer');
    if (closeDrawerBtn) {
      closeDrawerBtn.addEventListener('click', () => this.toggleDrawer(false));
    }

    const btnResetView = document.getElementById('btn-reset-view');
    if (btnResetView) {
      btnResetView.addEventListener('click', () => {
        this.fitView('msa');
      });
    }



    // Search Box
    const searchInput = document.getElementById('shutter-search');
    const doSearch = () => {
      const query = searchInput.value.trim();
      this.searchAndJump(query);
      searchInput.blur(); // Remove focus so Arrow keys and keyboard shortcuts work immediately
    };
    document.getElementById('search-btn').addEventListener('click', doSearch);
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        doSearch();
      }
    });

    // Inspector Interactive Title Input (Instant Jump)
    const inspInput = document.getElementById('insp-title-input');
    if (inspInput) {
      const doInspJump = () => {
        const val = inspInput.value.trim();
        if (val) this.searchAndJump(val);
      };
      inspInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          doInspJump();
          inspInput.blur();
        }
      });
      inspInput.addEventListener('change', doInspJump);
    }

    // Inspector Action Buttons
    document.getElementById('btn-toggle-open').addEventListener('click', () => {
      const key = `${this.selectedShutter.q}-${this.selectedShutter.x}-${this.selectedShutter.y}`;
      if (this.userOpenShutters.has(key)) {
        this.userOpenShutters.delete(key);
      } else {
        this.userOpenShutters.add(key);
      }
      this.updateInspector(this.selectedShutter.q, this.selectedShutter.x, this.selectedShutter.y);
      this.render();
    });

    document.getElementById('btn-open-slitlet-3').addEventListener('click', () => {
      const q = this.selectedShutter.q;
      const x = this.selectedShutter.x;
      const y = this.selectedShutter.y;
      // Slitlet across 3 consecutive rows: y-1, y, y+1
      for (let r = Math.max(1, y - 1); r <= Math.min(MSA_CONSTANTS.ROWS, y + 1); r++) {
        this.userOpenShutters.add(`${q}-${x}-${r}`);
      }
      this.updateInspector(q, x, y);
      this.render();
    });

    // Documentation Modal
    document.getElementById('btn-show-guide').addEventListener('click', () => {
      this.openGuideModal();
    });

    const modal = document.getElementById('guide-modal');
    document.getElementById('close-modal').addEventListener('click', () => {
      modal.classList.add('hidden');
    });

    modal.querySelector('.modal-backdrop').addEventListener('click', () => {
      modal.classList.add('hidden');
    });
  }

  async openGuideModal() {
    const modal = document.getElementById('guide-modal');
    if (!modal) return;
    try {
      const res = await fetch('MSA.md');
      const text = await res.text();
      document.getElementById('modal-content').innerHTML = this.renderMarkdown(text);
    } catch (e) {
      document.getElementById('modal-content').innerHTML = '<p>Unable to load MSA.md</p>';
    }
    modal.classList.remove('hidden');
  }


  // Convert screen coordinates to world coordinates (arcsec)
  screenToWorld(screenX, screenY) {
    const wx = this.camera.x + (screenX - this.width / 2) / this.camera.scale;
    // Since canvas renders with ctx.scale(scale, -scale), screen Y down is world Y down (-Y)
    const wy = this.camera.y - (screenY - this.height / 2) / this.camera.scale;
    return { x: wx, y: wy };
  }

  // Convert world coordinates (arcsec) to screen coordinates
  worldToScreen(wx, wy) {
    const sx = this.width / 2 + (wx - this.camera.x) * this.camera.scale;
    const sy = this.height / 2 - (wy - this.camera.y) * this.camera.scale;
    return { x: sx, y: sy };
  }

  zoomAt(screenX, screenY, factor) {
    const minScale = 1.0;
    const maxScale = 500.0;
    const newScale = Math.min(Math.max(this.camera.scale * factor, minScale), maxScale);

    if (newScale === this.camera.scale) return;

    // World coordinates under cursor before zoom
    const worldBefore = this.screenToWorld(screenX, screenY);

    this.camera.scale = newScale;

    // Adjust camera position so worldBefore stays exactly under screenX, screenY
    this.camera.x = worldBefore.x - (screenX - this.width / 2) / this.camera.scale;
    this.camera.y = worldBefore.y + (screenY - this.height / 2) / this.camera.scale;

    this.render();
  }

  // Geometry mappings
  // Quad coordinates (Q, col, row) -> Arcsec relative to MSA optical center (0,0)
  // Rows in CRDS files are indexed such that Row 1..10 are vignetted at the top of Q1/Q3,
  // and Row 161..171 are vignetted at the bottom of Q2/Q4.
  // Columns in CRDS files:
  // Q1: Col 1 is right (outer), Col 365 is left (gap)
  // Q2: Col 1 is right (outer), Col 365 is left (gap)
  // Q3: Col 1 is left (outer), Col 365 is right (gap)
  // Q4: Col 1 is left (outer), Col 365 is right (gap)
  quadToWorld(q, col, row) {
    const halfGapX = MSA_CONSTANTS.GAP_X / 2;
    const halfGapY = MSA_CONSTANTS.GAP_Y / 2;
    const qW = MSA_CONSTANTS.COLS * MSA_CONSTANTS.PITCH_X;
    const qH = MSA_CONSTANTS.ROWS * MSA_CONSTANTS.PITCH_Y;
    const pitchX = MSA_CONSTANTS.PITCH_X;
    const pitchY = MSA_CONSTANTS.PITCH_Y;

    let wx = 0;
    let wy = 0;

    switch (q) {
      case 1: // Top-Right (+X, +Y): row 1 is top outer, row 171 is center gap. col 1 is right outer, col 365 is center gap.
        wx = (halfGapX + qW) - (col - 0.5) * pitchX;
        wy = (halfGapY + qH) - (row - 0.5) * pitchY;
        break;
      case 2: // Bottom-Right (+X, -Y): row 1 is center gap, row 171 is bottom outer. col 1 is right outer, col 365 is center gap.
        wx = (halfGapX + qW) - (col - 0.5) * pitchX;
        wy = -halfGapY - (row - 0.5) * pitchY;
        break;
      case 3: // Top-Left (-X, +Y): row 1 is top outer, row 171 is center gap. col 1 is center gap, col 365 is left outer.
        wx = -halfGapX - (col - 0.5) * pitchX;
        wy = (halfGapY + qH) - (row - 0.5) * pitchY;
        break;
      case 4: // Bottom-Left (-X, -Y): row 1 is center gap, row 171 is bottom outer. col 1 is center gap, col 365 is left outer.
        wx = -halfGapX - (col - 0.5) * pitchX;
        wy = -halfGapY - (row - 0.5) * pitchY;
        break;
    }

    return { x: wx, y: wy };
  }

  // Arcsec world coordinates -> Quad coordinates (Q, col, row)
  worldToQuad(wx, wy) {
    const halfGapX = MSA_CONSTANTS.GAP_X / 2;
    const halfGapY = MSA_CONSTANTS.GAP_Y / 2;
    const qW = MSA_CONSTANTS.COLS * MSA_CONSTANTS.PITCH_X;
    const qH = MSA_CONSTANTS.ROWS * MSA_CONSTANTS.PITCH_Y;
    const pitchX = MSA_CONSTANTS.PITCH_X;
    const pitchY = MSA_CONSTANTS.PITCH_Y;

    let q = 0;
    let col = 0;
    let row = 0;

    if (wx >= halfGapX && wx <= halfGapX + qW && wy >= halfGapY && wy <= halfGapY + qH) {
      q = 1;
      col = Math.floor(((halfGapX + qW) - wx) / pitchX) + 1;
      row = Math.floor(((halfGapY + qH) - wy) / pitchY) + 1;
    } else if (wx >= halfGapX && wx <= halfGapX + qW && wy <= -halfGapY && wy >= -halfGapY - qH) {
      q = 2;
      col = Math.floor(((halfGapX + qW) - wx) / pitchX) + 1;
      row = Math.floor((-halfGapY - wy) / pitchY) + 1;
    } else if (wx <= -halfGapX && wx >= -halfGapX - qW && wy >= halfGapY && wy <= halfGapY + qH) {
      q = 3;
      col = Math.floor((-halfGapX - wx) / pitchX) + 1;
      row = Math.floor(((halfGapY + qH) - wy) / pitchY) + 1;
    } else if (wx <= -halfGapX && wx >= -halfGapX - qW && wy <= -halfGapY && wy >= -halfGapY - qH) {
      q = 4;
      col = Math.floor((-halfGapX - wx) / pitchX) + 1;
      row = Math.floor((-halfGapY - wy) / pitchY) + 1;
    }

    if (q >= 1 && q <= 4 && col >= 1 && col <= MSA_CONSTANTS.COLS && row >= 1 && row <= MSA_CONSTANTS.ROWS) {
      return { q, col, row };
    }

    return null;
  }

  // Convert (Q, col, row) to global 2x2 grid coordinates (1..730, 1..342)
  // Starting at Top-Right corner (Q1: col 1, row 1):
  // Global Col increases from Right to Left (Q1/Q2 -> Q3/Q4): 1..730
  // Global Row increases from Top to Bottom (Q1/Q3 -> Q2/Q4): 1..342
  quadToGlobal(q, col, row) {
    let globalCol = 0;
    let globalRow = 0;

    // Columns: Q1/Q2 (Right): 1..365, Q3/Q4 (Left): 366..730
    if (q === 1 || q === 2) {
      globalCol = col;
    } else {
      globalCol = MSA_CONSTANTS.COLS + col;
    }

    // Rows: Q1/Q3 (Top): 1..171, Q2/Q4 (Bottom): 172..342
    if (q === 1 || q === 3) {
      globalRow = row;
    } else {
      globalRow = MSA_CONSTANTS.ROWS + row;
    }

    return { globalCol, globalRow };
  }


  getShutterState(q, col, row) {
    if (q < 1 || q > 4 || col < 1 || col > MSA_CONSTANTS.COLS || row < 1 || row > MSA_CONSTANTS.ROWS) {
      return 0;
    }
    const idx = (col - 1) * MSA_CONSTANTS.ROWS + (row - 1);
    return this.quadGrids[q - 1][idx] || 0;
  }

  fitView(preset) {
    switch (preset) {
      case 'all': // Detectors + MSA: total width = 2*211 + 18 = 440", total height = 211" + frame margin
        this.camera.x = 0;
        this.camera.y = 0;
        this.camera.scale = Math.min(this.width / 460, this.height / 240);
        break;
      case 'msa': // Full 2x2 MSA active area
        this.camera.x = 0;
        this.camera.y = 0;
        this.camera.scale = Math.min(this.width / 240, this.height / 240);
        break;
      case 'q1': // Quadrant 1 (Top-Right)
        this.centerOnQuad(1);
        break;
      case 'q2': // Quadrant 2 (Bottom-Right)
        this.centerOnQuad(2);
        break;
      case 'q3': // Quadrant 3 (Top-Left)
        this.centerOnQuad(3);
        break;
      case 'q4': // Quadrant 4 (Bottom-Left)
        this.centerOnQuad(4);
        break;
      case 'fixed-slits':
        this.camera.x = 0;
        this.camera.y = 0;
        this.camera.scale = Math.min(this.width / 80, this.height / 45);
        break;
      case 'stuck-open':
        // Jump to famous stuck open shutter q2d88s116
        this.jumpToShutter(2, 88, 116, 60);
        break;
    }
    this.render();
  }

  centerOnQuad(q) {
    const centerShutter = this.quadToWorld(q, 183, 86);
    this.camera.x = centerShutter.x;
    this.camera.y = centerShutter.y;
    this.camera.scale = Math.min(this.width / 120, this.height / 110);
  }

  animateCameraTo(targetX, targetY, targetScale = this.camera.scale, duration = 300) {
    if (this._animFrame) cancelAnimationFrame(this._animFrame);

    const startX = this.camera.x;
    const startY = this.camera.y;
    const startScale = this.camera.scale;
    const startTime = performance.now();

    const step = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);

      this.camera.x = startX + (targetX - startX) * ease;
      this.camera.y = startY + (targetY - startY) * ease;
      this.camera.scale = startScale + (targetScale - startScale) * ease;

      this.render();

      if (progress < 1.0) {
        this._animFrame = requestAnimationFrame(step);
      } else {
        this._animFrame = null;
      }
    };

    this._animFrame = requestAnimationFrame(step);
  }

  jumpToShutter(q, col, row, zoomLevel = 80) {
    const world = this.quadToWorld(q, col, row);
    this.camera.x = world.x;
    this.camera.y = world.y;
    this.camera.scale = zoomLevel;
    this.selectedShutter = { q, x: col, y: row };
    this.updateInspector(q, col, row);
    this.render();
  }

  searchAndJump(query) {
    if (!query) return;
    const clean = query.trim().toLowerCase();

    // Match standard string or global string: e.g. q3d373s19, q2d88s116, q3d510s80, q4-12-234
    let match = clean.match(/q([1-4])\s*d?(\d+)\s*s?(\d+)/i);
    if (match) {
      const q = parseInt(match[1]);
      let col = parseInt(match[2]);
      let row = parseInt(match[3]);

      // If user provided global column (> 365):
      if (col > MSA_CONSTANTS.COLS && col <= MSA_CONSTANTS.COLS * 2) {
        col = col - MSA_CONSTANTS.COLS; // e.g. 373 -> 8, 510 -> 145
      }

      // If user provided global row (> 171):
      if (row > MSA_CONSTANTS.ROWS && row <= MSA_CONSTANTS.ROWS * 2) {
        row = row - MSA_CONSTANTS.ROWS;
      }

      if (col >= 1 && col <= MSA_CONSTANTS.COLS && row >= 1 && row <= MSA_CONSTANTS.ROWS) {
        this.jumpToShutter(q, col, row);
        return;
      }
    }

    // Match "1 38 25" or "Q1 38 25" or "3 373 19"
    match = clean.match(/([1-4])\s+(\d+)\s+(\d+)/);
    if (match) {
      const q = parseInt(match[1]);
      let col = parseInt(match[2]);
      let row = parseInt(match[3]);

      if (col > MSA_CONSTANTS.COLS && col <= MSA_CONSTANTS.COLS * 2) {
        col = col - MSA_CONSTANTS.COLS;
      }
      if (row > MSA_CONSTANTS.ROWS && row <= MSA_CONSTANTS.ROWS * 2) {
        row = row - MSA_CONSTANTS.ROWS;
      }

      if (col >= 1 && col <= MSA_CONSTANTS.COLS && row >= 1 && row <= MSA_CONSTANTS.ROWS) {
        this.jumpToShutter(q, col, row);
        return;
      }
    }

    // Match global 2x2 grid format e.g. "g510s80", "510 80"
    match = clean.match(/g?(\d+)\s*s?(\d+)/i);
    if (match) {
      const gCol = parseInt(match[1]);
      const gRow = parseInt(match[2]);
      if (gCol >= 1 && gCol <= MSA_CONSTANTS.COLS * 2 && gRow >= 1 && gRow <= MSA_CONSTANTS.ROWS * 2) {
        const q = (gCol <= MSA_CONSTANTS.COLS)
          ? (gRow <= MSA_CONSTANTS.ROWS ? 1 : 2)
          : (gRow <= MSA_CONSTANTS.ROWS ? 3 : 4);
        const col = (gCol <= MSA_CONSTANTS.COLS) ? gCol : (gCol - MSA_CONSTANTS.COLS);
        const row = (gRow <= MSA_CONSTANTS.ROWS) ? gRow : (gRow - MSA_CONSTANTS.ROWS);
        this.jumpToShutter(q, col, row);
        return;
      }

    }

    // Search fixed slit name
    const slit = FIXED_SLITS.find(s => s.name.toLowerCase().includes(clean));
    if (slit) {
      this.camera.x = slit.x;
      this.camera.y = slit.y;
      this.camera.scale = 80;
      this.render();
      return;
    }

    alert(`Shutter or slit "${query}" not recognized. Example formats: q3d373s19, q3d145s80, q2d88s116, S200A1`);
  }

  updateHover() {
    // Screen to world (arcsec)
    const { x: wx, y: wy } = this.screenToWorld(this.mousePos.x, this.mousePos.y);

    const hit = this.worldToQuad(wx, wy);
    const prevHover = this.hoveredShutter;
    this.hoveredShutter = hit ? { q: hit.q, x: hit.col, y: hit.row } : null;

    const hoverChanged = (!prevHover && this.hoveredShutter) ||
      (prevHover && !this.hoveredShutter) ||
      (prevHover && this.hoveredShutter && (prevHover.q !== this.hoveredShutter.q || prevHover.x !== this.hoveredShutter.x || prevHover.y !== this.hoveredShutter.y));

    if (hoverChanged && !this.isDragging) {
      this.render();
    }


    // Update HUD Bar
    const hudId = document.getElementById('hud-shutter-id');
    const hudBadge = document.getElementById('hud-status-badge');
    const hudQuad = document.getElementById('hud-quad');
    const hudQuadCol = document.getElementById('hud-quad-col');
    const hudQuadRow = document.getElementById('hud-quad-row');
    const hudGlobalCol = document.getElementById('hud-global-col');
    const hudGlobalRow = document.getElementById('hud-global-row');

    if (this.hoveredShutter) {
      const { q, x, y } = this.hoveredShutter;
      const global = this.quadToGlobal(q, x, y);
      const isUserOpen = this.userOpenShutters.has(`${q}-${x}-${y}`);
      const rawState = this.getShutterState(q, x, y);
      const stateInfo = STATE_NAMES[rawState] || STATE_NAMES[0];
      const isVig = (rawState === 2 || rawState === 3 || rawState === 5);

      hudId.innerText = `q${q}d${x}s${y}`;
      if (hudBadge) {
        hudBadge.style.display = 'inline-block';
        if (isUserOpen) {
          hudBadge.innerText = 'USER_OPEN';
          hudBadge.className = 'hud-badge font-mono badge-user-open';
        } else if (rawState === 4 || rawState === 5) {
          hudBadge.innerText = 'STUCK_OPEN';
          hudBadge.className = 'hud-badge font-mono badge-stuck-open';
        } else if (rawState === 1 || rawState === 3) {
          hudBadge.innerText = 'STUCK_CLOSED';
          hudBadge.className = 'hud-badge font-mono badge-stuck-closed';
        } else if (isVig) {
          hudBadge.innerText = 'VIGNETTED';
          hudBadge.className = 'hud-badge font-mono badge-vignetted';
        } else {
          hudBadge.innerText = 'OPERABLE';
          hudBadge.className = 'hud-badge font-mono badge-operable';
        }
      }
      if (hudQuad) hudQuad.innerText = `Q${q}`;
      if (hudQuadCol) hudQuadCol.innerText = x;
      if (hudQuadRow) hudQuadRow.innerText = y;
      if (hudGlobalCol) hudGlobalCol.innerText = global.globalCol;
      if (hudGlobalRow) hudGlobalRow.innerText = global.globalRow;
    } else {
      hudId.innerText = `(${wx.toFixed(1)}", ${wy.toFixed(1)}")`;
      if (hudBadge) hudBadge.style.display = 'none';
      if (hudQuad) hudQuad.innerText = '-';
      if (hudQuadCol) hudQuadCol.innerText = '-';
      if (hudQuadRow) hudQuadRow.innerText = '-';
      if (hudGlobalCol) hudGlobalCol.innerText = '-';
      if (hudGlobalRow) hudGlobalRow.innerText = '-';
    }


    // Update scale bar
    this.updateScaleBar();
  }

  updateScaleBar() {
    const zoomPct = Math.round(this.camera.scale * 10);
    document.getElementById('hud-zoom-level').innerText = `Zoom: ${zoomPct}%`;

    const visibleFovX = (this.width / this.camera.scale) / 60; // in arcmin
    const visibleFovY = (this.height / this.camera.scale) / 60;
    document.getElementById('hud-arcsec-scale').innerText = `FOV: ${visibleFovX.toFixed(2)}' × ${visibleFovY.toFixed(2)}'`;

    // Scale bar representation
    let barArcsec = 60; // 1 arcmin
    if (this.camera.scale > 50) barArcsec = 1; // 1 arcsec
    else if (this.camera.scale > 15) barArcsec = 10; // 10 arcsec
    else if (this.camera.scale > 5) barArcsec = 30; // 30 arcsec

    const barPx = barArcsec * this.camera.scale;
    const barEl = document.getElementById('scale-bar-line');
    const textEl = document.getElementById('scale-bar-text');
    barEl.style.width = `${Math.round(barPx)}px`;
    textEl.innerText = barArcsec >= 60 ? `${barArcsec / 60} arcmin (${barArcsec}")` : `${barArcsec} arcsec`;
  }

  updateInspector(q, col, row) {
    const badge = document.getElementById('insp-badge');
    const titleInput = document.getElementById('insp-title-input');
    const titleEl = document.getElementById('insp-title');

    if (!q || !col || !row) {
      if (badge) {
        badge.className = 'inspector-badge';
        badge.innerText = 'NO SELECTION';
      }
      if (titleInput) titleInput.value = '';
      if (titleEl) titleEl.innerText = 'No Shutter Selected';
      document.getElementById('insp-sub').innerText = 'Click on any microshutter to inspect';
      document.getElementById('insp-local').innerText = '—';
      document.getElementById('insp-global').innerText = '—';
      document.getElementById('insp-internal').innerText = '—';
      document.getElementById('insp-vig').innerText = '—';
      this.updateListHighlights(null);
      return;
    }

    const shutterId = `q${q}d${col}s${row}`;
    this.updateListHighlights(shutterId);

    const global = this.quadToGlobal(q, col, row);
    const isUserOpen = this.userOpenShutters.has(`${q}-${col}-${row}`);
    const rawState = this.getShutterState(q, col, row);
    const stateInfo = STATE_NAMES[rawState] || STATE_NAMES[0];
    const isVig = (rawState === 2 || rawState === 3 || rawState === 5);

    if (badge) {
      badge.className = 'inspector-badge';
      if (isUserOpen) {
        badge.innerText = 'USER OPEN';
        badge.classList.add('badge-user-open');
      } else {
        badge.innerText = stateInfo.label.toUpperCase();
        badge.classList.add(stateInfo.class);
      }
    }

    if (titleInput) {
      titleInput.value = `q${q}d${col}s${row}`;
    }
    if (titleEl) {
      titleEl.innerText = `q${q}d${col}s${row}`;
    }
    document.getElementById('insp-sub').innerText = `Quadrant ${q} • Column ${col} • Row ${row}`;
    document.getElementById('insp-local').innerText = `Q${q} Col ${col}, Row ${row}`;
    document.getElementById('insp-global').innerText = `Global Col ${global.globalCol}, Row ${global.globalRow}`;
    document.getElementById('insp-internal').innerText = (rawState === 4 || rawState === 5) ? 'failed open' : (rawState === 1 || rawState === 3 ? 'failed closed' : 'normal');
    document.getElementById('insp-vig').innerText = isVig ? 'Yes (Reduced/No Throughput)' : 'No (Full Throughput)';
  }

  populateStuckOpenList() {
    const container = document.getElementById('stuck-open-list');
    if (!container) return;
    container.innerHTML = '';
    container.className = 'stuck-quad-grid';

    const headerEl = document.getElementById('stuck-open-header');
    if (headerEl) headerEl.innerText = `Failed Open Shutters (${this.stuckOpenList.length})`;

    // Group items by quadrant Q1..Q4
    const quads = { 1: [], 2: [], 3: [], 4: [] };
    this.stuckOpenList.forEach(item => {
      if (quads[item.q]) quads[item.q].push(item);
    });

    for (let q = 1; q <= 4; q++) {
      const col = document.createElement('div');
      col.className = 'stuck-quad-col';

      const header = document.createElement('div');
      header.className = 'stuck-quad-header';
      header.innerHTML = `<span class="stuck-quad-title">Q${q}</span> <span class="stuck-quad-count font-mono">${quads[q].length}</span>`;
      col.appendChild(header);

      const list = document.createElement('div');
      list.className = 'stuck-quad-items';

      quads[q].forEach(item => {
        const btn = document.createElement('button');
        btn.className = `stuck-chip-btn ${item.vig ? 'is-vig' : ''}`;
        btn.dataset.shutterId = `q${item.q}d${item.x}s${item.y}`;
        btn.title = `Quadrant ${item.q}, Col ${item.x}, Row ${item.y} (${item.vig ? 'Vignetted' : 'Active'})`;
        btn.innerHTML = `<span class="chip-name font-mono">${item.label}</span>${item.vig ? '<span class="chip-dot"></span>' : ''}`;
        btn.addEventListener('click', () => {
          this.jumpToShutter(item.q, item.x, item.y, 80);
        });
        list.appendChild(btn);
      });

      if (quads[q].length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stuck-quad-empty';
        empty.innerText = 'None';
        list.appendChild(empty);
      }

      col.appendChild(list);
      container.appendChild(col);
    }
  }

  populateVignettedList() {
    const container = document.getElementById('vignetted-list');
    if (!container) return;
    container.innerHTML = '';
    container.className = 'stuck-quad-grid';

    // Collect first N vignetted shutters from each quadrant grid
    const quads = { 1: [], 2: [], 3: [], 4: [] };
    let totalVig = 0;

    for (let q = 1; q <= 4; q++) {
      const qGrid = this.quadGrids[q - 1];
      if (!qGrid) continue;

      let count = 0;
      for (let col = 1; col <= MSA_CONSTANTS.COLS; col++) {
        for (let row = 1; row <= MSA_CONSTANTS.ROWS; row++) {
          const idx = (col - 1) * MSA_CONSTANTS.ROWS + (row - 1);
          const stateCode = qGrid[idx];
          if (stateCode === 2 || stateCode === 3 || stateCode === 5) {
            totalVig++;
            // Include representative samples in each quadrant
            if (quads[q].length < 12) {
              quads[q].push({ q, x: col, y: row, label: `d${col}s${row}`, state: stateCode });
            }
          }
        }
      }
    }

    const headerEl = document.getElementById('vignetted-header');
    if (headerEl) {
      headerEl.innerText = `Vignetted Shutters (${totalVig.toLocaleString()})`;
    }

    for (let q = 1; q <= 4; q++) {
      const col = document.createElement('div');
      col.className = 'stuck-quad-col is-vig-col';

      const header = document.createElement('div');
      header.className = 'stuck-quad-header is-vig-header';
      header.innerHTML = `<span class="stuck-quad-title text-purple">Q${q}</span> <span class="stuck-quad-count font-mono">${quads[q].length}</span>`;
      col.appendChild(header);

      const list = document.createElement('div');
      list.className = 'stuck-quad-items';

      quads[q].forEach(item => {
        const btn = document.createElement('button');
        btn.className = 'stuck-chip-btn is-vig';
        btn.dataset.shutterId = `q${item.q}d${item.x}s${item.y}`;
        btn.title = `Quadrant ${item.q}, Col ${item.x}, Row ${item.y} (Vignetted)`;
        btn.innerHTML = `<span class="chip-name font-mono text-purple">${item.label}</span>`;
        btn.addEventListener('click', () => {
          this.jumpToShutter(item.q, item.x, item.y, 80);
        });
        list.appendChild(btn);
      });

      if (quads[q].length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stuck-quad-empty';
        empty.innerText = 'None';
        list.appendChild(empty);
      }

      col.appendChild(list);
      container.appendChild(col);
    }
  }

  updateListHighlights(selectedId) {
    document.querySelectorAll('.stuck-chip-btn').forEach(btn => {
      if (selectedId && btn.dataset.shutterId === selectedId) {
        btn.classList.add('selected');
      } else {
        btn.classList.remove('selected');
      }
    });
  }

  // Master Canvas Render Loop
  render() {
    const ctx = this.ctx;
    const dpr = this.dpr || 1;
    const w = this.width;
    const h = this.height;

    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Deep space dark background
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, w, h);

    // World-to-screen transform
    ctx.translate(w / 2, h / 2);
    ctx.scale(this.camera.scale, -this.camera.scale); // Invert Y so up is +Y
    ctx.translate(-this.camera.x, -this.camera.y);

    // 1. Render NIRSpec Detectors (NRS1 & NRS2)
    if (this.layers.detectors) {
      this.renderDetectors(ctx);
    }

    // 2. Render Microshutters (249,660 elements optimized)
    this.renderMicroshutters(ctx);

    // 3. Render MSA Structural Frame, Central Bar, & Crisp White Quadrant Borders (On Top)
    this.renderMSAFrame(ctx);

    // 4. Render Fixed Slits & IFU
    if (this.layers.fixedSlits) {
      this.renderFixedSlits(ctx);
    }

    // 5. Render Magnet Arm Sweep
    if (this.layers.magnetArm) {
      this.renderMagnetArm(ctx);
    }

    // 6. Render Coordinate Axes & Labels
    if (this.layers.coordinates) {
      this.renderCoordinates(ctx);
    }

    // 7. Render Selected & Hover Highlight
    this.renderHighlights(ctx);

    // 8. Render Older Map Warning Banner (Centered in X and Y around the central bar / fixed slits)
    const latestVersion = this.availableVersions ? this.availableVersions[this.availableVersions.length - 1] : null;
    if (latestVersion && this.currentVersion !== latestVersion && this.camera.scale < 25) {
      const mapMeta = this.allMapsData?.datasets?.[this.currentVersion];
      const dateStr = mapMeta?.useafter ? mapMeta.useafter.split('T')[0] : '';
      const vNum = parseInt(this.currentVersion, 10);
      const warningText = `⚠️ Viewing an older MSA operability map: v${vNum} (${dateStr})`;

      ctx.save();
      ctx.scale(1, -1); // Invert Y for text rendering
      const warnFontSize = 13.5 / this.camera.scale;
      ctx.font = `700 ${warnFontSize}px JetBrains Mono`;
      
      // Use the exact color from the map selector (red / orange / yellow)
      ctx.fillStyle = this.getMapColor(this.currentVersion);
      ctx.textAlign = 'center'; // Centered horizontally (X = 0)
      ctx.textBaseline = 'middle'; // Centered vertically (Y = 0)
      
      // Position at optical center (0, 0) inside the central horizontal bar
      ctx.fillText(warningText, 0, 0);
      ctx.restore();
    }

    ctx.restore();
  }

  renderDetectors(ctx) {
    const halfGap = MSA_CONSTANTS.DETECTOR_GAP / 2;
    const detSize = MSA_CONSTANTS.DETECTOR_SIZE; // 211.0" square
    const pxScale = MSA_CONSTANTS.DETECTOR_PIXEL_SCALE; // ~0.103027"/pixel
    const scale = this.camera.scale;

    ctx.save();

    // 1. NRS1 (Left Detector: centered on -halfGap - detSize/2)
    // Left edge = -halfGap - detSize, Right edge = -halfGap, Top = +detSize/2, Bottom = -detSize/2
    const nrs1X = -halfGap - detSize;
    const nrs1Y = -detSize / 2;
    ctx.fillStyle = 'rgba(34, 197, 94, 0.14)';
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 1.5 / scale;
    ctx.fillRect(nrs1X, nrs1Y, detSize, detSize);
    ctx.strokeRect(nrs1X, nrs1Y, detSize, detSize);

    // 2. NRS2 (Right Detector: centered on +halfGap + detSize/2)
    // Left edge = +halfGap, Right edge = +halfGap + detSize, Top = +detSize/2, Bottom = -detSize/2
    const nrs2X = halfGap;
    const nrs2Y = -detSize / 2;
    ctx.fillStyle = 'rgba(34, 197, 94, 0.14)';
    ctx.fillRect(nrs2X, nrs2Y, detSize, detSize);
    ctx.strokeRect(nrs2X, nrs2Y, detSize, detSize);

    // 3. Render 2048 x 2048 Pixel Grid (Strict True Pixel Dimensions: 0.103027" / pixel)
    if (this.layers.detectorPixels) {
      // Only render when zoom level is high enough that individual 0.103" pixels can be drawn (e.g. scale >= 14)
      if (scale >= 14) {
        const halfViewW = (this.width / 2) / scale;
        const halfViewH = (this.height / 2) / scale;
        const viewMinX = this.camera.x - halfViewW;
        const viewMaxX = this.camera.x + halfViewW;
        const viewMinY = this.camera.y - halfViewH;
        const viewMaxY = this.camera.y + halfViewH;

        ctx.save();
        ctx.strokeStyle = 'rgba(74, 222, 128, 0.25)';
        ctx.lineWidth = 0.5 / scale;

        const detectors = [
          { minX: nrs1X, maxX: nrs1X + detSize, minY: nrs1Y, maxY: nrs1Y + detSize },
          { minX: nrs2X, maxX: nrs2X + detSize, minY: nrs2Y, maxY: nrs2Y + detSize }
        ];

        const stepArcsec = pxScale; // Strictly 1 pixel = 0.103027"

        ctx.beginPath();
        for (const det of detectors) {
          const x0 = Math.max(det.minX, viewMinX);
          const x1 = Math.min(det.maxX, viewMaxX);
          const y0 = Math.max(det.minY, viewMinY);
          const y1 = Math.min(det.maxY, viewMaxY);

          if (x0 < x1 && y0 < y1) {
            // Vertical pixel grid lines
            const startPxCol = Math.floor((x0 - det.minX) / stepArcsec);
            const endPxCol = Math.ceil((x1 - det.minX) / stepArcsec);

            for (let i = startPxCol; i <= endPxCol; i++) {
              const lx = det.minX + i * stepArcsec;
              if (lx >= x0 && lx <= x1) {
                ctx.moveTo(lx, y0);
                ctx.lineTo(lx, y1);
              }
            }

            // Horizontal pixel grid lines
            const startPxRow = Math.floor((y0 - det.minY) / stepArcsec);
            const endPxRow = Math.ceil((y1 - det.minY) / stepArcsec);

            for (let j = startPxRow; j <= endPxRow; j++) {
              const ly = det.minY + j * stepArcsec;
              if (ly >= y0 && ly <= y1) {
                ctx.moveTo(x0, ly);
                ctx.lineTo(x1, ly);
              }
            }
          }
        }
        ctx.stroke();
        ctx.restore();
      }
    }

    // 4. Detector Labels in Top-Left (NRS1) and Top-Right (NRS2) Corners
    if (scale <= 20) {
      ctx.save();
      ctx.scale(1, -1);

      const fontSize = 14 / scale;
      ctx.font = `700 ${fontSize}px JetBrains Mono`;
      ctx.fillStyle = 'rgba(74, 222, 128, 0.9)';

      // NRS1 (Top-Left corner: x = nrs1X + margin, y = top of detector)
      const topDetY = -(nrs1Y + detSize) + (18 / scale);
      ctx.textAlign = 'left';
      ctx.fillText('NRS1', nrs1X + (12 / scale), topDetY);

      // NRS2 (Top-Right corner: x = nrs2X + detSize - margin, y = top of detector)
      ctx.textAlign = 'right';
      ctx.fillText('NRS2', nrs2X + detSize - (12 / scale), topDetY);

      ctx.restore();
    }

    ctx.restore();
  }

  renderMSAFrame(ctx) {
    const halfGapX = MSA_CONSTANTS.GAP_X / 2;
    const halfGapY = MSA_CONSTANTS.GAP_Y / 2;
    const qW = MSA_CONSTANTS.COLS * MSA_CONSTANTS.PITCH_X;
    const qH = MSA_CONSTANTS.ROWS * MSA_CONSTANTS.PITCH_Y;
    const scale = this.camera.scale;

    ctx.save();

    // 1. Central Mounting Bar Outline & Mounting Plate (faint pale yellow)
    ctx.strokeStyle = 'rgba(254, 240, 138, 0.12)';
    ctx.lineWidth = 0.75 / scale;
    ctx.strokeRect(-halfGapX - qW - 3.0, -halfGapY, 2 * (halfGapX + qW + 3.0), 2 * halfGapY);

    // 2. Overall MSA Mounting Frame perimeter outline (faint pale yellow)
    const frameMargin = 4.0; // arcsec
    const totalFrameW = 2 * (halfGapX + qW) + 2 * frameMargin;
    const totalFrameH = 2 * (halfGapY + qH) + 2 * frameMargin;
    ctx.strokeRect(-totalFrameW / 2, -totalFrameH / 2, totalFrameW, totalFrameH);

    // 3. Crisp White Border around the 4 Active MSA Quadrants
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)'; // Clean crisp white
    ctx.lineWidth = 1.2 / scale;

    const quads = [
      { q: 1, x: halfGapX, y: halfGapY, w: qW, h: qH },
      { q: 2, x: halfGapX, y: -halfGapY - qH, w: qW, h: qH },
      { q: 3, x: -halfGapX - qW, y: halfGapY, w: qW, h: qH },
      { q: 4, x: -halfGapX - qW, y: -halfGapY - qH, w: qW, h: qH }
    ];

    for (const q of quads) {
      ctx.strokeRect(q.x, q.y, q.w, q.h);
    }

    ctx.restore();
  }

  renderMicroshutters(ctx) {
    const scale = this.camera.scale;
    const w = this.width;
    const h = this.height;

    // Viewport bounds in world coordinates
    const minWx = this.camera.x - (w / 2) / scale;
    const maxWx = this.camera.x + (w / 2) / scale;
    const minWy = this.camera.y - (h / 2) / scale;
    const maxWy = this.camera.y + (h / 2) / scale;

    const shutterW = MSA_CONSTANTS.SHUTTER_WIDTH;
    const shutterH = MSA_CONSTANTS.SHUTTER_HEIGHT;
    const pitchX = MSA_CONSTANTS.PITCH_X;
    const pitchY = MSA_CONSTANTS.PITCH_Y;

    // Detailed rendering
    const op = this.shutterOpacity;

    // Render quadrant by quadrant
    for (let q = 1; q <= 4; q++) {
      const qGrid = this.quadGrids[q - 1];

      // Calculate quadrant bounding box
      const qOrigin = this.quadToWorld(q, 1, 1);
      const qEnd = this.quadToWorld(q, MSA_CONSTANTS.COLS, MSA_CONSTANTS.ROWS);
      const rx = Math.min(qOrigin.x, qEnd.x) - pitchX / 2;
      const ry = Math.min(qOrigin.y, qEnd.y) - pitchY / 2;
      const rw = MSA_CONSTANTS.COLS * pitchX;
      const rh = MSA_CONSTANTS.ROWS * pitchY;

      // Check if quadrant intersects viewport
      if (rx > maxWx || rx + rw < minWx || ry > maxWy || ry + rh < minWy) continue;

      // 1. When zoomed in (scale > 18), draw metal grid bars ultra-fast using batched path strokes
      if (this.layers.grid && scale > 18) {
        ctx.save();
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = MSA_CONSTANTS.BAR_WALL; // 0.07" bar wall thickness

        const qVisX0 = Math.max(rx, minWx);
        const qVisX1 = Math.min(rx + rw, maxWx);
        const qVisY0 = Math.max(ry, minWy);
        const qVisY1 = Math.min(ry + rh, maxWy);

        ctx.beginPath();
        // Vertical grid bars
        for (let col = 0; col <= MSA_CONSTANTS.COLS; col++) {
          let lineX;
          if (q === 1 || q === 2) {
            lineX = (MSA_CONSTANTS.GAP_X / 2) + col * pitchX;
          } else {
            lineX = (-MSA_CONSTANTS.GAP_X / 2) - col * pitchX;
          }
          if (lineX >= qVisX0 - pitchX && lineX <= qVisX1 + pitchX) {
            ctx.moveTo(lineX, qVisY0);
            ctx.lineTo(lineX, qVisY1);
          }
        }

        // Horizontal grid bars
        for (let row = 0; row <= MSA_CONSTANTS.ROWS; row++) {
          let lineY;
          if (q === 1 || q === 3) {
            lineY = (MSA_CONSTANTS.GAP_Y / 2) + row * pitchY;
          } else {
            lineY = (-MSA_CONSTANTS.GAP_Y / 2) - row * pitchY;
          }
          if (lineY >= qVisY0 - pitchY && lineY <= qVisY1 + pitchY) {
            ctx.moveTo(qVisX0, lineY);
            ctx.lineTo(qVisX1, lineY);
          }
        }
        ctx.stroke();
        ctx.restore();
      }

      // 2. Render Microshutter Apertures (Fast direct viewport bounds culling)
      for (let col = 1; col <= MSA_CONSTANTS.COLS; col++) {
        const samplePos = this.quadToWorld(q, col, 1);
        if (samplePos.x + pitchX < minWx || samplePos.x - pitchX > maxWx) continue;

        for (let row = 1; row <= MSA_CONSTANTS.ROWS; row++) {
          const pos = this.quadToWorld(q, col, row);
          if (pos.y + pitchY < minWy || pos.y - pitchY > maxWy) continue;

          const idx = (col - 1) * MSA_CONSTANTS.ROWS + (row - 1);
          const stateCode = qGrid[idx] || 0;
          const isUserOpen = this.userOpenShutters.has(`${q}-${col}-${row}`);
          const isStuckOpen = (stateCode === 4 || stateCode === 5);
          const isStuckClosed = (stateCode === 1 || stateCode === 3);
          const isVig = (stateCode === 2 || stateCode === 3 || stateCode === 5);

          const sx = pos.x - shutterW / 2;
          const sy = pos.y - shutterH / 2;

          if (isUserOpen) {
            if (op > 0) {
              ctx.fillStyle = `rgba(56, 189, 248, ${op})`;
              ctx.fillRect(sx, sy, shutterW, shutterH);
            }
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 1.0 / scale;
            ctx.strokeRect(sx, sy, shutterW, shutterH);
          } else if (isStuckOpen) {
            if (op > 0) {
              ctx.fillStyle = `rgba(239, 68, 68, ${op})`;
              ctx.fillRect(sx, sy, shutterW, shutterH);
            }
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 1.0 / scale;
            ctx.strokeRect(sx, sy, shutterW, shutterH);
          } else if (isStuckClosed && this.layers.stuckClosed) {
            // Stuck closed: if vignetted (stateCode 3), fill lighter muted purple-gray (#4e3d60), closer to standard slate (#555e69)
            if (isVig) {
              ctx.fillStyle = '#4e3d60'; // Lighter muted purple-gray (clearly visible door, close to #555e69 slate)
            } else {
              ctx.fillStyle = '#555e69'; // Standard slate gray for unvignetted stuck closed
            }
            ctx.fillRect(sx, sy, shutterW, shutterH);
          } else {
            // Operable shutter aperture (Open)
            if (isVig) {
              // Solid dark purple fill for vignetted open apertures (darker than failed closed, completely filled / non-transparent)
              ctx.fillStyle = '#2c2238'; // Dark opaque purple
              ctx.fillRect(sx, sy, shutterW, shutterH);
              ctx.strokeStyle = '#3d304c';
              ctx.lineWidth = 0.75 / scale;
              ctx.strokeRect(sx, sy, shutterW, shutterH);
            } else if (this.layers.fillOpen && op > 0) {
              ctx.fillStyle = `rgba(212, 184, 134, ${op})`; // Normal operable amber/gold aperture
              ctx.fillRect(sx, sy, shutterW, shutterH);
            }

          }
        }
      }
    }

    // Stuck Open glowing badges (always visible even at high zoom-out)
    if (this.layers.stuckOpen) {
      for (const item of this.stuckOpenList) {
        const pos = this.quadToWorld(item.q, item.x, item.y);
        ctx.save();
        ctx.fillStyle = '#ef4444';

        if (scale < 20) {
          // Draw clean 2x1 HxW red rectangle badge centered on shutter without white outline
          const w = Math.max(2.6 / scale, 0.35);
          const h = w * 2.0; // 2x1 HxW ratio
          const r = Math.min(w, h) * 0.15; // subtle rounded corner

          const rx = pos.x - w / 2;
          const ry = pos.y - h / 2;

          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(rx, ry, w, h, r);
          } else {
            ctx.rect(rx, ry, w, h);
          }
          ctx.fill();
        }
        ctx.restore();
      }
    }
  }

  renderFixedSlits(ctx) {
    ctx.save();
    for (const slit of FIXED_SLITS) {
      // Semi-transparent aperture opening allowing underlying detector pixel grid to show through
      ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5 / this.camera.scale;

      ctx.fillRect(slit.x - slit.w / 2, slit.y - slit.h / 2, slit.w, slit.h);
      ctx.strokeRect(slit.x - slit.w / 2, slit.y - slit.h / 2, slit.w, slit.h);

      // Only show text when zoomed in reasonably, and keep font size fixed in screen pixels
      if (this.camera.scale >= 8) {
        ctx.save();
        ctx.scale(1, -1);
        
        // Invert Y world-to-canvas coordinate:
        // World bottom of slit is (slit.y - slit.h / 2)
        // Canvas inverted Y for the bottom edge is -(slit.y - slit.h / 2)
        // Adding positive offset in canvas space moves the text downwards (clearly BELOW the slit bottom edge)
        const textBelowY = -(slit.y - slit.h / 2) + (14 / this.camera.scale);

        // Slit Name (Top-justified / baseline top, placed clearly below the slit)
        const nameFontSize = 11 / this.camera.scale;
        ctx.font = `600 ${nameFontSize}px JetBrains Mono`;
        ctx.fillStyle = '#fbbf24';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(slit.name, slit.x, textBelowY);

        // Slit Dimensions Subtitle (e.g. 0.2"×3.2" or 3.0"×3.0")
        const dimStr = `${slit.w.toFixed(2).replace(/\.?0+$/, '')}"×${slit.h.toFixed(2).replace(/\.?0+$/, '')}"`;
        const dimFontSize = 9.5 / this.camera.scale;
        ctx.font = `400 ${dimFontSize}px JetBrains Mono`;
        ctx.fillStyle = 'rgba(251, 191, 36, 0.8)';
        ctx.fillText(dimStr, slit.x, textBelowY + (13 / this.camera.scale));

        ctx.restore();
      }
    }
    ctx.restore();
  }

  renderMagnetArm(ctx) {
    ctx.save();
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.75)';
    ctx.lineWidth = 2.0 / this.camera.scale;
    ctx.setLineDash([4 / this.camera.scale, 4 / this.camera.scale]);

    // Primary Park (Left) <---> Secondary Park (Right)
    const yArm = 0;
    ctx.beginPath();
    ctx.moveTo(-130, yArm);
    ctx.lineTo(130, yArm);
    ctx.stroke();

    // Magnet bar representation
    ctx.fillStyle = 'rgba(22, 101, 52, 0.4)';
    ctx.fillRect(-6, -110, 12, 220);
    ctx.strokeStyle = '#22c55e';
    ctx.setLineDash([]);
    ctx.strokeRect(-6, -110, 12, 220);

    if (this.camera.scale < 25) {
      ctx.save();
      ctx.scale(1, -1);
      const fontSize = 11 / this.camera.scale;
      ctx.font = `${fontSize}px Inter`;
      ctx.fillStyle = '#4ade80';
      ctx.textAlign = 'center';
      ctx.fillText('Magnet Arm Sweep', 0, -100);
      ctx.fillText('◀ Primary Park', -115, 4);
      ctx.fillText('Secondary Park ▶', 115, 4);
      ctx.restore();
    }

    ctx.restore();
  }

  renderCoordinates(ctx) {
    const scale = this.camera.scale;
    if (scale > 25) return; // Only show quadrant labels at overview / medium zoom levels

    const halfGapX = MSA_CONSTANTS.GAP_X / 2;
    const halfGapY = MSA_CONSTANTS.GAP_Y / 2;
    const qW = MSA_CONSTANTS.COLS * MSA_CONSTANTS.PITCH_X;
    const qH = MSA_CONSTANTS.ROWS * MSA_CONSTANTS.PITCH_Y;

    ctx.save();
    ctx.scale(1, -1);

    const fontSize = 13 / scale;
    ctx.font = `700 ${fontSize}px JetBrains Mono`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'; // Clean crisp white matching quadrant borders

    // Q1 (Top-Right quadrant: label placed above top edge)
    const q1X = halfGapX + qW / 2;
    const q1Y = -(halfGapY + qH) - (6 / scale);
    ctx.textAlign = 'center';
    ctx.fillText('Q1', q1X, q1Y);

    // Q3 (Top-Left quadrant: label placed above top edge)
    const q3X = -halfGapX - qW / 2;
    const q3Y = -(halfGapY + qH) - (6 / scale);
    ctx.fillText('Q3', q3X, q3Y);

    // Q2 (Bottom-Right quadrant: label placed below bottom edge)
    const q2X = halfGapX + qW / 2;
    const q2Y = (halfGapY + qH) + (16 / scale);
    ctx.fillText('Q2', q2X, q2Y);

    // Q4 (Bottom-Left quadrant: label placed below bottom edge)
    const q4X = -halfGapX - qW / 2;
    const q4Y = (halfGapY + qH) + (16 / scale);
    ctx.fillText('Q4', q4X, q4Y);

    // --- Row and Column Arrows along Top-Right Corner (Q1) ---
    // Top-Right outer corner coordinates:
    const trX = halfGapX + qW; // outer right edge of Q1
    const trY = -(halfGapY + qH); // outer top edge of Q1

    const arrowFontSize = 10.5 / scale;
    ctx.font = `600 ${arrowFontSize}px JetBrains Mono`;
    ctx.fillStyle = 'rgba(56, 189, 248, 0.9)'; // Cyan
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.9)';
    ctx.lineWidth = 1.6 / scale;

    // 1. Column Axis Arrow (Pointing Leftwards along top of Q1: Col 1 -> Col 730)
    const colArrowY = trY - (6 / scale);
    const colStartX = trX;
    const colEndX = trX - (38 / scale);
    
    ctx.beginPath();
    ctx.moveTo(colStartX, colArrowY);
    ctx.lineTo(colEndX, colArrowY);
    // Arrow head pointing left
    ctx.lineTo(colEndX + (4 / scale), colArrowY - (3 / scale));
    ctx.moveTo(colEndX, colArrowY);
    ctx.lineTo(colEndX + (4 / scale), colArrowY + (3 / scale));
    ctx.stroke();

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('Col 1 ➔', colStartX + (42 / scale), colArrowY);

    // 2. Row Axis Arrow (Pointing Downwards along right of Q1: Row 1 -> Row 342)
    const rowArrowX = trX + (8 / scale);
    const rowStartY = trY;
    const rowEndY = trY + (38 / scale);

    ctx.beginPath();
    ctx.moveTo(rowArrowX, rowStartY);
    ctx.lineTo(rowArrowX, rowEndY);
    // Arrow head pointing downwards (in inverted canvas Y, down is +Y)
    ctx.lineTo(rowArrowX - (3 / scale), rowEndY - (4 / scale));
    ctx.moveTo(rowArrowX, rowEndY);
    ctx.lineTo(rowArrowX + (3 / scale), rowEndY - (4 / scale));
    ctx.stroke();

    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Row 1 ➔', rowArrowX + (4 / scale), rowStartY - (2 / scale));

    ctx.restore();
  }


  renderHighlights(ctx) {
    const shutterW = MSA_CONSTANTS.SHUTTER_WIDTH;  // 0.20" opening
    const shutterH = MSA_CONSTANTS.SHUTTER_HEIGHT; // 0.46" opening
    const pitchW = MSA_CONSTANTS.PITCH_X;          // 0.27" full pitch
    const pitchH = MSA_CONSTANTS.PITCH_Y;          // 0.53" full pitch

    // Hover Highlight (Only if not already selected)
    if (this.hoveredShutter) {
      const isSelected = this.selectedShutter &&
        this.selectedShutter.q === this.hoveredShutter.q &&
        this.selectedShutter.x === this.hoveredShutter.x &&
        this.selectedShutter.y === this.hoveredShutter.y;

      if (!isSelected) {
        const pos = this.quadToWorld(this.hoveredShutter.q, this.hoveredShutter.x, this.hoveredShutter.y);
        ctx.strokeStyle = '#38bdf8'; // Cyan hover outline
        ctx.lineWidth = 1.2 / this.camera.scale;
        ctx.strokeRect(pos.x - pitchW / 2, pos.y - pitchH / 2, pitchW, pitchH);
      }
    }

    // Selected Highlight
    if (this.selectedShutter) {
      const q = this.selectedShutter.q;
      const x = this.selectedShutter.x;
      const y = this.selectedShutter.y;
      const pos = this.quadToWorld(q, x, y);

      // 1. Outer Cyan Rectangle covering the full slit pitch (0.27" x 0.53") down the middles of the bars
      ctx.strokeStyle = '#00f2fe'; // Bright Electric Cyan
      ctx.lineWidth = 1.8 / this.camera.scale;
      ctx.strokeRect(pos.x - pitchW / 2, pos.y - pitchH / 2, pitchW, pitchH);

      // 2. Determine Inner Outline Color indicating the shutter's status
      const isUserOpen = this.userOpenShutters.has(`${q}-${x}-${y}`);
      const rawState = this.getShutterState(q, x, y);
      let statusColor = '#d4b886'; // Normal operable open/gold

      if (isUserOpen) {
        statusColor = '#38bdf8'; // Electric cyan / user open
      } else if (rawState === 4 || rawState === 5) {
        statusColor = '#ef4444'; // Vibrant red for failed open
      } else if (rawState === 2 || rawState === 3) {
        statusColor = '#a855f7'; // Purple for vignetted
      } else if (rawState === 1) {
        statusColor = '#94a3b8'; // Slate/gray for failed closed
      } else {
        statusColor = '#f59e0b'; // Amber/gold for operable
      }

      // 3. Inner Status Outline hugging the opening of the slit (0.20" x 0.46")
      ctx.strokeStyle = statusColor;
      ctx.lineWidth = 1.5 / this.camera.scale;
      ctx.strokeRect(pos.x - shutterW / 2, pos.y - shutterH / 2, shutterW, shutterH);
    }
  }

  // Full Markdown & GFM Table parser for guide modal
  renderMarkdown(md) {
    const lines = md.split('\n');
    let html = '';
    let inTable = false;
    let tableHeaderDone = false;
    let inCode = false;
    let codeContent = '';
    let inList = false;

    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];

      // Code blocks ```
      if (line.trim().startsWith('```')) {
        if (inCode) {
          html += `<pre><code>${codeContent}</code></pre>`;
          inCode = false;
          codeContent = '';
        } else {
          if (inList) { html += '</ul>'; inList = false; }
          if (inTable) { html += '</tbody></table>'; inTable = false; tableHeaderDone = false; }
          inCode = true;
          codeContent = '';
        }
        continue;
      }
      if (inCode) {
        codeContent += line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '\n';
        continue;
      }

      // Markdown Tables
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        const cells = line.trim().split('|').slice(1, -1).map(c => c.trim());
        const isSeparator = cells.every(c => /^:?-+:?$/.test(c));

        if (isSeparator) {
          tableHeaderDone = true;
          continue;
        }

        if (!inTable) {
          if (inList) { html += '</ul>'; inList = false; }
          inTable = true;
          tableHeaderDone = false;
          html += '<table class="guide-table"><thead><tr>';
          cells.forEach(c => {
            html += `<th>${this.formatInlineMd(c)}</th>`;
          });
          html += '</tr></thead><tbody>';
          continue;
        } else {
          html += '<tr>';
          cells.forEach(c => {
            html += `<td>${this.formatInlineMd(c)}</td>`;
          });
          html += '</tr>';
          continue;
        }
      } else if (inTable) {
        html += '</tbody></table>';
        inTable = false;
        tableHeaderDone = false;
      }

      const trimmed = line.trim();

      // Horizontal rules
      if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
        if (inList) { html += '</ul>'; inList = false; }
        html += '<hr class="guide-divider" />';
        continue;
      }

      // Headings
      if (trimmed.startsWith('# ')) {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<h1 class="guide-h1">${this.formatInlineMd(trimmed.substring(2))}</h1>`;
        continue;
      }
      if (trimmed.startsWith('## ')) {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<h2 class="guide-h2">${this.formatInlineMd(trimmed.substring(3))}</h2>`;
        continue;
      }
      if (trimmed.startsWith('### ')) {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<h3 class="guide-h3">${this.formatInlineMd(trimmed.substring(4))}</h3>`;
        continue;
      }

      // Lists
      if (/^[-*]\s+/.test(trimmed)) {
        if (!inList) { html += '<ul class="guide-list">'; inList = true; }
        html += `<li>${this.formatInlineMd(trimmed.replace(/^[-*]\s+/, ''))}</li>`;
        continue;
      } else if (inList && trimmed === '') {
        html += '</ul>';
        inList = false;
        continue;
      }

      // Paragraphs & Callout alerts
      if (trimmed.length > 0) {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<p class="guide-p">${this.formatInlineMd(trimmed)}</p>`;
      }
    }

    if (inTable) html += '</tbody></table>';
    if (inList) html += '</ul>';
    if (inCode) html += `<pre><code>${codeContent}</code></pre>`;

    return html;
  }

  // Format inline bold, italics, code, LaTeX math, and links
  formatInlineMd(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\\sim\s*/g, '~')
      .replace(/\\times\s*/g, '×')
      .replace(/\\mu\\text\{m\}/g, 'µm')
      .replace(/\\dots/g, '…')
      .replace(/\\text\{([^\}]+)\}/g, '$1')
      .replace(/\$([^\$]+)\$/g, '<span class="font-mono text-cyan">$1</span>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  }
}


// Initialize Application when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  window.msaApp = new MSAVisualizer();
});
