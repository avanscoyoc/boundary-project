/*
 * Script 05b: Earth Engine app for the visual inspection of the analysis-set PAs
 *
 * Shows one PA at a time on satellite imagery: its outline and its transect
 * points (interior, boundary, exterior), with an info panel. PAs come in
 * review order (biome, country, name); the PA number is the `order` column of
 * data/inspection/checklist.csv, where decisions are recorded
 * (blank = keep, exclude, check). Script 05 removes the PAs marked `exclude`.
 *
 * BEFORE RUNNING:
 * 1. Run scripts/05b_inspection_export.py and upload data/inspection/pas_final.zip
 *    as the table asset PAS_ASSET below. The transect points are the existing
 *    assets transect_points_NNN (script 02).
 * 2. Paste this file into the Code Editor (code.earthengine.google.com) and Run.
 *    To publish it: Apps > New App. Viewers of a published app need read access
 *    to the assets (share them, e.g. "Anyone can read").
 *
 * The page address keeps the current PA (…#n=123&pid=100204), so a reload or a
 * bookmark comes back to it; …#pid=100204 alone opens a PA by its WDPA_PID.
 */

// --- Settings ---------------------------------------------------------------
var ASSET_ROOT = 'projects/YOUR-PROJECT-ID/assets/';   // your Earth Engine project's assets
var PAS_ASSET = ASSET_ROOT + 'pas_final';                    // script 05b
var POINTS_PREFIX = ASSET_ROOT + 'transect_points_';         // script 02 (+ NNN)
var SEA_CUTOFF = 75;    // script 05: removed if median % complete transects < 75
var LAKE_CUTOFF = 75;   // script 05: removed if % transects with water > 75
var COLORS = {
  outline: 'FFFFFF',    // PA boundary (script 02 polygon)
  interior: '00FFFF',   // points -5 and -2.5 km (inside the PA)
  boundary: 'FFFF00',   // point 0 km (on the boundary)
  exterior: 'FF00FF'    // points +2.5 and +5 km (outside the PA)
};

var PAS = ee.FeatureCollection(PAS_ASSET);

// --- State --------------------------------------------------------------------
var N = null;          // number of PAs (read from the asset)
var current = 1;       // PA number on screen
var request = 0;       // answers to older requests are ignored
var lastBounds = null; // for "Re-centre"

// --- Helpers ------------------------------------------------------------------
function pointAsset(n) {
  return POINTS_PREFIX + ('00' + n).slice(-3);
}

// The page address (#n=…&pid=…); never lets a failure stop the app.
function setUrl(key, value) {
  try { ui.url.set(key, value); } catch (e) { /* address not available */ }
}
function getUrl(key, fallback) {
  try { return ui.url.get(key, fallback); } catch (e) { return fallback; }
}

function fmt(x, digits) {
  if (x === null || x === undefined || isNaN(x)) return '–';
  return Number(x).toLocaleString('en-US', {
    minimumFractionDigits: digits, maximumFractionDigits: digits
  });
}

function pointLayer(points, filter, color, name) {
  var styled = points.filter(filter).style({
    color: color, fillColor: color, pointSize: 3, width: 1
  });
  return ui.Map.Layer(styled, {}, name);
}

// --- Panel --------------------------------------------------------------------
var panel = ui.Panel({style: {width: '330px', padding: '8px'}});

var numBox = ui.Textbox({
  placeholder: 'PA number',
  style: {width: '110px'},
  onChange: function (value) {
    var n = parseInt(value, 10);
    if (!isNaN(n)) show(n);
  }
});
var pidBox = ui.Textbox({
  placeholder: 'WDPA_PID',
  style: {width: '110px'},
  onChange: function (value) { showPid(value); }
});

var status = ui.Label('Loading…', {color: '666666'});
var heading = ui.Label('', {fontWeight: 'bold', fontSize: '15px', margin: '8px 8px 2px 8px'});
var info = ui.Panel();

function row(label, value) {
  return ui.Panel([
    ui.Label(label, {width: '120px', margin: '2px 8px', color: '555555'}),
    ui.Label(value, {margin: '2px 8px'})
  ], ui.Panel.Layout.flow('horizontal'));
}

function swatch(color, text) {
  return ui.Panel([
    ui.Label('', {backgroundColor: '#' + color, padding: '7px', margin: '4px 8px',
                  border: '1px solid #888888'}),
    ui.Label(text, {margin: '4px 0px'})
  ], ui.Panel.Layout.flow('horizontal'));
}

panel.add(ui.Label('PA inspection', {fontWeight: 'bold', fontSize: '18px'}));
panel.add(ui.Panel([
  ui.Button('◀ Previous', function () { show(current - 1); }),
  ui.Button('Next ▶', function () { show(current + 1); })
], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Label('Go to #', {width: '70px', margin: '12px 8px'}), numBox],
                   ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Label('WDPA_PID', {width: '70px', margin: '12px 8px'}), pidBox],
                   ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Button('Re-centre', function () {
  if (lastBounds) Map.centerObject(ee.Geometry(lastBounds));
}));
panel.add(status);
panel.add(heading);
panel.add(info);
panel.add(ui.Label('Legend', {fontWeight: 'bold', margin: '12px 8px 2px 8px'}));
panel.add(swatch(COLORS.outline, 'PA boundary'));
panel.add(swatch(COLORS.interior, 'Interior points (−5, −2.5 km)'));
panel.add(swatch(COLORS.boundary, 'Boundary points (0 km)'));
panel.add(swatch(COLORS.exterior, 'Exterior points (+2.5, +5 km)'));
panel.add(ui.Label('Record decisions in data/inspection/checklist.csv (row = PA number).',
                   {fontSize: '11px', color: '666666', margin: '12px 8px'}));

ui.root.insert(0, panel);
Map.setOptions('SATELLITE');

// --- Show a PA ----------------------------------------------------------------
function show(n) {
  if (N !== null) n = Math.max(1, Math.min(N, n));
  current = n;
  numBox.setValue(String(n), false);  // false: don't fire onChange
  var token = ++request;
  status.setValue('Loading PA ' + n + '…');
  var pa = PAS.filter(ee.Filter.eq('ORDER', n));
  var f = ee.Feature(pa.first());
  ee.Dictionary({props: f.toDictionary(), bounds: f.geometry().bounds(100)})
    .evaluate(function (r, err) {
      if (token !== request) return;  // the user has moved on
      if (err || !r) {
        status.setValue('Error for PA ' + n + ': ' + (err || 'not found'));
        return;
      }
      draw(pa, r.props, r.bounds, n);
    });
}

function draw(pa, p, bounds, n) {
  var points = ee.FeatureCollection(pointAsset(p.ASSET))
    .filter(ee.Filter.eq('WDPA_PID', p.WDPA_PID));
  var outline = ee.Image().byte().paint(pa, 1, 2).visualize({palette: [COLORS.outline]});
  Map.layers().reset([
    ui.Map.Layer(outline, {}, 'PA boundary'),
    pointLayer(points, ee.Filter.lt('pointID', 0), COLORS.interior, 'Interior points'),
    pointLayer(points, ee.Filter.gt('pointID', 0), COLORS.exterior, 'Exterior points'),
    pointLayer(points, ee.Filter.eq('pointID', 0), COLORS.boundary, 'Boundary points')
  ]);
  lastBounds = bounds;
  Map.centerObject(ee.Geometry(bounds));
  setUrl('n', n);
  setUrl('pid', String(p.WDPA_PID));  // both name the same PA; pid wins on reload

  status.setValue('PA ' + n + ' of ' + (N === null ? '…' : fmt(N, 0)));
  heading.setValue(p.NAME);
  pidBox.setValue(String(p.WDPA_PID), false);
  info.clear();
  info.add(row('WDPA_PID', String(p.WDPA_PID)));
  info.add(row('Country', p.ISO3));
  info.add(row('Biome', p.BIOME));
  info.add(row('Area', fmt(p.AREA_KM2, 0) + ' km²'));
  info.add(row('Transects', fmt(p.N_TRNST, 0) + ' (' + fmt(p.PCT_KEPT, 1) + '% of generated kept)'));
  info.add(row('Missing data', fmt(p.MISS_PCT, 2) + '% of transect-years'));
  info.add(row('Sea rule', fmt(p.COMPL_PCT, 1) + '% complete transects (removed if < ' + SEA_CUTOFF + '%)'));
  info.add(row('Lake rule', fmt(p.WATER_PCT, 1) + '% transects with water (removed if > ' + LAKE_CUTOFF + '%)'));
}

function showPid(value) {
  var pid = String(value || '').trim();
  if (!pid) return;
  var token = ++request;
  status.setValue('Looking up WDPA_PID ' + pid + '…');
  PAS.filter(ee.Filter.eq('WDPA_PID', pid)).aggregate_array('ORDER').evaluate(function (a, err) {
    if (token !== request) return;
    if (err) {
      status.setValue('Error: ' + err);
    } else if (!a || a.length === 0) {
      status.setValue('WDPA_PID ' + pid + ' is not in the inspection set');
    } else {
      show(a[0]);
    }
  });
}

// --- Start --------------------------------------------------------------------
PAS.size().evaluate(function (n, err) {
  if (err) {
    status.setValue('Cannot read ' + PAS_ASSET + ': ' + err);
    return;
  }
  N = n;
  var pid = getUrl('pid', '');
  if (pid) {
    showPid(pid);
  } else {
    var start = parseInt(getUrl('n', 1), 10);
    show(isNaN(start) ? 1 : start);
  }
});
