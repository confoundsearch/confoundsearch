// Choosing an edge, a confound, or nothing. The diagram, the cards and the evidence panel always agree.
var mdColors = {};
document.querySelectorAll('.md-card').forEach(function(card) {
    mdColors[card.getAttribute('data-md')] = card.getAttribute('data-color');
});
var VARIABLE_OF = {to_iv: 'x', to_dv: 'y'};

function showEvidence(key) {
    var box = document.getElementById('evidence-display');
    box.innerHTML = EVIDENCE_BLOB[key] || EVIDENCE_BLOB['__deselect__'];
    var shown = box.querySelector('[data-color]');
    box.style.borderLeftColor = shown ? shown.getAttribute('data-color') : 'transparent';
    box.scrollTop = 0;
}

function forEachNode(selector, fn) {
    document.querySelectorAll('.dag-svg ' + selector).forEach(fn);
}

function highlight(md, leg) {
    var svg = document.querySelector('.dag-svg');
    forEachNode('.sel', function(el) { el.classList.remove('sel'); });
    forEachNode('line[data-md]', function(el) {
        el.setAttribute('stroke', '#888');
        el.setAttribute('marker-end', 'url(#arrowhead-arc)');
    });
    forEachNode('.dag-var', function(el) { el.setAttribute('stroke', '#333'); });
    forEachNode('.dag-ring', function(el) { el.setAttribute('stroke', '#333'); });
    svg.classList.toggle('has-sel', !!md);
    if (md) {
        var color = mdColors[md];
        var mine = function(el) { return el.getAttribute('data-md') === md; };
        var chosen = function(el) { return mine(el) && (!leg || el.getAttribute('data-leg') === leg); };
        forEachNode('.dag-md-circle', function(el) { if (mine(el)) el.classList.add('sel'); });
        forEachNode('.dag-md-label', function(el) { if (mine(el)) el.classList.add('sel'); });
        forEachNode('line[data-md]', function(el) {
            if (!chosen(el)) return;
            el.classList.add('sel');
            el.setAttribute('stroke', color);
            el.setAttribute('marker-end', 'url(#arrow-md-' + color.replace('#', '') + ')');
        });
        forEachNode('.dag-count', function(el) { if (chosen(el)) el.classList.add('sel'); });
        ['to_iv', 'to_dv'].forEach(function(which) {
            if (leg && leg !== which) return;
            forEachNode('[data-var="' + VARIABLE_OF[which] + '"]', function(el) {
                el.classList.add('sel');
                el.setAttribute('stroke', color);
            });
        });
    }
    document.querySelectorAll('.md-card').forEach(function(card) {
        var active = card.getAttribute('data-md') === md;
        var c = mdColors[card.getAttribute('data-md')];
        card.style.border = '2px solid ' + (active ? c : '#d1d5db');
        card.style.background = active ? c + '18' : 'transparent';
    });
    document.querySelectorAll('.leg-btn').forEach(function(btn) {
        var active = btn.getAttribute('data-md') === md && btn.getAttribute('data-leg') === leg;
        btn.classList.toggle('active', active);
        btn.style.background = active ? mdColors[md] : '';
    });
}

function dagBackgroundClick() { highlight(null, null); showEvidence('__deselect__'); }
function dagMetaClick(md) { highlight(md, null); showEvidence('__meta__:' + md); }
function dagLegClick(md, leg) { highlight(md, leg); showEvidence('__leg__:' + md + ':' + leg); }

document.addEventListener('click', function(ev) {
    var button = ev.target.closest('.leg-btn');
    if (button) { dagLegClick(button.getAttribute('data-md'), button.getAttribute('data-leg')); return; }
    var card = ev.target.closest('.md-card');
    if (card) dagMetaClick(card.getAttribute('data-md'));
});
highlight(null, null);
