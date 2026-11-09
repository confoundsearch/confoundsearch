// A short walkthrough of the page, step by step: one part of the page in focus, the rest blurred.
// TOUR_STEPS is a list of {html, focus, edge}: `focus` names a [data-tour] region (or none), `edge` is
// [confound, leg] for an arrow to point out. Needs highlight() from page.js.
(function () {
    var steps = TOUR_STEPS, at = 0;
    var card = document.getElementById('tour-card');
    var text = card.querySelector('.tour-text');
    var dots = card.querySelectorAll('.tour-dot');
    var back = card.querySelector('.tour-prev'), next = card.querySelector('.tour-next');

    function show() {
        var step = steps[at];
        document.body.classList.add('tour-on');
        document.documentElement.classList.add('tour-lock');
        document.querySelectorAll('[data-tour]').forEach(function (el) {
            el.classList.toggle('tour-focus', el.getAttribute('data-tour') === step.focus);
        });
        if (step.edge) {
            highlight(step.edge[0], step.edge[1]);  // the arrow to click: drawn strong in its confound's colour
        } else {
            highlight(null, null);
        }
        text.innerHTML = step.html;
        dots.forEach(function (dot, i) { dot.classList.toggle('on', i === at); });
        back.disabled = at === 0;
        next.textContent = at === steps.length - 1 ? 'Done' : '›';
        next.setAttribute('aria-label', at === steps.length - 1 ? 'Finish the tour' : 'Next');
        card.hidden = false;
    }

    function end() {
        document.body.classList.remove('tour-on');
        document.documentElement.classList.remove('tour-lock');
        document.querySelectorAll('.tour-focus').forEach(function (el) { el.classList.remove('tour-focus'); });
        card.hidden = true;
    }

    function go(by) {
        if (at + by >= steps.length) { end(); return; }
        at = Math.max(0, at + by);
        show();
    }

    back.addEventListener('click', function () { go(-1); });
    next.addEventListener('click', function () { go(1); });
    card.querySelector('.tour-close').addEventListener('click', end);
    dots.forEach(function (dot, i) { dot.addEventListener('click', function () { at = i; show(); }); });
    document.addEventListener('keydown', function (ev) {
        if (card.hidden) return;
        if (ev.key === 'ArrowRight') go(1);
        else if (ev.key === 'ArrowLeft') go(-1);
        else if (ev.key === 'Escape') end();
    });
    // a touch mouse or trackpad must not scroll the page (and so the graph) under the card while it is open
    ['wheel', 'touchmove'].forEach(function (kind) {
        window.addEventListener(kind, function (ev) { if (!card.hidden) ev.preventDefault(); }, { passive: false });
    });
    // clicking an arrow during the tour is what the last step asks for: the tour ends and the evidence shows
    document.addEventListener('click', function (ev) {
        if (!card.hidden && ev.target.closest('.dag-svg .edge')) end();
    }, true);
    var again = document.getElementById('tour-open');
    if (again) again.addEventListener('click', function () { at = 0; show(); });
    show();
})();
