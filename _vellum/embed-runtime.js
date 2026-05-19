/* Vellum embed runtime — Layer 1 protocol code shared by every HTML
 * file embedded via the :::{embed-html} MyST directive.
 *
 * Injected by the parent HtmlEmbed component into the iframe's
 * contentDocument after load; embed authors do NOT reference this
 * file directly. It does two things:
 *
 *   1. Adds an `.in-iframe` class to <html> so embeds can opt into
 *      iframe-aware styling. Also installs a style tag suppressing
 *      inner-document overflow — the parent page handles all scroll,
 *      so a redundant scrollbar inside the iframe is visually noisy.
 *
 *   2. Posts the document's measured height up to the parent via
 *      `postMessage({ type: 'vellum:height', value })` whenever it
 *      changes, so the iframe element can resize to fit and the
 *      surrounding prose layout stays correct.
 *
 * Standalone (opened directly in a browser, not via iframe) the
 * script no-ops on first check and the document scrolls normally.
 */
(function () {
  if (window.parent === window) return;
  document.documentElement.classList.add('in-iframe');

  var style = document.createElement('style');
  style.textContent =
    'html.in-iframe, html.in-iframe body { overflow: hidden; }';
  document.head.appendChild(style);

  var last = 0;
  function post() {
    var h = Math.max(
      document.documentElement.scrollHeight,
      document.body ? document.body.scrollHeight : 0,
    );
    if (h > 0 && Math.abs(h - last) > 1) {
      last = h;
      window.parent.postMessage({ type: 'vellum:height', value: h }, '*');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', post);
  } else {
    post();
  }
  window.addEventListener('load', post);
  window.addEventListener('resize', post);
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(post).observe(document.documentElement);
  }
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(post);
  }
})();
