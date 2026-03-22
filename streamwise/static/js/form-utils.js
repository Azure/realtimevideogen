/**
 * Shared JS utility functions for StreamWise templates.
 * Note: kept separate from apps/static/js/form-utils.js because apps and
 * streamwise run as independent Quart processes with separate static folders.
 */

/**
 * Escape a string for safe insertion into HTML to prevent XSS.
 * @param {string|number} str
 * @returns {string}
 */
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
