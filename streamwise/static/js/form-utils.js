/**
 * Shared form utility functions used across all StreamWise templates.
 */

/**
 * Escape a string for safe insertion into HTML content.
 * @param {*} str
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
