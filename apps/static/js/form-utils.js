/**
 * Shared form utility functions used across all StreamWise app submission forms.
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

/**
 * Read a File object and resolve with its base64-encoded content (without the data URL prefix).
 * @param {File} file
 * @returns {Promise<string>}
 */
const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]); // Strip "data:*/*;base64,"
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

/**
 * Format a duration in seconds as a human-readable string.
 * Returns at least "15 seconds" for very short durations.
 * @param {number} totalSeconds
 * @returns {string}
 */
function formatStringTime(totalSeconds) {
    if (totalSeconds < 15)
        return "15 seconds";
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return [
        hours > 0 ? `${hours} hours` : "",
        minutes > 0 ? `${minutes} minutes` : "",
        (seconds > 0 || (hours === 0 && minutes === 0)) ? `${seconds} seconds` : ""
    ].filter(Boolean).join(" ");
}
