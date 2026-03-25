/**
 * Main application JavaScript file
 */

/**
 * Toggle chunk content visibility
 * @param {string} chunkId - ID of the chunk
 */
function toggleChunk(chunkId) {
    const chunkElement = document.getElementById(`chunk-${chunkId}`);
    const chunkContent = document.getElementById(`chunk-content-${chunkId}`);
    
    if (chunkContent.style.display === 'none') {
        chunkContent.style.display = 'block';
        chunkElement.querySelector('.chunk-toggle').textContent = 'Hide';
    } else {
        chunkContent.style.display = 'none';
        chunkElement.querySelector('.chunk-toggle').textContent = 'Show';
    }
}

/**
 * Copy chunk text to clipboard
 * @param {string} chunkId - ID of the chunk
 */
function copyChunkText(chunkId) {
    const chunkElement = document.getElementById(chunkId);
    if (!chunkElement) return;
    
    const textElement = chunkElement.querySelector('.chunk-text');
    if (!textElement) return;
    
    const text = textElement.innerText;
    
    navigator.clipboard.writeText(text).then(() => {
        // Show tooltip or feedback
        const button = document.querySelector(`[onclick="copyChunkText('${chunkId}')"]`);
        if (button) {
            const originalTitle = button.getAttribute('title');
            button.setAttribute('title', 'Copied!');
            button.classList.add('btn-success');
            button.classList.remove('btn-outline-info');
            
            setTimeout(() => {
                button.setAttribute('title', originalTitle);
                button.classList.remove('btn-success');
                button.classList.add('btn-outline-info');
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    // Find all tooltip elements
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
    
    // Initialize Bootstrap tooltips
    if (window.bootstrap && window.bootstrap.Tooltip) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Initialize tooltips when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initTooltips();
});