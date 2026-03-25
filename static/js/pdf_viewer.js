/**
 * PDF Viewer Component
 * 
 * Uses PDF.js for rendering PDFs in the browser
 * Features:
 * - Page navigation
 * - Zoom controls
 * - Chunk highlighting
 */

class PDFViewer {
    /**
     * Initialize PDF viewer
     * @param {string} containerId - ID of the container element
     * @param {string} pdfUrl - URL to the PDF file
     * @param {Object} options - Configuration options
     */
    constructor(containerId, pdfUrl, options = {}) {
        this.container = document.getElementById(containerId);
        this.pdfUrl = pdfUrl;
        this.pdf = null;
        this.currentPage = options.initialPage || 1;
        this.scale = options.scale || 1.0;
        this.highlights = [];
        this.onPdfLoaded = null; // Callback for when PDF is loaded
        
        // Create viewer elements
        this.createViewerElements();
        
        // Load PDF
        this.init();
    }
    
    /**
     * Initialize the PDF viewer
     */
    async init() {
        try {
            // Load PDF.js
            await this.loadPDFJS();
            
            // Load PDF
            await this.loadPDF();
            
            // Add controls event listeners
            this.setupControls();
            
        } catch (error) {
            this.showError(`Failed to load PDF: ${error.message}`);
            console.error('PDF Viewer Error:', error);
        }
    }
    
    /**
     * Create the viewer DOM elements
     */
    createViewerElements() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create controls
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'pdf-controls d-flex justify-content-between align-items-center';
        
        // Page navigation
        const paginationDiv = document.createElement('div');
        paginationDiv.innerHTML = `
            <button id="prevPage" class="btn btn-sm btn-outline-secondary me-2">
                <span>&laquo; Previous</span>
            </button>
            <span id="pageInfo" class="pdf-page">Page 1 of 1</span>
            <button id="nextPage" class="btn btn-sm btn-outline-secondary ms-2">
                <span>Next &raquo;</span>
            </button>
        `;
        
        // Zoom controls
        const zoomDiv = document.createElement('div');
        zoomDiv.innerHTML = `
            <button id="zoomOut" class="btn btn-sm btn-outline-secondary me-2">
                <span>-</span>
            </button>
            <span id="zoomLevel" class="pdf-zoom">100%</span>
            <button id="zoomIn" class="btn btn-sm btn-outline-secondary ms-2">
                <span>+</span>
            </button>
        `;
        
        controlsDiv.appendChild(paginationDiv);
        controlsDiv.appendChild(zoomDiv);
        
        // Canvas container
        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'pdf-canvas-container overflow-auto';
        canvasContainer.style.height = 'calc(100% - 50px)';
        
        // Canvas for rendering PDF
        const canvas = document.createElement('canvas');
        canvas.id = 'pdfCanvas';
        
        canvasContainer.appendChild(canvas);
        
        // Add elements to container
        this.container.appendChild(controlsDiv);
        this.container.appendChild(canvasContainer);
        
        // Store references
        this.canvas = canvas;
        this.pageInfo = controlsDiv.querySelector('#pageInfo');
        this.zoomLevel = controlsDiv.querySelector('#zoomLevel');
        this.prevButton = controlsDiv.querySelector('#prevPage');
        this.nextButton = controlsDiv.querySelector('#nextPage');
        this.zoomInButton = controlsDiv.querySelector('#zoomIn');
        this.zoomOutButton = controlsDiv.querySelector('#zoomOut');
        this.canvasContainer = canvasContainer;
    }
    
    /**
     * Load PDF.js library
     */
    async loadPDFJS() {
        return new Promise((resolve, reject) => {
            // Check if PDF.js is already loaded
            if (window.pdfjsLib) {
                resolve();
                return;
            }
            
            // Load PDF.js script
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.min.js';
            script.async = true;
            script.onload = () => {
                // Configure worker
                window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.worker.min.js';
                resolve();
            };
            script.onerror = () => {
                reject(new Error('Failed to load PDF.js'));
            };
            
            document.head.appendChild(script);
        });
    }
    
    /**
     * Load PDF document
     */
    async loadPDF() {
        try {
            // Load the PDF document
            const loadingTask = window.pdfjsLib.getDocument(this.pdfUrl);
            this.pdf = await loadingTask.promise;
            
            // Update page count
            this.updatePageInfo();
            
            // Render first page
            this.renderPage(this.currentPage);
            
            // Enable controls
            this.prevButton.disabled = this.currentPage <= 1;
            this.nextButton.disabled = this.currentPage >= this.pdf.numPages;
            
            // Call onPdfLoaded callback if provided
            if (typeof this.onPdfLoaded === 'function') {
                this.onPdfLoaded();
            }
            
        } catch (error) {
            this.showError(`Failed to load PDF: ${error.message}`);
            console.error('PDF Load Error:', error);
        }
    }
    
    /**
     * Set up control event listeners
     */
    setupControls() {
        this.prevButton.addEventListener('click', () => this.prevPage());
        this.nextButton.addEventListener('click', () => this.nextPage());
        this.zoomInButton.addEventListener('click', () => this.zoomIn());
        this.zoomOutButton.addEventListener('click', () => this.zoomOut());
    }
    
    /**
     * Render a specific page
     * @param {number} pageNum - Page number to render
     */
    async renderPage(pageNum) {
        if (!this.pdf) return;
        
        try {
            this.currentPage = pageNum;
            
            // Get page
            const page = await this.pdf.getPage(pageNum);
            
            // Get viewport
            const viewport = page.getViewport({ scale: this.scale });
            
            // Set canvas dimensions
            this.canvas.height = viewport.height;
            this.canvas.width = viewport.width;
            
            // Render page
            const renderContext = {
                canvasContext: this.canvas.getContext('2d'),
                viewport: viewport
            };
            
            await page.render(renderContext).promise;
            
            // Apply highlights if any
            this.applyHighlights(page, viewport);
            
            // Update UI
            this.updatePageInfo();
            this.prevButton.disabled = this.currentPage <= 1;
            this.nextButton.disabled = this.currentPage >= this.pdf.numPages;
            
        } catch (error) {
            console.error('Error rendering page:', error);
        }
    }
    
    /**
     * Apply highlights to the rendered page
     * @param {Object} page - PDF.js page object
     * @param {Object} viewport - PDF.js viewport
     */
    applyHighlights(page, viewport) {
        const ctx = this.canvas.getContext('2d');
        
        this.highlights.forEach(highlight => {
            if (highlight.page !== this.currentPage) return;
            
            // Draw highlight
            ctx.fillStyle = highlight.color || 'rgba(255, 255, 0, 0.3)';
            
            if (highlight.bbox) {
                const [x, y, width, height] = highlight.bbox;
                const scaledRect = {
                    x: x * this.scale,
                    y: (viewport.height / this.scale - y - height) * this.scale,
                    width: width * this.scale,
                    height: height * this.scale
                };
                
                ctx.fillRect(scaledRect.x, scaledRect.y, scaledRect.width, scaledRect.height);
            }
        });
    }
    
    /**
     * Update page info display
     */
    updatePageInfo() {
        if (this.pdf) {
            this.pageInfo.textContent = `Page ${this.currentPage} of ${this.pdf.numPages}`;
        }
    }
    
    /**
     * Update zoom display
     */
    updateZoomDisplay() {
        this.zoomLevel.textContent = `${Math.round(this.scale * 100)}%`;
    }
    
    /**
     * Go to previous page
     */
    prevPage() {
        if (this.currentPage <= 1) return;
        this.queueRenderPage(this.currentPage - 1);
    }
    
    /**
     * Go to next page
     */
    nextPage() {
        if (!this.pdf || this.currentPage >= this.pdf.numPages) return;
        this.queueRenderPage(this.currentPage + 1);
    }
    
    /**
     * Queue page rendering to avoid multiple rapid renders
     * @param {number} pageNum - Page number to render
     */
    queueRenderPage(pageNum) {
        if (this.renderTask) {
            this.renderTask.cancel();
            this.renderTask = null;
        }
        
        this.renderTask = setTimeout(() => {
            this.renderPage(pageNum);
            this.renderTask = null;
        }, 50);
    }
    
    /**
     * Zoom in
     */
    zoomIn() {
        this.scale *= 1.2;
        this.updateZoomDisplay();
        this.renderPage(this.currentPage);
    }
    
    /**
     * Zoom out
     */
    zoomOut() {
        this.scale /= 1.2;
        this.updateZoomDisplay();
        this.renderPage(this.currentPage);
    }
    
    /**
     * Add highlight to a page
     * @param {number} page - Page number
     * @param {Array} bbox - Bounding box [x, y, width, height]
     * @param {string} color - Highlight color
     */
    highlightOnPage(page, bbox, color = 'rgba(255, 255, 0, 0.3)') {
        this.highlights.push({ page, bbox, color });
        if (page === this.currentPage) {
            this.renderPage(this.currentPage);
        }
    }
    
    /**
     * Clear all highlights
     */
    clearHighlights() {
        this.highlights = [];
        this.renderPage(this.currentPage);
    }
    
    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        this.container.innerHTML = `
            <div class="alert alert-danger m-3">
                ${message}
            </div>
        `;
    }
}