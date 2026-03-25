/**
 * Document Chunk Visualization
 * Uses D3.js for visualizing document chunks and their relationships
 */

/**
 * Create a 2D visualization of document chunks
 * @param {string} containerId - ID of the container element
 * @param {Object} data - Visualization data with projections and similarities
 */
function createChunkVisualization(containerId, data) {
    // Load D3 if not already loaded
    loadD3().then(() => {
        renderVisualization(containerId, data);
    }).catch(error => {
        console.error('Error loading D3:', error);
        document.getElementById(containerId).innerHTML = `
            <div class="alert alert-danger m-3">
                Failed to load visualization library: ${error.message}
            </div>
        `;
    });
}

/**
 * Load D3.js library dynamically
 * @returns {Promise} - Resolves when D3 is loaded
 */
function loadD3() {
    return new Promise((resolve, reject) => {
        // Check if D3 is already loaded
        if (window.d3) {
            resolve();
            return;
        }
        
        // Load D3 script
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/d3@7';
        script.async = true;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load D3.js'));
        
        document.head.appendChild(script);
    });
}

/**
 * Render the chunk visualization using D3
 * @param {string} containerId - ID of the container element
 * @param {Object} data - Visualization data
 */
function renderVisualization(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear container
    container.innerHTML = '';
    
    // Get container dimensions
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Add background
    svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#212529');
    
    // Create tooltip
    const tooltip = d3.select(container)
        .append('div')
        .attr('class', 'visualization-tooltip')
        .style('opacity', 0);
    
    // Scale for circle size
    const sizeScale = d3.scaleLinear()
        .domain([0, d3.max(data.projections, d => d.chunk.text.length)])
        .range([5, 15]);
    
    // Scale for X and Y coordinates
    const xExtent = d3.extent(data.projections, d => d.x);
    const yExtent = d3.extent(data.projections, d => d.y);
    
    const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([30, width - 30]);
    
    const yScale = d3.scaleLinear()
        .domain(yExtent)
        .range([30, height - 30]);
    
    // Draw connections (similarities)
    if (data.similarities && data.similarities.length > 0) {
        const links = svg.selectAll('.chunk-link')
            .data(data.similarities)
            .enter()
            .append('line')
            .attr('class', 'chunk-link')
            .attr('x1', d => xScale(data.projections[d.source].x))
            .attr('y1', d => yScale(data.projections[d.source].y))
            .attr('x2', d => xScale(data.projections[d.target].x))
            .attr('y2', d => yScale(data.projections[d.target].y))
            .attr('stroke', '#6c757d')
            .attr('stroke-opacity', d => d.score * 0.5)
            .attr('stroke-width', d => d.score * 2);
    }
    
    // Draw chunk nodes
    const nodes = svg.selectAll('.chunk-node')
        .data(data.projections)
        .enter()
        .append('circle')
        .attr('class', 'chunk-node')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', d => sizeScale(d.chunk.text.length))
        .attr('fill', d => {
            // Color by page if available
            if (d.chunk.page_num) {
                const hue = (d.chunk.page_num * 60) % 360;
                return `hsl(${hue}, 70%, 50%)`;
            }
            return '#0dcaf0';  // Default info color
        })
        .attr('opacity', 0.7)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('stroke-width', 2)
                .attr('opacity', 1);
                
            tooltip.transition()
                .duration(200)
                .style('opacity', 0.9);
                
            // Format tooltip text
            const text = d.chunk.text.length > 150 ? 
                d.chunk.text.substring(0, 150) + '...' : 
                d.chunk.text;
                
            tooltip.html(`
                <strong>Chunk #${d.chunk.id}</strong>
                ${d.chunk.page_num ? `<br>Page: ${d.chunk.page_num}` : ''}
                <br>Position: ${d.chunk.position}
                <br><small>${text}</small>
            `)
            .style('left', (event.pageX - container.getBoundingClientRect().left + 10) + 'px')
            .style('top', (event.pageY - container.getBoundingClientRect().top - 28) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this)
                .attr('stroke-width', 1)
                .attr('opacity', 0.7);
                
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        })
        .on('click', function(event, d) {
            // Highlight corresponding chunk in the chunk list
            const chunkElement = document.getElementById(`chunk-${d.chunk.id}`);
            if (chunkElement) {
                chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                chunkElement.classList.add('bg-dark');
                setTimeout(() => {
                    chunkElement.classList.remove('bg-dark');
                }, 1500);
                
                // Open collapse if it exists
                const collapseId = `chunk-collapse-${d.chunk.id}`;
                const collapseElement = document.getElementById(collapseId);
                if (collapseElement && !collapseElement.classList.contains('show')) {
                    // Find the button that controls this collapse
                    const button = document.querySelector(`[data-bs-target="#${collapseId}"]`);
                    if (button) button.click();
                }
            }
        });
    
    // Add labels for the first few chunks
    const labelCount = Math.min(10, data.projections.length);
    
    svg.selectAll('.chunk-label')
        .data(data.projections.slice(0, labelCount))
        .enter()
        .append('text')
        .attr('class', 'chunk-label')
        .attr('x', d => xScale(d.x))
        .attr('y', d => yScale(d.y) - 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .text(d => `#${d.chunk.id}`);
    
    // Add legend
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${width - 150}, 20)`);
    
    legend.append('rect')
        .attr('width', 140)
        .attr('height', 80)
        .attr('fill', 'rgba(0, 0, 0, 0.5)')
        .attr('rx', 5);
    
    legend.append('text')
        .attr('x', 10)
        .attr('y', 20)
        .attr('fill', '#fff')
        .text('Chunk Visualization');
    
    legend.append('circle')
        .attr('cx', 20)
        .attr('cy', 40)
        .attr('r', 6)
        .attr('fill', '#0dcaf0');
    
    legend.append('text')
        .attr('x', 35)
        .attr('y', 45)
        .attr('fill', '#fff')
        .text('Chunk');
    
    legend.append('line')
        .attr('x1', 10)
        .attr('y1', 60)
        .attr('x2', 30)
        .attr('y2', 60)
        .attr('stroke', '#6c757d')
        .attr('stroke-width', 2);
    
    legend.append('text')
        .attr('x', 35)
        .attr('y', 65)
        .attr('fill', '#fff')
        .text('Similarity');
}

/**
 * Create a distribution chart for chunk analysis
 * @param {string} containerId - ID of the container element
 * @param {Object} data - Distribution data
 */
function createDistributionChart(containerId, data) {
    // Load D3 if not already loaded
    loadD3().then(() => {
        renderDistributionChart(containerId, data);
    }).catch(error => {
        console.error('Error loading D3:', error);
        document.getElementById(containerId).innerHTML = `
            <div class="alert alert-danger m-3">
                Failed to load visualization library: ${error.message}
            </div>
        `;
    });
}

/**
 * Render the chunk distribution chart using D3
 * @param {string} containerId - ID of the container element
 * @param {Object} data - Distribution data
 */
function renderDistributionChart(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear container
    container.innerHTML = '';
    
    // Get container dimensions
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Add background
    svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#212529');
    
    // Add title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .text('Chunk Size Distribution');
    
    // Create scales
    const xScale = d3.scaleLinear()
        .domain([0, data.chunkSizes.length - 1])
        .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
        .domain([0, d3.max(data.chunkSizes)])
        .range([height - 50, 50]);
    
    // Create line generator
    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d))
        .curve(d3.curveMonotoneX);
    
    // Add the line path
    svg.append('path')
        .datum(data.chunkSizes)
        .attr('fill', 'none')
        .attr('stroke', '#0dcaf0')
        .attr('stroke-width', 2)
        .attr('d', line);
    
    // Add circles for each data point
    svg.selectAll('.data-point')
        .data(data.chunkSizes)
        .enter()
        .append('circle')
        .attr('class', 'data-point')
        .attr('cx', (d, i) => xScale(i))
        .attr('cy', d => yScale(d))
        .attr('r', 4)
        .attr('fill', '#fff');
    
    // Add axes
    const xAxis = d3.axisBottom(xScale)
        .ticks(Math.min(10, data.chunkSizes.length))
        .tickFormat(i => `Chunk ${i+1}`);
    
    const yAxis = d3.axisLeft(yScale);
    
    svg.append('g')
        .attr('transform', `translate(0, ${height - 50})`)
        .attr('color', '#fff')
        .call(xAxis);
    
    svg.append('g')
        .attr('transform', 'translate(50, 0)')
        .attr('color', '#fff')
        .call(yAxis);
    
    // Add axis labels
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height - 10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .text('Chunk');
    
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .text('Size (chars)');
}