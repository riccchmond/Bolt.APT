// AI Javelin Coach - Frontend JavaScript

class JavelinCoach {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.validateForm();
    }

    initializeElements() {
        this.form = document.getElementById('uploadForm');
        this.videoFile = document.getElementById('videoFile');
        this.height = document.getElementById('height');
        this.weight = document.getElementById('weight');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.btnText = document.getElementById('btnText');
        this.btnSpinner = document.getElementById('btnSpinner');
        this.logOutput = document.getElementById('logOutput');
        this.videoPreview = document.getElementById('videoPreview');
        this.fileInfo = document.getElementById('fileInfo');
        this.coachResults = document.getElementById('coachResults');
        this.waitingForCoach = document.getElementById('waitingForCoach');
        this.analysisResults = document.getElementById('analysisResults');
        this.coachVideo = document.getElementById('coachVideo');
    }

    attachEventListeners() {
        // Form validation
        [this.videoFile, this.height, this.weight].forEach(input => {
            input.addEventListener('input', () => this.validateForm());
            input.addEventListener('change', () => this.validateForm());
        });

        // Video preview
        this.videoFile.addEventListener('change', (e) => this.handleVideoPreview(e));

        // Form submission
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
    }

    validateForm() {
        const hasVideo = this.videoFile.files.length > 0;
        const hasHeight = this.height.value.trim() !== '';
        const hasWeight = this.weight.value.trim() !== '';
        
        const isValid = hasVideo && hasHeight && hasWeight;
        this.analyzeBtn.disabled = !isValid;
        
        if (isValid) {
            this.analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        } else {
            this.analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    }

    handleVideoPreview(event) {
        const file = event.target.files[0];
        if (file) {
            // Validate file size (100MB limit)
            const maxSize = 100 * 1024 * 1024; // 100MB
            if (file.size > maxSize) {
                this.addLogEntry('ERROR', 'File size too large. Maximum 100MB allowed.');
                this.videoFile.value = '';
                this.videoPreview.classList.add('hidden');
                this.validateForm();
                return;
            }

            // Show preview
            const video = this.videoPreview.querySelector('video');
            video.src = URL.createObjectURL(file);
            
            // Show file info
            const sizeInMB = (file.size / (1024 * 1024)).toFixed(1);
            this.fileInfo.textContent = `${file.name} (${sizeInMB} MB)`;
            
            this.videoPreview.classList.remove('hidden');
        } else {
            this.videoPreview.classList.add('hidden');
        }
    }

    async handleSubmit(event) {
        event.preventDefault();
        
        // Disable form and show loading state
        this.setLoadingState(true);
        this.clearLog();
        this.hideCoachResults();
        
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('video', this.videoFile.files[0]);
            formData.append('height', this.height.value);
            formData.append('weight', this.weight.value);

            this.addLogEntry('INFO', 'Starting analysis...');

            // Make streaming request
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Process streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        this.processStreamData(data);
                    }
                }
            }

            // Process any remaining data
            if (buffer.startsWith('data: ')) {
                const data = buffer.substring(6);
                this.processStreamData(data);
            }

        } catch (error) {
            this.addLogEntry('ERROR', `Request failed: ${error.message}`);
        } finally {
            this.setLoadingState(false);
        }
    }

    processStreamData(data) {
        // Try to parse as JSON (final result)
        try {
            const result = JSON.parse(data);
            if (result.tavus_video_url || result.analysis) {
                this.showCoachResults(result);
                return;
            }
        } catch (e) {
            // Not JSON, treat as log message
        }

        // Parse log message
        const match = data.match(/^\[(\w+)\]\s*(.*)$/);
        if (match) {
            const [, level, message] = match;
            this.addLogEntry(level, message);
        } else if (data.trim()) {
            this.addLogEntry('INFO', data);
        }
    }

    addLogEntry(level, message) {
        const timestamp = new Date().toLocaleTimeString();
        const colors = {
            'ERROR': 'text-red-400',
            'WARNING': 'text-yellow-400',
            'INFO': 'text-blue-400',
            'ANALYSIS': 'text-green-400',
            'TAVUS': 'text-purple-400',
            'UPLOAD': 'text-cyan-400',
            'CLEANUP': 'text-gray-400'
        };
        
        const color = colors[level] || 'text-white';
        
        const logEntry = document.createElement('div');
        logEntry.className = `${color} mb-1`;
        logEntry.innerHTML = `<span class="text-white/50">[${timestamp}]</span> <span class="font-semibold">[${level}]</span> ${message}`;
        
        this.logOutput.appendChild(logEntry);
        this.logOutput.scrollTop = this.logOutput.scrollHeight;
    }

    clearLog() {
        this.logOutput.innerHTML = '<div class="text-white/50">Starting analysis...</div>';
    }

    setLoadingState(loading) {
        if (loading) {
            this.analyzeBtn.disabled = true;
            this.btnText.classList.add('hidden');
            this.btnSpinner.classList.remove('hidden');
            this.form.classList.add('opacity-75');
        } else {
            this.analyzeBtn.disabled = false;
            this.btnText.classList.remove('hidden');
            this.btnSpinner.classList.add('hidden');
            this.form.classList.remove('opacity-75');
            this.validateForm(); // Re-validate to set correct state
        }
    }

    showCoachResults(result) {
        this.waitingForCoach.classList.add('hidden');
        this.coachResults.classList.remove('hidden');

        // Show analysis results
        const analysis = result.analysis || result;
        this.analysisResults.innerHTML = `
            <h3 class="text-white font-semibold mb-2">
                <i class="fas fa-chart-line mr-2"></i>Analysis Results
            </h3>
            <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                    <span class="text-white/70">Main Focus Area:</span>
                    <span class="text-white font-medium">${analysis.most_deviant_angle || 'Overall technique'}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-white/70">Deviation:</span>
                    <span class="text-white font-medium">${analysis.deviation || 0}°</span>
                </div>
                <div class="mt-3">
                    <span class="text-white/70">Recommendation:</span>
                    <p class="text-white mt-1">${analysis.llm_suggestion || 'Keep practicing!'}</p>
                </div>
            </div>
        `;

        // Show Tavus video if available
        if (result.tavus_video_url) {
            this.coachVideo.innerHTML = `
                <div class="mt-4">
                    <h3 class="text-white font-semibold mb-2">
                        <i class="fas fa-play-circle mr-2"></i>Personalized Coaching
                    </h3>
                    <iframe 
                        src="${result.tavus_video_url}" 
                        class="w-full h-64 rounded-lg"
                        frameborder="0" 
                        allowfullscreen>
                    </iframe>
                </div>
            `;
        } else {
            this.coachVideo.innerHTML = `
                <div class="mt-4 p-4 bg-yellow-500/20 rounded-lg">
                    <p class="text-yellow-200 text-sm">
                        <i class="fas fa-exclamation-triangle mr-2"></i>
                        Coaching video could not be generated, but analysis is complete.
                    </p>
                </div>
            `;
        }
    }

    hideCoachResults() {
        this.waitingForCoach.classList.remove('hidden');
        this.coachResults.classList.add('hidden');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new JavelinCoach();
});