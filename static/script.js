document.addEventListener('DOMContentLoaded', function() {
    // File upload preview
    const fileInput = document.getElementById('file-upload');
    const fileUploadArea = document.querySelector('.file-upload-area');
    
    if (fileInput && fileUploadArea) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    // Remove any existing preview
                    const existingPreview = fileUploadArea.querySelector('.image-preview');
                    if (existingPreview) {
                        existingPreview.remove();
                    }
                    
                    // Create and display the preview
                    const preview = document.createElement('img');
                    preview.src = e.target.result;
                    preview.classList.add('w-full', 'h-48', 'object-cover', 'rounded-md', 'mb-4', 'image-preview');
                    fileUploadArea.insertBefore(preview, fileUploadArea.firstChild);
                    
                    // Update the upload text
                    const uploadText = fileUploadArea.querySelector('.upload-text');
                    if (uploadText) {
                        uploadText.textContent = file.name;
                    }
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            fileUploadArea.classList.add('border-green-500', 'bg-green-50');
        }
        
        function unhighlight() {
            fileUploadArea.classList.remove('border-green-500', 'bg-green-50');
        }
        
        fileUploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            
            // Trigger change event
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    }
    
    // Form submission handling
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file-upload');
            const submitButton = form.querySelector('button[type="submit"]');
            
            if (fileInput.files.length === 0) {
                e.preventDefault();
                showAlert('Please select an image file to upload.', 'error');
                return;
            }
            
            // Show loading state
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = `
                    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing...
                `;
            }
        });
    }
    
    // Show alert function
    function showAlert(message, type = 'info') {
        // Remove any existing alerts
        const existingAlert = document.querySelector('.alert-message');
        if (existingAlert) {
            existingAlert.remove();
        }
        
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert-message fixed bottom-4 right-4 p-4 rounded-lg shadow-lg text-white ${
            type === 'error' ? 'bg-red-500' : 'bg-green-500'
        }`;
        alert.textContent = message;
        
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'ml-4';
        closeButton.innerHTML = '&times;';
        closeButton.onclick = () => alert.remove();
        alert.appendChild(closeButton);
        
        // Add to page
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    }
    
    // Initialize tooltips if using any
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
