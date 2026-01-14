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
                    const preview = document.createElement('div');
                    preview.className = 'relative inline-block';
                    preview.innerHTML = `
                        <div class="relative group">
                            <img src="${e.target.result}" class="h-32 w-32 object-cover rounded-lg" alt="Preview">
                            <button type="button" class="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity" onclick="this.parentElement.parentElement.remove(); fileInput.value = '';">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                </svg>
                            </button>
                        </div>
                    `;
                    
                    const existingPreview = fileUploadArea.querySelector('.relative.inline-block');
                    if (existingPreview) {
                        existingPreview.remove();
                    }
                    
                    fileUploadArea.appendChild(preview);
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
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
            
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    }
    
    // Show alert function
    function showAlert(message, type = 'info') {
        // Remove any existing alerts
        const existingAlert = document.querySelector('.alert-message');
        if (existingAlert) {
            existingAlert.remove();
        }
        
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-message p-4 mb-4 rounded-lg ${type === 'error' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'}`;
        alertDiv.role = 'alert';
        alertDiv.textContent = message;
        
        // Insert alert at the top of the form
        const form = document.querySelector('form');
        if (form) {
            form.prepend(alertDiv);
        } else {
            document.body.prepend(alertDiv);
        }
        
        // Auto-remove alert after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
    
    // Initialize tooltips if using any
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.forEach(function (tooltipTriggerEl) {
            new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});
