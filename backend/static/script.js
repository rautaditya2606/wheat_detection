document.addEventListener('DOMContentLoaded', function() {
    // File upload preview
    const fileInput = document.getElementById('file-upload');
    const fileUploadArea = document.querySelector('.file-upload-area');
    
    if (fileInput && fileUploadArea) {
        fileInput.addEventListener('change', function(e) {
            const files = Array.from(e.target.files);
            if (!files.length) return;

            // Clear old previews when a new selection is made
            fileUploadArea.querySelectorAll('.preview-item').forEach(el => el.remove());

            files.forEach((file) => {
                const reader = new FileReader();

                reader.onload = function(ev) {
                    const preview = document.createElement('div');
                    preview.className = 'preview-item relative inline-block m-1';

                    preview.innerHTML = `
                        <div class="relative group">
                            <img src="${ev.target.result}" class="h-32 w-32 object-cover rounded-lg" alt="Preview">
                            <button type="button" class="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity">×</button>
                        </div>
                    `;

                    // Remove only this preview
                    preview.querySelector('button').onclick = () => preview.remove();

                    // Click to enlarge
                    preview.querySelector('img').onclick = () => openImageModal(ev.target.result);

                    fileUploadArea.appendChild(preview);
                };

                reader.readAsDataURL(file);
            });
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

    // Simple image modal (shared)
    function openImageModal(src) {
        let modal = document.getElementById('image-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'image-modal';
            modal.className = 'fixed inset-0 bg-black/80 flex items-center justify-center z-50';
            modal.innerHTML = `
                <div class="relative max-w-4xl w-full p-4">
                    <img id="modal-img" class="w-full max-h-[80vh] object-contain rounded-lg" />
                    <button id="modal-close" class="absolute top-2 right-2 bg-white text-black rounded-full px-3 py-1">×</button>
                </div>
            `;
            document.body.appendChild(modal);

            modal.addEventListener('click', (e) => {
                if (e.target.id === 'image-modal' || e.target.id === 'modal-close') {
                    modal.remove();
                }
            });
        }

        modal.querySelector('#modal-img').src = src;
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
