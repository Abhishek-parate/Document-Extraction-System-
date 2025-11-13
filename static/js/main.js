// Global utility functions and common JavaScript code

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any global features
    initializeApp();
});

function initializeApp() {
    // Add any global initialization code here
    console.log('Smart Document Extractor initialized');
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility function to show loading state
function showLoading(element, text = 'Loading...') {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (element) {
        element.innerHTML = `
            <div class="flex items-center justify-center p-4">
                <div class="loader border-4 border-t-4 border-gray-200 rounded-full w-8 h-8 mr-3"></div>
                <span class="text-gray-600">${text}</span>
            </div>
        `;
    }
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Utility function to validate file type
function isValidFileType(file, allowedTypes) {
    return allowedTypes.includes(file.type);
}

// Utility function to validate file size
function isValidFileSize(file, maxSizeInMB) {
    const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
    return file.size <= maxSizeInBytes;
}

// Function to handle API errors
function handleAPIError(error, fallbackMessage = 'An error occurred') {
    console.error('API Error:', error);
    
    if (error.response && error.response.data && error.response.data.error) {
        return error.response.data.error;
    }
    
    return fallbackMessage;
}

// Function to debounce rapid function calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Function to create and show notifications
class NotificationManager {
    static show(message, type = 'info', duration = 3000) {
        // Remove existing notifications
        const existing = document.querySelectorAll('.notification');
        existing.forEach(notification => notification.remove());
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg transform transition-transform z-50 ${this.getTypeClasses(type)}`;
        
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="${this.getIconClass(type)} mr-2"></i>
                <span>${message}</span>
                <button class="ml-4 text-current opacity-70 hover:opacity-100" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => notification.remove(), 300);
            }, duration);
        }
    }
    
    static getTypeClasses(type) {
        const classes = {
            'success': 'bg-green-500 text-white',
            'error': 'bg-red-500 text-white',
            'warning': 'bg-yellow-500 text-white',
            'info': 'bg-blue-500 text-white'
        };
        return classes[type] || classes.info;
    }
    
    static getIconClass(type) {
        const icons = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Export for use in other files
window.NotificationManager = NotificationManager;
window.showLoading = showLoading;
window.formatFileSize = formatFileSize;
window.isValidFileType = isValidFileType;
window.isValidFileSize = isValidFileSize;
window.handleAPIError = handleAPIError;
window.debounce = debounce;