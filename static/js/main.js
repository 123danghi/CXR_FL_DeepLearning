document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const resultsCard = document.getElementById('results-card');
    const resultImage = document.getElementById('result-image');
    const predictionBadge = document.getElementById('prediction-badge');
    const normalPercentage = document.getElementById('normal-percentage');
    const tbPercentage = document.getElementById('tb-percentage');
    const normalProgress = document.getElementById('normal-progress');
    const tbProgress = document.getElementById('tb-progress');
    const resultMessage = document.getElementById('result-message');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            
            // Check if file is an image
            if (!file.type.match('image.*')) {
                alert('Please select an image file');
                return;
            }
            
            // Enable analyze button
            analyzeBtn.disabled = false;
            
            // Show file name in upload area
            uploadArea.classList.add('active');
            uploadArea.querySelector('.upload-text').textContent = file.name;
            
            // Clear previous results if any
            resultsCard.classList.add('d-none');
        } else {
            // Reset upload area
            uploadArea.classList.remove('active');
            uploadArea.querySelector('.upload-text').textContent = 'Drag & Drop or Click to Upload';
            analyzeBtn.disabled = true;
        }
    });

    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('active');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('active');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    }

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert('Please select a file first');
            return;
        }
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show loading state
        analyzeBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnSpinner.classList.remove('d-none');
        
        // Make API request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Handle error
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred: ' + error.message);
        })
        .finally(() => {
            // Reset button state
            analyzeBtn.disabled = false;
            btnText.textContent = 'Analyze X-Ray';
            btnSpinner.classList.add('d-none');
        });
    });

    // Function to display results
    function displayResults(data) {
        // Set image
        resultImage.src = data.image_path;
        
        // Set prediction badge
        let badgeClass = data.prediction === 'Normal' ? 'badge-normal' : 'badge-tuberculosis';
        predictionBadge.innerHTML = `<span class="prediction-badge ${badgeClass}">${data.prediction}</span>`;
        
        // Set percentages
        normalPercentage.textContent = `${data.normal_prob.toFixed(2)}%`;
        tbPercentage.textContent = `${data.tb_prob.toFixed(2)}%`;
        
        // Set progress bars
        normalProgress.style.width = `${data.normal_prob}%`;
        tbProgress.style.width = `${data.tb_prob}%`;
        
        // Set result message
        if (data.prediction === 'Normal') {
            resultMessage.className = 'alert alert-success';
            resultMessage.innerHTML = '<i class="fas fa-check-circle me-2"></i>No signs of tuberculosis detected.';
        } else {
            resultMessage.className = 'alert alert-danger';
            resultMessage.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Potential signs of tuberculosis detected. Please consult a healthcare professional.';
        }
        
        // Show results card
        resultsCard.classList.remove('d-none');
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    // Handle "Analyze Another X-Ray" button
    newAnalysisBtn.addEventListener('click', function() {
        // Reset form
        uploadForm.reset();
        uploadArea.classList.remove('active');
        uploadArea.querySelector('.upload-text').textContent = 'Drag & Drop or Click to Upload';
        analyzeBtn.disabled = true;
        
        // Hide results
        resultsCard.classList.add('d-none');
        
        // Scroll to top of form
        uploadForm.scrollIntoView({ behavior: 'smooth' });
    });
}); 