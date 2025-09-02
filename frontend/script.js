document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const imageUpload = document.getElementById('imageUpload');
    const enablePreprocessing = document.getElementById('enablePreprocessing');
    const savePreprocessedImage = document.getElementById('savePreprocessedImage');
    const padding = document.getElementById('padding');
    const verticalThreshold = document.getElementById('verticalThreshold');
    const maxNewTokens = document.getElementById('maxNewTokens');
    const numBeams = document.getElementById('numBeams');
    const earlyStopping = document.getElementById('earlyStopping');
    const noRepeatNgramSize = document.getElementById('noRepeatNgramSize');

    const statusText = document.getElementById('statusText');
    const progressBar = document.getElementById('progressBar');
    const intermediatePreprocessedImage = document.getElementById('intermediatePreprocessedImage'); // New element
    const preprocessedImage = document.getElementById('preprocessedImage');
    const recognizedTextList = document.getElementById('recognizedTextList');
    const logOutput = document.getElementById('logOutput'); // New element for logs

    // Initialize SSE connection for logs
    const eventSource = new EventSource('/progress_stream');
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.log) {
            const logItem = document.createElement('p');
            logItem.textContent = data.log;
            logOutput.appendChild(logItem);
            logOutput.scrollTop = logOutput.scrollHeight; // Auto-scroll to bottom
        }
    };
    eventSource.onerror = function(err) {
        console.error("EventSource failed:", err);
        logOutput.innerHTML += '<p style="color: red;">Error connecting to log stream. Please ensure the Flask server is running.</p>';
    };


    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        // Clear previous results and logs
        intermediatePreprocessedImage.src = ''; // Clear intermediate image
        intermediatePreprocessedImage.style.display = 'none'; // Hide intermediate image
        preprocessedImage.src = '';
        preprocessedImage.style.display = 'none';
        recognizedTextList.innerHTML = '';
        logOutput.innerHTML = ''; // Clear logs on new submission

        statusText.textContent = 'Uploading image...';
        progressBar.style.width = '10%';
        progressBar.style.backgroundColor = '#007bff';

        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);
        formData.append('enable_preprocessing', enablePreprocessing.checked);
        formData.append('save_preprocessed_image', savePreprocessedImage.checked);
        formData.append('padding', padding.value);
        formData.append('vertical_threshold', verticalThreshold.value);
        formData.append('max_new_tokens', maxNewTokens.value);
        formData.append('num_beams', numBeams.value);
        formData.append('early_stopping', earlyStopping.checked);
        formData.append('enable_preprocessing', enablePreprocessing.checked);
        // Only send image for preprocessing step

        try {
            statusText.textContent = 'Uploading and preprocessing image...';
            progressBar.style.width = '20%';

            // Step 1: Upload and Preprocess
            const preprocessResponse = await fetch('/upload_and_preprocess', {
                method: 'POST',
                body: formData
            });

            const preprocessData = await preprocessResponse.json();

            if (preprocessResponse.ok) {
                statusText.textContent = 'Image preprocessed. Performing OCR...';
                progressBar.style.width = '60%';

                if (preprocessData.intermediate_preprocessed_image_base64) {
                    intermediatePreprocessedImage.src = `data:image/png;base64,${preprocessData.intermediate_preprocessed_image_base64}`;
                    intermediatePreprocessedImage.style.display = 'block';
                }

                // Step 2: Perform OCR
                const ocrResponse = await fetch('/perform_ocr', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        preprocessed_image_id: preprocessData.preprocessed_image_id,
                        save_preprocessed_image: savePreprocessedImage.checked,
                        padding: padding.value,
                        vertical_threshold: verticalThreshold.value,
                        max_new_tokens: maxNewTokens.value,
                        num_beams: numBeams.value,
                        early_stopping: earlyStopping.checked,
                        no_repeat_ngram_size: noRepeatNgramSize.value
                    })
                });

                const ocrData = await ocrResponse.json();

                if (ocrResponse.ok) {
                    statusText.textContent = 'OCR processed successfully!';
                    progressBar.style.width = '100%';
                    progressBar.style.backgroundColor = '#28a745';

                    preprocessedImage.src = `data:image/png;base64,${ocrData.preprocessed_image_base64}`;
                    preprocessedImage.style.display = 'block';

                    recognizedTextList.innerHTML = '';
                    ocrData.recognized_texts_and_boxes.forEach(item => {
                        const li = document.createElement('li');
                        li.textContent = item.text;
                        recognizedTextList.appendChild(li);
                    });
                } else {
                    statusText.textContent = `Error during OCR: ${ocrData.error}`;
                    progressBar.style.width = '100%';
                    progressBar.style.backgroundColor = '#dc3545';
                    console.error('Error during OCR:', ocrData.error);
                }

            } else {
                statusText.textContent = `Error during preprocessing: ${preprocessData.error}`;
                progressBar.style.width = '100%';
                progressBar.style.backgroundColor = '#dc3545';
                console.error('Error during preprocessing:', preprocessData.error);
            }
        } catch (error) {
            statusText.textContent = `Network Error: ${error.message}`;
            progressBar.style.width = '100%';
            progressBar.style.backgroundColor = '#dc3545';
            console.error('Network error:', error);
        }
    });
});