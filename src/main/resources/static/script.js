const form = document.getElementById('upload-form');
const imageFileInput = document.getElementById('imageFile');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const predictionSummaryDiv = document.getElementById('prediction-summary');
const predictionDetailsCode = document.getElementById('prediction-details');
const detailsSection = document.getElementById('details-section');
const previewImg = document.getElementById('preview');
const spinner = document.getElementById('spinner');

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = imageFileInput.files[0];

    if (!file) {
        showError('Bitte wähle zuerst eine Datei aus.');
        return;
    }

    hideMessages();
    spinner.style.display = 'block';
    previewImg.style.display = 'none';
    detailsSection.style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        spinner.style.display = 'none';

        const responseText = await response.text();
        console.log("Raw response text:", responseText);

        if (!response.ok) {
            throw new Error(`Serverfehler: ${response.status} ${response.statusText} - ${responseText}`);
        }

        try {
            const resultData = JSON.parse(responseText);
            console.log("Parsed JSON data:", resultData);
            displayResult(resultData);
            displayPreview(file);
        } catch (parseError) {
            console.error("Error parsing JSON:", parseError);
            throw new Error(`Ungültige JSON-Antwort vom Server erhalten: ${responseText}`);
        }

    } catch (error) {
        spinner.style.display = 'none';
        console.error('Fehler beim Senden/Verarbeiten der Anfrage:', error);
        showError(`Ein Fehler ist aufgetreten: ${error.message}`);
    }
});

function displayResult(data) {
    if (Array.isArray(data) && data.length > 0) {
        let bestPrediction = data.reduce((max, current) =>
            (current.probability > max.probability) ? current : max, data[0]
        );

        const probabilityPercent = (bestPrediction.probability * 100).toFixed(2);
        predictionSummaryDiv.textContent = `Beim Bild handelt es sich um ein(e) ${bestPrediction.className} (Wahrscheinlichkeit: ${probabilityPercent}%).`;

        predictionDetailsCode.textContent = JSON.stringify(data, null, 2);
        resultDiv.style.display = 'block';
    } else {
        showError('Ungültige oder leere Antwortstruktur vom Server erhalten.');
        console.error('Received unexpected data structure:', data);
    }
}

function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
    }
    reader.readAsDataURL(file);
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideMessages() {
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    predictionSummaryDiv.textContent = '';
    predictionDetailsCode.textContent = '';
}

function toggleDetails() {
    if (detailsSection.style.display === 'none') {
        detailsSection.style.display = 'block';
    } else {
        detailsSection.style.display = 'none';
    }
}
