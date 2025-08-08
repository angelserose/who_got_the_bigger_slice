const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const captureButton = document.getElementById('captureButton');
const resetButton = document.getElementById('resetButton');
const resultContainer = document.getElementById('resultContainer');
const resultText = document.getElementById('resultText');
const pieceInfo = document.getElementById('pieceInfo');

let stream = null;

// Start webcam
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Error accessing webcam. Please make sure you have a webcam connected and have granted permission to use it.");
    }
}

// Initialize canvas
const ctx = canvas.getContext('2d');
canvas.width = 800;
canvas.height = 600;

// Start webcam when page loads
startWebcam();

// Capture and analyze image
captureButton.addEventListener('click', async () => {
    // Clear previous results
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultContainer.style.display = 'none';
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Add visual feedback
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#4ecdc4';
    ctx.font = 'bold 24px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Analyzing...', canvas.width/2, canvas.height/2);
    
    // Get image data from canvas
    const imageData = canvas.toDataURL('image/jpeg');

    try {
        // Send image to backend for analysis
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please try again.');
    }
});

// Display results
function displayResults(result) {
    resultContainer.style.display = 'block';
    
    if (result.pieces.length < 2) {
        resultText.textContent = "Please show at least 2 pieces of food!";
        pieceInfo.innerHTML = '';
        return;
    }

    // Sort pieces by area
    const sortedPieces = result.pieces.sort((a, b) => b.area - a.area);
    
    // Show which piece is biggest
    resultText.textContent = `Piece ${sortedPieces[0].id} is the biggest!`;
    
    // Create cards for each piece
    pieceInfo.innerHTML = '';
    sortedPieces.forEach((piece, index) => {
        const card = document.createElement('div');
        card.className = `piece-card ${index === 0 ? 'winner' : ''}`;
        card.innerHTML = `
            <h3>Piece ${piece.id}</h3>
            <p>Area: ${Math.round(piece.area)} pixels²</p>
            <p>Relative Size: ${Math.round(piece.area / sortedPieces[sortedPieces.length - 1].area * 100) / 100}x</p>
        `;
        pieceInfo.appendChild(card);
    });

    // Draw contours and highlights on canvas
    result.pieces.forEach((piece, index) => {
        // Create gradient for the fill
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, `rgba(${index === 0 ? '78, 205, 196' : '255, 107, 107'}, 0.2)`);
        gradient.addColorStop(1, `rgba(${index === 0 ? '78, 205, 196' : '255, 107, 107'}, 0.4)`);
        
        // Draw filled contour
        ctx.fillStyle = gradient;
        ctx.strokeStyle = index === 0 ? '#4ecdc4' : '#ff6b6b';
        ctx.lineWidth = index === 0 ? 4 : 3;
        
        ctx.beginPath();
        piece.contour.forEach((point, i) => {
            if (i === 0) {
                ctx.moveTo(point[0], point[1]);
            } else {
                ctx.lineTo(point[0], point[1]);
            }
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Add size comparison indicator
        const percentage = Math.round((piece.size_ratio - 1) * 100);
        if (index === 0 && percentage > 0) {
            ctx.fillStyle = '#4ecdc4';
            ctx.font = 'bold 16px Arial';
            ctx.fillText(`${percentage}% bigger`, piece.contour[0][0], piece.contour[0][1] - 10);
        }
    });
}

// Reset button handler
resetButton.addEventListener('click', () => {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Hide results
    resultContainer.style.display = 'none';
});

// Clean up when page is closed
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
