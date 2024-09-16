document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('file-input');
    
    if (fileInput.files.length === 0) {
        alert('Please upload an image file!');
        return;
    }

    formData.append('file', fileInput.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (response.ok) {
        document.getElementById('rgb-title').style.display = 'block';
        document.getElementById('mask-title').style.display = 'block';

        document.getElementById('rgb-image').src = `data:image/png;base64,${result.rgb_image}`;
        document.getElementById('rgb-image').style.display = 'block';

        document.getElementById('overlay-mask').src = `data:image/png;base64,${result.mask_overlay}`;
        document.getElementById('overlay-mask').style.display = 'block';
    } else {
        alert(result.error || 'Something went wrong!');
    }
});
