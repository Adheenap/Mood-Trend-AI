import kagglehub

print("Downloading RAVDESS dataset...")
ravdess_path = kagglehub.dataset_download(
    "uwrfkaggler/ravdess-emotional-speech-audio"
)

print("RAVDESS downloaded at:", ravdess_path)
