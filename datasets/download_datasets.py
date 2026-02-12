import kagglehub
import os

def download_text_datasets():
    print(" Downloading text emotion datasets...")

    # GoEmotions dataset (Google)
    goemotions_path = kagglehub.dataset_download(
        "debarshichanda/goemotions"
    )
    print(" GoEmotions downloaded at:")
    print(goemotions_path)

    # NLP Emotion dataset
    nlp_emotions_path = kagglehub.dataset_download(
        "praveengovi/emotions-dataset-for-nlp"
    )
    print(" NLP Emotion dataset downloaded at:")
    print(nlp_emotions_path)


def download_audio_datasets():
    print("\n Downloading audio emotion datasets...")

    # RAVDESS
    ravdess_path = kagglehub.dataset_download(
        "uwrfkaggler/ravdess-emotional-speech-audio"
    )
    print(" RAVDESS downloaded at:")
    print(ravdess_path)

    # CREMA-D
    cremad_path = kagglehub.dataset_download(
        "ejlok1/cremad"
    )
    print(" CREMA-D downloaded at:")
    print(cremad_path)


if __name__ == "__main__":
    print(" Starting dataset download for MoodTrend-AI\n")

    download_text_datasets()
    download_audio_datasets()

    print("\n All datasets downloaded successfully!")
    print(" Files are stored inside KaggleHub cache directory")
