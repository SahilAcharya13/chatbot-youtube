import os
import sys
import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import whisper


# -------------------------------
# Extract Video ID from URL
# -------------------------------
def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    else:
        return None


# -------------------------------
# Try YouTube Transcript API
# -------------------------------
def fetch_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t['text'] for t in transcript])
        print("✅ Transcript fetched using YouTube API")
        return text

    except TranscriptsDisabled:
        print("❌ Transcripts are disabled for this video.")
    except NoTranscriptFound:
        print("❌ No transcript found for this video.")
    except Exception as e:
        print(f"⚠️ API Error: {str(e)}")

    return None


# -------------------------------
# Download Audio using yt-dlp
# -------------------------------
def download_audio(video_url):
    print("⬇️ Downloading audio...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'quiet': False,

        # 🔥 THIS FIXES 403 ERROR
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        },

        # 🔥 FORCE WORKING FORMAT
        'format_sort': ['abr', 'ext:mp3:m4a'],
        'noplaylist': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        for file in os.listdir():
            if file.startswith("audio."):
                print(f"✅ Audio downloaded: {file}")
                return file

    except Exception as e:
        print(f"❌ Audio download failed: {str(e)}")

    return None


# -------------------------------
# Whisper Transcription
# -------------------------------
def whisper_transcribe(audio_file, model_size="base"):
    try:
        print(f"🧠 Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        print("📝 Transcribing audio...")
        result = model.transcribe(audio_file, verbose=True)

        print("✅ Whisper transcription completed")
        return result["text"]

    except Exception as e:
        print(f"❌ Whisper failed: {str(e)}")
        return None


# -------------------------------
# Main Pipeline
# -------------------------------
def get_transcript(video_url):
    video_id = get_video_id(video_url)

    if not video_id:
        print("❌ Invalid YouTube URL")
        return None

    print(f"\n🎥 Video ID: {video_id}")

    # Step 1: Try YouTube transcript
    text = fetch_youtube_transcript(video_id)
    if text:
        return text

    # Step 2: Fallback to audio + Whisper
    print("\n⚠️ Falling back to Whisper transcription...")

    audio_file = download_audio(video_url)
    if not audio_file:
        return None

    text = whisper_transcribe(audio_file, model_size="base")

    # Cleanup
    try:
        os.remove(audio_file)
        print("🧹 Audio file cleaned up")
    except:
        pass

    return text


# -------------------------------
# Run from Terminal
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL: ")

    start_time = time.time()

    transcript = get_transcript(url)

    if transcript:
        print("\n📄 TRANSCRIPT:\n")
        print(transcript[:2000])  # print first 2000 chars
    else:
        print("\n❌ Failed to get transcript.")

    print(f"\n⏱️ Time taken: {round(time.time() - start_time, 2)} seconds")


    if transcript:
        print("\n📄 TRANSCRIPT:\n")
        print(transcript[:2000])

        # 🔥 SAVE TO FILE
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        print("\n💾 Transcript saved to transcript.txt")