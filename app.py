import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Import utilities
from yt_transcript import get_transcript, get_video_id
from chatbot import process_transcript_to_pinecone, ask_question

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Use a random secret key if one isn't provided (for sessions)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

@app.route("/")
def index():
    # If the user has a video active, we can pass it to the UI
    active_video = session.get("video_id", None)
    return render_template("index.html", active_video=active_video)

@app.route("/process_video", methods=["POST"])
def process_video():
    data = request.get_json()
    video_url = data.get("video_url", "")
    
    if not video_url:
        return jsonify({"success": False, "error": "No URL provided."}), 400
        
    video_id = get_video_id(video_url)
    if not video_id:
        return jsonify({"success": False, "error": "Invalid YouTube URL."}), 400
        
    try:
        # Get the transcript
        transcript_text = get_transcript(video_url)
        if not transcript_text:
            return jsonify({"success": False, "error": "Could not extract transcript."}), 500
            
        # Index to Pinecone
        process_transcript_to_pinecone(transcript_text, video_id)
        
        # Save video_id to session so chat knows context
        session["video_id"] = video_id
        
        return jsonify({"success": True, "video_id": video_id})
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    video_id = data.get("video_id") or session.get("video_id")
    
    if not video_id:
        return jsonify({"success": False, "error": "No active video session. Please process a video first."}), 400
        
    if not user_message:
        return jsonify({"success": False, "error": "Empty message."}), 400
        
    try:
        answer = ask_question(video_id, user_message)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        print(f"Error during chat: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

    
