from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/")
def home():
    return jsonify(status="ok", message="Hello from Flask on Vercel!")
