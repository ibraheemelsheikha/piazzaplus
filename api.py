from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
from search_lib import search_top_k

app = Flask(__name__)
CORS(app)

AUTH_PATH = Path("auth.json")
AUTH_MAP = json.loads(AUTH_PATH.read_text(encoding="utf-8"))

@app.get("/api/is-registered")
def is_registered():
    nid = (request.args.get("network_id") or "").strip()
    return jsonify({"registered": nid in AUTH_MAP})

@app.post("/api/search")
def search():
    payload = request.get_json(force=True) or {}
    nid = (payload.get("network_id") or "").strip()
    query = (payload.get("query") or "").strip()
    k = int(payload.get("k", 10))

    if nid not in AUTH_MAP:
        return jsonify({"error": "unregistered course"}), 404

    try:
        results = search_top_k(nid, query, k)
        return jsonify({"results": results})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": repr(e)}), 500

@app.get("/api/health")
def health():
    return {"ok": True}
