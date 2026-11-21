import os
import uuid

from flask import jsonify, render_template, request, session

from agent import AnalystAgent
from ai_client import AIError

MAX_STORED_MESSAGES = 40
HISTORIES = {}


def register(app, get_df, step=None):
    """Analyst page aur API routes jodta hai. Agent wapas karta hai (testing ke liye)."""
    if not app.secret_key:
        app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)
    agent = AnalystAgent(get_df, step=step)

    def get_sid():
        if "sid" not in session:
            session["sid"] = uuid.uuid4().hex
        return session["sid"]

    @app.route("/analyst")
    def analyst_page():
        sid = get_sid()
        return render_template(
            "analyst.html",
            has_data=get_df() is not None,
            history=HISTORIES.get(sid, []),
        )

    @app.route("/api/ask", methods=["POST"])
    def api_ask():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Sawal JSON ki shakal mein bhejein."}), 400
        question = str(body.get("question", ""))
        sid = get_sid()
        history = HISTORIES.setdefault(sid, [])
        try:
            result = agent.ask(question, history)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except AIError as e:
            return jsonify({"error": str(e)}), 502
        except Exception as e:
            return jsonify({"error": f"Agent mein ghalti ({type(e).__name__})."}), 500
        history.append({"role": "user", "text": question.strip()})
        history.append({"role": "model", "text": result["answer"]})
        del history[:-MAX_STORED_MESSAGES]
        return jsonify(result)

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        HISTORIES.pop(get_sid(), None)
        return jsonify({"ok": True})

    return agent
