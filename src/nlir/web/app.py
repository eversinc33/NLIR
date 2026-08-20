"""Flask app factory for the local browser view.

The browser view lifts every prompt through the live model configured by
its caller. It stores nothing: each analysis lives only for the request
that produced it.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from nlir.web.inspector import Inspector, ViewerInputError


def create_app(*, live_config: Path, rules_directory: Path) -> Flask:
    """Build the Flask app, failing fast if the live configuration is invalid."""
    inspector = Inspector.from_live_config(live_config, rules_directory)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/rules")
    def list_rules():
        return jsonify({"rules": inspector.list_rules()})

    @app.get("/api/rules/<rule_id>")
    def rule_detail(rule_id: str):
        try:
            return jsonify(inspector.rule_detail(rule_id))
        except ViewerInputError as error:
            return jsonify({"error": str(error)}), 404

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"prompt"}:
            return jsonify({"error": "The request must contain only prompt text."}), 400
        try:
            return jsonify(inspector.analyze(payload["prompt"]))
        except ViewerInputError as error:
            return jsonify({"error": str(error)}), 400
        except Exception:
            app.logger.exception("Prompt analysis failed.")
            return jsonify({"error": "Prompt analysis failed."}), 500

    return app
