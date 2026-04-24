"""Backward-compatible entrypoint that runs the upgraded backend."""

from app_new import app, initialize_system

if __name__ == '__main__':
    initialize_system()
    app.run(host='127.0.0.1', port=5000, debug=True)
