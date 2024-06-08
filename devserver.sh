#!/bin/bash
export FLASK_APP=src/main.py
export FLASK_ENV=development
export PORT=${PORT:-3000}
python -m flask run --port=$PORT