#!/bin/bash
# Быстрый запуск без Docker
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 5000 --debug
