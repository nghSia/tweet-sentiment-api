#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Erreur : $PYTHON introuvable. Creez d'abord le venv (voir README)."
    exit 1
fi

CRON_LINE="0 3 * * 1 cd $PROJECT_DIR && $PYTHON scripts/retrain_model.py >> $PROJECT_DIR/logs/retrain.log 2>&1"

( crontab -l 2>/dev/null | grep -vF "retrain_model.py" ; echo "$CRON_LINE" ) | crontab -

echo "Cronjob installe :"
crontab -l | grep "retrain_model.py"
