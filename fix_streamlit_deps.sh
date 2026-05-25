#!/usr/bin/env bash
# Fix Streamlit / Altair vs TensorFlow typing_extensions conflict
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate

pip install --force-reinstall "typing-extensions>=4.10,<5"
python -c "from typing_extensions import TypeAliasType; print('typing_extensions OK')"

echo ""
echo "Restart Streamlit:  streamlit run app.py"
echo "Confirm app.py has no bar_chart:  grep -n bar_chart app.py || echo '(none — good)'"
