#!/usr/bin/env bash
set -Eeuo pipefail

# Apply the upgraded source to the EXISTING repository without touching Git or venv.
# Run this script from the root of your multilingual-text-classifier repository.
# Optional: set UPGRADE_ZIP=/path/to/multilingual-text-classifier-upgraded.zip

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: Run this script from inside your existing Git repository."
  exit 1
}
cd "$PROJECT_ROOT"

ZIP_PATH="${UPGRADE_ZIP:-$HOME/Downloads/multilingual-text-classifier-upgraded.zip}"
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Could not find the upgrade archive at:"
  echo "  $ZIP_PATH"
  echo
  echo "Download multilingual-text-classifier-upgraded.zip, then either move it to ~/Downloads"
  echo "or run with: UPGRADE_ZIP='/c/path/to/the/file.zip' bash apply_upgrade.sh"
  exit 1
fi

command -v unzip >/dev/null || {
  echo "ERROR: unzip is required. In Git Bash, install/use an unzip utility and retry."
  exit 1
}

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR=".upgrade-backup/$STAMP"
mkdir -p "$BACKUP_DIR"

# Back up the files most likely to be replaced. Keep .git and .venv untouched.
for f in app.py README.md requirements.txt runtime.txt language_model.pkl tfidf_vectorizer.pkl label_encoder.pkl; do
  if [[ -f "$f" ]]; then
    cp -p "$f" "$BACKUP_DIR/"
  fi
done

echo "Project: $PROJECT_ROOT"
echo "Archive: $ZIP_PATH"
echo "Backup:  $PROJECT_ROOT/$BACKUP_DIR"
echo
echo "Preview of archive files:"
unzip -l "$ZIP_PATH" | sed -n '1,45p'
echo
read -r -p "Apply these files to this repository? Type YES to continue: " CONFIRM
if [[ "$CONFIRM" != "YES" ]]; then
  echo "Cancelled. No project files were changed."
  exit 0
fi

# Exclude Git metadata, virtual environments, and the dataset so local history,
# installed packages, and the existing dataset are never overwritten.
unzip -o "$ZIP_PATH" -d "$PROJECT_ROOT" \
  -x '.git/*' '.venv/*' 'venv/*' 'data/train.csv'

echo
echo "Upgrade files copied. No commit or push was performed."
echo
echo "Next checks:"
echo "  git status --short"
echo "  ls src/language_intelligence"
echo "  ls tests"
echo
echo "If anything looks wrong, the pre-upgrade files are in: $BACKUP_DIR"
