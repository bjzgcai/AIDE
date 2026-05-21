#!/usr/bin/env bash
#
# Create a zip archive of the repo while excluding generated/virtual-env artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
REPO_NAME="$(basename "$REPO_ROOT")"
PARENT_DIR="$(dirname "$REPO_ROOT")"

timestamp="$(date +%Y%m%d-%H%M%S)"
requested_name="${1:-${REPO_NAME}-bundle-${timestamp}}"

if [[ "${requested_name}" == *.zip ]]; then
  zip_basename="${requested_name}"
else
  zip_basename="${requested_name}.zip"
fi

archive_path="${REPO_ROOT}/${zip_basename}"

if [[ -e "${archive_path}" ]]; then
  echo "Archive ${archive_path} already exists. Remove it or supply a different name." >&2
  exit 1
fi

excludes=(
  "${REPO_NAME}/.venv/*"
  "${REPO_NAME}/.vscode/*"
  "${REPO_NAME}/cache/*"
  "${REPO_NAME}/outputs/*"
  "${REPO_NAME}/log.txt"
  "${REPO_NAME}/**/*.pyc"
  "${REPO_NAME}/**/__pycache__/*"
  "${REPO_NAME}/release-bundle-*.zip"
)
excludes+=("${REPO_NAME}/${zip_basename}")

echo "Creating ${archive_path}..."
(cd "${PARENT_DIR}" && zip -r "${archive_path}" "${REPO_NAME}" -x "${excludes[@]}")
echo "Done."
