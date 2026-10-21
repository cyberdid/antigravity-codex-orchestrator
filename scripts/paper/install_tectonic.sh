#!/usr/bin/env bash
# Install the pinned official Linux x86_64 release after verifying its SHA-256.
set -euo pipefail
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "This helper supports Linux x86_64; install Tectonic 0.15.0 for your platform." >&2
  exit 2
fi
target_dir=${1:-"$HOME/.local/bin"}
tectonic_tmp=$(mktemp -d)
trap 'rm -rf "$tectonic_tmp"' EXIT
url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz'
curl --fail --location --silent --show-error --retry 3 "$url" -o "$tectonic_tmp/release.tar.gz"
printf '%s  %s\n' '875fbbc9ab48560d7776088c608e0beee49197b57ab4a2f6c5385b2c661c842f' "$tectonic_tmp/release.tar.gz" | sha256sum --check
tar -xzf "$tectonic_tmp/release.tar.gz" -C "$tectonic_tmp" tectonic
mkdir -p "$target_dir"
install -m 0755 "$tectonic_tmp/tectonic" "$target_dir/tectonic"
"$target_dir/tectonic" --version
