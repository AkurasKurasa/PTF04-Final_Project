#!/usr/bin/env bash
# Convert all .ipynb in notebooks/ to HTML in pages/ and generate manifest.json
set -e

mkdir -p pages
rm -f pages/*.html pages/manifest.json

manifest="["
first=1

for nb in notebooks/*.ipynb; do
  [ -e "$nb" ] || { echo "No .ipynb files in notebooks/"; exit 1; }
  base=$(basename "$nb" .ipynb)
  jupyter nbconvert --to html --template basic --output-dir pages "$nb"
  title=$(echo "$base" | sed 's/[_-]/ /g')
  if [ $first -eq 0 ]; then manifest+=","; fi
  manifest+="{\"title\":\"$title\",\"file\":\"${base}.html\"}"
  first=0
done

manifest+="]"
echo "$manifest" > pages/manifest.json
echo "Done. Manifest:"
cat pages/manifest.json
