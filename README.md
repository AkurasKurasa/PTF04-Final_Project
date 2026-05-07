# ML Notebook Portfolio

Self-hosted notebook-style site with page-flip animation between projects.

## Setup

1. Install nbconvert (one time):
   ```
   pip install nbconvert jupyter
   ```

2. Drop `.ipynb` files into `notebooks/`. File name = page title (underscores become spaces).

3. Convert notebooks to HTML pages:
   - Windows: `powershell -ExecutionPolicy Bypass -File convert.ps1`
   - macOS/Linux/Git Bash: `bash convert.sh`

4. Serve site:
   ```
   npx serve .
   ```
   or any static server. Open the URL it prints.

## Files

- `index.html` — page shell
- `style.css` — book styling
- `app.js` — loads `pages/manifest.json` and wires up `page-flip`
- `convert.ps1` / `convert.sh` — runs `jupyter nbconvert` over `notebooks/`
- `notebooks/` — drop `.ipynb` files here
- `pages/` — generated HTML + manifest

## Controls

- Drag page corner
- Arrow keys (Left/Right)
- Prev/Next buttons
