# tapestries

A static viewer for [felt](https://github.com/cailmdaley/felt) tapestry DAGs. Renders fibers, dependencies, evidence, and staleness as a navigable graph with a detail sidebar.

This repo is a GitHub template. Clone it, export your tapestry data with `felt tapestry export`, and serve it locally or via GitHub Pages.

## Quick Start

```bash
git clone --depth 1 https://github.com/cailmdaley/tapestries.git
cd tapestries
npx serve -s .
```

Open `http://localhost:3000` — the demo tapestry loads automatically.

## Adding Your Own Data

Use `felt tapestry export` to generate tapestry data from any project with `.felt/` fibers:

```bash
# In your project directory:
felt tapestry export
```

This writes `tapestry.json` and artifact images to `~/.felt/tapestries/data/{project}/`. Point that path at your clone of this repo (or set `--out` directly):

```bash
felt tapestry export --out /path/to/tapestries/data/myproject
```

The viewer reads `data/manifest.json` to discover available tapestries. `felt tapestry export` updates it automatically.

### Data Layout

```
data/
  manifest.json               # list of tapestries
  myproject/
    tapestry.json              # DAG: nodes, links, evidence, staleness
    tapestry/                  # artifact images
      specname/
        figure.png
    files/                     # linked files (optional)
```

## GitHub Pages

To serve your tapestries publicly:

1. Fork or clone this template into a new repo
2. Enable GitHub Pages in Settings → Pages → Source: **Deploy from a branch**, branch: `main`, folder: `/ (root)`
3. Export your data and push:

```bash
felt tapestry export --out /path/to/your-repo/data/myproject
cd /path/to/your-repo
git add -A && git commit -m "Update tapestry" && git push
```

Your tapestries will be at `https://<user>.github.io/<repo>/`.

### Keeping the repo small

Artifact images accumulate. To cap history size, you can periodically squash:

```bash
KEEP=5
COUNT=$(git rev-list --count HEAD)
if [ "$COUNT" -gt "$KEEP" ]; then
  CUTPOINT=$(git rev-list HEAD | sed -n "${KEEP}p")
  REMOTE_URL=$(git remote get-url origin)
  git replace --graft "$CUTPOINT"
  git filter-repo --force --quiet
  git remote add origin "$REMOTE_URL"
  git push --force --set-upstream origin main
fi
```

## Viewer Features

- **Fog + reveal**: nodes start fogged. Click to reveal a node and its 1-hop neighborhood. Click again to collapse.
- **Staleness**: nodes are colored by evidence freshness (green = fresh, red = stale, grey = no evidence).
- **Detail sidebar**: click a node to see its body, outcome, evidence metrics, and artifact images.
- **Fiber sidebar**: all fibers (not just tapestry-tagged) listed for context.
- **Shareable URLs**: clicking a node sets `#fiber-id` in the URL. Opening that URL auto-selects the node.

## Development

The viewer is built with Vite from the [portolan](https://github.com/cailmdaley/portolan) source. To rebuild:

```bash
cd /path/to/portolan
npm run build:static
```

This outputs `index.html`, `assets/`, `fonts/`, and `favicon.svg` into `docs/` (this repo). The built files are committed directly; no build step is needed to serve.

## License

[MIT](LICENSE)
