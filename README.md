# Tapestries

Public, static Vellum publications for felt fibers.

This repo is the GitHub Pages shell for published fibers: shared Vellum
assets live under `/_vellum/`, and each publication gets its own subdirectory
with `content/`, `fiber-graph.json`, `search-index.json`, and a tiny
`index.html` that boots the reader.

Current publications:

- [`vellum-on-vellum/`](./vellum-on-vellum/) — a self-referential tour of Vellum,
  rendered by Vellum.
- [`constitution-tapestries-publish/`](./constitution-tapestries-publish/) — the
  constitution that landed this publishing pipeline.

## Publishing workflow

Publishing stays user-owned shell; `vellum-reader` bakes a self-contained
publication directory and this repo keeps the shared Pages layout.

```bash
cd ~/Documents/projects/vellum
npm run build
npm run bake -- --root ~/loom --slug <fiber-slug> --out /tmp/<fiber-slug>

# Sync shared reader assets once per update
rsync -a --delete dist-static/ ~/Documents/projects/tapestries/_vellum/

# Sync one publication
rsync -a --delete \
  --exclude '_vellum' \
  /tmp/<fiber-slug>/ ~/Documents/projects/tapestries/<fiber-slug>/

cd ~/Documents/projects/tapestries
git add -A && git commit -m "Publish <fiber-slug>" && git push
```

GitHub Pages serves `main:/`. `.nojekyll` is present so the `/_vellum/`
directory is published verbatim.
