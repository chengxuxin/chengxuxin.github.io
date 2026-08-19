# chengxuxin.github.io

Personal academic homepage ([xuxin.ai](https://xuxin.ai)), built with Jekyll and deployed via GitHub Pages.

## Structure

```
_config.yml           Site config (name, collections)
_layouts/default.html Page shell: head, sticky nav, footer, hover-video JS
_includes/publication.html  One publication card (used by the homepage loop)
_pages/index.html     Homepage: hero, highlights, news, publications
_publications/        One file per paper (see format below)
_data/news.yml        News items
_data/highlights.yml  Featured videos on the homepage
_pages/research/      Media assets per paper (thumbnail, video, bib.txt)
style.scss            Entry point importing _sass/_base.scss (all layout + design tokens)
videos/               Highlight videos
```

## Add a paper

1. Drop the assets into `_pages/research/<abbr>/` — a `demo.jpg` thumbnail, optionally a `demo.mp4` (plays on hover) or `demo.gif`, and a `bib.txt`.
2. Create `_publications/YYYY-MM-DD-<abbr>.md` (the date prefix controls ordering — newest first):

```yaml
---
abbr: "parkour"
title: "Extreme Parkour with Legged Robots"
authors: "Xuxin Cheng*, Kexin Shi*, Ananye Agarwal, Deepak Pathak"
venue: "ICRA 2024"
award: "Oral"                      # optional, shown highlighted
note: "Also at CoRL 2023 Workshop" # optional, extra line under venue
image: "/_pages/research/parkour/demo.jpg"
video: "/_pages/research/parkour/demo.mp4"   # optional (or gif:)
website: "https://extreme-parkour.github.io"
links:
  - name: "arXiv"
    url: "https://arxiv.org/abs/2309.14341"
  - name: "Code"
    url: "https://github.com/chengxuxin/extreme-parkour"
bib: "/_pages/research/parkour/bib.txt"      # optional, adds a BibTeX expander
media:                                       # optional press links
  - name: "MIT Tech Review"
    url: "https://example.com"
---
```

Your name is bolded automatically in author lists (`name_bold` in `_config.yml`).

## Add news / highlights

Append an entry to `_data/news.yml` or `_data/highlights.yml` — the format is documented at the top of each file.

## Video guidelines

Videos should be H.264 / yuv420p, square pixels, with the moov atom up front,
or they may show up black or letterboxed in browsers. Normalize any new video with:

```
ffmpeg -i in.mp4 -c:v libx264 -crf 21 -vf "setsar=1" -pix_fmt yuv420p -movflags +faststart -an out.mp4
```

## Run locally

```
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>. GitHub Pages builds the site automatically on push to `master`.
