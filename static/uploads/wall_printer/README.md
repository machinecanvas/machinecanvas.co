# Wall printer demo

- `countryside_window.jpg` – AI-generated trompe-l'oeil window onto English countryside (the mural).
- `wall_printer_animation.mp4` – coded animation of a vertical wall printer printing the mural.
  The head runs up and down each strip while the machine steps left to right; the image only
  appears where the head has passed. Regenerate with:

      python scripts/wall_printer_animation.py static/uploads/wall_printer/countryside_window.jpg \
          static/uploads/wall_printer/wall_printer_animation.mp4

  Options: `--swath` (px per pass), `--pass-frames`, `--step-frames`, `--hold-frames`, `--fps`.
  Needs Pillow, numpy and ffmpeg.
- `wall_printer_ai_veo.mp4` – first AI-generated (Veo 3.1) photoreal take.
