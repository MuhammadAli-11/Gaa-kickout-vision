# 01 — Data collection protocol

This is the document a supervisor reads to decide whether your results
mean anything. Fill in every "RECORD:" line as you go; do not
reconstruct it afterwards from memory.

---

## 1. What footage is needed

| Requirement | Target | Minimum | Why |
|---|---|---|---|
| Total duration | 10 min | 5 min | Enough restarts to matter |
| Kickout contests | 15–20 | 12 | Below 12, every CI is uselessly wide |
| Camera type | Broadcast wide + cuts | — | The limitation being characterised *is* broadcast direction |
| Resolution | ≥ 1280×720 | 720×576 | Far-side players are already marginal at 720p |
| Frame rate | 25 or 50 fps | 25 fps | Irish/UK broadcast standard |
| Distinct passages | ≥ 2 halves or 2 matches | 1 | Guards against a single-camera-angle artefact |

**Density trick:** kickouts follow every score and every wide. A passage
with frequent scoring gives more restarts per minute than a defensive
grind. Choose the busy passage deliberately, then say in the write-up
that you did, because it biases the event rate upward.

**A second, contrasting clip is worth more than a longer first clip.**
Five minutes from a second match — different broadcaster, different
weather, different pitch — turns "it worked on my clip" into "performance
varied between these two conditions", which is a result rather than an
anecdote. If time is short, take 7 min + 3 min rather than 10 min from one game.

---

## 2. Sourcing and rights

**Rule for this project: footage stays on your machine.** No video, no
extracted frames, no overlay renders in the public repo. `.gitignore`
enforces this; verify with `git status` before every push.

Acceptable sources for a personal methods-development project:

1. Club or county footage you have permission to use. **Best option** —
   get one line of written permission by email and file it.
2. Publicly broadcast match footage recorded personally, used locally,
   never redistributed.
3. Freely licensed match footage where the licence permits analysis.

**RECORD:** for each file — source, date obtained, permission status, and
whether any person appearing is identifiable at the resolution used.

Wording for the README (adjust to your actual situation, do not copy blind):

> Footage was used locally for method development and is not redistributed
> with this repository. No video, frames, or rendered overlays are committed.
> Deployment beyond method development would require a data agreement with
> the rights holder.

**GDPR note.** Match footage of identifiable players is personal data.
For a personal, non-published methods project on already-broadcast
material the risk is low, but say the following in the write-up and mean
it: derived artefacts contain bounding-box coordinates and track indices
only, no identities, no biometric templates, and no attempt at
re-identification across clips. If you later publish, this becomes a
formal ethics question and TU Dublin's research ethics process is the
route — knowing that, and saying so unprompted, is exactly the instinct
the supervisor is hiring for.

---

## 3. Directory layout for footage

```
data/
├── raw/
│   ├── video_registry.csv          # THE index. Never edit by hand.
│   └── <video_id>.mp4              # untouched source, read-only after ingest
├── clips/
│   └── <video_id>_working.mp4      # standardised 25 fps 1280x720 segment
├── frames/
│   └── <video_id>/000000.jpg ...   # 6-digit, 0-based, contiguous
├── interim/
│   └── <video_id>/manifest.csv     # frame_idx -> timestamp_s -> path
└── gt/
    └── <video_id>/
        ├── gt_events.csv           # manual kickout coding (passes 1 and 2)
        ├── gt_boxes.txt            # MOT16 box annotation on the frame subset
        └── annotation_notes.md     # edge cases and how you resolved them
```

`video_id` convention: `<comp>_<yy>_<round>_<half>`, e.g. `nfl25_r3_h2`.
Lowercase, no spaces, stable forever. It appears in every downstream path.

---

## 4. Ingestion procedure

```bash
python src/s00_ingest.py \
  --file /path/to/source.mp4 --video-id nfl25_r3_h2 \
  --competition "Allianz NFL Div 1" --round "Round 3" \
  --match-date 2025-02-16 --broadcaster "TG4" \
  --capture-source broadcast --rights personal_local_only \
  --notes "second half, dry, floodlit"
```

This computes a SHA-256 of the file, probes the true codec/fps/resolution
with `ffprobe`, and appends a row to `video_registry.csv`. Nothing else in
the pipeline will touch a file that has no registry row.

**Why hash it?** Because in six weeks you will have three files called
`match_final.mp4` and no way to know which produced Table 3. The hash makes
that question answerable, and being able to answer it is what "data
architecture" means in practice.

---

## 5. Standardisation

```bash
python src/s01_prepare_footage.py --video-id nfl25_r3_h2
```

Fixes the working segment at constant 25 fps, 1280×720, and extracts
frames with 0-based contiguous indices, so that

```
timestamp_s == frame_idx / 25
```

holds exactly, by construction.

**This is the single most common place a sports-video project silently
goes wrong.** Broadcast files are frequently variable-frame-rate;
`-c copy` trims land on the nearest keyframe, not the requested time; and
`cv2.CAP_PROP_POS_MSEC` returns container timestamps that drift. The
pipeline therefore re-encodes rather than stream-copies, and never reads a
timestamp back out of the container. Say this out loud in the interview —
it is a two-sentence answer that demonstrates you have actually processed
video rather than read about it.

**RECORD:** source fps and whether it differed from 25. If the source was
50 fps and you decimated to 25, jump peaks are sampled half as often and
your temporal offsets inherit that resolution; note it in limitations.

---

## 6. Pre-flight checklist

Before annotating anything, confirm:

- [ ] `video_registry.csv` has a row with a real SHA-256
- [ ] Working clip plays, is the expected duration, has no audio track
- [ ] Frame count equals `duration × 25` (± 1)
- [ ] Frame `000000.jpg` is the first frame of the intended segment
- [ ] Spot-check: frame 1500 shows what is happening at 60.0 s in the clip
- [ ] `git status` shows **no** video, frames, or parquet staged
- [ ] You have counted the kickouts by eye and there are at least 12

The frame-1500 check takes forty seconds and catches an off-by-one that
would otherwise contaminate every temporal metric in the project.
