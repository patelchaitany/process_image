# Camera zone test page

A web page for trying the GPU pipeline with real cameras. It shows each camera with the boxes, track IDs, names and
zones the pipeline returns, and lets you draw zones on the picture.

## Run it

On the Mac, with the laptop switched on and on the same network:

```bash
./start.sh
```

It copies the page and the bridge to the laptop, starts the bridge inside WSL next to `fastpipe`, tunnels it through
ssh and opens `http://localhost:8765` (use Chrome). Ctrl-C in that terminal stops everything.

`start.sh` needs the GPU machine's address and ssh user: `LAPTOP=<address> LAPTOP_USER=<user> ./start.sh`, or put
those two lines in `live/local.env` once (the file is not in git).

The recogniser is EdgeFace-S. To try another engine: `FP_REC=rec_r50_fp16.plan ./start.sh`. The match threshold
follows the engine.

On the page:

1. **Allow cameras**, then pick a source for Camera 1 and Camera 2 (a Mac camera, a video file, or a test clip stored
   on the laptop).
2. **Draw zone**: click each corner on the video; double-click, press Enter or click the first corner to finish.
   Zones can be drawn before or while the pipeline runs, renamed and deleted. They are remembered per camera slot.
3. **Start pipeline**. After about 6 seconds boxes appear. Each box shows `#track-id name · zone`.
   **Faces are** (top bar) tells the detector how big faces are. *Auto* picks close-up for a Mac camera and far away
   for a clip or file. The detector misses faces wider than about 400 px of the 720p picture, so close-up shrinks
   the picture by half for the detector only; names are still read from the full picture. Close-up misses small,
   distant faces (15 of 22 found on the CCTV test clip).
4. Click a face to give it a name. The name then shows on both cameras.

## How it works

```
Mac browser                              laptop (WSL)                         GPU
camera -> 1280x720 -> H.264 encoder --WebSocket over ssh--> bridge.py --pipe--> fastpipe --live
  keeps each sent frame                                                         NVDEC -> detect -> track -> zone
  draws frame N with result N  <------------- JSON line per frame <------------ -> recognise -> names
```

- Every frame carries a number. The pipeline returns one result line per frame with the same number, so the page
  draws each frame with exactly the boxes computed for it. The video is shown a fraction of a second late (the page
  prints how much) instead of live with trailing boxes.
- The zone of a box is the zone under the **centre of the face box**. It is looked up on the GPU (kernel K7) for
  every delivered box, detected or Kalman-filled. Where zones overlap, the one drawn last wins.
- Faces are detected on 10 of the 30 frames per second; the Kalman filter fills the other 20.
- A name is stored by copying that face's embedding into the gallery on the GPU. Names last until **Stop**.

## Result line

```json
{"cam":0,"seq":123,"ms":21.4,"det":1,"b":[[x1,y1,x2,y2,track,recognised,label,zone], ...]}
```

`ms` is time inside the pipeline (frame arrived to result ready). `det` is 1 for a detected frame, 0 for a filled one.
`recognised` is 0 until the face has been through recognition; `label` is -1 for unknown; `zone` is 0 for no zone.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `start.sh` | Mac | Copy, start and tunnel |
| `index.html` | browser | The page |
| `bridge.py` | laptop (WSL) | WebSocket to pipes; Python standard library only |
| `wsl_run.sh` | laptop (WSL) | Stops an old session, starts the bridge |
| `local.env` | Mac | Your `LAPTOP` and `LAPTOP_USER` (not in git) |
| `laptop_known_hosts` | Mac | Optional, not in git: the GPU machine's ssh host key, so the Mac's own known-hosts file is not touched |

## Debugging

`FP_LIVE_DEBUG=1` in fastpipe's environment prints, every 10th batch, what the detector found and how many boxes were
delivered. Two test helpers are in `../../fastpipe-dev/tools/live/`: `test_client.mjs` checks the whole path without a
browser (`node test_client.mjs clip.h264 12 2`), and `livefeed.py` (run inside WSL) feeds an H.264 file straight into
`fastpipe --live`, with no bridge or browser.

## Limits

- Tested with recorded clips sent through the page and through `test_client.mjs`. Real cameras use the same path but
  were not tested here.
- Needs a browser with WebCodecs H.264 encoding (Chrome on macOS).
- All cameras are sent as 1280x720 at up to 30 fps; other shapes are letterboxed.
- Up to 4 cameras in the bridge; the page shows 2.
- The bridge listens only on the laptop's loopback address and has no login. It is reachable only through the ssh
  tunnel.
