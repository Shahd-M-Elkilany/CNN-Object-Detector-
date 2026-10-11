# Data

## Source

The *Airplanes* aerial-imagery set distributed with [1297rohit/RCNN](https://github.com/1297rohit/RCNN)
(Apache-2.0): 733 JPEG images and one annotation file per image. `airdet download` fetches the two archives at
commit `6c71fdc` and verifies their SHA-256 checksums before unpacking into `data/raw/`.

Annotation format: the first line is the number of boxes, then one `x1 y1 x2 y2` box per line, in pixels.

## Cleaning

Applied in `airplane_detector.data.load_dataset`:

| Rule | Affected | Reason |
|---|---|---|
| Drop `Lhasa` | 1 image | 2637×4800 scene with an empty annotation file although airplanes are visible; keeping it would teach the model that those airplanes are background |
| Drop `hotanairport5-rgb` | 1 image, 13 boxes | Byte-identical copy of `hotanairport5`; a duplicate could land on both sides of the split |
| Clip boxes to the image | 9 images | Boxes extend one pixel past the border |
| Drop boxes with a side under 3 px after clipping | 0 boxes | Safeguard; nothing in this dataset triggers it |

Result: **731 images, 2,893 airplanes**. 717 images are 256×256 tiles; the other 14 are larger scenes of up to
428×1061 pixels. Three images contain no airplane and are kept as pure negatives.

A search for near-duplicates (correlation of 16×16 grayscale thumbnails above 0.85) found only the pair listed
above.

## Object sizes

Size is √(box area) in pixels.

| Percentile | 1 % | 5 % | 25 % | 50 % | 75 % | 95 % | 99 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Size (px) | 9 | 15 | 23 | 34 | 51 | 92 | 116 |

## Splits

Images are split **by image**, 70 / 15 / 15, stratified by the number of airplanes per image (buckets 0, 1,
2–3, 4–9, 10–29, 30+) with seed 42. The split is stored in [`data/splits.json`](../data/splits.json) and is
read, not regenerated, by every command.

| Split | Images | Airplanes | Small < 20 px | Medium 20–45 px | Large ≥ 45 px | Most airplanes in one image |
|---|---:|---:|---:|---:|---:|---:|
| train | 511 | 1,877 | 295 | 1,002 | 580 | 72 |
| val | 110 | 456 | 131 | 185 | 140 | 105 |
| test | 110 | 560 | 75 | 315 | 170 | 206 |

The test split contains the densest scene in the dataset (`Planes6`, 206 airplanes). Because metrics are pooled
over objects, that single image carries 37 % of the test weight; the README therefore also reports results
without it.

## Known issues

- Some visible airplanes are not annotated, and a few boxes appear to cover a group of aircraft.
- The images come from a small number of airports, and neighbouring tiles of one airport can fall in different
  splits. Test performance therefore measures generalisation to unseen tiles, not to unseen airports.
- All annotations are axis-aligned boxes; airplane orientation is not labelled.
