"""Select a stratified, read-only pool of library frames for RTMDet teacher pseudo-labelling.

    python -m training.teacher.select_pseudo_label_pool --out data/pseudo_rtmdet_v0/pool.json \
        --exclude-cohort ../image-scoring-backend/docs/reports/detector-benchmark-2026-09/cohort.csv

Strata (the gaps the teacher is meant to fill):
  miss_birdkw  bird_detect_v0 found no box, but the image carries the `birds` keyword
  small_det    bird_detect_v0 box covers < 4% of the frame
  negative     no `birds` / `wildlife` / `animals` keyword (hard negatives; the student has seen none)
  random       everything else
Every image of the #377 benchmark cohort, and every folder it draws from, is excluded so that benchmark
stays a clean test. At most --per-folder frames per folder and stratum.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import psycopg2

SIZES = {"miss_birdkw": 800, "small_det": 400, "negative": 500, "random": 300}
KW = "(SELECT array_agg(k.keyword_norm) FROM image_keywords ik JOIN keywords_dim k USING (keyword_id) WHERE ik.image_id = i.id)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-cohort", required=True)
    ap.add_argument("--per-folder", type=int, default=5)
    ap.add_argument("--seed", default="teacher-v0")
    ap.add_argument("--dsn", default="host=127.0.0.1 port=5432 dbname=image_scoring user=postgres password=postgres")
    a = ap.parse_args()
    cohort = list(csv.DictReader(open(a.exclude_cohort)))
    ex_ids = [int(r["image_id"]) for r in cohort]
    ex_folders = sorted({int(r["folder_id"]) for r in cohort if r["folder_id"]})
    con = psycopg2.connect(a.dsn)
    con.set_session(readonly=True)
    q = con.cursor()
    base = f"""
      WITH c AS (
        SELECT i.id, i.file_path, i.folder_id, i.bird_bbox, {KW} AS kws
        FROM images i
        WHERE lower(i.file_path) LIKE '%%.nef' AND i.folder_id IS NOT NULL
          AND NOT (i.id = ANY(%s)) AND NOT (i.folder_id = ANY(%s))
      ), s AS (
        SELECT *, CASE
          WHEN (bird_bbox->>'detected') = 'false' AND 'birds' = ANY(kws) THEN 'miss_birdkw'
          WHEN (bird_bbox->>'area_frac')::float < 0.04 THEN 'small_det'
          WHEN kws IS NULL OR NOT (kws && ARRAY['birds','wildlife','animals']::varchar[]) THEN 'negative'
          ELSE 'random' END AS stratum
        FROM c
      ), r AS (
        SELECT *, row_number() OVER (PARTITION BY stratum, folder_id ORDER BY md5(id::text || %s)) AS rn
        FROM s
      )
      SELECT id, file_path, folder_id, stratum, kws FROM r WHERE rn <= %s
      ORDER BY md5(id::text || %s)"""
    q.execute(base, (ex_ids, ex_folders, a.seed, a.per_folder, a.seed))
    pool, counts = [], {k: 0 for k in SIZES}
    for iid, path, folder, stratum, kws in q.fetchall():
        if counts[stratum] >= SIZES[stratum]:
            continue
        counts[stratum] += 1
        kws = kws or []
        pool.append(dict(image_id=iid, path=path, folder_id=folder, stratum=stratum,
                         birds_kw="birds" in kws, animal_kw=bool({"birds", "wildlife", "animals"} & set(kws))))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(counts=counts, excluded_cohort_images=len(ex_ids),
                                   excluded_cohort_folders=len(ex_folders), seed=a.seed, pool=pool), indent=1))
    print(json.dumps(counts), "excluded folders:", len(ex_folders))


if __name__ == "__main__":
    main()
