import csv, shutil
from pathlib import Path

manifest = Path("/datastore/uittogether2/LuuTru/Thanhld/WSI/MergeSlide_TTA/notebooks/log/flatten_TCGA-NSCLC_svs_manifest.csv")
out = manifest.with_name("delete_TCGA-TGCT_source_dirs_manifest.csv")

dirs = sorted({
      Path(row["source"]).parent
      for row in csv.DictReader(manifest.open())
      if row.get("status") == "moved"
})

deleted, skipped = [], []
for d in dirs:
      if not d.exists():
          skipped.append((str(d), "missing"))
          continue
      if list(d.rglob("*.svs")):
          skipped.append((str(d), "svs_still_exists"))
          continue
      shutil.rmtree(d)
      deleted.append(str(d))

with out.open("w") as f:
      f.write("status,directory,reason\n")
      for d in deleted:
          f.write(f"deleted,{d},\n")
      for d, reason in skipped:
          f.write(f"skipped,{d},{reason}\n")

print("deleted_dirs", len(deleted))
print("skipped_dirs", len(skipped))
print("manifest", out)