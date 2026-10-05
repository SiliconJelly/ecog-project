# Rebuilds ECoG_Handpose.mat from the 5 .part files in this folder.
import glob, os, zlib

folder = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(folder, "ECoG_Handpose.mat.part*")))
out = os.path.join(folder, "ECoG_Handpose.mat")

crc = 0
with open(out, "wb") as f:
    for p in parts:
        data = open(p, "rb").read()
        crc = zlib.crc32(data, crc)
        f.write(data)

print("Joined", len(parts), "parts ->", out)
print("Size:", os.path.getsize(out), "bytes")
print("Checksum OK" if crc == 0xC4483A8B else "Checksum MISMATCH - something is missing")
