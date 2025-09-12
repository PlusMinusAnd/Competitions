import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.dataset import Scanner
import gc

dset = ds.dataset("./Project/Toss/train.parquet", format="parquet")
scanner = dset.scanner(["clicked"], batch_size=100000)

print("[train load]")

for rb in scanner.to_batches():
    train_table = pa.Table.from_batches([rb])
