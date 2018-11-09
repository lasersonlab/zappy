import pywren
import zappy.base as np
import zappy.executor

executor = zappy.executor.PywrenExecutor()
a = zappy.executor.ones(executor, (10, 2), chunks=(2, 2), dtype="i4")

import s3fs.mapping

s3 = s3fs.S3FileSystem()
path = "sc-tom-test-data/ones.zarr"
output_zarr = s3fs.mapping.S3Map(path, s3=s3)

out = np.log1p(a)
out.to_zarr(output_zarr, a.chunks)

s3.ls(path)

s3.rm(path, recursive=True)
