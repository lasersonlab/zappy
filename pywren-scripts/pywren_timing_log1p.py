import s3fs.mapping
import zappy.base as np
import zappy.executor

s3 = s3fs.S3FileSystem()
# using sc-tom-test-data/10x.zarr/X fails because the chunks sizes are too large for lambda memory
input_zarr = s3fs.mapping.S3Map("sc-tom-test-data/10x/anndata_zarr_2000/10x.zarr/X", s3=s3)
output_zarr = s3fs.mapping.S3Map("sc-tom-test-data/10x-log1p.zarr", s3=s3)

executor = zappy.executor.PywrenExecutor()
x = zappy.executor.from_zarr(executor, input_zarr)

out = np.log1p(x)
out.to_zarr(output_zarr, x.chunks)
