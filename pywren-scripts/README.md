Using the virtual env from the top-level, change to this directory.

Install packages:

```bash
pip install matplotlib pandas pywren seaborn s3fs
```

Run a test for Zappy on PyWren

```bash
python pywren_timing.py
```

If you log into the AWS console you should see the cloud invocation in CloudWatch.
Job information is written to `~/.zappy/logs`, and can be plotted with

```bash
python analyze.py
``` 

Try the same on a large dataset:

```bash
python pywren_timing_log1p.py
```
