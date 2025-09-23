This is a work-in-progress directory to host files being hipified.

Patch HIP runtime
=================
Do this before Configure and Build.

Refer to `runtime_patches/READMD.md`.

Apply the updated headers according to the documentation.


Configure and Build
===================
```bash
./configure

make -j -C build VERBOSE=1
```

