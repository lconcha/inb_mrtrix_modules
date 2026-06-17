# inb_mrtrix_modules

This is a repository for tools used at the Institute of Neurobiology, UNAM, Mexico.

## Building

This project links against the MRtrix3 core library checked out alongside it at `../mrtrix3`.

For day-to-day builds:

```
./configure_and_build
```

This just runs `./build` (the existing build script, symlinked from `../mrtrix3/build`).

If system libraries have changed (e.g. an apt upgrade removed/replaced libpng12 with libpng16,
or Qt was reinstalled) and `../mrtrix3/config` needs to be regenerated, run:

```
./configure_and_build --reconfigure
```

This reruns `../mrtrix3/configure` before building. It also makes sure `/usr/lib/qt5/bin`
(where `qmake` lives on this machine) is on `PATH` first — non-interactive shells don't pick
that up from `.bashrc`, which otherwise makes `./configure` fail with "Qt qmake not found!"
even though a plain `./build` works fine afterwards (it reuses the PATH that was saved inside
`config` the last time `configure` succeeded).

Any extra arguments are forwarded to `./build`, e.g. `./configure_and_build -verbose`.