# smallpaint_rust_port
Károly Zsolnai-Fehér's smallpaint ported to Rust

This is purely a learning project. I want to learn more about global illumination perhaps try my hand at implementing it after watching the [YouTube series on ray tracing](https://www.youtube.com/playlist?list=PLujxSBD-JXgnGmsn7gEyN28P1DnRZG7qi)uploaded by Two-minute Papers.

## Example Images

Check the `/examples` folder, or view the example images in the project wiki here: https://github.com/garfieldnate/smallpaint_rust_port/wiki/Example-Images

## Requirements

Just one: [Rust](https://www.rust-lang.org/tools/install)

## Building/Running

Only the "painterly" version of smallpaint is implemented (for now). The command prints PPM text data directly to stdout, so it should be redirected to a file and opened with a viewer:

    cargo run --release -- painterly > out.ppm && open out.ppm

Note that `open xyz.ppm` works fine on Mac OSX; you may need to use a different PPM viewer on your machine.

Add a `--help` after the `painterly` to see the available options.
