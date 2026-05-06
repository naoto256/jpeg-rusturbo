# Attribution: image crate

This crate (`jpeg-rusturbo`) consults the `image` crate
(https://github.com/image-rs/image, licensed MIT OR Apache-2.0) as
a public reference implementation of a baseline JPEG encoder. No
verbatim source from `image` was copied into this crate, but the
overall architecture (marker emission order, JFIF APP0 conventions,
component IDs, MCU iteration order, AAN-DCT folded into quantization)
is conventional baseline JPEG and matches `image`'s approach as well
as libjpeg/libjpeg-turbo's.

Algorithmic elements that are part of the JPEG standard (ITU-T T.81
Annex K — quantization and Huffman tables, zig-zag scan order,
canonical Huffman expansion) are reproduced directly from the spec.

The `image` crate's license texts are included alongside this notice
for reference and to satisfy attribution requirements for any future
incorporation of `image`-derived code in this crate.
