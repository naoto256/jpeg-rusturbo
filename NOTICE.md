# Third-party attribution

`jpeg-rusturbo` is licensed under MIT OR Apache-2.0 at your option (see
[LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE)). The
attributions below cover upstream code we have translated or
referenced.

---

## libjpeg-turbo

The four hot kernels under `src/arch/neon.rs` and `src/arch/x86_64.rs`
(color conversion, 4:2:0 chroma downsample, integer LL&M forward DCT,
reciprocal-multiply quantize) are translated from libjpeg-turbo's
`simd/arm/jc{color,sample}-neon.c`, `jfdctint-neon.c`, `jquanti-neon.c`
and `simd/x86_64/{jccolor,jcsample,jfdctint,jquanti}-avx2.asm`. The
`compute_reciprocal` formulation in `src/quant.rs` is a port of
libjpeg-turbo's `jcdctmgr.c::compute_reciprocal`. See those files'
header comments for the upstream filename in each case.

This software is based in part on the work of the Independent JPEG
Group.

The Modified (3-clause) BSD License covering libjpeg-turbo's SIMD code
is reproduced below verbatim:

> Copyright (C) 2009-2026 D. R. Commander
> Copyright (C) 2018-2023 Randy <randy408@protonmail.com>
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions
> are met:
>
> - Redistributions of source code must retain the above copyright
>   notice, this list of conditions and the following disclaimer.
> - Redistributions in binary form must reproduce the above copyright
>   notice, this list of conditions and the following disclaimer in
>   the documentation and/or other materials provided with the
>   distribution.
> - Neither the name of the libjpeg-turbo Project nor the names of its
>   contributors may be used to endorse or promote products derived
>   from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
> FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
> COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
> INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
> HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
> STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
> OF THE POSSIBILITY OF SUCH DAMAGE.

The full upstream license document is at
<https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md>.

---

## image (image-rs)

`jpeg-rusturbo` mirrors the public encoder API shape of
`image::codecs::jpeg::JpegEncoder` so call sites can swap with a `use`
change. The `image` crate (<https://github.com/image-rs/image>,
licensed MIT OR Apache-2.0) was consulted as a reference
implementation; no verbatim source from `image` was copied. JPEG-spec
elements (ITU-T T.81 Annex K — quantization and Huffman tables,
zig-zag scan order, canonical Huffman expansion) are reproduced
directly from the spec.
