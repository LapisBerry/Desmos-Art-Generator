# Desmos-Art-Generator

A single webpage that turn any image into desmos graph.

It uses RUST webassembly to achive the best performance possible.

The library I'm currently using is `vtracer`. This library is for turning images into vector art. This seems really nice at first glance but unfortunately vector art is not the goal we want. You can see the result for yourself that the line is full of "staircase" this is because the algorithm that's being used in `vtracer` is not returning the right format for this kind of task. What we actually want is `centerline tracing` algorithm like what is `potrace` using. This algorithm will never make it a weird staircase effect like `vtracer`. Unfornuately, as of right now `potrace` isn't being ported to rust crates yet. In the meantime, I will try to fix this problem by looking out for library that provide centerline algorithm.

> to be honest, I actually found out that [img2svg](https://crates.io/crates/img2svg) has centerline option but I have never tested this library out. I will look forward to it.

