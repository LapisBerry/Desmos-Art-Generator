use image::GrayImage;
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn process_image(
    bytes: &[u8],
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
    filter_speckle: usize,
    _corner_threshold: i32,
    path_precision: u32,
) -> Result<String, JsError> {
    let img = image::load_from_memory(bytes).map_err(|e| JsError::new(&e.to_string()))?;
    let gray: GrayImage = img.to_luma8();
    let blurred = gaussian_blur_f32(&gray, sigma);
    let edges = canny(&blurred, low_threshold, high_threshold);

    let polylines = trace_edges(&edges);

    // path_precision controls DP simplification: higher = finer detail (smaller epsilon).
    let epsilon = 1.0 / (path_precision.max(1) as f64);
    let min_len = filter_speckle.max(2);

    let mut eqs: Vec<String> = Vec::new();
    for poly in &polylines {
        if poly.len() < min_len {
            continue;
        }
        let pts: Vec<(f64, f64)> = poly.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
        let simplified = douglas_peucker(&pts, epsilon);
        if simplified.len() < 2 {
            continue;
        }
        for bz in catmull_rom_to_beziers(&simplified) {
            eqs.push(cubic_eq(bz[0], bz[1], bz[2], bz[3]));
        }
    }

    serde_json::to_string(&eqs).map_err(|e| JsError::new(&e.to_string()))
}

#[wasm_bindgen]
pub fn get_edge_png(
    bytes: &[u8],
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Vec<u8>, JsError> {
    let img = image::load_from_memory(bytes).map_err(|e| JsError::new(&e.to_string()))?;
    let gray: GrayImage = img.to_luma8();
    let blurred = gaussian_blur_f32(&gray, sigma);
    let edges = canny(&blurred, low_threshold, high_threshold);
    let mut buf = Vec::new();
    image::DynamicImage::from(edges)
        .write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(buf)
}

/// Walk 8-connected edge pixels into polylines (greedy chain following).
fn trace_edges(edges: &GrayImage) -> Vec<Vec<(i32, i32)>> {
    let w = edges.width() as i32;
    let h = edges.height() as i32;
    let mut visited = vec![false; (w * h) as usize];
    let mut polylines: Vec<Vec<(i32, i32)>> = Vec::new();

    const NBRS: [(i32, i32); 8] = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
    ];

    let is_edge = |x: i32, y: i32| -> bool {
        x >= 0 && y >= 0 && x < w && y < h && edges.get_pixel(x as u32, y as u32)[0] > 0
    };

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if visited[i] || !is_edge(x, y) {
                continue;
            }
            let mut poly = vec![(x, y)];
            visited[i] = true;
            let mut cur = (x, y);
            loop {
                let mut next: Option<(i32, i32, usize)> = None;
                for &(dx, dy) in &NBRS {
                    let nx = cur.0 + dx;
                    let ny = cur.1 + dy;
                    if !is_edge(nx, ny) {
                        continue;
                    }
                    let ni = (ny * w + nx) as usize;
                    if !visited[ni] {
                        next = Some((nx, ny, ni));
                        break;
                    }
                }
                match next {
                    Some((nx, ny, ni)) => {
                        visited[ni] = true;
                        poly.push((nx, ny));
                        cur = (nx, ny);
                    }
                    None => break,
                }
            }
            polylines.push(poly);
        }
    }

    polylines
}

fn douglas_peucker(points: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let start = points[0];
    let end = points[points.len() - 1];
    let mut max_dist = 0.0;
    let mut max_idx = 0;
    for i in 1..points.len() - 1 {
        let d = perp_dist(points[i], start, end);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }
    if max_dist > epsilon {
        let mut left = douglas_peucker(&points[..=max_idx], epsilon);
        let right = douglas_peucker(&points[max_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![start, end]
    }
}

fn perp_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        let px = p.0 - a.0;
        let py = p.1 - a.1;
        return (px * px + py * py).sqrt();
    }
    (dx * (a.1 - p.1) - (a.0 - p.0) * dy).abs() / len_sq.sqrt()
}

/// Uniform Catmull-Rom: turns a polyline into a sequence of cubic bezier curves
/// that smoothly pass through every input point. Endpoints are duplicated so
/// the curve starts/ends along the polyline tangent.
/// Each returned element is [start, c1, c2, end].
fn catmull_rom_to_beziers(pts: &[(f64, f64)]) -> Vec<[(f64, f64); 4]> {
    let n = pts.len();
    if n < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let p0 = if i == 0 { pts[0] } else { pts[i - 1] };
        let p1 = pts[i];
        let p2 = pts[i + 1];
        let p3 = if i + 2 >= n { pts[n - 1] } else { pts[i + 2] };

        let c1 = (p1.0 + (p2.0 - p0.0) / 6.0, p1.1 + (p2.1 - p0.1) / 6.0);
        let c2 = (p2.0 - (p3.0 - p1.0) / 6.0, p2.1 - (p3.1 - p1.1) / 6.0);
        out.push([p1, c1, c2, p2]);
    }
    out
}

fn n(v: f64) -> String {
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0');
    s.trim_end_matches('.').to_string()
}

fn cubic_eq(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)) -> String {
    format!(
        "((1-t)^3*{}+3*t*(1-t)^2*{}+3*t^2*(1-t)*{}+t^3*{},(1-t)^3*{}+3*t*(1-t)^2*{}+3*t^2*(1-t)*{}+t^3*{})",
        n(p0.0), n(p1.0), n(p2.0), n(p3.0),
        n(-p0.1), n(-p1.1), n(-p2.1), n(-p3.1)
    )
}
