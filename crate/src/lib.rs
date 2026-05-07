use image::GrayImage;
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
use svgtypes::{PathParser, PathSegment};
use visioncortex::PathSimplifyMode;
use vtracer::{ColorImage, ColorMode, Config, Hierarchical};
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
    corner_threshold: i32,
    path_precision: u32,
) -> Result<String, JsError> {
    let img = image::load_from_memory(bytes).map_err(|e| JsError::new(&e.to_string()))?;
    let gray: GrayImage = img.to_luma8();
    let blurred: GrayImage = gaussian_blur_f32(&gray, sigma);
    let edges: GrayImage = canny(&blurred, low_threshold, high_threshold);

    // vtracer Binary mode treats r < 128 as foreground; canny gives us white edges (255)
    // on black background, so invert before passing in.
    let width = edges.width() as usize;
    let height = edges.height() as usize;
    let pixels: Vec<u8> = edges
        .pixels()
        .flat_map(|p| {
            let v = 255 - p[0];
            [v, v, v, 255u8]
        })
        .collect();
    let color_img = ColorImage {
        pixels,
        width,
        height,
    };

    let config = Config {
        color_mode: ColorMode::Binary,
        hierarchical: Hierarchical::Cutout,
        mode: PathSimplifyMode::Spline,
        filter_speckle,
        color_precision: 6,
        layer_difference: 16,
        corner_threshold,
        length_threshold: 4.0,
        max_iterations: 10,
        splice_threshold: 45,
        path_precision: Some(path_precision),
    };

    let svg_file = vtracer::convert(color_img, config).map_err(|e| JsError::new(&e))?;
    let svg = svg_file.to_string();

    let equations = svg_to_desmos(&svg)?;

    serde_json::to_string(&equations).map_err(|e| JsError::new(&e.to_string()))
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

fn svg_to_desmos(svg: &str) -> Result<Vec<String>, JsError> {
    let doc = roxmltree::Document::parse(svg).map_err(|e| JsError::new(&e.to_string()))?;

    let mut equations = Vec::new();

    for node in doc.descendants() {
        if node.tag_name().name() == "path" {
            if let Some(d) = node.attribute("d") {
                // vtracer normalizes each path so its first point is (0,0) and stores the
                // real position in transform="translate(tx,ty)". Apply that offset here.
                let t = node
                    .attribute("transform")
                    .map(parse_translate)
                    .unwrap_or((0.0, 0.0));
                path_to_desmos(d, t, &mut equations);
            }
        }
    }

    Ok(equations)
}

fn parse_translate(transform: &str) -> (f64, f64) {
    let Some(start) = transform.find("translate(") else {
        return (0.0, 0.0);
    };
    let after = &transform[start + "translate(".len()..];
    let Some(end) = after.find(')') else {
        return (0.0, 0.0);
    };
    let inner = &after[..end];
    let parts: Vec<f64> = inner
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    match parts.len() {
        0 => (0.0, 0.0),
        1 => (parts[0], 0.0),
        _ => (parts[0], parts[1]),
    }
}

fn path_to_desmos(d: &str, t: (f64, f64), eqs: &mut Vec<String>) {
    let mut cur = (0.0f64, 0.0f64);
    let mut start = (0.0f64, 0.0f64);
    // Tracks the last cubic bezier c2 for smooth curve (S) reflection
    let mut prev_c2: Option<(f64, f64)> = None;

    for seg in PathParser::from(d) {
        let Ok(seg) = seg else { continue };

        match seg {
            PathSegment::MoveTo { abs, x, y } => {
                cur = abs_pt(abs, cur, x, y, t);
                start = cur;
                prev_c2 = None;
            }
            PathSegment::LineTo { abs, x, y } => {
                let p = abs_pt(abs, cur, x, y, t);
                eqs.push(line_eq(cur, p));
                cur = p;
                prev_c2 = None;
            }
            PathSegment::HorizontalLineTo { abs, x } => {
                let p = if abs { (x + t.0, cur.1) } else { (cur.0 + x, cur.1) };
                eqs.push(line_eq(cur, p));
                cur = p;
                prev_c2 = None;
            }
            PathSegment::VerticalLineTo { abs, y } => {
                let p = if abs { (cur.0, y + t.1) } else { (cur.0, cur.1 + y) };
                eqs.push(line_eq(cur, p));
                cur = p;
                prev_c2 = None;
            }
            PathSegment::CurveTo {
                abs,
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => {
                let c1 = abs_pt(abs, cur, x1, y1, t);
                let c2 = abs_pt(abs, cur, x2, y2, t);
                let p = abs_pt(abs, cur, x, y, t);
                eqs.push(cubic_eq(cur, c1, c2, p));
                prev_c2 = Some(c2);
                cur = p;
            }
            PathSegment::SmoothCurveTo { abs, x2, y2, x, y } => {
                // Reflect prev c2 through cur; fall back to cur if no prior cubic
                let c1 = prev_c2.map_or(cur, |pc2| (2.0 * cur.0 - pc2.0, 2.0 * cur.1 - pc2.1));
                let c2 = abs_pt(abs, cur, x2, y2, t);
                let p = abs_pt(abs, cur, x, y, t);
                eqs.push(cubic_eq(cur, c1, c2, p));
                prev_c2 = Some(c2);
                cur = p;
            }
            PathSegment::ClosePath { .. } => {
                if (cur.0 - start.0).abs() > 1e-6 || (cur.1 - start.1).abs() > 1e-6 {
                    eqs.push(line_eq(cur, start));
                }
                cur = start;
                prev_c2 = None;
            }
            _ => {
                prev_c2 = None;
            }
        }
    }
}

#[inline]
fn abs_pt(abs: bool, cur: (f64, f64), x: f64, y: f64, t: (f64, f64)) -> (f64, f64) {
    if abs {
        (x + t.0, y + t.1)
    } else {
        (cur.0 + x, cur.1 + y)
    }
}

fn n(v: f64) -> String {
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0');
    s.trim_end_matches('.').to_string()
}

fn line_eq(p0: (f64, f64), p1: (f64, f64)) -> String {
    format!(
        "((1-t)*{}+t*{},(1-t)*{}+t*{})",
        n(p0.0),
        n(p1.0),
        n(-p0.1),
        n(-p1.1)
    )
}

fn cubic_eq(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)) -> String {
    format!(
        "((1-t)^3*{}+3*t*(1-t)^2*{}+3*t^2*(1-t)*{}+t^3*{},(1-t)^3*{}+3*t*(1-t)^2*{}+3*t^2*(1-t)*{}+t^3*{})",
        n(p0.0), n(p1.0), n(p2.0), n(p3.0),
        n(-p0.1), n(-p1.1), n(-p2.1), n(-p3.1)
    )
}
