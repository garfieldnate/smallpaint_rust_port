use rand::distributions::OpenClosed01;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, Sub};
use thread_local::CachedThreadLocal;

const EPSILON: f64 = 1e-4;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Vector3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3d {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    pub fn norm(&self) -> Vector3d {
        let magnitude = self.magnitude();
        Vector3d {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude,
        }
    }
    pub fn dot(&self, other: Vector3d) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl Display for Vector3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "vector({}, {}, {})", self.x, self.y, self.z)?;
        Ok(())
    }
}
impl Add for Vector3d {
    type Output = Vector3d;
    fn add(self, other: Vector3d) -> Vector3d {
        Vector3d::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl AddAssign for Vector3d {
    fn add_assign(&mut self, other: Vector3d) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }
}

impl Sub for Vector3d {
    type Output = Vector3d;
    fn sub(self, other: Vector3d) -> Vector3d {
        Vector3d::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f64> for Vector3d {
    type Output = Vector3d;
    fn mul(self, scalar: f64) -> Vector3d {
        Vector3d {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// Rays have origin and direction.
// The direction vector should always be normalized.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ray {
    pub o: Vector3d,
    pub d: Vector3d,
}
impl Ray {
    fn new(o: Vector3d, d: Vector3d) -> Self {
        Self { o, d: d.norm() }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MaterialType {
    DIFFUSE,
    SPECULAR,
    REFRACTIVE,
}

// Objects have color, emission, type (diffuse, specular, refractive)
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material {
    pub color: Vector3d,
    pub emission: f64,
    pub material_type: MaterialType,
}
impl Material {
    fn new(color: Vector3d, emission: f64, material_type: MaterialType) -> Self {
        Self {
            color,
            emission,
            material_type,
        }
    }
}

// All object should be intersectable and should be able to compute their surface normals.
pub trait Obj: Sync {
    fn intersect(&self, ray: &Ray) -> f64;
    fn normal(&self, point: Vector3d) -> Vector3d;
    fn get_material(&self) -> Material;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Plane {
    position: f64,
    n: Vector3d,
    material: Material,
}
impl Plane {
    fn new(position: f64, n: Vector3d, material: Material) -> Self {
        Self {
            position,
            n,
            material,
        }
    }
}

impl Obj for Plane {
    fn intersect(&self, ray: &Ray) -> f64 {
        let position_0: f64 = self.n.dot(ray.d);
        if position_0 != 0. {
            let t: f64 = -1. * (((self.n.dot(ray.o)) + self.position) / position_0);
            if t > EPSILON {
                t
            } else {
                0.
            }
        } else {
            0.
        }
    }
    fn normal(&self, _point: Vector3d) -> Vector3d {
        self.n
    }
    fn get_material(&self) -> Material {
        self.material
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sphere {
    radius: f64,
    center: Vector3d,
    material: Material,
}
impl Sphere {
    fn new(radius: f64, center: Vector3d, material: Material) -> Self {
        Self {
            radius,
            center,
            material,
        }
    }
}

impl Obj for Sphere {
    fn intersect(&self, ray: &Ray) -> f64 {
        let b: f64 = ((ray.o - self.center) * 2.).dot(ray.d);
        // TODO: reuse calculation here
        let c_: f64 = (ray.o - self.center).dot(ray.o - self.center) - (self.radius * self.radius);
        let mut disc: f64 = b * b - 4. * c_;
        if disc < 0. {
            return 0.;
        } else {
            disc = disc.sqrt();
        }
        let sol1: f64 = -b + disc;
        let sol2: f64 = -b - disc;
        if sol2 > EPSILON {
            sol2 / 2.
        } else {
            if sol1 > EPSILON {
                sol1 / 2.
            } else {
                0.
            }
        }
    }
    fn normal(&self, p0: Vector3d) -> Vector3d {
        return (p0 - self.center).norm();
    }
    fn get_material(&self) -> Material {
        self.material
    }
}

pub struct Intersection<'a> {
    t: f64,
    object: &'a dyn Obj,
}
impl<'a> Intersection<'a> {
    fn new(t: f64, object: &'a dyn Obj) -> Intersection<'a> {
        Intersection { t, object: object }
    }
}

pub struct Scene {
    objects: Vec<Box<dyn Obj>>,
}
impl Scene {
    fn new() -> Self {
        Self { objects: vec![] }
    }
    fn add(&mut self, object: Box<dyn Obj>) {
        self.objects.push(object);
    }
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        let mut closest_object: Option<&dyn Obj> = None;
        let mut closest_t = std::f64::INFINITY;
        for o in &self.objects {
            let t = o.intersect(ray);
            // ignore intersections at or behind camera
            if t > EPSILON && t < closest_t {
                closest_object = Some(o.as_ref());
                closest_t = t;
            }
        }
        match closest_object {
            Some(o) => Some(Intersection::new(closest_t, o)),
            None => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Halton {
    value: f64,
    inv_base: f64,
}

impl Halton {
    fn new(mut index: i32, base: i32) -> Halton {
        let inv_base = 1. / (base as f64);
        let mut fraction = 1.;
        let mut value = 0.;
        while index > 0 {
            fraction *= inv_base;
            value += fraction * (index % base) as f64;
            index /= base;
        }
        Halton { value, inv_base }
    }

    // TODO: I tried to make `new` clearer, but this one I don't understand at all
    fn next(&mut self) {
        let r: f64 = 1. - self.value - 1e-7;
        self.value += if self.inv_base < r {
            self.inv_base
        } else {
            let mut h: f64 = self.inv_base;
            loop {
                let hh = h;
                h *= self.inv_base;
                if h < r {
                    break hh + h - 1.;
                }
            }
        }
    }

    fn get(&self) -> f64 {
        self.value
    }
}

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    data: Vec<Vec<Vector3d>>,
}

impl Canvas {
    pub fn write_data_parallel<F>(&mut self, processor: F)
    where
        F: Fn(usize, usize, &mut Vector3d) + Sync,
    {
        self.data
            .par_iter_mut()
            .enumerate() // generate an index for each column we're iterating
            .for_each(|(col_index, row)| {
                for (row_index, pixel) in row.iter_mut().enumerate() {
                    processor(row_index, col_index, pixel);
                }
            });
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> Vector3d {
        self.data[y][x]
    }

    // scale/clamp color values from 0-1 to 0-255
    fn scale_color(&self, norm_by: u64, rgb: f64) -> u8 {
        (rgb as u64 / norm_by).min(MAX_PPM_COLOR_VAL).max(0) as u8
    }

    // If current line has no more room for more RGB values, add it to the PPM string and clear it;
    // otherwise, add a space separator in preparation for the next RGB value
    fn write_rgb_separator(&self, line: &mut String, ppm: &mut String) {
        if line.len() < MAX_PPM_LINE_LENGTH - MAX_COLOR_VAL_STR_LEN {
            (*line).push(' ');
        } else {
            ppm.push_str(&line);
            ppm.push('\n');
            line.clear();
        }
    }

    // Return string containing PPM (portable pixel map) data representing current canvas
    pub fn to_ppm(&self, samples_per_pixel: u64) -> String {
        let mut ppm = String::new();
        // write header
        ppm.push_str("P3\n");
        ppm.push_str(&(format!("{} {}\n", self.width, self.height)));
        ppm.push_str(&(format!("{}\n", MAX_PPM_COLOR_VAL)));

        // Write pixel data. Each pixel RGB value is written with a separating space or newline;
        // new rows are written on new lines for human reading convenience, but lines longer than
        // MAX_PPM_LINE_LENGTH must also be split.
        let mut current_line = String::new();
        for row in 0..self.height {
            current_line.clear();
            for (i, column) in (0..self.width).enumerate() {
                let color = self.pixel_at(column, row);
                let r = self.scale_color(samples_per_pixel, color.x);
                let g = self.scale_color(samples_per_pixel, color.y);
                let b = self.scale_color(samples_per_pixel, color.z);

                current_line.push_str(&r.to_string());
                self.write_rgb_separator(&mut current_line, &mut ppm);

                current_line.push_str(&g.to_string());
                self.write_rgb_separator(&mut current_line, &mut ppm);

                current_line.push_str(&b.to_string());

                // if not at end of row yet, write a space or newline if the next point will be on this line
                if i != self.width - 1 {
                    self.write_rgb_separator(&mut current_line, &mut ppm);
                }
            }
            if !current_line.is_empty() {
                ppm.push_str(&current_line);
                ppm.push('\n');
            }
        }
        ppm
    }
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub width: usize,
    pub height: usize,
}

unsafe impl Sync for Canvas {}

const MAX_PPM_COLOR_VAL: u64 = 255;
const MAX_PPM_LINE_LENGTH: usize = 70;
// length of "255" is 3
// TODO: this should be evaluated programmatically, but "no matching in consts allowed" error prevented this
const MAX_COLOR_VAL_STR_LEN: usize = 3;
impl Camera {
    // Create a canvas initialized to all black
    pub fn new(width: usize, height: usize) -> Camera {
        Camera { width, height }
    }
    fn jitter(&self) -> f64 {
        let sample: f64 = thread_rng().sample(OpenClosed01);
        return sample / 700.;
    }

    pub fn camcr(&self, x: usize, y: usize) -> Ray {
        let w: f64 = self.width as f64;
        let h: f64 = self.height as f64;
        let fovx: f32 = std::f32::consts::FRAC_PI_4;
        let fovy: f32 = (h / w) as f32 * fovx;
        let mut cam = Vector3d::new(
            ((2. * x as f64 - w) / w) * fovx.tan() as f64,
            ((2. * y as f64 - h) / h) * fovy.tan() as f64,
            -1.,
        );
        cam.x += self.jitter();
        cam.y += self.jitter();
        Ray::new(Vector3d::default(), cam.norm())
    }

    pub fn get_blank_canvas(&self) -> Canvas {
        Canvas {
            width: self.width,
            height: self.height,
            data: vec![vec![Vector3d::default(); self.width]; self.height],
        }
    }
}

// a messed up sampling function (at least in this context).
// courtesy of http://www.rorydriscoll.com/2009/01/07/better-sampling/
fn hemisphere(u1: f64, u2: f64) -> Vector3d {
    let r: f64 = (1. - u1 * u1).sqrt();
    let phi: f64 = 2. * std::f64::consts::PI * u2;
    Vector3d::new(phi.cos() * r, phi.sin() * r, u1)
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Params {
    pub refractive_index: f64,
    pub samples_per_pixel: u64,
    pub width: usize,
    pub height: usize,
}

fn trace(
    ray: &mut Ray,
    scene: &Scene,
    depth: u8,
    color: &mut Vector3d,
    params: Params,
    hal1: &mut Halton,
    hal2: &mut Halton,
    log: bool,
) {
    if depth >= 20 {
        return;
    }

    match scene.intersect(ray) {
        None => return,
        Some(intersection) => {
            // Travel the ray to the hit point where the closest object lies and compute the surface normal there.
            let hit_point: Vector3d = ray.o + ray.d * intersection.t;
            let mut normal: Vector3d = intersection.object.normal(hit_point);
            ray.o = hit_point;
            if log {
                eprintln!("hit_point={}, normal={}", hit_point, normal);
            }

            let material: Material = intersection.object.get_material();
            *color += Vector3d::new(material.emission, material.emission, material.emission) * 2.;

            match material.material_type {
                MaterialType::DIFFUSE => {
                    hal1.next();
                    hal2.next();
                    ray.d = normal + hemisphere(hal1.get(), hal2.get());
                    let cost: f64 = ray.d.dot(normal);
                    let mut tmp = Vector3d::default();
                    trace(ray, scene, depth + 1, &mut tmp, params, hal1, hal2, log);
                    color.x += cost * (tmp.x * material.color.x) * 0.1;
                    color.y += cost * (tmp.y * material.color.y) * 0.1;
                    color.z += cost * (tmp.z * material.color.z) * 0.1;
                }
                MaterialType::SPECULAR => {
                    let cost: f64 = ray.d.dot(normal);
                    ray.d = (ray.d - normal * (cost * 2.)).norm();
                    let mut tmp = Vector3d::default();
                    trace(ray, scene, depth + 1, &mut tmp, params, hal1, hal2, log);
                    *color += tmp;
                }
                MaterialType::REFRACTIVE => {
                    let mut n: f64 = params.refractive_index;
                    if normal.dot(ray.d) > 0. {
                        normal = normal * -1.;
                        // TODO: wouldn't this just mean we should skip both 1/n calculations?
                        n = 1. / n;
                    }
                    n = 1. / n;
                    let cost1: f64 = (normal.dot(ray.d)) * -1.;
                    let cost2: f64 = 1.0 - n * n * (1.0 - cost1 * cost1);
                    if cost2 > 0. {
                        ray.d = (ray.d * n) + (normal * (n * cost1 - cost2.sqrt()));
                        ray.d = ray.d.norm();
                        let mut tmp = Vector3d::default();
                        trace(ray, scene, depth + 1, &mut tmp, params, hal1, hal2, log);
                        *color += tmp;
                    } else {
                        return;
                    }
                }
            }
        }
    }
}

fn render(params: Params) -> Canvas {
    let mut scene = Scene::new();

    // Middle sphere
    scene.add(Box::new(Sphere::new(
        1.05,
        Vector3d::new(1.45, -0.75, -4.4),
        Material::new(Vector3d::new(4., 8., 4.), 0., MaterialType::SPECULAR),
    )));
    // Right sphere
    scene.add(Box::new(Sphere::new(
        0.5,
        Vector3d::new(2.05, 2.0, -3.7),
        Material::new(Vector3d::new(10., 10., 1.), 0., MaterialType::REFRACTIVE),
    )));
    // Left sphere
    scene.add(Box::new(Sphere::new(
        0.6,
        Vector3d::new(1.95, -1.75, -3.1),
        Material::new(Vector3d::new(4., 4., 12.), 0., MaterialType::DIFFUSE),
    )));
    // Bottom plane
    scene.add(Box::new(Plane::new(
        2.5,
        Vector3d::new(-1., 0., 0.),
        Material::new(Vector3d::new(6., 6., 6.), 0., MaterialType::DIFFUSE),
    )));
    // Back plane
    scene.add(Box::new(Plane::new(
        5.5,
        Vector3d::new(0., 0., 1.),
        Material::new(Vector3d::new(6., 6., 6.), 0., MaterialType::DIFFUSE),
    )));
    // Left plane
    scene.add(Box::new(Plane::new(
        2.75,
        Vector3d::new(0., 1., 0.),
        Material::new(Vector3d::new(10., 2., 2.), 0., MaterialType::DIFFUSE),
    )));
    // Right plane
    scene.add(Box::new(Plane::new(
        2.75,
        Vector3d::new(0., -1., 0.),
        Material::new(Vector3d::new(2., 10., 2.), 0., MaterialType::DIFFUSE),
    )));
    // Ceiling plane
    scene.add(Box::new(Plane::new(
        3.0,
        Vector3d::new(1., 0., 0.),
        Material::new(Vector3d::new(6., 6., 6.), 0., MaterialType::DIFFUSE),
    )));
    // Front plane
    scene.add(Box::new(Plane::new(
        0.5,
        Vector3d::new(0., 0., -1.),
        Material::new(Vector3d::new(6., 6., 6.), 0., MaterialType::DIFFUSE),
    )));
    // Light
    scene.add(Box::new(Sphere::new(
        0.5,
        Vector3d::new(-1.9, 0., -3.),
        Material::new(Vector3d::new(0., 0., 0.), 120., MaterialType::DIFFUSE),
    )));

    let camera = Camera::new(params.width, params.height);
    let mut canvas = camera.get_blank_canvas();

    let hal1_thread_local: CachedThreadLocal<Halton> = CachedThreadLocal::new();
    let hal2_thread_local: CachedThreadLocal<Halton> = CachedThreadLocal::new();
    canvas.write_data_parallel(|row_index, col_index, pixel| {
        // correlated Halton-sequence dimensions
        let mut hal1 = *hal1_thread_local.get_or(|| Halton::new(0, 2));
        let mut hal2 = *hal2_thread_local.get_or(|| Halton::new(0, 2));
        for _ in 0..params.samples_per_pixel {
            let mut color = Vector3d::default();
            let mut ray = camera.camcr(col_index, row_index);
            trace(
                &mut ray, &scene, 0, &mut color, params, &mut hal1, &mut hal2, false,
            );
            *pixel += color;
        }
    });

    canvas
}

pub fn run(params: Params) {
    let canvas = render(params);
    println!("{}", canvas.to_ppm(params.samples_per_pixel));
}
