use rand::distributions::OpenClosed01;
use rand::{thread_rng, Rng};
use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

const EPSILON: f64 = 1e-4;

fn rnd() -> f64 {
	thread_rng().sample(OpenClosed01)
}

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
	pub fn cross(&self, other: Vector3d) -> Vector3d {
		Vector3d {
			x: self.y * other.z - self.z * other.y,
			y: self.z * other.x - self.x * other.z,
			z: self.x * other.y - self.y * other.x,
		}
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

impl Mul<Vector3d> for f64 {
	type Output = Vector3d;
	fn mul(self, other: Vector3d) -> Vector3d {
		other * self
	}
}

impl Div<f64> for Vector3d {
	type Output = Vector3d;
	fn div(self, scalar: f64) -> Vector3d {
		Vector3d {
			x: self.x / scalar,
			y: self.y / scalar,
			z: self.z / scalar,
		}
	}
}

impl Neg for Vector3d {
	type Output = Vector3d;
	fn neg(self) -> Vector3d {
		Vector3d {
			x: -self.x,
			y: -self.y,
			z: -self.z,
		}
	}
}
// Wow, that vector implementation is waaaaay longer than it was in C++

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

// Objects have color, emission, type (diffuse, specular, refractive)
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Material {
	pub color: Vector3d,
	pub emission: f64,
	pub material_type: u8,
}

// All object should be intersectable and should be able to compute their surface normals.
pub trait Obj {
	fn intersect(&self, ray: &Ray) -> f64;
	fn normal(&self, point: Vector3d) -> Vector3d;
	fn set_material(&mut self, m: Material);
	fn get_material(&self) -> Material;
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Plane {
	material: Material,
	d: f64,
	n: Vector3d,
}
impl Plane {
	fn new(d: f64, n: Vector3d) -> Self {
		Self {
			d,
			n,
			..Default::default()
		}
	}
}

impl Obj for Plane {
	fn intersect(&self, ray: &Ray) -> f64 {
		let d0: f64 = self.n.dot(ray.d);
		if d0 != 0. {
			let t: f64 = -1. * (((self.n.dot(ray.o)) + self.d) / d0);
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
	fn set_material(&mut self, m: Material) {
		self.material = m;
	}
	fn get_material(&self) -> Material {
		self.material
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Sphere {
	material: Material,
	c: Vector3d,
	r: f64,
}
impl Sphere {
	fn new(r: f64, c: Vector3d) -> Self {
		Self {
			r,
			c,
			..Default::default()
		}
	}
}

impl Obj for Sphere {
	fn intersect(&self, ray: &Ray) -> f64 {
		let b: f64 = ((ray.o - self.c) * 2.).dot(ray.d);
		let c_: f64 = (ray.o - self.c).dot(ray.o - self.c) - (self.r * self.r);
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
		return (p0 - self.c).norm();
	}
	fn set_material(&mut self, m: Material) {
		self.material = m;
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
		if self.inv_base < r {
			self.value += self.inv_base;
		} else {
			let mut h: f64 = self.inv_base;
			let mut hh: f64;
			loop {
				hh = h;
				h *= self.inv_base;
				if h < r {
					break;
				}
			}
			self.value += hh + h - 1.;
		}
	}

	fn get(&self) -> f64 {
		self.value
	}
}

#[derive(Clone, Debug)]
pub struct Canvas {
	pub width: usize,
	pub height: usize,
	data: Vec<Vec<Vector3d>>,
}

const MAX_COLOR_VAL: u16 = 255;
const MAX_PPM_LINE_LENGTH: usize = 70;
// length of "255" is 3
// TODO: this should be evaluated programmatically, but "no matching in consts allowed" error prevented this
const MAX_COLOR_VAL_STR_LEN: usize = 3;
impl Canvas {
	// Create a canvas initialized to all black
	pub fn new(width: usize, height: usize) -> Canvas {
		Canvas {
			width,
			height,
			data: vec![vec![Vector3d::default(); width]; height],
		}
	}
	fn camcr(&self, x: usize, y: usize) -> Vector3d {
		let w: f64 = self.width as f64;
		let h: f64 = self.height as f64;
		let fovx: f32 = std::f32::consts::PI / 4.;
		let fovy: f32 = (h / w) as f32 * fovx;
		return Vector3d::new(
			((2. * x as f64 - w) / w) * fovx.tan() as f64,
			((2. * y as f64 - h) / h) * fovy.tan() as f64,
			-1.,
		);
	}
	pub fn add_color(&mut self, x: usize, y: usize, color: Vector3d) {
		if x <= self.width && y <= self.height {
			self.data[y][x] += color;
		} else {
			// TODO: return fail result
		}
	}

	pub fn pixel_at(&self, x: usize, y: usize) -> Vector3d {
		self.data[y][x]
	}

	// scale/clamp color values from 0-1 to 0-255
	fn scale_color(&self, rgb: f64) -> u8 {
		(rgb * MAX_COLOR_VAL as f64)
			.min(MAX_COLOR_VAL as f64)
			.max(0.0) as u8
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
	pub fn to_ppm(&self) -> String {
		let mut ppm = String::new();
		// write header
		ppm.push_str("P3\n");
		ppm.push_str(&(format!("{} {}\n", self.width, self.height)));
		ppm.push_str(&(format!("{}\n", MAX_COLOR_VAL)));

		// Write pixel data. Each pixel RGB value is written with a separating space or newline;
		// new rows are written on new lines for human reading convenience, but lines longer than
		// MAX_PPM_LINE_LENGTH must also be split.
		let mut current_line = String::new();
		for row in 0..self.height {
			current_line.clear();
			for (i, column) in (0..self.width).enumerate() {
				let color = self.pixel_at(column, row);
				let r = self.scale_color(color.x);
				let g = self.scale_color(color.y);
				let b = self.scale_color(color.z);

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

// a messed up sampling function (at least in this context).
// courtesy of http://www.rorydriscoll.com/2009/01/07/better-sampling/
fn hemisphere(u1: f64, u2: f64) -> Vector3d {
	let r: f64 = (1. - u1 * u1).sqrt();
	let phi: f64 = 2. * std::f64::consts::PI * u2;
	Vector3d::new(phi.cos() * r, phi.sin() * r, u1)
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Params {
	refractive_index: f64,
	samples_per_pixel: i32,
}

fn trace(
	ray: &mut Ray,
	scene: &Scene,
	depth: u8,
	color: &mut Vector3d,
	params: Params,
	mut hal1: Halton,
	mut hal2: Halton,
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

			let material: Material = intersection.object.get_material();
			*color += Vector3d::new(material.emission, material.emission, material.emission) * 2.;

			if material.material_type == 1 {
				hal1.next();
				hal2.next();
				ray.d = normal + hemisphere(hal1.get(), hal2.get());
				let cost: f64 = ray.d.dot(normal);
				let mut tmp = Vector3d::default();
				trace(ray, scene, depth + 1, &mut tmp, params, hal1, hal2);
				color.x += cost * (tmp.x * material.color.x) * 0.1;
				color.y += cost * (tmp.y * material.color.y) * 0.1;
				color.z += cost * (tmp.z * material.color.z) * 0.1;
			} else if material.material_type == 2 {
				let cost: f64 = ray.d.dot(normal);
				ray.d = (ray.d - normal * (cost * 2.)).norm();
				let mut tmp = Vector3d::default();
				trace(ray, scene, depth + 1, &mut tmp, params, hal1, hal2);
				*color += tmp;
			} else if material.material_type == 3 {
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
					trace(ray, scene, depth + 1, &mut tmp, params, hal1, hal2);
					*color += tmp;
				} else {
					return;
				}
			}
		}
	}
}

fn render(size: usize, params: Params) -> Canvas {
	// srand(time(NULL));

	let mut scene = Scene::new();
	let mut add = |mut obj: Box<dyn Obj>, color: Vector3d, emission: f64, material_type: u8| {
		let material = Material {
			color,
			emission,
			material_type,
		};
		obj.set_material(material);
		scene.add(obj);
	};

	// Radius, position, color, emission, type (1=diff, 2=spec, 3=refr) for spheres
	add(
		Box::new(Sphere::new(1.05, Vector3d::new(1.45, -0.75, -4.4))),
		Vector3d::new(4., 8., 4.),
		0.,
		2,
	); // Middle sphere
	add(
		Box::new(Sphere::new(0.5, Vector3d::new(2.05, 2.0, -3.7))),
		Vector3d::new(10., 10., 1.),
		0.,
		3,
	); // Right sphere
	add(
		Box::new(Sphere::new(0.6, Vector3d::new(1.95, -1.75, -3.1))),
		Vector3d::new(4., 4., 12.),
		0.,
		1,
	); // Left sphere
   // Position, normal, color, emission, type for planes
	add(
		Box::new(Plane::new(2.5, Vector3d::new(-1., 0., 0.))),
		Vector3d::new(6., 6., 6.),
		0.,
		1,
	); // Bottom plane
	add(
		Box::new(Plane::new(5.5, Vector3d::new(0., 0., 1.))),
		Vector3d::new(6., 6., 6.),
		0.,
		1,
	); // Back plane
	add(
		Box::new(Plane::new(2.75, Vector3d::new(0., 1., 0.))),
		Vector3d::new(10., 2., 2.),
		0.,
		1,
	); // Left plane
	add(
		Box::new(Plane::new(2.75, Vector3d::new(0., -1., 0.))),
		Vector3d::new(2., 10., 2.),
		0.,
		1,
	); // Right plane
	add(
		Box::new(Plane::new(3.0, Vector3d::new(1., 0., 0.))),
		Vector3d::new(6., 6., 6.),
		0.,
		1,
	); // Ceiling plane
	add(
		Box::new(Plane::new(0.5, Vector3d::new(0., 0., -1.))),
		Vector3d::new(6., 6., 6.),
		0.,
		1,
	); // Front plane
	add(
		Box::new(Sphere::new(0.5, Vector3d::new(-1.9, 0., -3.))),
		Vector3d::new(0., 0., 0.),
		120.,
		1,
	); // Light

	let mut canvas = Canvas::new(size, size);

	// correlated Halton-sequence dimensions
	let hal1 = Halton::new(0, 2);
	let hal2 = Halton::new(0, 2);

	for s in 0..params.samples_per_pixel {
		println!("sample={}", s);
		// #pragma omp parallel for schedule(dynamic) firstprivate(hal, hal2)
		for i in 0..canvas.width {
			for j in 0..canvas.height {
				let mut color = Vector3d::default();
				let mut cam = canvas.camcr(i, j);
				cam.x = cam.x + rnd() / 700.;
				cam.y = cam.y + rnd() / 700.;
				// original had (cam - ray.o).norm(), but ray.o was always vec(0,0,0)
				let mut ray = Ray::new(Vector3d::default(), cam.norm());
				trace(&mut ray, &scene, 0, &mut color, params, hal1, hal2);
				canvas.add_color(i, j, color);
			}
		}
	}
	canvas
}

fn main() {
	let canvas = render(
		512,
		Params {
			refractive_index: 1.5,
			samples_per_pixel: 50,
		},
	);
	println!("{}", canvas.to_ppm());
}
