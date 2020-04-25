// smallpaint by Károly Zsolnai-Fehér - zsolnai@cg.tuwien.ac.at
// Modified to be run standalone on OSX
// Dependency: `brew install libomp`
// Compiling: `clang++ -std=c++11 -O3 -Xpreprocessor -fopenmp smallpaint_osx.cpp -o painterly -lomp`

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <ctime>
#include <vector>
#include <string>
#include <unordered_map>

unsigned int seed = time(NULL);
#define RND (2.0 * (double)rand_r(&seed) / RAND_MAX - 1.0)

#define PI 3.1415926536

int width, height;
const double inf = 1e9;
const double eps = 1e-4;
using namespace std;
typedef unordered_map<string, double> pl;

struct Vec
{
    double x, y, z;
    Vec(double x0 = 0, double y0 = 0, double z0 = 0)
    {
        x = x0;
        y = y0;
        z = z0;
    }
    Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
    Vec operator/(double b) const { return Vec(x / b, y / b, z / b); }
    Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
    double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }

    friend ostream &operator<<(ostream &output, const Vec &v)
    {
        output << "(" << v.x << "," << v.y << "," << v.z << ")";
        return output;
    }
};

// Rays have origin and direction.
// The direction vector should always be normalized.
struct Ray
{
    Vec o, d;
    Ray(Vec o0 = 0, Vec d0 = 0) { o = o0, d = d0.norm(); }

    friend ostream &operator<<(ostream &output, const Ray &r)
    {
        output << "[origin=" << r.o << ", direction=" << r.d << "]";
        return output;
    }
};

// Objects have color, emission, type (diffuse, specular, refractive)
// All object should be intersectable and should be able to compute their surface normals.
class Obj
{
public:
    Vec cl;
    double emission;
    int type;
    void setMat(Vec cl_ = 0, double emission_ = 0, int type_ = 0)
    {
        cl = cl_;
        emission = emission_;
        type = type_;
    }
    virtual double intersect(const Ray &) const = 0;
    virtual Vec normal(const Vec &) const = 0;
};

class Plane : public Obj
{
public:
    Vec n;
    double d;
    Plane(double d_ = 0, Vec n_ = 0)
    {
        d = d_;
        n = n_;
    }
    double intersect(const Ray &ray) const
    {
        double d0 = n.dot(ray.d);
        if (d0 != 0)
        {
            double t = -1 * (((n.dot(ray.o)) + d) / d0);
            return (t > eps) ? t : 0;
        }
        else
            return 0;
    }
    Vec normal(const Vec &p0) const { return n; }
};

class Sphere : public Obj
{
public:
    Vec c;
    double r;

    Sphere(double r_ = 0, Vec c_ = 0)
    {
        c = c_;
        r = r_;
    }
    double intersect(const Ray &ray) const
    {
        double b = ((ray.o - c) * 2).dot(ray.d);
        double c_ = (ray.o - c).dot((ray.o - c)) - (r * r);
        double disc = b * b - 4 * c_;
        if (disc < 0)
            return 0;
        else
            disc = sqrt(disc);
        double sol1 = -b + disc;
        double sol2 = -b - disc;
        return (sol2 > eps) ? sol2 / 2 : ((sol1 > eps) ? sol1 / 2 : 0);
    }

    Vec normal(const Vec &p0) const
    {
        return (p0 - c).norm();
    }
};

class Intersection
{
public:
    Intersection()
    {
        t = inf;
        object = nullptr;
    }
    Intersection(double t_, Obj *object_)
    {
        t = t_;
        object = object_;
    }
    operator bool() { return object != nullptr; }
    double t;
    Obj *object;
};

class Scene
{
    vector<Obj *> objects;

public:
    void add(Obj *object)
    {
        objects.push_back(object);
    }

    Intersection intersect(const Ray &ray) const
    {
        Intersection closestIntersection;
        // intersect all objects, one after the other
        for (auto iter = objects.begin(); iter != objects.end(); ++iter)
        {
            double t = (*iter)->intersect(ray);
            if (t > eps && t < closestIntersection.t)
            {
                closestIntersection.t = t;
                closestIntersection.object = *iter;
            }
        }
        return closestIntersection;
    }
};

class Halton
{
    double value, inv_base;

public:
    void number(int i, int base)
    {
        double f = inv_base = 1.0 / base;
        value = 0.0;
        while (i > 0)
        {
            value += f * (double)(i % base);
            i /= base;
            f *= inv_base;
        }
    }
    void next()
    {
        double r = 1.0 - value - 0.0000001;
        if (inv_base < r)
            value += inv_base;
        else
        {
            double h = inv_base, hh;
            do
            {
                hh = h;
                h *= inv_base;
            } while (h >= r);
            value += hh + h - 1.0;
        }
    }
    double get() { return value; }
};

Vec camcr(const double x, const double y)
{
    double w = width;
    double h = height;
    float fovx = PI / 4;
    float fovy = (h / w) * fovx;
    return Vec(((2 * x - w) / w) * tan(fovx),
               ((2 * y - h) / h) * tan(fovy),
               -1.0);
}

// a messed up sampling function (at least in this context).
// courtesy of http://www.rorydriscoll.com/2009/01/07/better-sampling/
Vec hemisphere(double u1, double u2)
{
    const double r = sqrt(1.0 - u1 * u1);
    const double phi = 2 * PI * u2;
    return Vec(cos(phi) * r, sin(phi) * r, u1);
}

void trace(Ray &ray, const Scene &scene, int depth, Vec &clr, pl &params, Halton &hal, Halton &hal2)
{
    if (depth >= 20)
        return;

    Intersection intersection = scene.intersect(ray);
    if (!intersection)
        return;

    // Travel the ray to the hit point where the closest object lies and compute the surface normal there.
    Vec hp = ray.o + ray.d * intersection.t;
    Vec N = intersection.object->normal(hp);
    ray.o = hp;

    clr = clr + Vec(intersection.object->emission, intersection.object->emission, intersection.object->emission) * 2;

    if (intersection.object->type == 1)
    {
        hal.next();
        hal2.next();
        ray.d = (N + hemisphere(hal.get(), hal2.get()));
        double cost = ray.d.dot(N);
        Vec tmp = Vec();
        trace(ray, scene, depth + 1, tmp, params, hal, hal2);
        clr.x += cost * (tmp.x * intersection.object->cl.x) * 0.1;
        clr.y += cost * (tmp.y * intersection.object->cl.y) * 0.1;
        clr.z += cost * (tmp.z * intersection.object->cl.z) * 0.1;
    }

    if (intersection.object->type == 2)
    {
        double cost = ray.d.dot(N);
        ray.d = (ray.d - N * (cost * 2)).norm();
        Vec tmp = Vec(0, 0, 0);
        trace(ray, scene, depth + 1, tmp, params, hal, hal2);
        clr = clr + tmp;
    }

    if (intersection.object->type == 3)
    {
        double n = params["refr_index"];
        if (N.dot(ray.d) > 0)
        {
            N = N * -1;
            n = 1 / n;
        }
        n = 1 / n;
        double cost1 = (N.dot(ray.d)) * -1;
        double cost2 = 1.0 - n * n * (1.0 - cost1 * cost1);
        if (cost2 > 0)
        {
            ray.d = (ray.d * n) + (N * (n * cost1 - sqrt(cost2)));
            ray.d = ray.d.norm();
            Vec tmp = Vec(0, 0, 0);
            trace(ray, scene, depth + 1, tmp, params, hal, hal2);
            clr = clr + tmp;
        }
        else
            return;
    }
}

void render(int id, int size, int spp, double refr_index)
{

    srand(time(NULL));
    pl params;

    Scene scene;
    auto add = [&scene](Obj *s, Vec cl, double emission, int type) {
        s->setMat(cl, emission, type);
        scene.add(s);
    };

    // Radius, position, color, emission, type (1=diff, 2=spec, 3=refr) for spheres
    add(new Sphere(1.05, Vec(1.45, -0.75, -4.4)), Vec(4, 8, 4), 0, 2); // Middle sphere
    add(new Sphere(0.5, Vec(2.05, 2.0, -3.7)), Vec(10, 10, 1), 0, 3);  // Right sphere
    add(new Sphere(0.6, Vec(1.95, -1.75, -3.1)), Vec(4, 4, 12), 0, 1); // Left sphere
    // Position, normal, color, emission, type for planes
    add(new Plane(2.5, Vec(-1, 0, 0)), Vec(6, 6, 6), 0, 1);       // Bottom plane
    add(new Plane(5.5, Vec(0, 0, 1)), Vec(6, 6, 6), 0, 1);        // Back plane
    add(new Plane(2.75, Vec(0, 1, 0)), Vec(10, 2, 2), 0, 1);      // Left plane
    add(new Plane(2.75, Vec(0, -1, 0)), Vec(2, 10, 2), 0, 1);     // Right plane
    add(new Plane(3.0, Vec(1, 0, 0)), Vec(6, 6, 6), 0, 1);        // Ceiling plane
    add(new Plane(0.5, Vec(0, 0, -1)), Vec(6, 6, 6), 0, 1);       // Front plane
    add(new Sphere(0.5, Vec(-1.9, 0, -3)), Vec(0, 0, 0), 120, 1); // Light

    params["refr_index"] = refr_index;
    params["spp"] = spp; // samples per pixel

    width = size;
    height = size;

    Vec **pix = new Vec *[width];
    for (int i = 0; i < width; i++)
    {
        pix[i] = new Vec[height];
    }

    // correlated Halton-sequence dimensions
    Halton hal, hal2;
    hal.number(0, 2);
    hal2.number(0, 2);

    bool running = true;

    for (int s = 0; s < spp; s++)
    {
        cout << "sample=" << s << "\n";
#pragma omp parallel for schedule(dynamic) firstprivate(hal, hal2)
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                Vec c;
                Ray ray;
                ray.o = (Vec(0, 0, 0));
                Vec cam = camcr(i, j);
                cam.x = cam.x + RND / 700;
                cam.y = cam.y + RND / 700;
                ray.d = (cam - ray.o).norm();
                trace(ray, scene, 0, c, params, hal, hal2);
                pix[j][i].x += c.x;
                pix[j][i].y += c.y;
                pix[j][i].z += c.z;
                // if (i == 359 && j == 420)
                // {
                //     cout << "final_color=" << c << "\n";
                // }
            }
        }
    }

    // Save the results to file
    FILE *f = fopen("ray.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n ", width, height, 12000); // Note: 	 is a random guess. We should be doing 255 and normalizing the output values.
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            fprintf(f, "%d %d %d ", (int)pix[col][row].x, (int)pix[col][row].y, (int)pix[col][row].z);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main()
{
    render(0, 512, 50, 1.5);
}
