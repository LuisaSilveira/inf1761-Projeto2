from __future__ import annotations

from typing import Any
import os
import time

import graphics_math as gm
import numpy as np
import wgpu
from PIL import Image, ImageFilter
from rendercanvas.glfw import RenderCanvas, loop

from camera3d import Camera3D
from engine import Engine
from node import Node
from phongmaterial import PhongMaterial
from pipeline import Pipeline
from pointlight import PointLight
from renderer import Renderer
from sampler import Sampler
from scene import Scene
from shader import Shader
from skybox import SkyBox, SkyBoxTransform
from sphere import Sphere
from textureset import TextureSet
from transform import Transform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG  = os.path.join(ROOT, "images")
SHD  = os.path.join(ROOT, "shaders", "solar")


R_SOL, R_MERCURIO, R_TERRA, R_LUA = 1.5, 0.24, 0.50, 0.16
D_MERCURIO, D_TERRA, D_LUA = 3.4, 6.2, 1.6

W_SOL_ROT     = 6.0  
W_MERC_TRANS  = 33.0   
W_MERC_ROT    = 12.0
W_TERRA_TRANS = 8.0   
W_TERRA_ROT   = 90.0   
W_LUA_TRANS   = 60.0   
INCL_TERRA    = 23.4  
INCL_LUA      = 5.1    
EARTH_CAM_FOVY = 28.0  
BUMP_TERRA, BUMP_LUA, BUMP_MERCURIO = 1.0, 1.6, 1.3


canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
scene: Scene
cameras: list[Camera3D] = []
camera_names = ["Camera global (arcball)", "Camera na Terra -> Lua"]
current_camera = 0
planet_shader: Shader
last_t: float = 0.0

paused      = False
bump_on     = True
_title_dirty = True



def _load_rgba(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def _make_seamless(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x   = np.linspace(0.0, 1.0, w, dtype=np.float32)
    t   = 1.0 - np.abs(2.0 * x - 1.0)
    wgt = (t * t * (3.0 - 2.0 * t))[None, :, None]
    shifted = np.roll(img, w // 2, axis=1).astype(np.float32)
    out = img.astype(np.float32) * wgt + shifted * (1.0 - wgt)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _fade_poles(img: np.ndarray, frac: float = 0.18) -> np.ndarray:
    out = img.astype(np.float32).copy()
    h   = img.shape[0]
    n   = max(1, int(h * frac))
    top = out[:n].mean(axis=(0, 1))
    bot = out[-n:].mean(axis=(0, 1))
    for i in range(n):
        w = (1.0 - i / n) ** 1.5
        out[i]         = out[i]         * (1 - w) + top * w
        out[h - 1 - i] = out[h - 1 - i] * (1 - w) + bot * w
    return np.clip(out, 0, 255).astype(np.uint8)


def _height_to_normal(img: np.ndarray, strength: float = 3.0) -> np.ndarray:
    lum = (img[..., :3].astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    h8  = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    h8  = h8.filter(ImageFilter.GaussianBlur(1.0))
    h   = np.asarray(h8, dtype=np.float32) / 255.0
    dx  = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    dy  = np.vstack([h[:1], h[:-1]]) - np.vstack([h[1:], h[-1:]])
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(dx)
    inv = 1.0 / np.sqrt(nx*nx + ny*ny + nz*nz)
    rgb = ((np.stack([nx*inv, ny*inv, nz*inv], axis=-1) * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)
    return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)


def _mip_chain(img: np.ndarray) -> list[np.ndarray]:
    levels = [img]
    cur = Image.fromarray(img)
    while cur.width > 1 or cur.height > 1:
        cur = cur.resize((max(1, cur.width // 2), max(1, cur.height // 2)), Image.BOX)
        levels.append(np.array(cur))
    return levels


def _upload_texture(dev: wgpu.GPUDevice, varname: str, data: np.ndarray,
                    srgb: bool = True) -> Any:
    h, w = data.shape[:2]
    levels = _mip_chain(data) if (w > 1 and h > 1) else [data]
    fmt = "rgba8unorm-srgb" if srgb else "rgba8unorm"
    tex = dev.create_texture(
        size=(w, h, 1), format=fmt,
        mip_level_count=len(levels),
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    for lv, arr in enumerate(levels):
        lh, lw = arr.shape[:2]
        dev.queue.write_texture(
            {"texture": tex, "mip_level": lv},
            np.ascontiguousarray(arr).tobytes(),
            {"bytes_per_row": lw * 4, "rows_per_image": lh}, (lw, lh, 1),
        )
    view = tex.create_view()

    class _Tex:
        def __init__(self, vn, v):
            self.varname = vn
            self._view   = v
        @property
        def resource(self):
            return self._view

    return _Tex(varname, view)


def _prep(path: str, seamless: bool = True, fade: bool = True) -> np.ndarray:

    img = _load_rgba(path)
    if seamless:
        img = _make_seamless(img)
    if fade:
        img = _fade_poles(img)
    return img


def _flat_normal() -> np.ndarray:
    return np.array([[[128, 128, 255, 255]]], dtype=np.uint8)


def _build_space_cubemap(dev: wgpu.GPUDevice, varname: str, path: str,
                         face_size: int = 512, seed: int = 7) -> Any:
    """Monta cubemap de estrelas a partir de uma unica imagem (espaco.png).
    Cada face e um mosaico 3x3 de recortes aleatorios para variar o ceu."""
    src = np.array(Image.open(path).convert("RGBA"))
    rng = np.random.default_rng(seed)
    ts  = face_size // 3 + 1
    faces = []
    for _ in range(6):
        face = np.zeros((3 * ts, 3 * ts, 4), dtype=np.uint8)
        for ty in range(3):
            for tx in range(3):
                y0   = int(rng.integers(0, max(1, src.shape[0] - ts)))
                x0   = int(rng.integers(0, max(1, src.shape[1] - ts)))
                crop = src[y0:y0+ts, x0:x0+ts]
                if rng.random() < 0.5: crop = crop[:, ::-1]
                if rng.random() < 0.5: crop = crop[::-1, :]
                ch, cw = crop.shape[:2]
                face[ty*ts:ty*ts+ch, tx*ts:tx*ts+cw] = crop
        faces.append(np.ascontiguousarray(face[:face_size, :face_size]))

    chains = [_mip_chain(f) for f in faces]
    nlev   = len(chains[0])
    tex    = dev.create_texture(
        size=(face_size, face_size, 6), format="rgba8unorm-srgb", dimension="2d",
        mip_level_count=nlev,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    for layer, chain in enumerate(chains):
        for lv, arr in enumerate(chain):
            lh, lw = arr.shape[:2]
            dev.queue.write_texture(
                {"texture": tex, "mip_level": lv, "origin": (0, 0, layer)},
                np.ascontiguousarray(arr).tobytes(),
                {"bytes_per_row": lw * 4, "rows_per_image": lh}, (lw, lh, 1),
            )
    view = tex.create_view(dimension="cube")

    class _Cube:
        def __init__(self, vn, v):
            self.varname = vn
            self._view   = v
        @property
        def resource(self): return self._view

    return _Cube(varname, view)


class Revolve(Engine):

    def __init__(self, trf: Transform, dist: float, speed_deg_s: float,
                 phase_deg: float = 0.0, incl_deg: float = 0.0,
                 keep_orientation: bool = True) -> None:
        self.trf, self.dist, self.speed = trf, dist, speed_deg_s
        self.angle, self.incl, self.keep = phase_deg, incl_deg, keep_orientation

    def update(self, dt: float) -> None:
        self.angle = (self.angle + self.speed * dt) % 360.0
        t = self.trf
        t.load_identity()
        if self.incl:
            t.rotate(self.incl, 0, 0, 1)
        t.rotate(self.angle, 0, 1, 0)
        t.translate(self.dist, 0, 0)
        if self.keep:
            t.rotate(-self.angle, 0, 1, 0)


class Spin(Engine):

    def __init__(self, trf: Transform, radius: float, speed_deg_s: float,
                 tilt_deg: float = 0.0, offset: tuple = (0, 0, 0),
                 phase_deg: float = 0.0) -> None:
        self.trf, self.radius, self.speed = trf, radius, speed_deg_s
        self.tilt, self.offset, self.angle = tilt_deg, offset, phase_deg

    def update(self, dt: float) -> None:
        self.angle = (self.angle + self.speed * dt) % 360.0
        t = self.trf
        t.load_identity()
        t.translate(*self.offset)
        if self.tilt:
            t.rotate(self.tilt, 0, 0, 1)
        t.rotate(self.angle, 0, 1, 0)
        t.scale(self.radius, self.radius, self.radius)


class EarthMoonCamera(Engine):

    def __init__(self, camera: Camera3D, earth: Node, moon: Node) -> None:
        self.camera, self.earth, self.moon = camera, earth, moon

    @staticmethod
    def _center(node: Node) -> np.ndarray:
        return (node.get_model_matrix() @ gm.vec4(0, 0, 0, 1))[:3]

    def update(self, dt: float) -> None:
        earth = self._center(self.earth)
        moon  = self._center(self.moon)
        dir_  = gm.normalize(moon - earth)
        eye   = earth + dir_ * (R_TERRA * 1.04)
        self.camera.set_eye(float(eye[0]), float(eye[1]), float(eye[2]))
        self.camera.set_center(float(moon[0]), float(moon[1]), float(moon[2]))
        self.camera.set_up_dir(0.0, 1.0, 0.0)


def initialize(dev: wgpu.GPUDevice, target_format: str) -> None:
    global scene, cameras, planet_shader

    cam_global = Camera3D(0.0, 7.5, 11.5)
    cam_global.set_center(0.0, 0.0, 0.0)
    cam_global.create_arcball()
    cam_earth = Camera3D(D_TERRA + 1.0, 0.0, 0.0)
    cam_earth.set_angle(EARTH_CAM_FOVY)
    cam_earth.set_z_planes(0.02, 1000.0)
    cameras = [cam_global, cam_earth]

    light = PointLight(0.0, 0.0, 0.0, space="world")
    light.set_ambient(0.05, 0.05, 0.06)
    light.set_diffuse(1.0, 1.0, 1.0)
    light.set_specular(1.0, 1.0, 1.0)

    planet_shader = Shader(dev, os.path.join(SHD, "planet.wgsl"), light=light, space="world")
    planet_shader.set_vertex_buffers([
        {"array_stride": 3 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
        {"array_stride": 3 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
        {"array_stride": 3 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x3", "offset": 0, "var_name": "tangent"}]},
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
    ])
    pipeline_planet = Pipeline(planet_shader, target_format, primitive={"cull_mode": "none"})

    planet_shader.set_value("bump_on", 1.0)

    sampler = Sampler(dev, "tex_sampler",
                      address_mode_u="repeat", address_mode_v="clamp-to-edge")
    flat_nm = _upload_texture(dev, "normal_texture", _flat_normal(), srgb=False)

    def tex_set(decal_arr: np.ndarray, normal_obj=None) -> TextureSet:
        d  = _upload_texture(dev, "decal_texture", decal_arr, srgb=True)
        ts = TextureSet([d, normal_obj or flat_nm, sampler])
        planet_shader.add_texture_set(ts)
        return ts

    def bump_nm(img: np.ndarray, strength: float):
        nm = _height_to_normal(img, strength)
        return _upload_texture(dev, "normal_texture", nm, srgb=False)

    ts_sol = tex_set(_prep(os.path.join(IMG, "sol.png")))

    merc_img = _prep(os.path.join(IMG, "mercurio.png"))
    ts_merc  = tex_set(merc_img, bump_nm(merc_img, 5.0))

    lua_img = _prep(os.path.join(IMG, "lua.png"))
    ts_lua  = tex_set(lua_img, bump_nm(lua_img, 5.0))

    earth_img = _load_rgba(os.path.join(IMG, "earth.jpg"))
    earth_nm  = _load_rgba(os.path.join(IMG, "earth-normal.png"))
    earth_nm[..., 0] = 255 - earth_nm[..., 0] 
    earth_nm = np.array(Image.fromarray(earth_nm).resize((1024, 512), Image.LANCZOS))
    ts_terra = tex_set(earth_img,
                       _upload_texture(dev, "normal_texture", earth_nm, srgb=False))

    def material(base: float, spec: float, shin: float,
                 bump: float = 0.0, emissive: float = 0.0, ocean: float = 0.0) -> PhongMaterial:
        m = PhongMaterial(base, base, base)
        m.set_specular(spec, spec, spec)
        m.set_shininess(shin)
        m.set("emissive",   emissive)
        m.set("bump_scale", bump)
        m.set("ocean_spec", ocean)
        planet_shader.add_material(m)
        return m

    mat_sol  = material(1.0, 0.0,  1.0, emissive=1.0)
    mat_merc = material(0.95, 0.08, 12.0, bump=BUMP_MERCURIO)
    mat_terra= material(1.0,  0.60, 60.0, bump=BUMP_TERRA, ocean=1.0)
    mat_lua  = material(0.92, 0.03,  8.0, bump=BUMP_LUA)

    sphere = Sphere(dev, 96, 96)

    trf_sol  = Transform()
    sol_node = Node(trf=trf_sol, apps=[mat_sol, ts_sol], shps=[sphere])

    trf_merc_anchor, trf_merc = Transform(), Transform()
    merc_body   = Node(trf=trf_merc,        apps=[mat_merc, ts_merc], shps=[sphere])
    merc_anchor = Node(trf=trf_merc_anchor, nodes=[merc_body])

    trf_terra_anchor, trf_terra = Transform(), Transform()
    trf_lua_pivot, trf_lua      = Transform(), Transform()
    terra_body  = Node(trf=trf_terra,       apps=[mat_terra, ts_terra], shps=[sphere])
    lua_body    = Node(trf=trf_lua,         apps=[mat_lua,   ts_lua],   shps=[sphere])
    lua_pivot   = Node(trf=trf_lua_pivot,   nodes=[lua_body])
    terra_anchor= Node(trf=trf_terra_anchor,nodes=[terra_body, lua_pivot])

    system = Node(pipeline_planet, nodes=[sol_node, merc_anchor, terra_anchor])

    sky_shader = Shader(dev, os.path.join(SHD, "sky.wgsl"))
    sky_shader.set_vertex_buffers([
        {"array_stride": 3 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
    ])
    pipeline_sky = Pipeline(sky_shader, target_format,
                            primitive={"cull_mode": "none"},
                            depth_stencil={"depth_write_enabled": False})
    sky_tex = _build_space_cubemap(dev, "sky_texture", os.path.join(IMG, "espaco.png"))
    sky_smp = Sampler(dev, "sky_sampler",
                      address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
    ts_sky  = TextureSet([sky_tex, sky_smp])
    sky_shader.add_texture_set(ts_sky)
    sky_node = Node(pipeline_sky, SkyBoxTransform(), [ts_sky], [SkyBox(dev)])

    scene = Scene(Node(None, nodes=[sky_node, system]))

    scene.add_engine(Spin(trf_sol, R_SOL, W_SOL_ROT))
    scene.add_engine(Revolve(trf_merc_anchor, D_MERCURIO, W_MERC_TRANS, phase_deg=40.0))
    scene.add_engine(Spin(trf_merc, R_MERCURIO, W_MERC_ROT))
    scene.add_engine(Revolve(trf_terra_anchor, D_TERRA, W_TERRA_TRANS, phase_deg=200.0))
    scene.add_engine(Spin(trf_terra, R_TERRA, W_TERRA_ROT, tilt_deg=INCL_TERRA))
    scene.add_engine(Revolve(trf_lua_pivot, 0.0, W_LUA_TRANS,
                             incl_deg=INCL_LUA, keep_orientation=False))

    scene.add_engine(Spin(trf_lua, R_LUA, 0.0, offset=(D_LUA, 0, 0), phase_deg=90.0))

    scene.add_engine(EarthMoonCamera(cam_earth, terra_body, lua_body))
    scene.update(0.0)

def apply_bump() -> None:
    global _title_dirty
    planet_shader.set_value("bump_on", 1.0 if bump_on else 0.0)
    _title_dirty = True


def update_title() -> None:
    global _title_dirty
    if not _title_dirty:
        return
    _title_dirty = False
    parts = [f"Sistema Solar 3D - {camera_names[current_camera]}"]
    parts.append(f"bump: {'on' if bump_on else 'off'}")
    parts.append("PAUSADO" if paused else "ANIMANDO")
    try:
        canvas.set_title("  |  ".join(parts))
    except Exception:
        pass


def on_key(event: Any) -> None:
    global current_camera, paused, bump_on
    global _title_dirty
    k = event["key"]
    if k == "q":
        canvas.close()
    elif k == "c":
        current_camera = 1 - current_camera
        _title_dirty = True
    elif k in (" ", "Space"):
        paused = not paused
        _title_dirty = True
    elif k == "b":
        bump_on = not bump_on
        apply_bump()


def attach_arcball(cnv: Any) -> None:
    """Liga o arcball da camera global ao mouse."""
    arcball = cameras[0].get_arcball()
    assert arcball is not None
    state = {"drag": False, "zoom": 0.0}

    def on_down(e: Any) -> None:
        if current_camera != 0:
            return
        state["drag"] = True
        _, h = cnv.get_logical_size()
        arcball.init_mouse_motion(e["x"], h - e["y"])

    def on_up(e: Any) -> None:
        state["drag"] = False

    def on_move(e: Any) -> None:
        if state["drag"] and current_camera == 0:
            w, h = cnv.get_logical_size()
            arcball.accumulate_mouse_motion(e["x"], h - e["y"], w, h)

    def on_wheel(e: Any) -> None:
        if current_camera != 0:
            return
        dz = -float(e.get("dy", 0.0)) * 0.0005
        z  = min(max(state["zoom"] + dz, -0.6), 0.75)
        arcball.translate(0.0, 0.0, z - state["zoom"])
        state["zoom"] = z

    cnv.add_event_handler(on_down,  "pointer_down")
    cnv.add_event_handler(on_up,    "pointer_up")
    cnv.add_event_handler(on_move,  "pointer_move")
    cnv.add_event_handler(on_wheel, "wheel")


def draw() -> None:
    global last_t
    now = time.perf_counter()
    dt  = min(now - last_t, 0.1)
    last_t = now
    if not paused:
        scene.update(dt)
    update_title()
    renderer.render(context.get_current_texture(), scene, cameras[current_camera])


def main() -> None:
    global canvas, device, context, renderer, last_t

    canvas = RenderCanvas(size=(1100, 760), title="Sistema Solar 3D - INF1761",
                          update_mode="continuous", max_fps=60)
    adapter = wgpu.gpu.request_adapter_sync()
    device  = adapter.request_device_sync()
    context = canvas.get_context("wgpu")
    target_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=target_format)
    renderer = Renderer(device, depth_test=True, clear_value=(0.0, 0.0, 0.02, 1.0))

    initialize(device, target_format)
    apply_bump()
    attach_arcball(canvas)
    canvas.add_event_handler(on_key, "key_down")

    last_t = time.perf_counter()
    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
