struct Matrix {
  vertex: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,  
}
@group(0) @binding(0) var<uniform> global: Global;

@group(2) @binding(0) var sky_texture: texture_cube<f32>;
@group(2) @binding(1) var sky_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) dir: vec3<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) coord: vec3<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  out.dir = coord;
  out.clip_position = global.projection * (m.vertex * vec4<f32>(coord, 1.0));
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(sky_texture, sky_sampler, normalize(in.dir));
}
