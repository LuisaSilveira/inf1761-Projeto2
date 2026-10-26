struct Matrix {
  vertex: mat4x4<f32>,   
  normal: mat4x4<f32>,   
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,    
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,  
  camera_position: vec4<f32>, 
  bump_on: f32,             
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,
  specular_color: vec3<f32>,
  shininess: f32,
  emissive: f32,   
  bump_scale: f32,  
  ocean_spec: f32,   
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var normal_texture: texture_2d<f32>;
@group(3) @binding(2) var tex_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) wpos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec3<f32>,
  @location(3) texcoord: vec2<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) coord: vec3<f32>,
            @location(1) normal: vec3<f32>,
            @location(2) tangent: vec3<f32>,
            @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let p = m.vertex * vec4<f32>(coord, 1.0);
  out.wpos = p.xyz;
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.tangent = (m.vertex * vec4<f32>(tangent, 0.0)).xyz;
  out.texcoord = texcoord;
  out.clip_position = global.projection * p;
  return out;
}

@fragment
fn fs_main (in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
  let decal = textureSample(decal_texture, tex_sampler, in.texcoord);
  let nmap = textureSample(normal_texture, tex_sampler, in.texcoord).xyz;

  if (!front) {
    discard;
  }

  if (material.emissive > 0.5) {
    return vec4<f32>(min(decal.rgb * material.base_color * 1.15, vec3<f32>(1.0)), 1.0);
  }

  var n = normalize(in.normal);
  if (global.bump_on > 0.5 && material.bump_scale > 0.0) {
    var t = normalize(in.tangent);
    t = normalize(t - n * dot(n, t));        
    let b = cross(n, t);                  
    var tn = nmap * 2.0 - vec3<f32>(1.0);    
    tn = vec3<f32>(tn.xy * material.bump_scale, tn.z);
    n = normalize(t * tn.x + b * tn.y + n * tn.z);
  }

  var l: vec3<f32>;
  if (global.light_position.w == 0.0) {
    l = normalize(global.light_position.xyz);
  } else {
    l = normalize(global.light_position.xyz - in.wpos);
  }
  let v = normalize(global.camera_position.xyz - in.wpos);
  let albedo = decal.rgb * material.base_color;

  var color = albedo * global.light_ambient;
  let ndotl = dot(n, l);
  if (ndotl > 0.0) {
    color += albedo * global.light_diffuse * ndotl;
    let h = normalize(l + v);
    var spec_mask = 1.0;
    if (material.ocean_spec > 0.5) {
      spec_mask = clamp((decal.b - max(decal.r, decal.g)) * 6.0, 0.0, 1.0);
    }
    color += spec_mask * material.specular_color * global.light_specular
             * pow(max(dot(n, h), 0.0), material.shininess);
  }
  return vec4<f32>(color, 1.0);
}
