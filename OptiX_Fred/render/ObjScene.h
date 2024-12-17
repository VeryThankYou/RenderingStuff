#pragma once

#include <string>
#include <vector>
#include <optix.h>
#include <sutil/Scene.h>
#include <sutil/CUDAOutputBuffer.h>

#include "structs.h"

class ObjScene
{
public:
  ObjScene(const std::vector<std::string>& obj_filenames,
           const std::string& shader_name, const std::string& camera_name, const std::string& env_filename,
           int32_t frame_width, int32_t frame_height,
           const float3& light_direction, const float3& light_emission)
    : filenames(obj_filenames), shadername(shader_name), cameraname(camera_name), envfile(env_filename),
      resize_dirty(false), minimized(false),
      width(frame_width), height(frame_height), scene_scale(1.0e-2f),
      light_dir(light_direction), light_rad(light_emission),
      has_translucent(false), use_sunsky(false),
      day_of_year(255.0f), time_of_day(12.0f), latitude(55.78f), angle_with_south(0.0f)
  { 
    if(shadername.empty())
      shadername = "normals";
    if(cameraname.empty())
      cameraname = "pinhole";
    use_envmap = !envfile.empty();
    handleSunSkyUpdate(time_of_day, sun_sky.getOvercast());
  }

  ~ObjScene();

  void initScene(bool bsdf = false);
  void initLaunchParams(const sutil::Scene& scene);
  void initCameraState();
  void initSunSky(float ordinal_day, float solar_time, float globe_latitude,
                  float sky_turbidity, float overcast, float sky_angle, const float3& sky_up);

  void handleCameraUpdate();
  void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, int32_t w, int32_t h);
  void handleSunSkyUpdate(float& solar_time, float overcast);
  void handleLightUpdate() { add_default_light(); }

  void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer);

  // Window resize state
  bool resize_dirty;
  bool minimized;

  // Camera state
  sutil::Camera camera;
  int32_t width;
  int32_t height;
  float scene_scale;

  // Default light configuration
  float3 light_dir;
  float3 light_rad;

private:
  void loadObjs();
  void scanMeshObj(std::string m_filename);
  void addImage3D(const int32_t width, const int32_t height, const int32_t depth, const int32_t bits_per_component, const int32_t num_components, void* data);
  void addSampler3D(cudaTextureAddressMode address_mode, cudaTextureFilterMode  filter_mode, const int32_t image_idx, const bool is_hdr = false);
  cudaArray_t getImage3D(int32_t image_index) const { return images_3d[image_index]; }
  cudaTextureObject_t getSampler3D(int32_t sampler_index) const { return samplers_3d[sampler_index]; }
  void add_default_light();
  void compute_diffuse_reflectance(MtlData& mtl);
  void analyze_materials();
  unsigned int extract_area_lights();
  sutil::Matrix4x4 get_object_transform(std::string filename) const;
  void createPTXModule();
  void createProgramGroups(bool bsdf = false);
  void createPipeline();
  void createSBT();

  OptixProgramGroup createShader(int illum, std::string name);
  void setShader(int illum, OptixProgramGroup closest_hit_program);
  OptixProgramGroup getShader(int illum);

  std::vector<MtlData> m_materials;
  std::vector<std::string> mtl_names;
  OptixShaderBindingTable m_sbt = {};
  OptixShaderBindingTable m_sample_sbt = {};
  OptixShaderBindingTable m_env_luminance_sbt = {};
  OptixShaderBindingTable m_env_marginal_sbt = {};
  OptixShaderBindingTable m_env_pdf_sbt = {};
  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixPipeline m_pipeline = 0;
  OptixModule m_ptx_module = 0;
  OptixProgramGroup m_raygen_prog_group = 0;
  OptixProgramGroup m_sample_prog_group = 0;
  OptixProgramGroup m_env_luminance_prog_group = 0;
  OptixProgramGroup m_env_marginal_prog_group = 0;
  OptixProgramGroup m_env_pdf_prog_group = 0;
  OptixProgramGroup m_radiance_miss_group = 0;
  OptixProgramGroup m_occlusion_miss_group = 0;
  OptixProgramGroup m_feeler_miss_group = 0;
  std::vector<OptixProgramGroup> shaders;
  OptixProgramGroup m_occlusion_hit_group = 0;
  OptixProgramGroup m_feeler_hit_group = 0;

  struct Surface
  {
    unsigned int no_of_faces = 0;
    std::vector<uint3> indices;
    std::vector<float3> positions;
    std::vector<float3> normals;
  };

  std::vector<std::string> filenames;
  std::vector<Surface> surfaces;
  std::string shadername;
  std::string cameraname;
  std::string envfile;
  sutil::Scene scene;
  sutil::Aabb bbox;
  bool has_translucent;
  bool use_envmap;
  bool use_sunsky;

  PreethamSunSky sun_sky;
  float day_of_year;
  float time_of_day;
  float latitude;
  float angle_with_south;

  std::vector<cudaArray_t> images_3d;
  std::vector<cudaTextureObject_t> samplers_3d;
};

