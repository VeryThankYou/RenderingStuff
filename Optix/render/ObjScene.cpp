#include <vector>
#include <string>
#include <map>

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Scene.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>

#include <GLFW/glfw3.h>

#include <optprops/Medium.h>
#include <optprops/Interface.h>
#include <optprops/load_mpml.h>
#include <optprops/milk.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <support/tinyobjloader/tiny_obj_loader.h>

#include "Directional.h"
#include "HDRLoader.h"
#include "SunSky.h"
#include "ObjScene.h"
#include "fresnel.h"

using namespace sutil;
using namespace std;

LaunchParams launch_params;
LaunchParams* d_params = nullptr;

namespace
{
  unsigned int MAX_DEPTH = 10u;
  unsigned int TRANSLUCENT_SAMPLES = 1000u;
  
  typedef Record<HitGroupData> HitGroupRecord;

  vector<tinyobj::shape_t> obj_shapes;
  vector<tinyobj::material_t> obj_materials;
  int32_t bufidx = 0;
}

ObjScene::~ObjScene()
{
  try
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch_params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch_params.lights.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

    // Destroy 3D textures 
    for(cudaTextureObject_t& texture : samplers_3d)
      CUDA_CHECK(cudaDestroyTextureObject(texture));
    samplers_3d.clear();

    for(cudaArray_t& image : images_3d)
      CUDA_CHECK(cudaFreeArray(image));
    images_3d.clear();
  }
  catch(exception& e)
  {
    cerr << "Caught exception: " << e.what() << "\n";
  }
}

void ObjScene::initScene(bool bsdf)
{
  scene.cleanup();
  if(use_envmap)
  {
    bool is_hdr = envfile.compare(envfile.length() - 3, 3, "hdr") == 0;
    if(is_hdr)
    {
      HDRLoader hdr(envfile);
      if(hdr.failed()) 
      {
        cerr << "Could not load HDR environment map called: " << envfile << endl;
        use_envmap = false;
      }
      scene.addImage(hdr.width(), hdr.height(), 32, 4, hdr.raster());
      launch_params.env_width = hdr.width();
      launch_params.env_height = hdr.height();
    }
    else
    {
      ImageBuffer img = loadImage(envfile.c_str());
      if(img.pixel_format != UNSIGNED_BYTE4)
      {
        cerr << "Environment map texture image with unknown pixel format: " << envfile << endl;
        use_envmap = false;
      }
      scene.addImage(img.width, img.height, 8, 4, img.data);
      launch_params.env_width = img.width;
      launch_params.env_height = img.height;
    }
    if(use_envmap)
      scene.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, 0, is_hdr);
  }
  if(!filenames.empty())
  {
    //loadScene(filenames[0], scene);
    loadObjs();
    analyze_materials();
    scene.createContext();
    scene.buildMeshAccels();
    scene.buildInstanceAccel();

    OPTIX_CHECK(optixInit()); // Need to initialize function table
    createPTXModule();
    createProgramGroups(bsdf);
    createPipeline();
    createSBT();
    
    bbox.invalidate();
    for(const auto instance : scene.instances())
      if(instance->world_aabb.area() < 1.0e5f) // Objects with a very large bounding box are considered background
        bbox.include(instance->world_aabb);
    cout << "Scene bounding box maximum extent: " << bbox.maxExtent() << endl;
    
    initCameraState();
    initLaunchParams(scene);
  }
}

void ObjScene::initLaunchParams(const Scene& scene)
{
  LaunchParams& lp = launch_params;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.accum_buffer), width*height*sizeof(float4)));
  lp.frame_buffer = nullptr; // Will be set when output buffer is mapped
  lp.translucent_no_of_samples = TRANSLUCENT_SAMPLES;
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&lp.translucent_samples), lp.translucent_no_of_samples*sizeof(PositionSample)
  ));
  lp.subframe_index = 0u;
  lp.max_depth = MAX_DEPTH;

  // Add light sources depending on chosen shader
  if(shadername == "arealight")
  {
    if(!extract_area_lights())
    {
      cerr << "Error: no area lights in scene. "
           << "You cannot use the area light shader if there are no emissive objects in the scene. "
           << "Objects are emissive if their ambient color is not zero."
           << endl;
      exit(0);
    }
  }
  else
    add_default_light();

  if(use_envmap)
  {
    lp.envmap = scene.getSampler(0);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.env_luminance), lp.env_width*lp.env_height*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.marginal_f), lp.env_height*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.marginal_pdf), lp.env_height*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.marginal_cdf), lp.env_height*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.conditional_pdf), lp.env_width*lp.env_height*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lp.conditional_cdf), lp.env_width*lp.env_height*sizeof(float)));
  }

  //CUDA_CHECK( cudaStreamCreate( &stream ) );
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));

  lp.handle = scene.traversableHandle();
}

void ObjScene::initCameraState()
{
  camera.setFovY(45.0f);
  camera.setLookat(bbox.center());
  camera.setEye(bbox.center() + make_float3(0.0f, 0.0f, 1.8f*bbox.maxExtent()));
}

void ObjScene::initSunSky(float ordinal_day, float solar_time, float globe_latitude,
                          float turbidity, float overcast, float sky_angle, const float3& sky_up)
{
  day_of_year = ordinal_day;
  latitude = globe_latitude;
  angle_with_south = sky_angle;
  sun_sky.setUpDir(sky_up);
  sun_sky.setTurbidity(turbidity);
  sun_sky.setOvercast(overcast);
  handleSunSkyUpdate(solar_time, overcast);
  use_sunsky = true;
}

void ObjScene::handleCameraUpdate()
{
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  launch_params.eye = camera.eye();
  camera.UVWFrame(launch_params.U, launch_params.V, launch_params.W);
  /*
  cerr
      << "Updating camera:\n"
      << "\tU: " << launch_params.U.x << ", " << launch_params.U.y << ", " << launch_params.U.z << endl
      << "\tV: " << launch_params.V.x << ", " << launch_params.V.y << ", " << launch_params.V.z << endl
      << "\tW: " << launch_params.W.x << ", " << launch_params.W.y << ", " << launch_params.W.z << endl;
      */
}

void ObjScene::handleResize(CUDAOutputBuffer<uchar4>& output_buffer, int32_t w, int32_t h)
{
  width = w;
  height = h;
  output_buffer.resize(width, height);

  // Realloc accumulation buffer
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch_params.accum_buffer)));
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&launch_params.accum_buffer),
    width*height*sizeof(float4)
  ));
}

void ObjScene::handleSunSkyUpdate(float& solar_time, float overcast)
{
  sun_sky.setOvercast(clamp(overcast, 0.0f, 1.0f));
  if(solar_time >= 24.0f)
  {
    ++day_of_year;
    solar_time -= 24.0f;
  }
  else if(solar_time < 0.01f)
  {
    --day_of_year;
    solar_time += 24.0f;
  }
  time_of_day = solar_time;

  // Use the ordinal day (day_of_year), the solar time (time_of_day), and the latitude
  // to find the spherical coordinates of the sun position (theta_sun, phi_sun).
  float theta_sun = 0.0f;
  float phi_sun = 0.0f;
  sun_sky.setSunTheta(theta_sun);
  sun_sky.setSunPhi(phi_sun + angle_with_south*M_PIf/180.0f);
  sun_sky.init();
  launch_params.sunsky = sun_sky.getParams();
  add_default_light();
}

void ObjScene::launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
  static bool first = true;

  // Launch
  uchar4* result_buffer_data = output_buffer.map();
  launch_params.frame_buffer = result_buffer_data;
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
    &launch_params,
    sizeof(LaunchParams),
    cudaMemcpyHostToDevice,
    0 // stream
  ));

  if(has_translucent)
  {
    OPTIX_CHECK(optixLaunch(
      m_pipeline,
      0,             // stream
      reinterpret_cast<CUdeviceptr>(d_params),
      sizeof(LaunchParams),
      &m_sample_sbt,
      TRANSLUCENT_SAMPLES, // launch width
      1,                   // launch height
      1                    // launch depth
    ));
  }

  if(first && use_envmap)
  {
    first = false;
    OPTIX_CHECK(optixLaunch(
      m_pipeline,
      0,             // stream
      reinterpret_cast<CUdeviceptr>(d_params),
      sizeof(LaunchParams),
      &m_env_luminance_sbt,
      launch_params.env_width,    // launch width
      launch_params.env_height,   // launch height
      1                           // launch depth
    ));
    OPTIX_CHECK(optixLaunch(
      m_pipeline,
      0,             // stream
      reinterpret_cast<CUdeviceptr>(d_params),
      sizeof(LaunchParams),
      &m_env_marginal_sbt,
      1,                        // launch width
      launch_params.env_height, // launch height
      1                         // launch depth
    ));
    OPTIX_CHECK(optixLaunch(
      m_pipeline,
      0,             // stream
      reinterpret_cast<CUdeviceptr>(d_params),
      sizeof(LaunchParams),
      &m_env_pdf_sbt,
      launch_params.env_width,  // launch width
      launch_params.env_height, // launch height
      1                         // launch depth
    ));
  }

  OPTIX_CHECK(optixLaunch(
    m_pipeline,
    0,             // stream
    reinterpret_cast<CUdeviceptr>(d_params),
    sizeof(LaunchParams),
    &m_sbt,
    width,  // launch width
    height, // launch height
    1       // launch depth
  ));
  output_buffer.unmap();
  CUDA_SYNC_CHECK();
}

void ObjScene::createPTXModule()
{
  OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

  m_pipeline_compile_options = {};
  m_pipeline_compile_options.usesMotionBlur = false;
  m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  m_pipeline_compile_options.numPayloadValues = NUM_PAYLOAD_VALUES;
  m_pipeline_compile_options.numAttributeValues = 2; // TODO
  m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  m_pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

  size_t inputSize = 0;
  const char* input = nullptr;
  if(CUDA_NVRTC_ENABLED && SAMPLES_INPUT_GENERATE_PTX)
    input = getInputData("", "render", "shaders.cu", inputSize);
  else
    input = getInputData("render", "", "shaders.cu", inputSize);

  m_ptx_module = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixModuleCreate(
    scene.context(),
    &module_compile_options,
    &m_pipeline_compile_options,
    input,
    inputSize,
    log,
    &sizeof_log,
    &m_ptx_module
  ));
}

void ObjScene::createProgramGroups(bool bsdf)
{
  OptixProgramGroupOptions program_group_options = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);

  //
  // Ray generation
  //
  {
    string raygen = "__raygen__" + cameraname;
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = m_ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = bsdf ? "__raygen__bsdf" : raygen.c_str();

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &raygen_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_raygen_prog_group
    ));
  }
  if(has_translucent)
  {

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = m_ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__sample_translucent";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &raygen_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_sample_prog_group
    ));
  }

  if(use_envmap)
  {
    {
      OptixProgramGroupDesc raygen_prog_group_desc = {};
      raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_prog_group_desc.raygen.module = m_ptx_module;
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__env_luminance";

      OPTIX_CHECK_LOG(optixProgramGroupCreate(
        scene.context(),
        &raygen_prog_group_desc,
        1,                             // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &m_env_luminance_prog_group
      ));
    }
    {
      OptixProgramGroupDesc raygen_prog_group_desc = {};
      raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_prog_group_desc.raygen.module = m_ptx_module;
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__env_marginal";

      OPTIX_CHECK_LOG(optixProgramGroupCreate(
        scene.context(),
        &raygen_prog_group_desc,
        1,                             // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &m_env_marginal_prog_group
      ));
    }
    {
      OptixProgramGroupDesc raygen_prog_group_desc = {};
      raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_prog_group_desc.raygen.module = m_ptx_module;
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__env_pdf";

      OPTIX_CHECK_LOG(optixProgramGroupCreate(
        scene.context(),
        &raygen_prog_group_desc,
        1,                             // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &m_env_pdf_prog_group
      ));
    }
  }

  //
  // Miss
  //
  {
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = m_ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = bsdf ? "__miss__ray_direction"
      : (use_envmap ? "__miss__envmap_radiance" : (use_sunsky ? "__miss__sunsky_radiance" : "__miss__constant_radiance"));
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &miss_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_radiance_miss_group
    ));

    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = nullptr;  // NULL miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &miss_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_occlusion_miss_group
    ));
  }

  //
  // Hit group
  //
  {
    // Associate the shader selected in the command line with illum 0, 1, and 2
    OptixProgramGroup m_radiance_hit_group = createShader(1, shadername);
    setShader(0, m_radiance_hit_group);
    setShader(2, m_radiance_hit_group);
    createShader(3, "mirror");            // associate the mirror shader with illum 3
    createShader(4, "transparent");       // associate the transparent shader with illum 4
    createShader(5, "glossy");            // associate the glossy shader with illum 5
    createShader(11, "metal");            // associate the metal shader with illum 11
    createShader(30, "holdout");          // associate the holdout shader with illum 30

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
      scene.context(),
      &hit_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_occlusion_hit_group
    ));
  }
}

void ObjScene::createPipeline()
{
  OptixProgramGroup program_groups[] =
  {
      m_raygen_prog_group,
      m_radiance_miss_group,
      m_occlusion_miss_group,
      getShader(1),
      m_occlusion_hit_group,
  };

  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = MAX_DEPTH + 1u;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixPipelineCreate(
    scene.context(),
    &m_pipeline_compile_options,
    &pipeline_link_options,
    program_groups,
    sizeof(program_groups)/sizeof(program_groups[0]),
    log,
    &sizeof_log,
    &m_pipeline
  ));
}


void ObjScene::createSBT()
{
  {
    const size_t raygen_record_size = sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), raygen_record_size));

    EmptyRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sbt.raygenRecord),
      &rg_sbt,
      raygen_record_size,
      cudaMemcpyHostToDevice
    ));
  }

  {
    const unsigned int ray_type_count = RAY_TYPE_COUNT;
    const size_t miss_record_size = sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&m_sbt.missRecordBase),
      miss_record_size*ray_type_count
    ));

    vector<EmptyRecord> ms_sbt(ray_type_count);
    OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_group, &ms_sbt[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_miss_group, &ms_sbt[RAY_TYPE_OCCLUSION]));

    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sbt.missRecordBase),
      &ms_sbt[0],
      miss_record_size*ray_type_count,
      cudaMemcpyHostToDevice
    ));
    m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_sbt.missRecordCount = ray_type_count;
  }

  {
    vector<HitGroupRecord> hitgroup_records;
    for(const auto mesh : scene.meshes())
    {
      for(size_t i = 0; i < mesh->material_idx.size(); ++i)
      {
        HitGroupRecord rec = {};
        const int32_t mat_idx = mesh->material_idx[i];
        if(mat_idx >= 0)
        {
          rec.data.mtl_inside = m_materials[mat_idx];
          if(rec.data.mtl_inside.opposite >= 0)
            rec.data.mtl_outside = m_materials[rec.data.mtl_inside.opposite];
          else
            rec.data.mtl_outside = MtlData();
        }
        else
        {
          rec.data.mtl_inside = MtlData();
          rec.data.mtl_outside = MtlData();
        }

        OptixProgramGroup m_radiance_hit_group = getShader(rec.data.mtl_inside.illum);
        OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hit_group, &rec));
        GeometryData::TriangleMesh triangle_mesh = {};
        triangle_mesh.positions = mesh->positions[i];
        triangle_mesh.normals = mesh->normals[i];
        triangle_mesh.texcoords[0] = mesh->texcoords[0][i];
        //triangle_mesh.colors = mesh->colors[i]; // no vertex colors in OBJ
        triangle_mesh.indices = mesh->indices[i];
        rec.data.geometry.setTriangleMesh(triangle_mesh);
        hitgroup_records.push_back(rec);

        OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_hit_group, &rec));
        hitgroup_records.push_back(rec);
      }
    }

    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase),
      hitgroup_record_size*hitgroup_records.size()
    ));
    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sbt.hitgroupRecordBase),
      hitgroup_records.data(),
      hitgroup_record_size*hitgroup_records.size(),
      cudaMemcpyHostToDevice
    ));

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
  }

  if(has_translucent)
  {
    m_sample_sbt = m_sbt;
    const size_t raygen_record_size = sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sample_sbt.raygenRecord), raygen_record_size));

    EmptyRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_sample_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sample_sbt.raygenRecord),
      &rg_sbt,
      raygen_record_size,
      cudaMemcpyHostToDevice
    ));
  }

  if(use_envmap)
  {
    m_env_luminance_sbt = m_env_marginal_sbt = m_env_pdf_sbt = m_sbt;
    {
      const size_t raygen_record_size = sizeof(EmptyRecord);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_env_luminance_sbt.raygenRecord), raygen_record_size));

      EmptyRecord rg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(m_env_luminance_prog_group, &rg_sbt));
      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_env_luminance_sbt.raygenRecord),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
      ));
    }
    {
      const size_t raygen_record_size = sizeof(EmptyRecord);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_env_marginal_sbt.raygenRecord), raygen_record_size));

      EmptyRecord rg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(m_env_marginal_prog_group, &rg_sbt));
      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_env_marginal_sbt.raygenRecord),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
      ));
    }
    {
      const size_t raygen_record_size = sizeof(EmptyRecord);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_env_pdf_sbt.raygenRecord), raygen_record_size));

      EmptyRecord rg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(m_env_pdf_prog_group, &rg_sbt));
      CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_env_pdf_sbt.raygenRecord),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
      ));
    }
  }
}

OptixProgramGroup ObjScene::createShader(int illum, string name)
{
  OptixProgramGroupOptions program_group_options = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);
  string shader = "__closesthit__" + name;
  OptixProgramGroup m_radiance_hit_group = 0;
  OptixProgramGroupDesc hit_prog_group_desc = {};
  hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
  hit_prog_group_desc.hitgroup.entryFunctionNameCH = shader.c_str();
  sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(
    scene.context(),
    &hit_prog_group_desc,
    1,                             // num program groups
    &program_group_options,
    log,
    &sizeof_log,
    &m_radiance_hit_group
  ));
  setShader(illum, m_radiance_hit_group);
  return m_radiance_hit_group;
}

void ObjScene::setShader(int illum, OptixProgramGroup closest_hit_program)
{
  if(illum < 0)
  {
    cerr << "Error: Negative identification numbers are not supported for illumination models." << endl;
    return;
  }
  while(illum >= static_cast<int>(shaders.size()))
    shaders.push_back(0);
  shaders[illum] = closest_hit_program;
}

OptixProgramGroup ObjScene::getShader(int illum)
{
  OptixProgramGroup shader = 0;
  if(illum >= 0 && illum < static_cast<int>(shaders.size()))
    shader = shaders[illum];
  
  if(!shader)
  {
    cerr << "Warning: An object uses a material with an unsupported illum identifier. Using the default shader instead." << endl;
    shader = shaders[0];
  }
  return shader;
}

void ObjScene::loadObjs()
{
  int mtl_count = 0;
  int mesh_count = 0;
  int tex_count = use_envmap ? 1 : 0;
  int tex3d_count = 0;
  for(string filename : filenames)
  {
    scanMeshObj(filename);

    for(tinyobj::material_t& mtl : obj_materials)
    {
      MtlData m_mtl;
      m_mtl.rho_d = make_float3(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);
      m_mtl.rho_s = make_float3(mtl.specular[0], mtl.specular[1], mtl.specular[2]);
      m_mtl.emission = make_float3(mtl.ambient[0], mtl.ambient[1], mtl.ambient[2]);
      m_mtl.shininess = mtl.shininess;
      m_mtl.ior = mtl.ior;
      m_mtl.illum = mtl.illum;
      if(!mtl.diffuse_texname.empty())
      {
        string path;
        size_t idx = filename.find_last_of("/\\");
        if(idx < filename.length())
          path = filename.substr(0, idx + 1);

        string textype;
        idx = mtl.diffuse_texname.find_last_of(".");
        if(idx < mtl.diffuse_texname.length())
          textype = mtl.diffuse_texname.substr(idx + 1, mtl.diffuse_texname.length());

        if(textype == "raw")
        {
          string sdf_filename = path + mtl.diffuse_texname;
          ifstream sdf_file(sdf_filename, ios::binary);
          if(!sdf_file)
            cout << "File not found: " << sdf_filename;

          cout << "Loading volume data (SDF) from " << sdf_filename << endl;
          size_t b_width = 128, b_height = 128, b_depth = 128;
          vector<float> sdf(b_width*b_height*b_depth);
          sdf_file.read(reinterpret_cast<char*>(sdf.data()), sdf.size()*sizeof(float));

          addImage3D(static_cast<int32_t>(b_width), static_cast<int32_t>(b_height), static_cast<int32_t>(b_depth), 32, 1, reinterpret_cast<void*>(sdf.data()));
          addSampler3D(cudaAddressModeMirror, cudaFilterModeLinear, tex3d_count, true);
          m_mtl.sdf_tex = getSampler3D(tex3d_count++);
        }
        else
        {
          ImageBuffer img = loadImage((path + mtl.diffuse_texname).c_str());
          if(img.pixel_format != UNSIGNED_BYTE4)
            cerr << "Texture image with unknown pixel format: " << mtl.diffuse_texname << endl;
          else
          {
            cout << "Loaded texture image " << mtl.diffuse_texname << endl;
            scene.addImage(img.width, img.height, 8, 4, img.data);
            scene.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, tex_count);
            m_mtl.base_color_tex = scene.getSampler(tex_count++);
          }
        }
      }
      m_materials.push_back(m_mtl);
      mtl_names.push_back(mtl.name);
    }
    for(vector<tinyobj::shape_t>::const_iterator it = obj_shapes.begin(); it < obj_shapes.end(); ++it)
    {
      const tinyobj::shape_t& shape = *it;
      CUdeviceptr buffer;
      auto mesh = std::make_shared<Scene::MeshGroup>();
      scene.addMesh(mesh);
      mesh->name = shape.name;
      {
        BufferView<unsigned int> buffer_view;
        scene.addBuffer(shape.mesh.indices.size()*sizeof(unsigned int), reinterpret_cast<const void*>(&shape.mesh.indices[0]));
        buffer = scene.getBuffer(bufidx++);
        buffer_view.data = buffer;
        buffer_view.byte_stride = 0;
        buffer_view.count = static_cast<uint32_t>(shape.mesh.indices.size());
        buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(unsigned int));
        mesh->indices.push_back(buffer_view);
      }
      {
        BufferView<float3> buffer_view;
        scene.addBuffer(shape.mesh.positions.size()*sizeof(float), reinterpret_cast<const void*>(&shape.mesh.positions[0]));
        buffer = scene.getBuffer(bufidx++);
        buffer_view.data = buffer;
        buffer_view.byte_stride = 0;
        buffer_view.count = static_cast<uint32_t>(shape.mesh.positions.size()/3);
        buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
        mesh->positions.push_back(buffer_view);
      }
      {
        BufferView<float3> buffer_view;
        if(shape.mesh.normals.size() > 0)
        {
          scene.addBuffer(shape.mesh.normals.size()*sizeof(float), reinterpret_cast<const void*>(&shape.mesh.normals[0]));
          buffer = scene.getBuffer(bufidx++);
          buffer_view.data = buffer;
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(shape.mesh.normals.size()/3);
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
        }
        mesh->normals.push_back(buffer_view);
      }
      {
        BufferView<Vec2f> buffer_view;
        if(shape.mesh.texcoords.size() > 0)
        {
          scene.addBuffer(shape.mesh.texcoords.size()*sizeof(float), reinterpret_cast<const void*>(&shape.mesh.texcoords[0]));
          buffer = scene.getBuffer(bufidx++);
          buffer_view.data = buffer;
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(shape.mesh.texcoords.size()/2);
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(Vec2f));
        }
        mesh->texcoords[0].push_back(buffer_view);
      }
      mesh->material_idx.push_back(shape.mesh.material_ids[0] + mtl_count);
      cerr << "\t\tNum triangles: " << mesh->indices.back().count/3 << endl;
      auto instance = std::make_shared<Scene::Instance>();
      instance->transform = get_object_transform(filename);

      Surface surface;
      surface.indices.resize(shape.mesh.indices.size()/3);
      copy(shape.mesh.indices.begin(), shape.mesh.indices.end(), &surface.indices.front().x);
      surface.positions.resize(shape.mesh.positions.size()/3);
      for(unsigned int i = 0; i < surface.positions.size(); ++i)
      {
        float4 pos = make_float4(shape.mesh.positions[i*3], shape.mesh.positions[i*3 + 1], shape.mesh.positions[i*3 + 2], 1.0f);
        surface.positions[i] = make_float3(instance->transform*pos);
      }
      //copy(shape.mesh.positions.begin(), shape.mesh.positions.end(), &surface.positions.front().x);
      if(shape.mesh.normals.size() > 0)
      {
        surface.normals.resize(shape.mesh.normals.size()/3);
        for(unsigned int i = 0; i < surface.normals.size(); ++i)
        {
          float4 normal = make_float4(shape.mesh.normals[i*3], shape.mesh.normals[i*3 + 1], shape.mesh.normals[i*3 + 2], 0.0f);
          surface.normals[i] = make_float3(instance->transform*normal);
        }
        //copy(shape.mesh.normals.begin(), shape.mesh.normals.end(), &surface.normals.front().x);
      }
      surface.no_of_faces = static_cast<unsigned int>(surface.indices.size());
      surfaces.push_back(surface);

      mesh->object_aabb.invalidate();
      for(unsigned int i = 0; i < shape.mesh.positions.size()/3; ++i)
        mesh->object_aabb.include(make_float3(shape.mesh.positions[i*3], shape.mesh.positions[i*3 + 1], shape.mesh.positions[i*3 + 2]));

      instance->mesh_idx = mesh_count++;
      instance->world_aabb = mesh->object_aabb;
      instance->world_aabb.transform(instance->transform);
      scene.addInstance(instance);
    }
    mtl_count += static_cast<int>(obj_materials.size());
    obj_materials.clear();
    obj_shapes.clear();
  }
}

void ObjScene::scanMeshObj(string m_filename)
{
  int32_t num_triangles = 0;
  int32_t num_vertices = 0;
  int32_t num_materials = 0;
  bool has_normals = false;
  bool has_texcoords = false;

  if(obj_shapes.empty())
  {
    std::string err;
    bool ret = tinyobj::LoadObj(
      obj_shapes,
      obj_materials,
      err,
      m_filename.c_str(),
      m_filename.substr(0, m_filename.find_last_of("\\/") + 1).c_str()
    );

    if(!err.empty())
      cerr << err << endl;

    if(!ret)
      throw runtime_error("MeshLoader: " + err);
  }

  //
  // Iterate over all shapes and sum up number of vertices and triangles
  //
  uint64_t num_groups_with_normals = 0;
  uint64_t num_groups_with_texcoords = 0;
  for(vector<tinyobj::shape_t>::const_iterator it = obj_shapes.begin(); it < obj_shapes.end(); ++it)
  {
    const tinyobj::shape_t& shape = *it;

    num_triangles += static_cast<int32_t>(shape.mesh.indices.size())/3;
    num_vertices += static_cast<int32_t>(shape.mesh.positions.size())/3;

    if(!shape.mesh.normals.empty())
      ++num_groups_with_normals;

    if(!shape.mesh.texcoords.empty())
      ++num_groups_with_texcoords;
  }

  //
  // We ignore normals and texcoords unless they are present for all shapes
  //
  if(num_groups_with_normals != 0)
  {
    if(num_groups_with_normals != obj_shapes.size())
      cerr << "MeshLoader - WARNING: mesh '" << m_filename
           << "' has normals for some groups but not all.  "
           << "Ignoring all normals." << endl;
    else
      has_normals = true;
  }
  if(num_groups_with_texcoords != 0)
  {
    if(num_groups_with_texcoords != obj_shapes.size())
      cerr << "MeshLoader - WARNING: mesh '" << m_filename
           << "' has texcoords for some groups but not all.  "
           << "Ignoring all texcoords." << endl;
    else
      has_texcoords = true;
  }
  num_materials = (int32_t)m_materials.size();
}

void ObjScene::addImage3D(const int32_t width, const int32_t height, const int32_t depth, const int32_t bits_per_component, const int32_t num_components, void* data)
{
  // Allocate CUDA array in device memory
  int32_t               pitch;
  cudaChannelFormatDesc channel_desc;
  switch(bits_per_component)
  {
  case 8:
    pitch = width*num_components*sizeof(uint8_t);
    switch(num_components)
    {
    case 1: channel_desc = cudaCreateChannelDesc<unsigned char>(); break;
    case 2: channel_desc = cudaCreateChannelDesc<uchar2>(); break;
    case 4: channel_desc = cudaCreateChannelDesc<uchar4>(); break;
    }
    break;
  case 32:
    pitch = width*num_components*sizeof(float);
    switch(num_components)
    {
    case 1: channel_desc = cudaCreateChannelDesc<float>(); break;
    case 2: channel_desc = cudaCreateChannelDesc<float2>(); break;
    case 4: channel_desc = cudaCreateChannelDesc<float4>(); break;
    }
    break;
  default:
    throw Exception("Unsupported bits/component in texture image");
  }

  // Default initialization for a 3D texture. There are no layers.
  cudaExtent extent = make_cudaExtent(width, height, depth);

  cudaArray_t cuda_array = nullptr;
  CUDA_CHECK(cudaMalloc3DArray(&cuda_array, &channel_desc, extent, cudaArrayDefault));

  cudaMemcpy3DParms params = { 0 };
  params.srcPtr = make_cudaPitchedPtr(data, pitch, width, height);
  params.dstArray = cuda_array;
  params.extent = extent;
  params.kind = cudaMemcpyHostToDevice;

  CUDA_CHECK(cudaMemcpy3D(&params));
  images_3d.push_back(cuda_array);
}

void ObjScene::addSampler3D(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter, const int32_t image_idx, const bool is_hdr)
{
  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = getImage3D(image_idx);

  cudaTextureDesc tex_desc = {};
  tex_desc.addressMode[0] = address_mode;
  tex_desc.addressMode[1] = address_mode;
  tex_desc.addressMode[2] = address_mode;
  tex_desc.filterMode = filter;
  tex_desc.readMode = is_hdr ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
  tex_desc.normalizedCoords = 1;
  tex_desc.maxAnisotropy = 1;
  tex_desc.maxMipmapLevelClamp = 99;
  tex_desc.minMipmapLevelClamp = 0;
  tex_desc.mipmapFilterMode = cudaFilterModePoint;
  tex_desc.borderColor[0] = 1.0f;
  tex_desc.sRGB = 0; // Is the renderer using conversion to sRGB?

  // Create texture object
  cudaTextureObject_t cuda_tex = 0;
  CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
  samplers_3d.push_back(cuda_tex);
}

void ObjScene::add_default_light()
{
  // The radiance of a directional source modeling the Sun should be equal
  // to the irradiance at the surface of the Earth.
  // We convert radiance to irradiance at the surface of the Earth using the
  // solid angle 6.74e-5 subtended by the solar disk as seen from Earth.

  // Default directional light
  vector<Directional> dir_lights(1);
  dir_lights[0].emission = use_sunsky ? sun_sky.sunColor()*6.74e-5f*(1.0f - sun_sky.getOvercast()) : light_rad;
  dir_lights[0].direction = use_sunsky ? -sun_sky.getSunDir() : normalize(light_dir);

  if(launch_params.lights.count == 0)
  {
    launch_params.lights.count = static_cast<uint32_t>(dir_lights.size());
    CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&launch_params.lights.data),
      dir_lights.size()*sizeof(Directional)
    ));
  }
  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(launch_params.lights.data),
    dir_lights.data(),
    dir_lights.size()*sizeof(Directional),
    cudaMemcpyHostToDevice
  ));
}

void ObjScene::compute_diffuse_reflectance(MtlData& mtl)
{
  float n1_over_n2 = 1.0f/mtl.ior;
  float3 sca = mtl.alb*mtl.ext;
  float3 abs = mtl.ext - sca;
  float3 sca_p = sca*(1.0f - mtl.asym);
  float3 alb_p = sca_p/(sca_p + abs);
  float F_dr = fresnel_diffuse(n1_over_n2);
  float A = (1.0f + F_dr)/(1.0f - F_dr);
  float3 transport = sqrtf(3.0f*(1.0f - alb_p));
  mtl.rho_d = alb_p*0.5f*(1.0f + expf(-4.0f/3.0f*A*transport))*expf(-transport);
}

void ObjScene::analyze_materials()
{
  vector<int> adv_mtls;
  auto& meshes = scene.meshes();
  unsigned int mesh_idx = 0;
  bool calculate_area = false;
  for(auto mesh : meshes)
  {
    bool translucent = false;
    for(unsigned int j = 0; j < mesh->material_idx.size(); ++j)
    {
      int mtl_idx = mesh->material_idx[j];
      if(mtl_idx >= 0)
      {
        const MtlData& mtl = m_materials[mtl_idx];
        if(mtl.illum == 5 || mtl.illum > 10)
          adv_mtls.push_back(mtl_idx);
        if(mtl.illum == 14)
        {
          translucent = true;
          calculate_area = true;
        }
        if(mtl.illum == 20)
          calculate_area = true;
      }
    }
    if(calculate_area)
    {
      const Surface& surface = surfaces[mesh_idx];
      float surface_area = 0.0f;
      vector<float> face_areas(surface.indices.size());
      vector<float> face_area_cdf(surface.indices.size());
      for(unsigned int i = 0; i < surface.indices.size(); ++i)
      {
        uint3 face = surface.indices[i];
        float3 p0 = surface.positions[face.x];
        float3 a = surface.positions[face.y] - p0;
        float3 b = surface.positions[face.z] - p0;
        face_areas[i] = 0.5f*length(cross(a, b));
        face_area_cdf[i] = surface_area + face_areas[i];
        surface_area += face_areas[i];
      }
      launch_params.surface_area = surface_area;

      if(translucent && !has_translucent)
      {
        has_translucent = true;
        {
          BufferView<uint3> buffer_view;
          scene.addBuffer(surface.indices.size()*sizeof(uint3), reinterpret_cast<const void*>(&surface.indices[0]));
          buffer_view.data = scene.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(surface.indices.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(uint3));
          launch_params.translucent_idxs = buffer_view;
        }
        {
          BufferView<float3> buffer_view;
          scene.addBuffer(surface.positions.size()*sizeof(float3), reinterpret_cast<const void*>(&surface.positions[0]));
          buffer_view.data = scene.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(surface.positions.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
          launch_params.translucent_verts = buffer_view;
        }
        {
          BufferView<float3> buffer_view;
          scene.addBuffer(surface.normals.size()*sizeof(float3), reinterpret_cast<const void*>(&surface.normals[0]));
          buffer_view.data = scene.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(surface.normals.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
          launch_params.translucent_norms = buffer_view;
        }
        if(surface.normals.size() == 0)
          cerr << "Warning: Translucent object was loaded for surface sampling but has no vertex normals." << endl;
        if(surface_area > 0.0f)
          for(unsigned int i = 0; i < surface.indices.size(); ++i)
            face_area_cdf[i] /= surface_area;
        {
          BufferView<float> buffer_view;
          scene.addBuffer(face_area_cdf.size()*sizeof(float), reinterpret_cast<const void*>(&face_area_cdf[0]));
          buffer_view.data = scene.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(face_area_cdf.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
          launch_params.translucent_face_area_cdf = buffer_view;
        }
      }
    }
    ++mesh_idx;
  }
  if(adv_mtls.size() > 0)
  {
    load_mpml("../../models/media.mpml");
    map<string, Medium>& media = get_media();
    map<string, Interface>& interfaces = get_interfaces();
    unhomogenized_milk(media["milk"], 0.5);
    for(unsigned int j = 0; j < adv_mtls.size(); ++j)
    {
      int mtl_idx = adv_mtls[j];
      MtlData& mtl = m_materials[mtl_idx];
      auto i_iter = interfaces.find(mtl_names[mtl_idx]);
      if(i_iter != interfaces.end() && i_iter->second.med_out)
      {
        unsigned int i = 0;
        while(i < mtl_names.size())
        {
          if(i_iter->second.med_out->name == mtl_names[i])
          {
            mtl.opposite = i;
            break;
          }
          ++i;
        }
        if(i == mtl_names.size())
        {
          mtl_names.push_back(i_iter->second.med_out->name);
          MtlData outside;
          outside.illum = mtl.illum;
          outside.opposite = mtl_idx;
          m_materials.push_back(outside);
          adv_mtls.push_back(i);
          mtl.opposite = i;
        }
      }
    }
    for(unsigned int j = 0; j < adv_mtls.size(); ++j)
    {
      int mtl_idx = adv_mtls[j];
      MtlData& mtl = m_materials[mtl_idx];
      auto m_iter = media.find(mtl_names[mtl_idx]);
      auto i_iter = interfaces.find(mtl_names[mtl_idx]);
      Medium* med = nullptr;
      if(m_iter != media.end())
        med = &m_iter->second;
      else if(i_iter != interfaces.end() && !i_iter->second.med_in->name.empty())
        med = i_iter->second.med_in;
      if(med)
      {
        med->fill_rgb_data();
        Color< complex<double> >& ior = med->get_ior(rgb);
        Color<double>& alb = med->get_albedo(rgb);
        Color<double>& ext = med->get_extinction(rgb);
        Color<double>& asym = med->get_asymmetry(rgb);
        m_complex ior_x = { static_cast<float>(ior[0].real()), static_cast<float>(ior[0].imag()) };
        m_complex ior_y = { static_cast<float>(ior[1].real()), static_cast<float>(ior[1].imag()) };
        m_complex ior_z = { static_cast<float>(ior[2].real()), static_cast<float>(ior[2].imag()) };
        cout << "Complex IoR: (" << ior_x.re << " + i " << ior_x.im << ", " << ior_y.re << " + i " << ior_y.im << ", " << ior_z.re << " + i " << ior_z.im << ")" << endl;
        mtl.ior = static_cast<float>(ior[0].real() + ior[1].real() + ior[2].real())/3.0f;
        mtl.c_recip_ior = { 1.0f/ior_x, 1.0f/ior_y, 1.0f/ior_z };
        mtl.alb = make_float3(static_cast<float>(alb[0]), static_cast<float>(alb[1]), static_cast<float>(alb[2]));
        mtl.ext = make_float3(static_cast<float>(ext[0]), static_cast<float>(ext[1]), static_cast<float>(ext[2]))*scene_scale;
        mtl.asym = make_float3(static_cast<float>(asym[0]), static_cast<float>(asym[1]), static_cast<float>(asym[2]));
        cout << "Albedo:     (" << mtl.alb.x << ", " << mtl.alb.y << ", " << mtl.alb.z << ")" << endl;
        cout << "Extinction: (" << mtl.ext.x << ", " << mtl.ext.y << ", " << mtl.ext.z << ")" << endl;
        cout << "Asymmetry:  (" << mtl.asym.x << ", " << mtl.asym.y << ", " << mtl.asym.z << ")" << endl;
        compute_diffuse_reflectance(mtl);
      }
      else
      {
        m_complex c_recip_ior = { 1.0f/mtl.ior, 0.0f };
        mtl.c_recip_ior = { c_recip_ior, c_recip_ior, c_recip_ior };
      }
    }
  }
}

unsigned int ObjScene::extract_area_lights()
{
  vector<uint2> lights;
  auto& meshes = scene.meshes();
  int mesh_idx = 0;
  for(auto mesh : meshes)
  {
    for(unsigned int j = 0; j < mesh->material_idx.size(); ++j)
    {
      int mtl_idx = mesh->material_idx[j];
      const MtlData& mtl = m_materials[mtl_idx];
      bool emissive = false;
      for(unsigned int k = 0; k < 3; ++k)
        emissive = emissive || *(&mtl.emission.x + k) > 0.0f;
      if(emissive)
        lights.push_back(make_uint2(mesh_idx, mtl_idx));
    }
    ++mesh_idx;
  }
  Surface lightsurf;
  vector<float3> emission;
  for(unsigned int j = 0; j < lights.size(); ++j)
  {
    uint2 light = lights[j];
    auto mesh = meshes[light.x];
    const Surface& surface = surfaces[light.x];
    unsigned int no_of_verts = static_cast<unsigned int>(lightsurf.positions.size());
    lightsurf.positions.insert(lightsurf.positions.end(), surface.positions.begin(), surface.positions.end());
    lightsurf.normals.insert(lightsurf.normals.end(), surface.positions.begin(), surface.positions.end());
    lightsurf.indices.insert(lightsurf.indices.end(), surface.indices.begin(), surface.indices.end());
    if(surface.normals.size() > 0)
      for(unsigned int k = no_of_verts; k < lightsurf.normals.size(); ++k)
        lightsurf.normals[k] = surface.normals[k - no_of_verts];
    for(unsigned int k = lightsurf.no_of_faces; k < lightsurf.indices.size(); ++k)
    {
      lightsurf.indices[k] += make_uint3(no_of_verts);
      if(surface.normals.size() == 0)
      {
        uint3 face = lightsurf.indices[k];
        float3 p0 = lightsurf.positions[face.x];
        float3 a = lightsurf.positions[face.y] - p0;
        float3 b = lightsurf.positions[face.z] - p0;
        lightsurf.normals[face.x] = lightsurf.normals[face.y] = lightsurf.normals[face.z] = normalize(cross(a, b));
      }
    }
    emission.insert(emission.end(), surface.no_of_faces, m_materials[light.y].emission);
    lightsurf.no_of_faces += surface.no_of_faces;
  }
  {
    BufferView<uint3> buffer_view;
    scene.addBuffer(lightsurf.indices.size()*sizeof(uint3), reinterpret_cast<const void*>(&lightsurf.indices[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.indices.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(uint3));
    launch_params.light_idxs = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    scene.addBuffer(lightsurf.positions.size()*sizeof(float3), reinterpret_cast<const void*>(&lightsurf.positions[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.positions.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    launch_params.light_verts = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    scene.addBuffer(lightsurf.normals.size()*sizeof(float3), reinterpret_cast<const void*>(&lightsurf.normals[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.normals.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    launch_params.light_norms = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    scene.addBuffer(emission.size()*sizeof(float3), reinterpret_cast<const void*>(&emission[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(emission.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    launch_params.light_emission = buffer_view;
  }
  float surface_area = 0.0f;
  vector<float> face_areas(lightsurf.no_of_faces);
  vector<float> face_area_cdf(lightsurf.no_of_faces);
  for(unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
  {
    uint3 face = lightsurf.indices[i];
    float3 p0 = lightsurf.positions[face.x];
    float3 a = lightsurf.positions[face.y] - p0;
    float3 b = lightsurf.positions[face.z] - p0;
    face_areas[i] = 0.5f*length(cross(a, b));
    face_area_cdf[i] = surface_area + face_areas[i];
    surface_area += face_areas[i];
  }
  if(surface_area > 0.0f)
    for(unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
      face_area_cdf[i] /= surface_area;
  launch_params.light_area = surface_area;
  {
    BufferView<float> buffer_view;
    scene.addBuffer(face_area_cdf.size()*sizeof(float), reinterpret_cast<const void*>(&face_area_cdf[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(face_area_cdf.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
    launch_params.light_face_area_cdf = buffer_view;
  }
  return static_cast<unsigned int>(lights.size());
}

Matrix4x4 ObjScene::get_object_transform(string filename) const
{
  size_t idx = filename.find_last_of("\\/") + 1;
  if(idx < filename.length())
  {
    if(filename.compare(idx, 7, "cornell") == 0)
      return Matrix4x4::scale(make_float3(0.025f))*Matrix4x4::rotate(M_PIf, make_float3(0.0f, 1.0f, 0.0f));
    else if(filename.compare(idx, 6, "dragon") == 0)
      return Matrix4x4::rotate(-M_PI_2f, make_float3(1.0, 0.0, 0.0));
    else if(filename.compare(idx, 5, "bunny") == 0)
      return Matrix4x4::translate(make_float3(-3.0f, -0.84f, -8.0f))*Matrix4x4::scale(make_float3(25.0f))*Matrix4x4::rotate(0.02f, make_float3(1.0f, 0.0f, 0.0f));
    else if(filename.compare(idx, 6, "closed") == 0)
      return Matrix4x4::scale(make_float3(25.0f));
    else if(filename.compare(idx, 12, "justelephant") == 0)
      return Matrix4x4::translate(make_float3(-10.0f, 3.0f, -2.0f))*Matrix4x4::rotate(0.5f, make_float3(0.0f, 1.0f, 0.0f));
    else if(filename.compare(idx, 10, "glass_wine") == 0)
      return Matrix4x4::scale(make_float3(5.0f));
  }
  return Matrix4x4::identity();
}
