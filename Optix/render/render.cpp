//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/whitted.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm> 

#include "QuatTrackBall.h"
#include "ObjScene.h"

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

using namespace std;
using namespace sutil;

extern LaunchParams launch_params;

namespace
{
  int32_t width = 720;
  int32_t height = 720;

  double render_time_split = 0.0;

  bool resize_dirty = false;
  bool minimized = false;
  bool save_image = false;
  bool export_raw = false;
  bool import_raw = false;
  bool progressive = true;
  
  // Sky state
  bool sky_changed = false;
  float time_of_day = 12.0f;
  float clouds = 0.0f;

  // Light state
  bool light_changed = false;
  float theta_i = 54.7356f;
  float phi_i = 45.0f;

  // Mouse state
  bool camera_changed = true;
  bool is_spinning = false;
  QuatTrackBall* trackball = 0;
  int32_t mouse_button = -1;
  float cam_const = 0.41421356f;
  float key_spin_speed = 0.0f;

  void save_view(const string& filename)
  {
    if(trackball)
    {
      ofstream ofs(filename.c_str(), ofstream::binary);
      if(ofs)
      {
        ofs.write(reinterpret_cast<const char*>(trackball), sizeof(QuatTrackBall));
        ofs.write(reinterpret_cast<const char*>(&cam_const), sizeof(float));
      }
      ofs.close();
      cout << "Camera settings stored in a file called " << filename << endl;
    }
  }

  void load_view(const string& filename)
  {
    if(trackball)
    {
      ifstream ifs_view(filename.c_str(), ifstream::binary);
      if(ifs_view)
      {
        ifs_view.read(reinterpret_cast<char*>(trackball), sizeof(QuatTrackBall));
        ifs_view.read(reinterpret_cast<char*>(&cam_const), sizeof(float));
      }
      ifs_view.close();
      float3 eye, lookat, up;
      float vfov = atanf(cam_const)*360.0f*M_1_PIf;
      trackball->get_view_param(eye, lookat, up);
      cout << "Loaded view: eye [" << eye.x << ", " << eye.y << ", " << eye.z
           << "], lookat [" << lookat.x << ", " << lookat.y << ", " << lookat.z
           << "], up [" << up.x << ", " << up.y << ", " << up.z
           << "], vfov " << vfov << endl;
      camera_changed = true;
    }
  }

  float3 get_light_direction()
  {
    float theta = theta_i*M_PIf/180.0f;
    float phi = phi_i*M_PIf/180.0f;
    float sin_theta = sinf(theta);
    return -make_float3(sin_theta*cosf(phi), sin_theta*sinf(phi), cosf(theta));
  }
}

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);

  if(action == GLFW_PRESS)
  {
    mouse_button = button;
    switch(button)
    {
    case GLFW_MOUSE_BUTTON_LEFT:
      trackball->grab_ball(ORBIT_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      trackball->grab_ball(DOLLY_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      trackball->grab_ball(PAN_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    }
  }
  else
  {
    mouse_button = -1;
    if(trackball->get_current_action() == ORBIT_ACTION)
    {
      if(!trackball->is_spinning())
      {
        trackball->release_ball();
        trackball->stop_spin();
        is_spinning = false;
      }
      else
        is_spinning = true;
    }
  }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
  if(mouse_button >= 0)
  {
    trackball->roll_ball(make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
    camera_changed = true;
  }
}

static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
  // Keep rendering at the current resolution when the window is minimized.
  if(minimized)
    return;

  // Output dimensions must be at least 1 in both x and y.
  ensureMinimumSize(res_x, res_y);

  width = res_x;
  height = res_y;
  camera_changed = true;
  resize_dirty = true;
}

static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
  minimized = (iconified > 0);
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t mods)
{
  if(action == GLFW_PRESS || action == GLFW_REPEAT)
  {
    switch(key)
    {
    case GLFW_KEY_Q:      // Quit the program using <Q>
    case GLFW_KEY_ESCAPE: // Quit the program using <esc>
      glfwSetWindowShouldClose(window, true);
      break;
    case GLFW_KEY_S:      // Save the rendered image using <S>
      if(action == GLFW_PRESS)
        save_image = true;
      break;
    case GLFW_KEY_R:      // Toggle progressive rendering using <R>
      if(action == GLFW_PRESS)
      {
        cout << "Samples per pixel: " << launch_params.subframe_index << endl;
        progressive = !progressive;
        cout << "Progressive sampling is " << (progressive ? "on." : "off.") << endl;
      }
      break;
    case GLFW_KEY_Z:      // Zoom using 'z' or 'Z'
    {
      int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
      int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
      cam_const *= rshift || lshift ? 1.05f : 1.0f/1.05f;
      camera_changed = true;
      cout << "Vertical field of view: " << atanf(cam_const)*360.0f*M_1_PIf << endl;
      break;
    }
    case GLFW_KEY_O:      // Save current view to a file called view using <O> (output)
      if(action == GLFW_PRESS)
        save_view("view");
      break;
    case GLFW_KEY_I:      // Load current view from a file called view using <I> (input)
      if(action == GLFW_PRESS)
        load_view("view");
      break;
    case GLFW_KEY_H:      // Change the time of the day in half-hour steps using 'h' or 'H'
      {
        int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
        int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
        time_of_day += rshift || lshift ? -0.5f : 0.5f;
        sky_changed = true;
        cout << "The solar time is now: " << time_of_day << endl;
        break;
      }
    case GLFW_KEY_C:      // Change the cloud cover using 'c' or 'C'
      {
        int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
        int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
        clouds += rshift || lshift ? -0.05f : 0.05f;
        clouds = clamp(clouds, 0.0f, 1.0f);
        sky_changed = true;
        cout << "The cloud cover is now: " << clouds*100.0f << "%" << endl;
        break;
      }
    case GLFW_KEY_E:      // Export raw image data using 'e'
      {
        int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
        int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
        if(action == GLFW_PRESS)
        {
          if(rshift || lshift)
            import_raw = true;
          else
          {
            save_image = true;
            export_raw = true;
          }
        }
        break;
      }
    case GLFW_KEY_V:
      {
        float3 eye, lookat, up;
        trackball->get_view_param(eye, lookat, up);
        
        cout << "eye:    [" << eye.x << ", " << eye.y << ", " << eye.z << "]" << endl
             << "lookat: [" << lookat.x << ", " << lookat.y << ", " << lookat.z << "]" << endl
             << "up:     [" << up.x << ", " << up.y << ", " << up.z << "]" << endl
             << "vfov:   " << atanf(cam_const)*360.0f*M_1_PIf << endl;
        break;
      }
    case GLFW_KEY_KP_ADD: // Increment the angle of incidence using '+'
      {
        theta_i = fminf(theta_i + 1.0f, 90.0f);
        cout << "Angle of incidence: " << static_cast<int>(theta_i) << endl;
        light_changed = true;
        break;
      }
    case GLFW_KEY_KP_SUBTRACT: // Decrement the angle of incidence using '-'
      {
        theta_i = fmaxf(theta_i - 1.0f, -90.0f);
        cout << "Angle of incidence: " << static_cast<int>(theta_i) << endl;
        light_changed = true;
        break;
      }
    case GLFW_KEY_UP:
    case GLFW_KEY_DOWN:
    case GLFW_KEY_LEFT:
    case GLFW_KEY_RIGHT:
      {
        float s = 1.0f - 2.0f*(key == GLFW_KEY_LEFT || key == GLFW_KEY_UP);
        is_spinning = mods&GLFW_MOD_SHIFT;
        key_spin_speed += s*static_cast<float>(is_spinning)*0.1f;
        key_spin_speed *= static_cast<float>(is_spinning);
        float speed = is_spinning ? key_spin_speed : s*10.0f;
        float v_x = speed*static_cast<float>(key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT);
        float v_y = speed*static_cast<float>(key == GLFW_KEY_UP || key == GLFW_KEY_DOWN);
        trackball->grab_ball(ORBIT_ACTION, make_float2(width*0.5f, height*0.5f));
        trackball->roll_ball(make_float2(width*0.5f + v_x, height*0.5f + v_y));
        if(!is_spinning)
        {
          trackball->release_ball();
          trackball->stop_spin();
        }
        mouse_button = -1;
        camera_changed = true;
        break;
      }
    }
  }
}

static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
  //if(trackball.wheelEvent((int)yscroll))
  //  camera_changed = true;
}

//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
  cerr << "Usage  : " << argv0 << " [options] any_object.obj [another.obj ...]" << endl
    << "Options: --help    | -h                 Print this usage message" << endl
    << "         --shader  | -sh <shader>       Specify the closest hit program to be used for shading" << endl
    << "         --camera  | -cam <raygen>      Specify the ray generation program" << endl
    << "                   | -env <filename>    Specify the environment map to be loaded in panoramic format" << endl
    << "                   | -bgc <r> <g> <b>   Specify RGB background color (not used if env is available)" << endl
    << "         --dim=<width>x<height>         Set image dimensions; defaults to 768x768" << endl
    << "         --no-gl-interop                Disable GL interop for display" << endl
    << "         --file    | -f <filename>      File for image output" << endl
    << "         --bsdf                         Render a BSDF slice in [-1,1]x[-1,1] of the xy-plane" << endl
    << "         --samples | -s                 Number of samples per pixel if rendering to file (default 16)" << endl
    << "                   | -sky <l> <t> <o>   Use the Preetham sun and sky model (latitude <l>, turbidity <t>, overcast <o>)" << endl
    << "                   | -t <d> <t>         Set the ordinal day <d> and solar time <t> for the sun and sky model" << endl
    << "                   | -r <a> <x> <y> <z> Angle and axis (up vector) for rotation of environment" << endl
    << "                   | -sc <s>            Scene scale <s> for scaling optical properties defined per meter (default 1e-2)" << endl
    << "                   | -dir <th> <ph>     Direction of default light in spherical coordinates (polar <th>, azimuth <ph>)" << endl
    << "                   | -rad <r> <g> <b>   Specify RGB radiance of default directional light (default PI)" << endl
    << "                   | -srgb              Convert output image to sRGB" << endl
    << "                   | -bf <s>            Beam factor in [0,1]: the ratio of beam to microgeometry width (default 1)" << endl;
  exit(0);
}

// Avoiding case sensitivity
void lower_case(char& x)
{
  x = tolower(x);
}
inline void lower_case_string(std::string& s)
{
  for_each(s.begin(), s.end(), lower_case);
}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, ObjScene& scene)
{
  bool reset = camera_changed || sky_changed || light_changed || resize_dirty;

  if(resize_dirty)
  {
    trackball->set_screen_window(width, height);
    scene.handleResize(output_buffer, width, height);
    resize_dirty = false;
  }
  if(camera_changed)
  {
    float3 eye, lookat, up;
    trackball->get_view_param(eye, lookat, up);
    scene.camera.setEye(eye);
    scene.camera.setLookat(lookat);
    scene.camera.setUp(up);
    scene.camera.setFovY(atanf(cam_const)*360.0f*M_1_PIf);
    scene.handleCameraUpdate();
    camera_changed = false;
  }
  if(sky_changed)
  {
    scene.handleSunSkyUpdate(time_of_day, clouds);
    sky_changed = false;
  }
  if(light_changed)
  {
    scene.light_dir = get_light_direction();
    scene.handleLightUpdate();
    light_changed = false;
  }

  // Update params on device
  if(reset)
  {
    LaunchParams& lp = launch_params;
    lp.subframe_index = 0;
    int size_buffer = width*height*4;
    uchar4* result_buffer_data = output_buffer.map();
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(result_buffer_data), 0, size_buffer*sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(lp.accum_buffer), 0, size_buffer*sizeof(float)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(lp.translucent_samples), 0, lp.translucent_no_of_samples*sizeof(PositionSample)));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
    ::render_time_split = 0.0;
  }
}

void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO());
}

void saveImage(sutil::CUDAOutputBuffer<uchar4>& output_buffer, string outfile)
{
  sutil::ImageBuffer buffer;
  buffer.data = output_buffer.getHostPointer();
  buffer.width = output_buffer.width();
  buffer.height = output_buffer.height();
  buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
  sutil::saveImage(outfile.c_str(), buffer, true);
  cout << "Rendered image stored in " << outfile << endl;
}

void exportRawImage(string outfile)
{
  // Get image info
  size_t name_end = outfile.find_last_of('.');
  string name = outfile.substr(0, name_end);

  // Write image info in .txt-file 
  ofstream ofs_data(name + ".txt");
  if(ofs_data.bad())
    return;
  ofs_data << launch_params.subframe_index << " " << ::render_time_split << endl << width << " " << height << endl;
  ofs_data << theta_i << " " << phi_i;
  ofs_data.close();

  // Copy buffer data from device to host
  int size_buffer = width*height*4;
  float* mapped = new float[size_buffer];
  CUDA_CHECK(cudaMemcpyAsync(mapped, launch_params.accum_buffer, size_buffer*sizeof(float), cudaMemcpyDeviceToHost, 0));

  // Export image data to binary .raw-file
  ofstream ofs_image;
  ofs_image.open(name + ".raw", ios::binary);
  if(ofs_image.bad())
  {
    cerr << "Error when exporting file" << endl;
    return;
  }

  int size_image = width*height*3;
  float* converted = new float[size_image];
  float average = 0.0f;
  for(int i = 0; i < size_image/3; ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      float value = mapped[i*4 + j];
      converted[i*3 + j] = value;
      average += value;
    }
  }
  average /= size_image;
  delete[] mapped;
  ofs_image.write(reinterpret_cast<const char*>(converted), size_image*sizeof(float));
  ofs_image.close();
  delete[] converted;
  cout << "Exported buffer to " << name << ".raw (avg: " << average << ")" << endl;
}

void importRawImage(string infile, sutil::CUDAOutputBuffer<uchar4>& output_buffer, ObjScene& scene)
{
  import_raw = false;
  size_t name_end = infile.find_last_of('.');
  string name = infile.substr(0, name_end);

  // import view
  cout << "Importing " << name << "." << endl;
  load_view("view");

  // import render data
  int32_t sample_count, old_width, old_height;
  double render_time = 0.0f;
  ifstream ifs_data(name + ".txt");
  if(!ifs_data)
  {
    cout << "Could not find " << name + ".txt" << " for raw image import." << endl;
    return;
  }
  //ifs_data >> sample_count >> old_width >> old_height; // for old exports
  ifs_data >> sample_count >> render_time >> old_width >> old_height;
  ifs_data.close();

  if(old_width != width || old_height != height)
  {
    cout << "Resolution mismatch between current render resolution and that of the render result to be imported." << endl;
    return;
  }

  vector<float> img_vec3(width*height*3);
  ifstream ifs_image(name + ".raw", ifstream::binary);
  if(!ifs_image)
  {
    cout << "Could not find " << name + ".raw" << " for raw image import." << endl;
    return;
  }
  ifs_image.read(reinterpret_cast<char*>(img_vec3.data()), img_vec3.size()*sizeof(float));
  ifs_image.close();

  vector<float> img_vec4(width*height*4);
  vector<unsigned char> display_img(width*height*4);
  for(size_t i = 0; i < width*height; ++i)
  {
    for(size_t j = 0; j < 3; ++j)
    {
      //img_vec4[i*4 + j] = img_vec3[i*3 + j]*sample_count; // for old exports
      img_vec4[i*4 + j] = img_vec3[i*3 + j];
      display_img[i*4 + j] = static_cast<unsigned char>(img_vec4[i*4 + j]*255.0f);
    }
    img_vec4[i*4 + 3] = 1.0f;
    display_img[i*4 + 3] = 255;
  }

  updateState(output_buffer, scene);
  LaunchParams& lp = launch_params;
  lp.subframe_index = sample_count;
  ::render_time_split = render_time;
  size_t size_buffer = img_vec4.size();
  uchar4* result_buffer_data = output_buffer.map();
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(result_buffer_data),
    reinterpret_cast<void*>(display_img.data()),
    size_buffer*sizeof(unsigned char),
    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(lp.accum_buffer),
    reinterpret_cast<void*>(img_vec4.data()),
    size_buffer*sizeof(float),
    cudaMemcpyHostToDevice));
  output_buffer.unmap();
  CUDA_SYNC_CHECK();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

  //
  // Parse command line options
  //
  vector<string> filenames;
  string filename;
  string shadername;
  string cameraname;
  string envname;
  string outfile;
  bool outfile_selected = false;
  bool use_sunsky = false;
  bool bsdf = false;
  float latitude = 55.78f;
  float turbidity = 2.4f;
  float ordinal_day = 255.0f;
  float sky_angle = 0.0f;
  float3 sky_up = make_float3(0.0f, 1.0f, 0.0f);
  float3 light_dir = make_float3(-1.0f);
  float3 emission = make_float3(M_PIf);
  unsigned int samples = 16;
  float scene_scale = 1.0e-2f;
  launch_params.miss_color = make_float3(0.8f, 0.9f, 1.0f);
  launch_params.use_srgb = false;
  launch_params.beam_factor = 1.0f;

  for( int i = 1; i < argc; ++i )
  {
    const std::string arg = argv[i];
    if(arg == "--help" || arg == "-h")
    {
      printUsageAndExit( argv[0] );
    }
    else if(arg == "-sh" || arg == "--shader")
    {
      if(i == argc - 1)
        printUsageAndExit(argv[0]);
      shadername = argv[++i];
      lower_case_string(shadername);
    }
    else if(arg == "-cam" || arg == "--camera")
    {
      if(i == argc - 1)
        printUsageAndExit(argv[0]);
      cameraname = argv[++i];
      lower_case_string(cameraname);
    }
    else if(arg == "-env")
    {
      if(i == argc - 1)
        printUsageAndExit(argv[0]);

      envname = argv[++i];
      string file_extension;
      size_t idx = envname.find_last_of('.');
      if(idx < envname.length())
      {
        file_extension = envname.substr(idx, envname.length() - idx);
        lower_case_string(file_extension);
      }
      if(file_extension == ".png" || file_extension == ".ppm" || file_extension == ".hdr")
        lower_case_string(envname);
      else
      {
        cerr << "Please use environment maps in .png or .ppm  or .hdr format. Received: '" << envname << "'" << endl;
        printUsageAndExit(argv[0]);
      }
    }
    else if(arg == "-bgc")
    {
      if(i >= argc - 3)
        printUsageAndExit(argv[0]);
      launch_params.miss_color.x = static_cast<float>(atof(argv[++i]));
      launch_params.miss_color.y = static_cast<float>(atof(argv[++i]));
      launch_params.miss_color.z = static_cast<float>(atof(argv[++i]));
    }
    else if(arg.substr(0, 6) == "--dim=")
    {
      const std::string dims_arg = arg.substr(6);
      sutil::parseDimensions(dims_arg.c_str(), width, height);
    }
    else if(arg == "--no-gl-interop")
    {
      output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    }
    else if(arg == "--file" || arg == "-f")
    {
      if(i >= argc - 1)
        printUsageAndExit(argv[0]);
      outfile = argv[++i];
      outfile_selected = true;
    }
    else if(arg == "--samples" || arg == "-s")
    {
      if(i >= argc - 1)
        printUsageAndExit(argv[0]);
      samples = atoi(argv[++i]);
    }
    else if(arg == "--bsdf")
    {
      bsdf = true;
    }
    else if(arg == "-sky")
    {
      if(i >= argc - 3)
        printUsageAndExit(argv[0]);
      latitude = static_cast<float>(atof(argv[++i]));
      turbidity = static_cast<float>(atof(argv[++i]));
      clouds = static_cast<float>(atof(argv[++i]));
      use_sunsky = true;
    }
    else if(arg == "-t")
    {
      if(i >= argc - 2)
        printUsageAndExit(argv[0]);
      ordinal_day = static_cast<float>(atoi(argv[++i]));
      time_of_day = static_cast<float>(atof(argv[++i]));
      use_sunsky = true;
    }
    else if(arg == "-r")
    {
      if(i >= argc - 4)
        printUsageAndExit(argv[0]);
      sky_angle = static_cast<float>(atof(argv[++i]));
      sky_up.x = static_cast<float>(atof(argv[++i]));
      sky_up.y = static_cast<float>(atof(argv[++i]));
      sky_up.z = static_cast<float>(atof(argv[++i]));
      use_sunsky = true;
    }
    else if(arg == "-sc")
    {
      if(i >= argc - 1)
        printUsageAndExit(argv[0]);
      scene_scale = static_cast<float>(atof(argv[++i]));
    }
    else if(arg == "-dir")
    {
      if(i >= argc - 2)
        printUsageAndExit(argv[0]);
      theta_i = static_cast<float>(atof(argv[++i]));
      phi_i = static_cast<float>(atof(argv[++i]));
      light_dir = get_light_direction();
    }
    else if(arg == "-rad")
    {
      if(i >= argc - 3)
        printUsageAndExit(argv[0]);
      emission.x = static_cast<float>(atof(argv[++i]));
      emission.y = static_cast<float>(atof(argv[++i]));
      emission.z = static_cast<float>(atof(argv[++i]));
    }
    else if(arg == "-srgb")
    {
      launch_params.use_srgb = true;
    }
    else if(arg == "-bf")
    {
      if(i >= argc - 1)
        printUsageAndExit(argv[0]);
      launch_params.beam_factor = static_cast<float>(atof(argv[++i]));
    }
    else
    {
      filename = argv[i];
      string file_extension;
      size_t idx = filename.find_last_of('.');
      if(idx < filename.length())
      {
        file_extension = filename.substr(idx, filename.length() - idx);
        lower_case_string(file_extension);
      }
      if(file_extension == ".obj")
      {
        filenames.push_back(filename);
        lower_case_string(filenames.back());
      }
      else
      {
        cerr << "Unknown option or not an obj file: '" << arg << "'" << endl;
        printUsageAndExit(argv[0]);
      }
    }
  }
  if(filenames.size() == 0)
    filenames.push_back(string(SAMPLES_DIR) + "/models/cow_vn.obj");
  if(!outfile_selected)
  {
    size_t name_start = filenames.back().find_last_of("\\/") + 1;
    size_t name_end = filenames.back().find_last_of('.');
    outfile = filenames.back().substr(name_start < name_end ? name_start : 0, name_end - name_start) + ".png";
  }

  try
  {
    ObjScene scene(filenames, shadername, cameraname, envname, width, height, light_dir, emission);
    scene.scene_scale = scene_scale;
    if(use_sunsky)
      scene.initSunSky(ordinal_day, time_of_day, latitude, turbidity, clouds, sky_angle, sky_up);
    scene.initScene(bsdf);

    camera_changed = true;
    trackball = new QuatTrackBall(scene.camera.lookat(), length(scene.camera.lookat() - scene.camera.eye()), width, height);

    if(!outfile_selected)
    {
      GLFWwindow* window = sutil::initUI( "render_OptiX", scene.width, scene.height );
      glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
      glfwSetCursorPosCallback    ( window, cursorPosCallback     );
      glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
      glfwSetWindowIconifyCallback( window, windowIconifyCallback );
      glfwSetKeyCallback          ( window, keyCallback           );
      glfwSetScrollCallback       ( window, scrollCallback        );
      glfwSetWindowUserPointer    ( window, &launch_params         );

      //
      // Render loop
      //
      {
        sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, scene.width, scene.height);
        sutil::GLDisplay gl_display;

        std::chrono::steady_clock::time_point spin_timer = std::chrono::steady_clock::now();
        std::chrono::duration<double> state_update_time(0.0);
        std::chrono::duration<double> render_time(0.0);
        std::chrono::duration<double> display_time(0.0);

        do
        {
          auto t0 = std::chrono::steady_clock::now();
          glfwPollEvents();

          if(is_spinning)
          {
            if((t0 - spin_timer).count() > 16666667UL)
            {
              trackball->do_spin();
              camera_changed = true;
              spin_timer = t0;
            }
          }
          if(import_raw)
            importRawImage(outfile, output_buffer, scene);
          else
            updateState(output_buffer, scene);
          auto t1 = std::chrono::steady_clock::now();
          state_update_time += t1 - t0;
          t0 = t1;

          if(progressive || launch_params.subframe_index == 0)
            scene.launchSubframe(output_buffer);
          t1 = std::chrono::steady_clock::now();
          render_time += t1 - t0;
          render_time_split += render_time.count();
          t0 = t1;

          displaySubframe(output_buffer, gl_display, window);
          t1 = std::chrono::steady_clock::now();
          display_time += t1 - t0;

          if(progressive)
            sutil::displayStats( state_update_time, render_time, display_time );

          glfwSwapBuffers(window);

          if(progressive)
            ++launch_params.subframe_index;

          if(save_image)
          {
            if(export_raw)
            {
              exportRawImage(outfile);
              export_raw = false;
            }
            else
              saveImage(output_buffer, outfile);
            save_image = false;
          }
        }
        while( !glfwWindowShouldClose( window ) );
        CUDA_SYNC_CHECK();
      }

      sutil::cleanupUI( window );
    }
    else
    {
		  if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
		  {
			  sutil::initGLFW(); // For GL context
			  sutil::initGL();
		  }

		  sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, scene.width, scene.height);
      if(import_raw)
        importRawImage(outfile, output_buffer, scene);
      else
      {
        load_view("view");
        updateState(output_buffer, scene);
      }

      cout << "Rendering";
      unsigned int dot = max(samples/20u, 20u);
      chrono::duration<double> render_time(0.0);
      auto t0 = chrono::steady_clock::now();
      for(unsigned int i = 0; i < samples; ++i)
      {
        scene.launchSubframe(output_buffer);
        ++launch_params.subframe_index;
        if((i + 1)%dot == 0) cerr << ".";
      }
      auto t1 = chrono::steady_clock::now();
      render_time = t1 - t0;
      render_time_split += render_time.count();
      cout << endl << "Time: " << render_time.count() << endl;

      exportRawImage(outfile);
      saveImage(output_buffer, outfile);
      if(output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
      {
        output_buffer.deletePBO();
        glfwTerminate();
      }
    }
    delete trackball;
  }
  catch( std::exception& e )
  {
      std::cerr << "Caught exception: " << e.what() << "\n";
      return 1;
  }
  return 0;
}
