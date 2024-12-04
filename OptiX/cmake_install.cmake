# Install script for directory: C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/SDK

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/OptiX-Samples")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixBoundValues/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixCallablePrograms/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixCompileWithTasks/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixConsole/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixCurves/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixCustomPrimitive/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixCutouts/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixDenoiser/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixDisplacedMicromesh/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixDynamicGeometry/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixDynamicMaterials/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixHair/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixHello/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixMeshViewer/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixMixSDKs/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixModuleCreateAbort/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixMotionGeometry/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixMultiGPU/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixNVLink/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixOpticalFlow/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixOpacityMicromap/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixPathTracer/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixRaycasting/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixRibbons/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixSimpleMotionBlur/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixSphere/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixTriangle/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixVolumeViewer/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/optixWhitted/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/sutil/cmake_install.cmake")
  include("C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/support/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/frede/Documents/GitHub/RenderingStuff/OptiX/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
