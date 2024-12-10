#
# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

set(sources
  Interface.h
  Medium.h
  Medium.cpp
  load_mpml.h
  load_mpml.cpp
  Parse.h
  Parse.cpp
  XmlParser.h
  XmlParser.cpp
  string_utils.h
  string_utils.cpp
  spectrum2rgb.h
  spectrum2xyz.h
  cdf_bsearch.h
  LorenzMie.h
  LorenzMie.cpp
  apple_juice.h
  apple_juice.cpp
  glass.h
  glass.cpp
  incandescent.h
  incandescent.cpp
  milk.h
  milk.cpp
  orange_juice.h
  orange_juice.cpp
  sun.h
  sun.cpp
  water.h
  water.cpp
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Make the library.
set(optprops_target "optprops")
add_library(${optprops_target} ${sources})

# Use gcc rather than g++ to link if we are linking statically against libgcc_s and libstdc++
if(USING_GNU_C OR USING_GNU_CXX)
  if(GCC_LIBSTDCPP_HACK)
    set_target_properties(${optprops_target} PROPERTIES LINKER_LANGUAGE "C")
    target_link_libraries(${optprops_target} LINK_PRIVATE ${STATIC_LIBSTDCPP})
  endif()
endif()

if(RELEASE_INSTALL_BINARY_SAMPLES AND NOT RELEASE_STATIC_BUILD)
  # If performing a release install, we want to use rpath for our install name.
  # The executables' rpaths will then be set to @executable_path so we can invoke
  # the samples from an arbitrary location and it will still find this library.
  set_target_properties(${optprops_target} PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON
    )
  install(TARGETS ${optprops_target}
    RUNTIME DESTINATION ${SDK_BINARY_INSTALL_DIR}
    LIBRARY DESTINATION ${SDK_BINARY_INSTALL_DIR}
    )
endif()

# Make the list of sources available to the parent directory for installation needs.
set(optprops_sources "${sources}" PARENT_SCOPE)
