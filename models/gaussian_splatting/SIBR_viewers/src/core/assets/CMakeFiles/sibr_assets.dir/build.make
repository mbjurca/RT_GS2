# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1361/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1361/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers

# Include any dependencies generated for this target.
include src/core/assets/CMakeFiles/sibr_assets.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.make

# Include the progress variables for this target.
include src/core/assets/CMakeFiles/sibr_assets.dir/progress.make

# Include the compile flags for this target's objects.
include src/core/assets/CMakeFiles/sibr_assets.dir/flags.make

src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/flags.make
src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o: src/core/assets/ActiveImageFile.cpp
src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o -MF CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o.d -o CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o -c /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/ActiveImageFile.cpp

src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.i"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/ActiveImageFile.cpp > CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.i

src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.s"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/ActiveImageFile.cpp -o CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.s

src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/flags.make
src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o: src/core/assets/CameraRecorder.cpp
src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o -MF CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o.d -o CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o -c /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/CameraRecorder.cpp

src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.i"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/CameraRecorder.cpp > CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.i

src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.s"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/CameraRecorder.cpp -o CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.s

src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/flags.make
src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o: src/core/assets/ImageListFile.cpp
src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o -MF CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o.d -o CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o -c /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/ImageListFile.cpp

src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sibr_assets.dir/ImageListFile.cpp.i"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/ImageListFile.cpp > CMakeFiles/sibr_assets.dir/ImageListFile.cpp.i

src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sibr_assets.dir/ImageListFile.cpp.s"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/ImageListFile.cpp -o CMakeFiles/sibr_assets.dir/ImageListFile.cpp.s

src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/flags.make
src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.o: src/core/assets/InputCamera.cpp
src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.o"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.o -MF CMakeFiles/sibr_assets.dir/InputCamera.cpp.o.d -o CMakeFiles/sibr_assets.dir/InputCamera.cpp.o -c /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/InputCamera.cpp

src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sibr_assets.dir/InputCamera.cpp.i"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/InputCamera.cpp > CMakeFiles/sibr_assets.dir/InputCamera.cpp.i

src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sibr_assets.dir/InputCamera.cpp.s"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/InputCamera.cpp -o CMakeFiles/sibr_assets.dir/InputCamera.cpp.s

src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/flags.make
src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.o: src/core/assets/Resources.cpp
src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.o"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.o -MF CMakeFiles/sibr_assets.dir/Resources.cpp.o.d -o CMakeFiles/sibr_assets.dir/Resources.cpp.o -c /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/Resources.cpp

src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sibr_assets.dir/Resources.cpp.i"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/Resources.cpp > CMakeFiles/sibr_assets.dir/Resources.cpp.i

src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sibr_assets.dir/Resources.cpp.s"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/Resources.cpp -o CMakeFiles/sibr_assets.dir/Resources.cpp.s

src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/flags.make
src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o: src/core/assets/UVUnwrapper.cpp
src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o: src/core/assets/CMakeFiles/sibr_assets.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o -MF CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o.d -o CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o -c /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/UVUnwrapper.cpp

src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.i"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/UVUnwrapper.cpp > CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.i

src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.s"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/UVUnwrapper.cpp -o CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.s

# Object files for target sibr_assets
sibr_assets_OBJECTS = \
"CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o" \
"CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o" \
"CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o" \
"CMakeFiles/sibr_assets.dir/InputCamera.cpp.o" \
"CMakeFiles/sibr_assets.dir/Resources.cpp.o" \
"CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o"

# External object files for target sibr_assets
sibr_assets_EXTERNAL_OBJECTS =

src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/ActiveImageFile.cpp.o
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/CameraRecorder.cpp.o
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/ImageListFile.cpp.o
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/InputCamera.cpp.o
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/Resources.cpp.o
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/UVUnwrapper.cpp.o
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/build.make
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libassimp.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libGLX.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libGLU.so
src/core/assets/libsibr_assets.so: src/core/graphics/libsibr_graphics.so
src/core/assets/libsibr_assets.so: extlibs/xatlas/build/libxatlas.so.1.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libassimp.so
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_gapi.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_stitching.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_alphamat.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_aruco.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_bgsegm.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_bioinspired.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_ccalib.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_dnn_superres.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_dpm.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_highgui.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_face.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_freetype.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_fuzzy.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_hfs.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_img_hash.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_intensity_transform.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_line_descriptor.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_mcc.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_quality.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_rapid.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_reg.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_rgbd.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_saliency.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_stereo.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_structured_light.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_superres.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_optflow.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_surface_matching.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_tracking.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_datasets.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_plot.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_text.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_dnn.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_videostab.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_videoio.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_xfeatures2d.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_ml.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_shape.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_ximgproc.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_video.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_xobjdetect.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_imgcodecs.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_objdetect.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_calib3d.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_features2d.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_flann.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_xphoto.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_photo.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_imgproc.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/local/lib/libopencv_core.so.4.5.0
src/core/assets/libsibr_assets.so: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libpthread.so
src/core/assets/libsibr_assets.so: extlibs/imgui/build/libimgui.a
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libGLX.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libGLU.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
src/core/assets/libsibr_assets.so: src/core/system/libsibr_system.so
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
src/core/assets/libsibr_assets.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
src/core/assets/libsibr_assets.so: extlibs/nativefiledialog/build/libnativefiledialog.a
src/core/assets/libsibr_assets.so: src/core/assets/CMakeFiles/sibr_assets.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library libsibr_assets.so"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sibr_assets.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/core/assets/CMakeFiles/sibr_assets.dir/build: src/core/assets/libsibr_assets.so
.PHONY : src/core/assets/CMakeFiles/sibr_assets.dir/build

src/core/assets/CMakeFiles/sibr_assets.dir/clean:
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets && $(CMAKE_COMMAND) -P CMakeFiles/sibr_assets.dir/cmake_clean.cmake
.PHONY : src/core/assets/CMakeFiles/sibr_assets.dir/clean

src/core/assets/CMakeFiles/sibr_assets.dir/depend:
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/assets/CMakeFiles/sibr_assets.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/core/assets/CMakeFiles/sibr_assets.dir/depend

