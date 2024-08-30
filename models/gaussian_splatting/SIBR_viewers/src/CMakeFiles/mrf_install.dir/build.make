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

# Utility rule file for mrf_install.

# Include any custom commands dependencies for this target.
include src/CMakeFiles/mrf_install.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/mrf_install.dir/progress.make

src/CMakeFiles/mrf_install: extlibs/mrf/mrf/block.h
src/CMakeFiles/mrf_install: extlibs/mrf/mrf/config.h
src/CMakeFiles/mrf_install: extlibs/mrf/mrf/graph.cpp
src/CMakeFiles/mrf_install: extlibs/mrf/mrf/graph.h
src/CMakeFiles/mrf_install: extlibs/mrf/mrf/instances.inc
src/CMakeFiles/mrf_install: extlibs/mrf/mrf/maxflow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "run the installation only for mrf"
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src && /snap/cmake/1361/bin/cmake -DBUILD_TYPE=Release -DCOMPONENT=mrf_install -P /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/cmake_install.cmake

mrf_install: src/CMakeFiles/mrf_install
mrf_install: src/CMakeFiles/mrf_install.dir/build.make
.PHONY : mrf_install

# Rule to build all files generated by this target.
src/CMakeFiles/mrf_install.dir/build: mrf_install
.PHONY : src/CMakeFiles/mrf_install.dir/build

src/CMakeFiles/mrf_install.dir/clean:
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src && $(CMAKE_COMMAND) -P CMakeFiles/mrf_install.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/mrf_install.dir/clean

src/CMakeFiles/mrf_install.dir/depend:
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/CMakeFiles/mrf_install.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/mrf_install.dir/depend

